from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from threading import Lock

from jarvis.presence.adapters.wake_word_adapter import WakeWordDetection
from jarvis.presence.models import AudioFrame, VoiceActivity, VoiceActivityState
from jarvis.runtime.observability.structured_logger import get_logger


@dataclass(frozen=True, slots=True)
class AudioEnergyFeatures:
    """
    Lightweight PCM audio features for real-time VAD/wake decisions.
    """

    rms: float
    peak: int
    zero_crossing_rate: float
    sample_count: int


@dataclass(frozen=True, slots=True)
class EnergyVoiceActivityConfig:
    """
    Dependency-free energy VAD configuration.

    This is a safe local baseline for real microphone testing. It is not a
    neural VAD, but it gives the Presence runtime a real-time speech gate
    without adding binary model/runtime complexity yet.
    """

    sample_width_bytes: int = 2
    speech_rms_threshold: float = 500.0
    silence_rms_threshold: float = 250.0
    speech_start_frames: int = 2
    speech_end_frames: int = 8
    min_zero_crossing_rate: float = 0.005
    max_zero_crossing_rate: float = 0.35
    adaptive_noise_floor: bool = True
    noise_floor_learning_rate: float = 0.05
    threshold_multiplier: float = 3.0

    def validate(self) -> None:
        if self.sample_width_bytes != 2:
            raise ValueError("Only 16-bit PCM audio is supported.")

        if self.speech_rms_threshold <= 0:
            raise ValueError("speech_rms_threshold must be greater than zero.")

        if self.silence_rms_threshold < 0:
            raise ValueError("silence_rms_threshold cannot be negative.")

        if self.speech_start_frames <= 0:
            raise ValueError("speech_start_frames must be greater than zero.")

        if self.speech_end_frames <= 0:
            raise ValueError("speech_end_frames must be greater than zero.")

        if not 0 <= self.min_zero_crossing_rate <= 1:
            raise ValueError("min_zero_crossing_rate must be between 0 and 1.")

        if not 0 <= self.max_zero_crossing_rate <= 1:
            raise ValueError("max_zero_crossing_rate must be between 0 and 1.")

        if self.min_zero_crossing_rate > self.max_zero_crossing_rate:
            raise ValueError(
                "min_zero_crossing_rate cannot exceed max_zero_crossing_rate."
            )

        if not 0 < self.noise_floor_learning_rate <= 1:
            raise ValueError(
                "noise_floor_learning_rate must be greater than 0 and <= 1."
            )

        if self.threshold_multiplier <= 0:
            raise ValueError("threshold_multiplier must be greater than zero.")


class EnergyVoiceActivityAdapter:
    """
    Dependency-free real-time VAD adapter.

    Responsibilities:
    - analyze AudioFrame PCM energy
    - return VoiceActivity objects
    - maintain a small speech/silence state machine
    - adapt slowly to background noise

    Non-responsibilities:
    - no microphone capture
    - no wake-word phrase recognition
    - no STT
    - no event publishing
    """

    def __init__(
        self,
        *,
        config: EnergyVoiceActivityConfig | None = None,
        name: str = "energy_vad_adapter",
    ) -> None:
        clean_name = name.strip()

        if not clean_name:
            raise ValueError("EnergyVoiceActivityAdapter name cannot be empty.")

        self._config = config or EnergyVoiceActivityConfig()
        self._config.validate()

        self._name = clean_name
        self._lock = Lock()
        self._in_speech = False
        self._speech_frames = 0
        self._silence_frames = 0
        self._noise_floor = self._config.silence_rms_threshold
        self._last_features: AudioEnergyFeatures | None = None

        self._logger = get_logger("presence.energy_vad")

    @property
    def name(self) -> str:
        return self._name

    @property
    def noise_floor(self) -> float:
        with self._lock:
            return self._noise_floor

    @property
    def last_features(self) -> AudioEnergyFeatures | None:
        with self._lock:
            return self._last_features

    def reset(self) -> None:
        with self._lock:
            self._in_speech = False
            self._speech_frames = 0
            self._silence_frames = 0
            self._noise_floor = self._config.silence_rms_threshold
            self._last_features = None

        self._logger.info("energy_vad_reset", adapter=self._name)

    def detect(self, frame: AudioFrame) -> VoiceActivity:
        features = extract_int16_audio_features(frame.audio_data)

        with self._lock:
            self._last_features = features

            speech_threshold = self._speech_threshold_locked()
            speech_like = self._is_speech_like(
                features=features,
                speech_threshold=speech_threshold,
            )

            if not speech_like:
                self._learn_noise_floor_locked(features.rms)

            state = self._next_state_locked(speech_like=speech_like)
            noise_floor = self._noise_floor

        return VoiceActivity(
            frame_id=frame.frame_id,
            state=state,
            is_speech=state
            in {
                VoiceActivityState.SPEECH_STARTED,
                VoiceActivityState.SPEECH_CONTINUING,
            },
            confidence=self._confidence(
                rms=features.rms,
                threshold=speech_threshold,
            ),
            energy=features.rms,
            metadata={
                "adapter": self._name,
                "peak": features.peak,
                "zero_crossing_rate": features.zero_crossing_rate,
                "noise_floor": noise_floor,
                "speech_threshold": speech_threshold,
            },
        )

    def _next_state_locked(self, *, speech_like: bool) -> VoiceActivityState:
        if speech_like:
            self._speech_frames += 1
            self._silence_frames = 0

            if not self._in_speech:
                if self._speech_frames >= self._config.speech_start_frames:
                    self._in_speech = True
                    return VoiceActivityState.SPEECH_STARTED

                return VoiceActivityState.SILENCE

            return VoiceActivityState.SPEECH_CONTINUING

        self._silence_frames += 1
        self._speech_frames = 0

        if self._in_speech:
            if self._silence_frames >= self._config.speech_end_frames:
                self._in_speech = False
                return VoiceActivityState.SPEECH_ENDED

            return VoiceActivityState.SPEECH_CONTINUING

        return VoiceActivityState.SILENCE

    def _is_speech_like(
        self,
        *,
        features: AudioEnergyFeatures,
        speech_threshold: float,
    ) -> bool:
        if features.rms < speech_threshold:
            return False

        return (
            self._config.min_zero_crossing_rate
            <= features.zero_crossing_rate
            <= self._config.max_zero_crossing_rate
        )

    def _speech_threshold_locked(self) -> float:
        if not self._config.adaptive_noise_floor:
            return self._config.speech_rms_threshold

        adaptive_threshold = self._noise_floor * self._config.threshold_multiplier

        return max(self._config.speech_rms_threshold, adaptive_threshold)

    def _learn_noise_floor_locked(self, rms: float) -> None:
        if not self._config.adaptive_noise_floor:
            return

        alpha = self._config.noise_floor_learning_rate
        self._noise_floor = ((1.0 - alpha) * self._noise_floor) + (alpha * rms)

    @staticmethod
    def _confidence(*, rms: float, threshold: float) -> float:
        if threshold <= 0:
            return 0.0

        ratio = rms / threshold

        return max(0.0, min(1.0, ratio / 2.0))


@dataclass(frozen=True, slots=True)
class EnergyWakeWordConfig:
    """
    Lightweight real-audio wake trigger.

    This is an energy gate, not phrase recognition. It is useful for early
    real microphone testing. Later, Porcupine/OpenWakeWord can replace this
    through the same WakeWordAdapter contract.
    """

    rms_threshold: float = 700.0
    required_consecutive_frames: int = 3
    cooldown_frames: int = 20
    min_zero_crossing_rate: float = 0.005
    max_zero_crossing_rate: float = 0.35
    wake_word: str = "jarvis"

    def validate(self) -> None:
        if self.rms_threshold <= 0:
            raise ValueError("rms_threshold must be greater than zero.")

        if self.required_consecutive_frames <= 0:
            raise ValueError("required_consecutive_frames must be greater than zero.")

        if self.cooldown_frames < 0:
            raise ValueError("cooldown_frames cannot be negative.")

        if not self.wake_word.strip():
            raise ValueError("wake_word cannot be empty.")

        if not 0 <= self.min_zero_crossing_rate <= 1:
            raise ValueError("min_zero_crossing_rate must be between 0 and 1.")

        if not 0 <= self.max_zero_crossing_rate <= 1:
            raise ValueError("max_zero_crossing_rate must be between 0 and 1.")

        if self.min_zero_crossing_rate > self.max_zero_crossing_rate:
            raise ValueError(
                "min_zero_crossing_rate cannot exceed max_zero_crossing_rate."
            )


class EnergyWakeWordAdapter:
    """
    Dependency-free real audio wake trigger.

    Existing wake contract:
    - return WakeWordDetection when wake is detected
    - return None when wake is not detected
    """

    def __init__(
        self,
        *,
        config: EnergyWakeWordConfig | None = None,
        name: str = "energy_wake_word_adapter",
    ) -> None:
        clean_name = name.strip()

        if not clean_name:
            raise ValueError("EnergyWakeWordAdapter name cannot be empty.")

        self._config = config or EnergyWakeWordConfig()
        self._config.validate()

        self._name = clean_name
        self._lock = Lock()
        self._consecutive_hits = 0
        self._cooldown_remaining = 0
        self._detections = 0
        self._recent_rms: deque[float] = deque(maxlen=16)
        self._last_features: AudioEnergyFeatures | None = None

        self._logger = get_logger("presence.energy_wake")

    @property
    def name(self) -> str:
        return self._name

    @property
    def detections(self) -> int:
        with self._lock:
            return self._detections

    @property
    def last_features(self) -> AudioEnergyFeatures | None:
        with self._lock:
            return self._last_features

    def reset(self) -> None:
        with self._lock:
            self._consecutive_hits = 0
            self._cooldown_remaining = 0
            self._detections = 0
            self._recent_rms.clear()
            self._last_features = None

        self._logger.info("energy_wake_reset", adapter=self._name)

    def detect(self, frame: AudioFrame) -> WakeWordDetection | None:
        features = extract_int16_audio_features(frame.audio_data)

        with self._lock:
            self._last_features = features
            self._recent_rms.append(features.rms)

            if self._cooldown_remaining > 0:
                self._cooldown_remaining -= 1
                self._consecutive_hits = 0
                return None

            hit = self._is_wake_like(features)

            if hit:
                self._consecutive_hits += 1
            else:
                self._consecutive_hits = 0

            detected = (
                self._consecutive_hits
                >= self._config.required_consecutive_frames
            )

            if not detected:
                return None

            self._detections += 1
            self._cooldown_remaining = self._config.cooldown_frames
            self._consecutive_hits = 0
            confidence = self._confidence(features.rms)

        return WakeWordDetection(
            frame_id=frame.frame_id,
            wake_word=self._config.wake_word,
            confidence=confidence,
            metadata={
                "adapter": self._name,
                "reason": "energy_gate",
                "rms": features.rms,
                "peak": features.peak,
                "zero_crossing_rate": features.zero_crossing_rate,
            },
        )

    def _is_wake_like(self, features: AudioEnergyFeatures) -> bool:
        if features.rms < self._config.rms_threshold:
            return False

        return (
            self._config.min_zero_crossing_rate
            <= features.zero_crossing_rate
            <= self._config.max_zero_crossing_rate
        )

    def _confidence(self, rms: float) -> float:
        ratio = rms / self._config.rms_threshold

        return max(0.0, min(1.0, ratio / 2.0))


def extract_int16_audio_features(audio_data: bytes) -> AudioEnergyFeatures:
    """
    Extract PCM int16 audio features without external dependencies.
    """

    if not audio_data:
        return AudioEnergyFeatures(
            rms=0.0,
            peak=0,
            zero_crossing_rate=0.0,
            sample_count=0,
        )

    usable_length = len(audio_data) - (len(audio_data) % 2)

    if usable_length <= 0:
        return AudioEnergyFeatures(
            rms=0.0,
            peak=0,
            zero_crossing_rate=0.0,
            sample_count=0,
        )

    total_square = 0.0
    peak = 0
    sample_count = 0
    zero_crossings = 0
    previous_sign = 0

    for index in range(0, usable_length, 2):
        sample = int.from_bytes(
            audio_data[index : index + 2],
            byteorder="little",
            signed=True,
        )
        magnitude = abs(sample)

        total_square += float(sample * sample)
        peak = max(peak, magnitude)
        sample_count += 1

        current_sign = 1 if sample >= 0 else -1

        if previous_sign != 0 and current_sign != previous_sign:
            zero_crossings += 1

        previous_sign = current_sign

    if sample_count == 0:
        rms = 0.0
        zero_crossing_rate = 0.0
    else:
        rms = math.sqrt(total_square / sample_count)
        zero_crossing_rate = zero_crossings / sample_count

    return AudioEnergyFeatures(
        rms=rms,
        peak=peak,
        zero_crossing_rate=zero_crossing_rate,
        sample_count=sample_count,
    )