from __future__ import annotations

import importlib
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import Any, Protocol

from jarvis.voice.contracts import (
    VoiceInputFrame,
    VoiceInputFrameKind,
    utc_now,
)


class VoiceAudioPreprocessingStatus(StrEnum):
    READY = "ready"
    DROPPED = "dropped"
    FAILED = "failed"


class VoiceAudioPreprocessingOperation(StrEnum):
    PROCESS_FRAME = "process_frame"


@dataclass(frozen=True, slots=True)
class VoiceAudioPreprocessingCapabilities:
    echo_cancellation: bool = False
    noise_suppression: bool = False
    auto_gain_control: bool = False
    voice_activity_detection: bool = False
    provider: str = "passthrough"
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class VoiceAudioPreprocessingPolicy:
    require_echo_cancellation: bool = False
    require_noise_suppression: bool = False
    require_auto_gain_control: bool = False
    drop_non_speech: bool = False
    vad_aggressiveness: int = 2
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not 0 <= self.vad_aggressiveness <= 3:
            raise ValueError("vad_aggressiveness must be 0..3.")


@dataclass(frozen=True, slots=True)
class VoiceAudioPreprocessingResult:
    status: VoiceAudioPreprocessingStatus
    operation: VoiceAudioPreprocessingOperation
    frame: VoiceInputFrame | None
    message: str
    created_at: datetime
    capabilities: VoiceAudioPreprocessingCapabilities
    metadata: dict[str, object] = field(default_factory=dict)

    @property
    def succeeded(self) -> bool:
        return self.status in {
            VoiceAudioPreprocessingStatus.READY,
            VoiceAudioPreprocessingStatus.DROPPED,
        }


class VoiceAudioPreprocessingAdapter(Protocol):
    def capabilities(self) -> VoiceAudioPreprocessingCapabilities:
        raise NotImplementedError

    def process_frame(
        self,
        frame: VoiceInputFrame,
        policy: VoiceAudioPreprocessingPolicy,
    ) -> VoiceAudioPreprocessingResult:
        raise NotImplementedError


class PassthroughAudioPreprocessor:
    def capabilities(self) -> VoiceAudioPreprocessingCapabilities:
        return VoiceAudioPreprocessingCapabilities(provider="passthrough")

    def process_frame(
        self,
        frame: VoiceInputFrame,
        policy: VoiceAudioPreprocessingPolicy,
    ) -> VoiceAudioPreprocessingResult:
        capabilities = self.capabilities()
        unavailable = _missing_required_features(
            capabilities=capabilities,
            policy=policy,
        )
        if unavailable:
            return _failed_result(
                frame=frame,
                capabilities=capabilities,
                message="required audio preprocessing feature unavailable",
                metadata={"missing_features": unavailable},
            )
        return VoiceAudioPreprocessingResult(
            status=VoiceAudioPreprocessingStatus.READY,
            operation=VoiceAudioPreprocessingOperation.PROCESS_FRAME,
            frame=frame,
            message="audio frame passed through",
            created_at=utc_now(),
            capabilities=capabilities,
            metadata={"audio_preprocessed": False},
        )


class WebRTCVADAudioPreprocessor:
    """
    Real WebRTC VAD pre-gate.

    This adapter is intentionally honest: the Python `webrtcvad` package gives
    VAD only. It does not provide AEC, NS, or AGC, so those features remain
    unavailable unless a real audio-processing engine is added.
    """

    def __init__(self) -> None:
        self._vad: Any | None = None

    def capabilities(self) -> VoiceAudioPreprocessingCapabilities:
        return VoiceAudioPreprocessingCapabilities(
            voice_activity_detection=True,
            provider="webrtcvad",
        )

    def process_frame(
        self,
        frame: VoiceInputFrame,
        policy: VoiceAudioPreprocessingPolicy,
    ) -> VoiceAudioPreprocessingResult:
        capabilities = self.capabilities()
        unavailable = _missing_required_features(
            capabilities=capabilities,
            policy=policy,
        )
        if unavailable:
            return _failed_result(
                frame=frame,
                capabilities=capabilities,
                message="required audio preprocessing feature unavailable",
                metadata={"missing_features": unavailable},
            )
        if frame.kind != VoiceInputFrameKind.PCM16_MONO:
            return _failed_result(
                frame=frame,
                capabilities=capabilities,
                message="WebRTC VAD requires PCM16 mono frames",
                metadata={"frame_kind": frame.kind.value},
            )
        if frame.duration_ms not in {10, 20, 30}:
            return _failed_result(
                frame=frame,
                capabilities=capabilities,
                message="WebRTC VAD requires 10, 20, or 30ms frames",
                metadata={"duration_ms": frame.duration_ms},
            )
        if frame.sample_rate_hz not in {8_000, 16_000, 32_000, 48_000}:
            return _failed_result(
                frame=frame,
                capabilities=capabilities,
                message="WebRTC VAD sample rate unsupported",
                metadata={"sample_rate_hz": frame.sample_rate_hz},
            )

        vad = self._vad_for_policy(policy)
        try:
            is_speech = bool(
                vad.is_speech(
                    frame.data,
                    frame.sample_rate_hz,
                )
            )
        except Exception as exc:
            return _failed_result(
                frame=frame,
                capabilities=capabilities,
                message="WebRTC VAD failed",
                metadata={"error": str(exc)},
            )

        metadata = {
            **frame.metadata,
            "audio_preprocessed": True,
            "audio_preprocessor": "webrtcvad",
            "webrtc_vad_speech": is_speech,
            "vad_aggressiveness": policy.vad_aggressiveness,
        }
        if not is_speech and policy.drop_non_speech:
            return VoiceAudioPreprocessingResult(
                status=VoiceAudioPreprocessingStatus.DROPPED,
                operation=VoiceAudioPreprocessingOperation.PROCESS_FRAME,
                frame=None,
                message="WebRTC VAD dropped non-speech frame",
                created_at=utc_now(),
                capabilities=capabilities,
                metadata=metadata,
            )

        return VoiceAudioPreprocessingResult(
            status=VoiceAudioPreprocessingStatus.READY,
            operation=VoiceAudioPreprocessingOperation.PROCESS_FRAME,
            frame=VoiceInputFrame(
                frame_id=frame.frame_id,
                session_id=frame.session_id,
                kind=frame.kind,
                sample_rate_hz=frame.sample_rate_hz,
                channels=frame.channels,
                data=frame.data,
                captured_at=frame.captured_at,
                duration_ms=frame.duration_ms,
                metadata=metadata,
            ),
            message="WebRTC VAD inspected audio frame",
            created_at=utc_now(),
            capabilities=capabilities,
            metadata=metadata,
        )

    def _vad_for_policy(self, policy: VoiceAudioPreprocessingPolicy) -> Any:
        if self._vad is None:
            module = importlib.import_module("webrtcvad")
            self._vad = module.Vad(policy.vad_aggressiveness)
        return self._vad


class VoiceAudioPreprocessingRuntime:
    def __init__(
        self,
        *,
        adapter: VoiceAudioPreprocessingAdapter | None = None,
        policy: VoiceAudioPreprocessingPolicy | None = None,
    ) -> None:
        self._adapter = adapter or PassthroughAudioPreprocessor()
        self._policy = policy or VoiceAudioPreprocessingPolicy()
        self._processed_frames = 0
        self._dropped_frames = 0
        self._failed_frames = 0
        self._last_result: VoiceAudioPreprocessingResult | None = None

    def process_frame(self, frame: VoiceInputFrame) -> VoiceAudioPreprocessingResult:
        result = self._adapter.process_frame(frame, self._policy)
        self._processed_frames += 1
        if result.status == VoiceAudioPreprocessingStatus.DROPPED:
            self._dropped_frames += 1
        elif result.status == VoiceAudioPreprocessingStatus.FAILED:
            self._failed_frames += 1
        self._last_result = result
        return result

    def snapshot(self) -> dict[str, object]:
        capabilities = self._adapter.capabilities()
        return {
            "provider": capabilities.provider,
            "processed_frames": self._processed_frames,
            "dropped_frames": self._dropped_frames,
            "failed_frames": self._failed_frames,
            "capabilities": {
                "echo_cancellation": capabilities.echo_cancellation,
                "noise_suppression": capabilities.noise_suppression,
                "auto_gain_control": capabilities.auto_gain_control,
                "voice_activity_detection": capabilities.voice_activity_detection,
            },
            "last_result": (
                None
                if self._last_result is None
                else {
                    "status": self._last_result.status.value,
                    "message": self._last_result.message,
                    **self._last_result.metadata,
                }
            ),
        }


def _missing_required_features(
    *,
    capabilities: VoiceAudioPreprocessingCapabilities,
    policy: VoiceAudioPreprocessingPolicy,
) -> tuple[str, ...]:
    missing: list[str] = []
    if policy.require_echo_cancellation and not capabilities.echo_cancellation:
        missing.append("echo_cancellation")
    if policy.require_noise_suppression and not capabilities.noise_suppression:
        missing.append("noise_suppression")
    if policy.require_auto_gain_control and not capabilities.auto_gain_control:
        missing.append("auto_gain_control")
    return tuple(missing)


def _failed_result(
    *,
    frame: VoiceInputFrame,
    capabilities: VoiceAudioPreprocessingCapabilities,
    message: str,
    metadata: dict[str, object],
) -> VoiceAudioPreprocessingResult:
    return VoiceAudioPreprocessingResult(
        status=VoiceAudioPreprocessingStatus.FAILED,
        operation=VoiceAudioPreprocessingOperation.PROCESS_FRAME,
        frame=frame,
        message=message,
        created_at=utc_now(),
        capabilities=capabilities,
        metadata=metadata,
    )
