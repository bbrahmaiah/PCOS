from __future__ import annotations

import json
import os
import signal
import socket
import subprocess
import sys
import threading
import time
import webbrowser
from dataclasses import dataclass
from http.server import ThreadingHTTPServer
from pathlib import Path
from types import FrameType
from typing import cast
from urllib import request as http_request

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_VISUAL_CONSOLE_DIR = Path(
    os.environ.get(
        "JARVIS_VISUAL_CONSOLE_DIR",
        str(
            Path.home()
            / "Documents"
            / "Codex"
            / "2026-06-21"
            / "hi"
            / "ai-studio-app"
        ),
    )
)
DEFAULT_VISUAL_NODE_PATH = Path(
    os.environ.get(
        "JARVIS_VISUAL_NODE_PATH",
        str(
            Path.home()
            / ".cache"
            / "codex-runtimes"
            / "codex-primary-runtime"
            / "dependencies"
            / "node"
            / "bin"
            / "node.exe"
        ),
    )
)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from jarvis.runtime import (  # noqa: E402
    JarvisHttpBridgeConfig,
    JarvisHttpBridgeRuntime,
    JarvisRuntimeBindingVerificationMode,
    JarvisRuntimeBindingVerificationReport,
    JarvisRuntimeBindingVerificationStatus,
    JarvisRuntimeBindingVerifier,
    JarvisRuntimeBindingVerifierConfig,
    JarvisStartControlConfig,
    JarvisStartControlResult,
    JarvisStartControlRuntime,
    JarvisStartControlStatus,
    build_connected_start_control_from_plan,
    build_plan_from_import_bindings,
    create_jarvis_http_server,
    read_runtime_binding_imports,
    summarize_binding_report,
)
from jarvis.system import (  # noqa: E402
    PureJarvisPreflightConfig,
    PureJarvisPreflightReport,
    PureJarvisPreflightRuntime,
    PureJarvisPreflightStatus,
    default_pure_jarvis_manifest,
)
from jarvis.voice.audio_preprocessing import (  # noqa: E402
    PassthroughAudioPreprocessor,
    VoiceAudioPreprocessingPolicy,
    VoiceAudioPreprocessingRuntime,
    WebRTCVADAudioPreprocessor,
)
from jarvis.voice.awareness_cognition_bridge import (  # noqa: E402
    VoiceAwarenessCognitionBridge,
)
from jarvis.voice.cognition_response import (  # noqa: E402
    VoiceCognitionLatencyBudget,
    VoiceCognitionPolicy,
    VoiceCognitionRequest,
    VoiceCognitionResponseRuntime,
    VoiceOllamaGeneratorConfig,
    VoiceOllamaResponseGenerator,
)
from jarvis.voice.contracts import (  # noqa: E402
    VoiceRuntimeConfig,
    VoiceTranscript,
    VoiceTranscriptKind,
    make_voice_segment_id,
    make_voice_session_id,
    make_voice_transcript_id,
    utc_now,
)
from jarvis.voice.microphone_capture import (  # noqa: E402
    PyAudioMicrophoneAdapter,
    VoiceMicrophoneCaptureRuntime,
)
from jarvis.voice.reflex_response import (  # noqa: E402
    VoiceReflexResponsePolicy,
    VoiceReflexResponseRuntime,
)
from jarvis.voice.runtime_launcher import (  # noqa: E402
    VoiceRuntimeLauncher,
    VoiceRuntimeLauncherConfig,
    VoiceRuntimeLauncherSnapshot,
)
from jarvis.voice.session_loop import (  # noqa: E402
    VoiceSessionLoopPolicy,
    VoiceSessionLoopRuntime,
    VoiceSessionPlayback,
    VoiceSessionTTS,
)
from jarvis.voice.spine_validation import (  # noqa: E402
    VoiceSpineValidationRuntime,
    VoiceSpineValidationStatus,
)
from jarvis.voice.stt_calibration import load_voice_capture_profile  # noqa: E402
from jarvis.voice.stt_runtime import VoiceSTTPolicy, VoiceSTTRuntime  # noqa: E402
from jarvis.voice.transcript_attention_gate import (  # noqa: E402
    TranscriptAttentionGatePolicy,
)
from jarvis.voice.tts_runtime import (  # noqa: E402
    VoiceTTSPolicy,
    VoiceTTSRuntime,
)
from jarvis.voice.voice_activity import (  # noqa: E402
    VoiceActivityPolicy,
    VoiceActivityRuntime,
)
from jarvis.voice.windows_audible_playback import WindowsAudiblePlayback  # noqa: E402


@dataclass(frozen=True, slots=True)
class PureConnectedJarvisLauncherConfig:
    bindings_path: Path = PROJECT_ROOT / "config" / "runtime_bindings.env"
    piper_executable_path: Path = PROJECT_ROOT / ".venv" / "Scripts" / "piper.exe"
    piper_model_path: Path = (
        PROJECT_ROOT / "models" / "piper" / "en_US-lessac-medium.onnx"
    )
    piper_config_path: Path = (
        PROJECT_ROOT / "models" / "piper" / "en_US-lessac-medium.onnx.json"
    )
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.2:3b"
    require_ollama: bool = True
    source_fingerprint_paths: tuple[Path, ...] = (
        PROJECT_ROOT / "scripts" / "run_connected_jarvis.py",
        PROJECT_ROOT / "jarvis" / "voice" / "microphone_capture.py",
        PROJECT_ROOT / "jarvis" / "voice" / "voice_activity.py",
        PROJECT_ROOT / "jarvis" / "voice" / "stt_runtime.py",
        PROJECT_ROOT / "jarvis" / "voice" / "session_loop.py",
        PROJECT_ROOT / "jarvis" / "voice" / "reflex_response.py",
        PROJECT_ROOT / "jarvis" / "voice" / "transcript_attention_gate.py",
        PROJECT_ROOT / "jarvis" / "voice" / "cognition_response.py",
        PROJECT_ROOT / "jarvis" / "voice" / "windows_audible_playback.py",
        PROJECT_ROOT / "jarvis" / "system" / "pure_jarvis_contract.py",
        PROJECT_ROOT / "jarvis" / "system" / "pure_jarvis_preflight.py",
    )
    verify_factory_dry_run: bool = False
    allow_degraded_start: bool = False
    recover_on_degraded_health: bool = True
    health_interval_seconds: float = 1.0
    max_consecutive_health_failures: int = 3
    voice_idle_sleep_seconds: float = 0.01
    diagnostic_mode: str = "off"
    diagnostic_every_health_checks: int = 5
    reflex_responses_enabled: bool = True
    reflex_min_confidence: float = 0.70
    require_wake_word_for_companion_speech: bool = True
    min_words_without_wake: int = 2
    min_words_when_attention_active: int = 1
    run_startup_self_test: bool = False
    startup_self_test_mode: str = "off"
    play_startup_audio_probe: bool = False
    synthesize_startup_tts_probe: bool = False
    startup_audio_probe_text: str = "Audio channel ready."
    speak_ready_on_start: bool = False
    live_voice_events_enabled: bool = True
    voice_profile_enabled: bool = True
    voice_profile_path: Path = PROJECT_ROOT / "config" / "voice_profile.json"
    background_ollama_warmup: bool = True
    background_ollama_warmup_timeout_seconds: int = 20
    ollama_readiness_probe_timeout_seconds: float = 0.75
    ollama_keep_alive: str = "20m"
    ollama_prewarm_on_start: bool = False
    ollama_timeout_seconds: int = 8
    ollama_max_sentences: int = 2
    ollama_temperature: float = 0.25
    ollama_num_predict: int = 96
    ollama_stream_response: bool = True
    stt_partial_model_name: str = "base.en"
    stt_final_model_name: str = "small"
    stt_partial_beam_size: int = 1
    stt_final_beam_size: int = 1
    stt_min_partial_confidence: float = 0.40
    stt_min_final_confidence: float = 0.70
    stt_min_transcript_chars: int = 3
    stt_max_no_speech_prob: float = 0.55
    stt_max_compression_ratio: float = 2.35
    stt_min_avg_logprob: float = -1.15
    stt_prewarm_on_prepare: bool = True
    background_stt_warmup: bool = False
    background_tts_warmup: bool = False
    voice_max_silence_ms: int = 900
    partial_transcript_every_frames: int = 6
    vad_min_energy: float = 1_100.0
    vad_speech_start_ratio: float = 5.0
    vad_start_trigger_frames: int = 5
    vad_min_speech_ms: int = 400
    vad_end_silence_ms: int = 650
    vad_max_segment_ms: int = 15_000
    vad_noise_adaptation_rate: float = 0.05
    audio_preprocessor: str = "webrtcvad"
    audio_preprocessor_drop_non_speech: bool = False
    require_echo_cancellation: bool = False
    require_noise_suppression: bool = False
    require_auto_gain_control: bool = False
    webrtc_vad_aggressiveness: int = 2
    tts_max_chars_per_chunk: int = 90
    tts_max_total_chars: int = 720
    tts_timeout_seconds: int = 35
    tts_target_first_chunk_ms: int = 350
    visual_console_enabled: bool = True
    visual_bridge_enabled: bool = True
    visual_console_dir: Path = DEFAULT_VISUAL_CONSOLE_DIR
    visual_console_node_path: Path = DEFAULT_VISUAL_NODE_PATH
    visual_console_auto_open: bool = False
    visual_console_url_marker_path: Path = (
        DEFAULT_VISUAL_CONSOLE_DIR / "visual_console.url"
    )
    visual_console_host: str = "127.0.0.1"
    visual_console_port: int = 3000
    visual_bridge_host: str = "127.0.0.1"
    visual_bridge_port: int = 8765
    visual_bridge_start_timeout_seconds: float = 4.0
    visual_console_start_timeout_seconds: float = 2.0
    wait_for_voice_listening_before_ready: bool = True
    voice_ready_timeout_seconds: float = 45.0
    run_ten_turn_spine_gate: bool = False

    def __post_init__(self) -> None:
        if self.health_interval_seconds <= 0:
            raise ValueError("health_interval_seconds must be positive.")
        if self.max_consecutive_health_failures < 1:
            raise ValueError("max_consecutive_health_failures must be positive.")
        if self.voice_idle_sleep_seconds < 0:
            raise ValueError("voice_idle_sleep_seconds cannot be negative.")
        if self.diagnostic_mode not in {"off", "quiet", "normal", "verbose"}:
            raise ValueError(
                "diagnostic_mode must be off, quiet, normal, or verbose."
            )
        if self.diagnostic_every_health_checks < 1:
            raise ValueError("diagnostic_every_health_checks must be positive.")
        if not 0.0 <= self.reflex_min_confidence <= 1.0:
            raise ValueError("reflex_min_confidence must be 0..1.")
        if self.min_words_without_wake < 1:
            raise ValueError("min_words_without_wake must be positive.")
        if self.min_words_when_attention_active < 1:
            raise ValueError("min_words_when_attention_active must be positive.")
        if self.startup_self_test_mode not in {"fast", "deep", "off"}:
            raise ValueError("startup_self_test_mode must be fast, deep, or off.")
        if (
            self.run_startup_self_test
            and self.startup_self_test_mode != "off"
            and not self.startup_audio_probe_text.strip()
        ):
            raise ValueError("startup_audio_probe_text cannot be empty.")
        if self.voice_profile_enabled and not str(self.voice_profile_path).strip():
            raise ValueError("voice_profile_path cannot be empty.")
        if self.background_ollama_warmup_timeout_seconds <= 0:
            raise ValueError(
                "background_ollama_warmup_timeout_seconds must be positive."
            )
        if self.ollama_readiness_probe_timeout_seconds <= 0:
            raise ValueError(
                "ollama_readiness_probe_timeout_seconds must be positive."
            )
        if not self.ollama_base_url.strip():
            raise ValueError("ollama_base_url cannot be empty.")
        if not self.ollama_model.strip():
            raise ValueError("ollama_model cannot be empty.")
        if not self.ollama_keep_alive.strip():
            raise ValueError("ollama_keep_alive cannot be empty.")
        if self.ollama_timeout_seconds <= 0:
            raise ValueError("ollama_timeout_seconds must be positive.")
        if self.ollama_max_sentences < 1:
            raise ValueError("ollama_max_sentences must be positive.")
        if not 0.0 <= self.ollama_temperature <= 2.0:
            raise ValueError("ollama_temperature must be 0..2.")
        if self.ollama_num_predict < 1:
            raise ValueError("ollama_num_predict must be positive.")
        if not self.stt_partial_model_name.strip():
            raise ValueError("stt_partial_model_name cannot be empty.")
        if not self.stt_final_model_name.strip():
            raise ValueError("stt_final_model_name cannot be empty.")
        if self.stt_partial_beam_size < 1:
            raise ValueError("stt_partial_beam_size must be positive.")
        if self.stt_final_beam_size < 1:
            raise ValueError("stt_final_beam_size must be positive.")
        if not 0.0 <= self.stt_min_partial_confidence <= 1.0:
            raise ValueError("stt_min_partial_confidence must be 0..1.")
        if not 0.0 <= self.stt_min_final_confidence <= 1.0:
            raise ValueError("stt_min_final_confidence must be 0..1.")
        if self.stt_min_transcript_chars < 1:
            raise ValueError("stt_min_transcript_chars must be positive.")
        if not 0.0 <= self.stt_max_no_speech_prob <= 1.0:
            raise ValueError("stt_max_no_speech_prob must be 0..1.")
        if self.stt_max_compression_ratio <= 0:
            raise ValueError("stt_max_compression_ratio must be positive.")
        if self.voice_max_silence_ms <= 0:
            raise ValueError("voice_max_silence_ms must be positive.")
        if self.partial_transcript_every_frames < 1:
            raise ValueError("partial_transcript_every_frames must be positive.")
        if self.vad_min_energy <= 0:
            raise ValueError("vad_min_energy must be positive.")
        if self.vad_speech_start_ratio <= 1.0:
            raise ValueError("vad_speech_start_ratio must be greater than 1.")
        if self.vad_start_trigger_frames < 1:
            raise ValueError("vad_start_trigger_frames must be positive.")
        if self.vad_min_speech_ms <= 0:
            raise ValueError("vad_min_speech_ms must be positive.")
        if self.vad_end_silence_ms <= 0:
            raise ValueError("vad_end_silence_ms must be positive.")
        if self.vad_max_segment_ms <= self.vad_min_speech_ms:
            raise ValueError("vad_max_segment_ms must exceed vad_min_speech_ms.")
        if not 0.0 < self.vad_noise_adaptation_rate <= 1.0:
            raise ValueError("vad_noise_adaptation_rate must be 0..1.")
        if self.audio_preprocessor not in {"off", "webrtcvad"}:
            raise ValueError("audio_preprocessor must be off or webrtcvad.")
        if not 0 <= self.webrtc_vad_aggressiveness <= 3:
            raise ValueError("webrtc_vad_aggressiveness must be 0..3.")
        if self.tts_max_chars_per_chunk < 40:
            raise ValueError("tts_max_chars_per_chunk must be at least 40.")
        if self.tts_max_total_chars < self.tts_max_chars_per_chunk:
            raise ValueError(
                "tts_max_total_chars must exceed tts_max_chars_per_chunk."
            )
        if self.tts_timeout_seconds <= 0:
            raise ValueError("tts_timeout_seconds must be positive.")
        if self.tts_target_first_chunk_ms <= 0:
            raise ValueError("tts_target_first_chunk_ms must be positive.")
        if self.visual_console_enabled and not str(self.visual_console_dir).strip():
            raise ValueError("visual_console_dir cannot be empty.")
        if self.visual_console_enabled and not str(
            self.visual_console_node_path
        ).strip():
            raise ValueError("visual_console_node_path cannot be empty.")
        if self.visual_console_enabled and not str(
            self.visual_console_url_marker_path
        ).strip():
            raise ValueError("visual_console_url_marker_path cannot be empty.")
        if not self.visual_console_host.strip():
            raise ValueError("visual_console_host cannot be empty.")
        if not 0 < self.visual_console_port < 65_536:
            raise ValueError("visual_console_port must be 1..65535.")
        if not self.visual_bridge_host.strip():
            raise ValueError("visual_bridge_host cannot be empty.")
        if not 0 < self.visual_bridge_port < 65_536:
            raise ValueError("visual_bridge_port must be 1..65535.")
        if self.visual_bridge_start_timeout_seconds <= 0:
            raise ValueError("visual_bridge_start_timeout_seconds must be positive.")
        if self.visual_console_start_timeout_seconds <= 0:
            raise ValueError("visual_console_start_timeout_seconds must be positive.")
        if self.voice_ready_timeout_seconds <= 0:
            raise ValueError("voice_ready_timeout_seconds must be positive.")
        if not self.source_fingerprint_paths:
            raise ValueError("source_fingerprint_paths cannot be empty.")


class PureConnectedJarvisLauncher:
    """
    Production connected JARVIS launcher.

    This is the single live entrypoint for the personal cognitive OS:
    Phase 1-9 bindings -> Start Control -> supervised Step 51 voice loop.

    It does not generate conversational speech. Spoken words still come only
    from microphone -> STT -> awareness -> cognition/response boundary -> TTS.
    """

    def __init__(self, *, config: PureConnectedJarvisLauncherConfig) -> None:
        self._config = config
        self._start_control: JarvisStartControlRuntime | None = None
        self._voice_launcher: VoiceRuntimeLauncher | None = None
        self._stt_runtime: VoiceSTTRuntime | None = None
        self._tts_runtime: VoiceTTSRuntime | None = None
        self._stop_requested = False
        self._health_checks = 0
        self._last_health_status: JarvisStartControlStatus | None = None
        self._last_voice_event_signature: tuple[object, ...] | None = None
        self._visual_bridge_server: ThreadingHTTPServer | None = None
        self._visual_bridge_thread: threading.Thread | None = None
        self._visual_console_process: subprocess.Popen[bytes] | None = None
        self._visual_console_port = config.visual_console_port
        self._visual_console_opened = False

    def run(self) -> int:
        try:
            preflight_report = self._run_preflight()
            self._print_preflight_report(preflight_report)
            if preflight_report.status == PureJarvisPreflightStatus.FAILED:
                return 1

            self._verify_pure_manifest()
            self._verify_bindings(
                mode=JarvisRuntimeBindingVerificationMode.RESOLVE_ONLY
            )
            if self._config.verify_factory_dry_run:
                self._verify_bindings(
                    mode=JarvisRuntimeBindingVerificationMode.FACTORY_DRY_RUN
                )

            if (
                self._config.run_startup_self_test
                and self._config.startup_self_test_mode != "off"
            ):
                self._run_startup_self_test()
            if self._config.run_ten_turn_spine_gate:
                self._run_ten_turn_spine_gate()

            self._start_control = self._build_start_control()
            start_result = self._start_control.start_all()
            self._print_start_result(start_result)

            if start_result.status == JarvisStartControlStatus.FAILED:
                return 1
            if (
                start_result.status == JarvisStartControlStatus.DEGRADED
                and not self._config.allow_degraded_start
            ):
                return 1

            self._start_background_stt_warmup()
            self._start_background_tts_warmup()
            self._start_background_ollama_warmup()
            self._start_visual_console_stack()
            self._wait_for_voice_listening_if_configured()
            self._print_ready_status()
            self._print_ready_voice_policy()

            return self._supervise_until_stopped()

        except KeyboardInterrupt:
            self.request_stop()
            return 0
        except Exception as exc:
            print(f"[JARVIS_DIAGNOSTIC] fatal_error={exc}", flush=True)
            return 1
        finally:
            self._shutdown()

    def request_stop(self) -> None:
        self._stop_requested = True
        if self._voice_launcher is not None:
            self._voice_launcher.request_stop()

    def _verify_pure_manifest(self) -> None:
        manifest = default_pure_jarvis_manifest()
        if not manifest.ready_for_pure_runtime:
            missing = ", ".join(
                kind.value for kind in manifest.missing_required_capabilities()
            )
            raise RuntimeError(
                "Pure JARVIS manifest is not ready; missing=" + missing
            )

    def _verify_bindings(
        self,
        *,
        mode: JarvisRuntimeBindingVerificationMode,
    ) -> JarvisRuntimeBindingVerificationReport:
        verifier = JarvisRuntimeBindingVerifier(
            config=JarvisRuntimeBindingVerifierConfig(
                bindings_path=self._config.bindings_path,
                mode=mode,
                metadata={"entrypoint": "pure_connected_jarvis"},
            )
        )
        report = verifier.verify()
        if report.status != JarvisRuntimeBindingVerificationStatus.PASSED:
            print(summarize_binding_report(report), flush=True)
            raise RuntimeError(f"runtime binding verification failed: {mode.value}")
        return report

    def _build_start_control(self) -> JarvisStartControlRuntime:
        import_bindings = read_runtime_binding_imports(self._config.bindings_path)
        self._voice_launcher = self._build_voice_launcher()
        plan = build_plan_from_import_bindings(
            import_bindings=import_bindings,
            voice_launcher=self._voice_launcher,
        )
        return build_connected_start_control_from_plan(
            plan,
            config=JarvisStartControlConfig(
                metadata={
                    "entrypoint": "pure_connected_jarvis",
                    "fixed_conversational_responses_allowed": False,
                }
            ),
        )

    def _build_voice_launcher(self) -> VoiceRuntimeLauncher:
        return VoiceRuntimeLauncher(
            session_loop=self._build_voice_session_loop(),
            config=VoiceRuntimeLauncherConfig(
                run_forever=True,
                run_daily_driver_gate=False,
                idle_sleep_seconds=self._config.voice_idle_sleep_seconds,
                stop_on_session_failure=True,
                metadata={
                    "entrypoint": "pure_connected_jarvis",
                    "response_origin": "cognition_response_boundary",
                    "fixed_conversational_responses_allowed": False,
                },
            ),
        )

    def _build_voice_runtime_config(
        self,
        *,
        microphone_device_index: int | None,
    ) -> VoiceRuntimeConfig:
        return VoiceRuntimeConfig(
            user_label="Tony",
            assistant_name="JARVIS",
            wake_word="jarvis",
            max_silence_ms=self._config.voice_max_silence_ms,
            metadata={
                "entrypoint": "pure_connected_jarvis",
                "microphone_device_index": microphone_device_index,
            },
        )

    def _build_microphone_runtime(
        self,
        *,
        voice_config: VoiceRuntimeConfig,
        microphone_device_index: int | None,
    ) -> VoiceMicrophoneCaptureRuntime:
        return VoiceMicrophoneCaptureRuntime(
            config=voice_config,
            adapter=PyAudioMicrophoneAdapter(
                device_index=microphone_device_index,
            ),
        )

    def _build_stt_runtime(
        self,
        *,
        voice_config: VoiceRuntimeConfig,
    ) -> VoiceSTTRuntime:
        runtime = VoiceSTTRuntime(
            config=voice_config,
            policy=VoiceSTTPolicy(
                partial_model_name=self._config.stt_partial_model_name,
                final_model_name=self._config.stt_final_model_name,
                partial_beam_size=self._config.stt_partial_beam_size,
                final_beam_size=self._config.stt_final_beam_size,
                min_partial_confidence=self._config.stt_min_partial_confidence,
                min_final_confidence=self._config.stt_min_final_confidence,
                min_transcript_chars=self._config.stt_min_transcript_chars,
                max_no_speech_prob=self._config.stt_max_no_speech_prob,
                max_compression_ratio=self._config.stt_max_compression_ratio,
                min_avg_logprob=self._config.stt_min_avg_logprob,
                prewarm_on_prepare=self._config.stt_prewarm_on_prepare,
                metadata={
                    "profile": "connected_jarvis_live_room",
                    "speed_tuned": True,
                    "voice_profile_enabled": self._config.voice_profile_enabled,
                    "voice_profile_path": str(self._config.voice_profile_path),
                },
            ),
        )
        self._stt_runtime = runtime
        return runtime

    def _build_audio_preprocessor(self) -> VoiceAudioPreprocessingRuntime:
        adapter = (
            WebRTCVADAudioPreprocessor()
            if self._config.audio_preprocessor == "webrtcvad"
            else PassthroughAudioPreprocessor()
        )
        return VoiceAudioPreprocessingRuntime(
            adapter=adapter,
            policy=VoiceAudioPreprocessingPolicy(
                require_echo_cancellation=self._config.require_echo_cancellation,
                require_noise_suppression=self._config.require_noise_suppression,
                require_auto_gain_control=self._config.require_auto_gain_control,
                drop_non_speech=self._config.audio_preprocessor_drop_non_speech,
                vad_aggressiveness=self._config.webrtc_vad_aggressiveness,
                metadata={"profile": "connected_jarvis_audio_preprocessing"},
            ),
        )

    def _build_voice_session_loop(self) -> VoiceSessionLoopRuntime:
        microphone_device_index = _env_int("JARVIS_MIC_DEVICE_INDEX")
        voice_config = self._build_voice_runtime_config(
            microphone_device_index=microphone_device_index
        )
        return VoiceSessionLoopRuntime(
            config=voice_config,
            microphone=self._build_microphone_runtime(
                voice_config=voice_config,
                microphone_device_index=microphone_device_index,
            ),
            audio_preprocessor=self._build_audio_preprocessor(),
            vad=VoiceActivityRuntime(
                policy=VoiceActivityPolicy(
                    min_energy=self._config.vad_min_energy,
                    speech_start_ratio=self._config.vad_speech_start_ratio,
                    start_trigger_frames=self._config.vad_start_trigger_frames,
                    min_speech_ms=self._config.vad_min_speech_ms,
                    end_silence_ms=self._config.vad_end_silence_ms,
                    max_segment_ms=self._config.vad_max_segment_ms,
                    noise_adaptation_rate=self._config.vad_noise_adaptation_rate,
                    metadata={"profile": "connected_jarvis_live_room"},
                )
            ),
            stt=self._build_stt_runtime(voice_config=voice_config),
            cognition=self._build_companion_cognition_runtime(),
            reflex=VoiceReflexResponseRuntime(
                policy=VoiceReflexResponsePolicy(
                    enabled=self._config.reflex_responses_enabled,
                    min_confidence=self._config.reflex_min_confidence,
                    wake_words=("jarvis", "jervis", "jarves"),
                    metadata={"profile": "connected_jarvis_reflex_lane"},
                )
            ),
            tts=cast(VoiceSessionTTS, self._build_real_piper_tts_runtime()),
            playback=cast(
                VoiceSessionPlayback,
                self._build_startup_playback_runtime(),
            ),
            transcript_gate_policy=self._build_companion_attention_policy(),
            policy=VoiceSessionLoopPolicy(
                partial_transcript_every_frames=(
                    self._config.partial_transcript_every_frames
                ),
                idle_sleep_seconds=self._config.voice_idle_sleep_seconds,
                barge_in_completion_silence_ms=220,
                metadata={"profile": "connected_jarvis_low_latency"},
            ),
        )

    def _build_companion_attention_policy(self) -> TranscriptAttentionGatePolicy:
        return TranscriptAttentionGatePolicy(
            wake_words=("jarvis", "jervis", "jarves"),
            min_final_confidence=0.55,
            min_words_without_wake=self._config.min_words_without_wake,
            min_words_when_attention_active=(
                self._config.min_words_when_attention_active
            ),
            require_attention_for_promoted_partials=(
                self._config.require_wake_word_for_companion_speech
            ),
            require_wake_or_attention=(
                self._config.require_wake_word_for_companion_speech
            ),
        )

    def _build_companion_cognition_runtime(self) -> VoiceAwarenessCognitionBridge:
        return VoiceAwarenessCognitionBridge(
            cognition=VoiceCognitionResponseRuntime(
                response_generator=VoiceOllamaResponseGenerator(
                    config=VoiceOllamaGeneratorConfig(
                        base_url=self._config.ollama_base_url,
                        model=self._config.ollama_model,
                        timeout_seconds=self._config.ollama_timeout_seconds,
                        max_sentences=self._config.ollama_max_sentences,
                        temperature=self._config.ollama_temperature,
                        num_predict=self._config.ollama_num_predict,
                        keep_alive=self._config.ollama_keep_alive,
                        prewarm_on_prepare=self._config.ollama_prewarm_on_start,
                        stream_response=self._config.ollama_stream_response,
                        metadata={"profile": "connected_jarvis_fast_voice"},
                    )
                ),
                policy=VoiceCognitionPolicy(
                    min_dialogue_confidence=0.55,
                    max_response_sentences=self._config.ollama_max_sentences,
                    require_wake_word_when_sleeping=(
                        self._config.require_wake_word_for_companion_speech
                    ),
                    metadata={"profile": "connected_jarvis_companion"},
                ),
                latency_budget=VoiceCognitionLatencyBudget(
                    response_budget_ms=1_500,
                    total_budget_ms=2_000,
                    metadata={"profile": "connected_jarvis_fast_voice"},
                ),
            )
        )

    def _build_real_piper_tts_runtime(self) -> VoiceTTSRuntime:
        if self._tts_runtime is not None:
            return self._tts_runtime

        if not self._config.piper_executable_path.exists():
            raise RuntimeError(
                f"Piper executable missing: {self._config.piper_executable_path}"
            )
        if not self._config.piper_model_path.exists():
            raise RuntimeError(
                f"Piper model missing: {self._config.piper_model_path}"
            )
        if not self._config.piper_config_path.exists():
            raise RuntimeError(
                f"Piper config missing: {self._config.piper_config_path}"
            )
        runtime = VoiceTTSRuntime(
            policy=VoiceTTSPolicy(
                voice_name="en_US-lessac-medium",
                piper_executable=str(self._config.piper_executable_path),
                piper_model_path=self._config.piper_model_path,
                piper_config_path=self._config.piper_config_path,
                max_chars_per_chunk=self._config.tts_max_chars_per_chunk,
                max_total_chars=self._config.tts_max_total_chars,
                timeout_seconds=self._config.tts_timeout_seconds,
                target_first_chunk_ms=self._config.tts_target_first_chunk_ms,
                metadata={"profile": "connected_jarvis_fast_voice"},
            )
        )
        self._tts_runtime = runtime
        return runtime

    def _build_startup_playback_runtime(self) -> WindowsAudiblePlayback:
        return WindowsAudiblePlayback()

    def _run_startup_self_test(self) -> None:
        started = time.perf_counter()
        mode = self._config.startup_self_test_mode
        print(f"[JARVIS_SELF_TEST] status=starting mode={mode}", flush=True)
        microphone_device_index = _env_int("JARVIS_MIC_DEVICE_INDEX")
        voice_config = self._build_voice_runtime_config(
            microphone_device_index=microphone_device_index
        )
        response_text = "skipped_fast_mode"

        try:
            self._self_test_microphone(
                voice_config=voice_config,
                microphone_device_index=microphone_device_index,
            )
            self._self_test_stt(voice_config=voice_config)
            chunks = self._self_test_tts(
                synthesize_audio=(
                    mode == "deep"
                    or self._config.play_startup_audio_probe
                    or self._config.synthesize_startup_tts_probe
                )
            )
            if mode == "deep":
                response_text = self._self_test_cognition()
            if self._config.play_startup_audio_probe:
                self._self_test_playback(chunks=chunks)
        except Exception as exc:
            print(f"[JARVIS_SELF_TEST] status=failed error={exc}", flush=True)
            raise

        latency_ms = (time.perf_counter() - started) * 1000.0
        print(
            "[JARVIS_SELF_TEST] "
            f"status=passed mode={mode} latency_ms={latency_ms:.1f} "
            f"cognition_probe={_clip_diagnostic_text(response_text, limit=96)}",
            flush=True,
        )

    def _run_ten_turn_spine_gate(self) -> None:
        started = time.perf_counter()
        print("[JARVIS_SPINE_GATE] status=starting turns=10", flush=True)
        session = self._build_voice_session_loop()
        try:
            report = VoiceSpineValidationRuntime(session=session).run()
        finally:
            stop = getattr(session, "stop", None)
            if callable(stop):
                stop()

        latency_ms = (time.perf_counter() - started) * 1000.0
        print(
            "[JARVIS_SPINE_GATE] "
            f"status={report.status.value} passed_turns={report.passed_turns} "
            f"total_turns={len(report.turns)} latency_ms={latency_ms:.1f} "
            f"message={_clip_diagnostic_text(report.message)}",
            flush=True,
        )
        if report.status != VoiceSpineValidationStatus.PASSED:
            raise RuntimeError(f"ten-turn voice spine gate failed: {report.message}")

    def _self_test_microphone(
        self,
        *,
        voice_config: VoiceRuntimeConfig,
        microphone_device_index: int | None,
    ) -> None:
        microphone = self._build_microphone_runtime(
            voice_config=voice_config,
            microphone_device_index=microphone_device_index,
        )
        try:
            start = microphone.start()
            self._require_self_test(
                "microphone_start",
                start.succeeded,
                _result_message(start),
                {
                    "status": _result_status_value(start),
                    "device": getattr(getattr(start, "device", None), "name", None),
                },
            )
            capture = microphone.capture_once()
            self._require_self_test(
                "microphone_capture",
                capture.succeeded and capture.frame is not None,
                _result_message(capture),
                {
                    "status": _result_status_value(capture),
                    "bytes": len(capture.frame.data) if capture.frame else 0,
                },
            )
        finally:
            microphone.stop()

    def _self_test_stt(self, *, voice_config: VoiceRuntimeConfig) -> None:
        stt = self._build_stt_runtime(voice_config=voice_config)
        result = stt.prepare()
        self._require_self_test(
            "stt_prepare",
            _result_status_value(result) == "ready",
            _result_message(result),
            {
                "status": _result_status_value(result),
                "partial_model": self._config.stt_partial_model_name,
                "final_model": self._config.stt_final_model_name,
                "partial_beam_size": self._config.stt_partial_beam_size,
                "final_beam_size": self._config.stt_final_beam_size,
                "prewarm_on_prepare": self._config.stt_prewarm_on_prepare,
                "error": result.metadata.get("error"),
            },
        )

    def _self_test_cognition(self) -> str:
        cognition = self._build_companion_cognition_runtime()
        cognition.prepare(user_label="Tony", assistant_name="JARVIS")
        transcript = VoiceTranscript(
            transcript_id=make_voice_transcript_id(),
            session_id=make_voice_session_id(),
            segment_id=make_voice_segment_id(),
            kind=VoiceTranscriptKind.FINAL,
            text="startup readiness check",
            confidence=0.99,
            created_at=utc_now(),
            metadata={"source": "startup_self_test"},
        )
        result = cognition.think_from_transcript(
            VoiceCognitionRequest(
                transcript=transcript,
                user_label="Tony",
                assistant_name="JARVIS",
            )
        )
        response_text = result.response.text if result.response is not None else ""
        self._require_self_test(
            "ollama_cognition_probe",
            bool(response_text.strip()),
            result.message,
            {
                "status": result.status.value,
                "safety": result.safety.value,
                "response": _clip_diagnostic_text(response_text, limit=96),
            },
        )
        return response_text

    def _self_test_tts(self, *, synthesize_audio: bool) -> tuple[object, ...]:
        tts = self._build_real_piper_tts_runtime()
        prepared = tts.prepare()
        self._require_self_test(
            "piper_prepare",
            _result_status_value(prepared) == "ready",
            _result_message(prepared),
            {
                "status": _result_status_value(prepared),
                "error": prepared.metadata.get("error"),
            },
        )
        if not synthesize_audio:
            return ()

        synthesized = tts.synthesize_text(
            text=self._config.startup_audio_probe_text,
            session_id=make_voice_session_id(),
            metadata={"source": "startup_self_test"},
        )
        self._require_self_test(
            "piper_synthesize",
            bool(synthesized.chunks),
            synthesized.message,
            {
                "status": synthesized.status.value,
                "chunks": len(synthesized.chunks),
                "bytes": sum(len(chunk.audio) for chunk in synthesized.chunks),
            },
        )
        return cast(tuple[object, ...], synthesized.chunks)

    def _self_test_playback(self, *, chunks: tuple[object, ...]) -> None:
        playback = self._build_startup_playback_runtime()
        prepared = playback.prepare()
        self._require_self_test(
            "windows_playback_prepare",
            _result_succeeded(prepared),
            _result_message(prepared),
            {"status": _result_status_value(prepared)},
        )
        enqueued = playback.enqueue_chunks(chunks)
        self._require_self_test(
            "windows_playback_enqueue",
            _result_succeeded(enqueued),
            _result_message(enqueued),
            {"status": _result_status_value(enqueued)},
        )
        played = playback.play_all()
        self._require_self_test(
            "windows_playback_play",
            _result_succeeded(played),
            _result_message(played),
            {"status": _result_status_value(played)},
        )
        self._wait_for_async_playback_probe(playback=playback, played=played)

    def _wait_for_async_playback_probe(
        self,
        *,
        playback: object,
        played: object,
    ) -> None:
        metadata = getattr(played, "metadata", {})
        if not isinstance(metadata, dict) or metadata.get("async_playback") is not True:
            return

        wait_seconds = 2.0
        estimated_duration_ms = metadata.get("estimated_duration_ms")
        if isinstance(estimated_duration_ms, int | float):
            wait_seconds += min(5.0, max(0.0, float(estimated_duration_ms) / 1000.0))

        snapshot = getattr(playback, "snapshot", None)
        if not callable(snapshot):
            return

        deadline = time.perf_counter() + wait_seconds
        while time.perf_counter() < deadline:
            current_snapshot = snapshot()
            if (
                _result_status_value(current_snapshot) != "playing"
                and getattr(current_snapshot, "current_playback", None) is None
            ):
                return
            time.sleep(0.02)

    def _print_ready_voice_policy(self) -> None:
        if self._config.speak_ready_on_start:
            raise RuntimeError(
                "Fixed startup speech is disabled. Ready speech must be "
                "generated through cognition, not launcher text."
            )
        print(
            "[JARVIS_READY_VOICE] "
            "status=disabled reason=no_fixed_startup_speech",
            flush=True,
        )

    def _require_self_test(
        self,
        check: str,
        passed: bool,
        message: str,
        metadata: dict[str, object] | None = None,
    ) -> None:
        status = "passed" if passed else "failed"
        print(
            "[JARVIS_SELF_TEST] "
            f"check={check} status={status} "
            f"message={_clip_diagnostic_text(message, limit=120)} "
            f"metadata={metadata or {}}",
            flush=True,
        )
        if not passed:
            raise RuntimeError(f"startup self-test failed: {check}: {message}")

    def _start_background_stt_warmup(self) -> None:
        if not self._config.background_stt_warmup:
            return
        if self._stt_runtime is None:
            return

        thread = threading.Thread(
            target=self._run_background_stt_warmup,
            name="jarvis_stt_warmup",
            daemon=True,
        )
        thread.start()

    def _run_background_stt_warmup(self) -> None:
        started = time.perf_counter()
        stt = self._stt_runtime
        if stt is None:
            return

        try:
            result = stt.warm_models()
        except Exception as exc:
            latency_ms = (time.perf_counter() - started) * 1000.0
            print(
                "[JARVIS_WARMUP] "
                f"subsystem=stt status=failed latency_ms={latency_ms:.1f} "
                f"error={_clip_diagnostic_text(exc)}",
                flush=True,
            )
            return

        latency_ms = (time.perf_counter() - started) * 1000.0
        status = "passed" if _result_status_value(result) == "ready" else "degraded"
        print(
            "[JARVIS_WARMUP] "
            f"subsystem=stt status={status} latency_ms={latency_ms:.1f} "
            "partial_model="
            f"{_clip_diagnostic_text(self._config.stt_partial_model_name)} "
            f"final_model={_clip_diagnostic_text(self._config.stt_final_model_name)} "
            f"message={_clip_diagnostic_text(_result_message(result))}",
            flush=True,
        )

    def _start_background_tts_warmup(self) -> None:
        if not self._config.background_tts_warmup:
            return
        if self._tts_runtime is None:
            return

        thread = threading.Thread(
            target=self._run_background_tts_warmup,
            name="jarvis_tts_warmup",
            daemon=True,
        )
        thread.start()

    def _run_background_tts_warmup(self) -> None:
        started = time.perf_counter()
        tts = self._tts_runtime
        if tts is None:
            return

        try:
            result = tts.synthesize_text(
                text="Ready.",
                session_id=make_voice_session_id(),
                metadata={"source": "background_tts_warmup"},
            )
        except Exception as exc:
            latency_ms = (time.perf_counter() - started) * 1000.0
            print(
                "[JARVIS_WARMUP] "
                f"subsystem=tts status=failed latency_ms={latency_ms:.1f} "
                f"error={_clip_diagnostic_text(exc)}",
                flush=True,
            )
            return

        latency_ms = (time.perf_counter() - started) * 1000.0
        snapshot = tts.snapshot()
        status = "passed" if result.succeeded else "degraded"
        error = result.metadata.get("error") or snapshot.last_error
        print(
            "[JARVIS_WARMUP] "
            f"subsystem=tts status={status} latency_ms={latency_ms:.1f} "
            f"chunks={len(result.chunks)} "
            f"first_chunk_ms={_clip_diagnostic_number(result.first_chunk_latency_ms)} "
            f"message={_clip_diagnostic_text(result.message)} "
            f"error={_clip_diagnostic_text(error)}",
            flush=True,
        )

    def _start_background_ollama_warmup(self) -> None:
        if not self._config.background_ollama_warmup:
            return
        if not self._config.require_ollama:
            return

        thread = threading.Thread(
            target=self._run_background_ollama_warmup,
            name="jarvis_ollama_warmup",
            daemon=True,
        )
        thread.start()

    def _run_background_ollama_warmup(self) -> None:
        started = time.perf_counter()
        payload = {
            "model": self._config.ollama_model,
            "prompt": "Ready.",
            "stream": False,
            "keep_alive": self._config.ollama_keep_alive,
            "options": {
                "temperature": 0.0,
                "num_predict": 1,
            },
        }
        request = http_request.Request(
            url=f"{self._config.ollama_base_url.rstrip('/')}/api/generate",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with http_request.urlopen(
                request,
                timeout=self._config.background_ollama_warmup_timeout_seconds,
            ) as response:
                response.read()
        except Exception as exc:
            latency_ms = (time.perf_counter() - started) * 1000.0
            print(
                "[JARVIS_WARMUP] "
                f"status=degraded latency_ms={latency_ms:.1f} "
                f"error={_clip_diagnostic_text(exc)}",
                flush=True,
            )
            return

        latency_ms = (time.perf_counter() - started) * 1000.0
        print(
            "[JARVIS_WARMUP] "
            f"status=passed latency_ms={latency_ms:.1f} "
            f"model={_clip_diagnostic_text(self._config.ollama_model)}",
            flush=True,
        )

    def _wait_for_voice_listening_if_configured(self) -> None:
        if not self._config.wait_for_voice_listening_before_ready:
            return
        if self._voice_launcher is None:
            return

        print(
            "[JARVIS_STARTING] "
            "status=warming_voice_loop "
            "message='loading speech, cognition, and voice before ready'",
            flush=True,
        )
        deadline = time.perf_counter() + self._config.voice_ready_timeout_seconds
        last_status: object = None
        while time.perf_counter() < deadline:
            snapshot = self._voice_launcher.snapshot()
            session = snapshot.session_snapshot
            if session is not None:
                last_status = session.status.value
                if session.status.value in {
                    "listening",
                    "user_speaking",
                    "speaking",
                }:
                    return
                if session.status.value in {"failed", "stopped"}:
                    print(
                        "[JARVIS_STARTING] "
                        f"status=degraded voice_status={session.status.value} "
                        f"error={_clip_diagnostic_text(session.last_error)}",
                        flush=True,
                    )
                    return
            time.sleep(0.1)

        print(
            "[JARVIS_STARTING] "
            "status=degraded "
            f"voice_status={_clip_diagnostic_text(last_status)} "
            "message='voice loop did not report listening before timeout'",
            flush=True,
        )

    def _start_visual_console_stack(self) -> None:
        if not (
            self._config.visual_bridge_enabled
            or self._config.visual_console_enabled
        ):
            return

        if self._config.visual_bridge_enabled:
            self._start_visual_bridge()
        if self._config.visual_console_enabled:
            self._start_visual_console()

    def _start_visual_bridge(self) -> None:
        if self._visual_bridge_server is not None:
            return

        bridge_config = JarvisHttpBridgeConfig(
            host=self._config.visual_bridge_host,
            port=self._config.visual_bridge_port,
            user_label="Tony",
            assistant_name="JARVIS",
            ollama_base_url=self._config.ollama_base_url,
            ollama_model=self._config.ollama_model,
            ollama_timeout_seconds=self._config.ollama_timeout_seconds,
            ollama_max_sentences=self._config.ollama_max_sentences,
            ollama_temperature=self._config.ollama_temperature,
            ollama_num_predict=self._config.ollama_num_predict,
            ollama_keep_alive=self._config.ollama_keep_alive,
            ollama_prewarm_on_start=False,
            ollama_stream_response=self._config.ollama_stream_response,
            metadata={
                "entrypoint": "pure_connected_jarvis_visual_bridge",
                "fixed_conversational_responses_allowed": False,
            },
        )
        runtime = JarvisHttpBridgeRuntime(config=bridge_config)
        try:
            server = create_jarvis_http_server(
                config=bridge_config,
                runtime=runtime,
            )
        except OSError as exc:
            health = self._probe_http_json(
                f"{self._visual_bridge_base_url()}/api/health",
                timeout_seconds=self._config.visual_bridge_start_timeout_seconds,
            )
            if health is not None and health.get("fakeFallbackEnabled") is False:
                print(
                    "[JARVIS_VISUAL] "
                    f"bridge=external url={self._visual_bridge_base_url()} "
                    "fakeFallback=false",
                    flush=True,
                )
                return
            print(
                "[JARVIS_VISUAL] "
                f"bridge=degraded url={self._visual_bridge_base_url()} "
                f"error={_clip_diagnostic_text(exc)}",
                flush=True,
            )
            return

        server.daemon_threads = True
        thread = threading.Thread(
            target=server.serve_forever,
            kwargs={"poll_interval": 0.1},
            name="jarvis_visual_http_bridge",
            daemon=True,
        )
        self._visual_bridge_server = server
        self._visual_bridge_thread = thread
        thread.start()
        print(
            "[JARVIS_VISUAL] "
            f"bridge=ready url={self._visual_bridge_base_url()} "
            "source=real_jarvis_runtime fakeFallback=false",
            flush=True,
        )

    def _start_visual_console(self) -> None:
        if self._visual_console_process is not None:
            if self._visual_console_process.poll() is None:
                return
            self._visual_console_process = None

        selected_port = self._select_visual_console_port()
        if selected_port is None:
            self._clear_visual_console_url_marker()
            print(
                "[JARVIS_VISUAL] "
                "console=disabled reason=no_free_visual_console_port "
                f"start_port={self._config.visual_console_port}",
                flush=True,
            )
            return
        if selected_port == 0:
            return

        self._visual_console_port = selected_port
        console_url = self._visual_console_url_for_port(selected_port)

        console_dir = self._config.visual_console_dir
        server_entry = console_dir / "dist" / "server.cjs"
        if not server_entry.exists():
            self._clear_visual_console_url_marker()
            print(
                "[JARVIS_VISUAL] "
                "console=disabled reason=missing_console_build "
                f"path={_clip_diagnostic_text(server_entry)}",
                flush=True,
            )
            return

        node_command = self._resolve_visual_node_command()
        env = os.environ.copy()
        env["JARVIS_REAL_BACKEND_URL"] = self._visual_bridge_base_url()
        env["JARVIS_ENABLE_GEMINI_AUXILIARY"] = env.get(
            "JARVIS_ENABLE_GEMINI_AUXILIARY",
            "false",
        )
        env["NODE_ENV"] = env.get("NODE_ENV", "production")
        env["PORT"] = str(selected_port)

        stdout_path = console_dir / "server.out.log"
        stderr_path = console_dir / "server.err.log"
        try:
            with stdout_path.open("ab") as stdout_log, stderr_path.open(
                "ab"
            ) as stderr_log:
                process = subprocess.Popen(
                    [node_command, str(server_entry)],
                    cwd=console_dir,
                    env=env,
                    stdout=stdout_log,
                    stderr=stderr_log,
                )
        except OSError as exc:
            self._clear_visual_console_url_marker()
            print(
                "[JARVIS_VISUAL] "
                f"console=failed url={console_url} "
                f"error={_clip_diagnostic_text(exc)}",
                flush=True,
            )
            return

        self._visual_console_process = process
        deadline = (
            time.perf_counter() + self._config.visual_console_start_timeout_seconds
        )
        while time.perf_counter() < deadline:
            exit_code = process.poll()
            if exit_code is not None:
                self._visual_console_process = None
                self._clear_visual_console_url_marker()
                print(
                    "[JARVIS_VISUAL] "
                    f"console=failed url={console_url} exit_code={exit_code} "
                    f"logs={_clip_diagnostic_text(stderr_path)}",
                    flush=True,
                )
                return
            health = self._probe_http_json(
                f"{console_url}/api/health",
                timeout_seconds=0.4,
            )
            if health is not None:
                self._publish_visual_console_url(console_url, state="ready")
                print(
                    "[JARVIS_VISUAL] "
                    f"console=ready url={console_url} "
                    f"backend={self._visual_bridge_base_url()} "
                    "fakeFallback=false",
                    flush=True,
                )
                return
            time.sleep(0.2)

        self._publish_visual_console_url(console_url, state="starting")
        print(
            "[JARVIS_VISUAL] "
            f"console=starting url={console_url} "
            f"backend={self._visual_bridge_base_url()} "
            f"logs={_clip_diagnostic_text(stderr_path)}",
            flush=True,
        )

    def _stop_visual_console_stack(self) -> None:
        process = self._visual_console_process
        if process is not None:
            was_running = process.poll() is None
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=5.0)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=2.0)
            print(
                "[JARVIS_VISUAL] "
                f"console=stopped reason="
                f"{'launcher_shutdown' if was_running else 'process_exited'} "
                f"exit_code={process.poll()}",
                flush=True,
            )
            self._visual_console_process = None

        self._clear_visual_console_url_marker()
        server = self._visual_bridge_server
        if server is not None:
            server.shutdown()
            server.server_close()
            thread = self._visual_bridge_thread
            if thread is not None:
                thread.join(timeout=2.0)
            print("[JARVIS_VISUAL] bridge=stopped", flush=True)
            self._visual_bridge_server = None
            self._visual_bridge_thread = None

    def _visual_bridge_base_url(self) -> str:
        return (
            f"http://{self._config.visual_bridge_host}:"
            f"{self._config.visual_bridge_port}"
        )

    def _visual_console_base_url(self) -> str:
        return self._visual_console_url_for_port(self._visual_console_port)

    def _visual_console_url_for_port(self, port: int) -> str:
        return f"http://{self._config.visual_console_host}:{port}"

    def _select_visual_console_port(self) -> int | None:
        for port in _visual_port_candidates(self._config.visual_console_port):
            console_url = self._visual_console_url_for_port(port)
            existing_health = self._probe_http_json(
                f"{console_url}/api/health",
                timeout_seconds=0.5,
            )
            if (
                existing_health is not None
                and existing_health.get("fakeFallbackEnabled") is False
            ):
                self._visual_console_port = port
                self._publish_visual_console_url(console_url, state="external")
                print(
                    "[JARVIS_VISUAL] "
                    f"console=external url={console_url} "
                    f"backend={self._visual_bridge_base_url()} fakeFallback=false",
                    flush=True,
                )
                return 0

            if _tcp_port_open(self._config.visual_console_host, port):
                print(
                    "[JARVIS_VISUAL] "
                    f"console=port_busy port={port} "
                    "reason=occupied_by_non_jarvis_process",
                    flush=True,
                )
                continue

            return port
        return None

    def _publish_visual_console_url(self, console_url: str, *, state: str) -> None:
        self._write_visual_console_url_marker(console_url, state=state)
        if not self._config.visual_console_auto_open:
            return
        if self._visual_console_opened:
            return

        try:
            opened = webbrowser.open(console_url, new=2, autoraise=True)
        except Exception as exc:
            print(
                "[JARVIS_VISUAL] "
                f"console_browser=failed url={console_url} "
                f"error={_clip_diagnostic_text(exc)}",
                flush=True,
            )
            return

        self._visual_console_opened = True
        print(
            "[JARVIS_VISUAL] "
            f"console_browser=opened url={console_url} state={state} "
            f"opened={opened}",
            flush=True,
        )

    def _write_visual_console_url_marker(
        self,
        console_url: str,
        *,
        state: str,
    ) -> None:
        marker_path = self._config.visual_console_url_marker_path
        try:
            marker_path.parent.mkdir(parents=True, exist_ok=True)
            marker_path.write_text(
                f"{console_url}\nstate={state}\n",
                encoding="utf-8",
            )
        except OSError as exc:
            print(
                "[JARVIS_VISUAL] "
                f"console_url_marker=failed path={_clip_diagnostic_text(marker_path)} "
                f"error={_clip_diagnostic_text(exc)}",
                flush=True,
            )

    def _clear_visual_console_url_marker(self) -> None:
        marker_path = self._config.visual_console_url_marker_path
        try:
            marker_path.unlink(missing_ok=True)
        except OSError:
            return

    def _resolve_visual_node_command(self) -> str:
        configured = self._config.visual_console_node_path
        if configured.exists() or not configured.is_absolute():
            return str(configured)
        return "node"

    def _probe_http_json(
        self,
        url: str,
        *,
        timeout_seconds: float,
    ) -> dict[str, object] | None:
        try:
            with http_request.urlopen(url, timeout=timeout_seconds) as response:
                payload = response.read(64 * 1024)
        except Exception:
            return None

        try:
            decoded = json.loads(payload.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            return None
        if not isinstance(decoded, dict):
            return None
        return cast(dict[str, object], decoded)

    def _supervise_until_stopped(self) -> int:
        if self._start_control is None:
            raise RuntimeError("start control was not built")

        consecutive_failures = 0

        while not self._stop_requested:
            time.sleep(self._config.health_interval_seconds)
            result = self._start_control.health()
            self._health_checks += 1
            self._print_health_if_needed(result)

            if result.status == JarvisStartControlStatus.FAILED:
                consecutive_failures += 1
                if consecutive_failures >= self._config.max_consecutive_health_failures:
                    return 1
                self._recover_if_allowed()
                continue

            if result.status == JarvisStartControlStatus.DEGRADED:
                consecutive_failures = 0
                self._recover_if_allowed()
                continue

            consecutive_failures = 0

        return 0

    def _recover_if_allowed(self) -> None:
        if not self._config.recover_on_degraded_health:
            return
        if self._start_control is None:
            return
        result = self._start_control.recover()
        print(
            "[JARVIS_DIAGNOSTIC] "
            f"recovery_status={result.status.value} "
            f"reason={result.reason}",
            flush=True,
        )

    def _shutdown(self) -> None:
        try:
            if self._start_control is not None:
                result = self._start_control.stop_all()
                print(
                    "[JARVIS_DIAGNOSTIC] "
                    f"shutdown_status={result.status.value} "
                    f"reason={result.reason}",
                    flush=True,
                )
        finally:
            self._stop_visual_console_stack()

    def _print_start_result(self, result: JarvisStartControlResult) -> None:
        print(
            "[JARVIS_DIAGNOSTIC] "
            f"start_status={result.status.value} "
            f"organs={len(result.health)} "
            f"reason={result.reason}",
            flush=True,
        )
        self._print_non_running_organs(result)

    def _print_ready_status(self) -> None:
        ollama_status = self._probe_ollama_readiness()
        print(
            "[JARVIS_READY] "
            "status=listening "
            f"diagnostics={self._config.diagnostic_mode} "
            f"reflex={'enabled' if self._config.reflex_responses_enabled else 'off'} "
            f"streaming={'enabled' if self._config.ollama_stream_response else 'off'} "
            f"stt={_clip_diagnostic_text(self._config.stt_partial_model_name)}/"
            f"{_clip_diagnostic_text(self._config.stt_final_model_name)} "
            f"model={_clip_diagnostic_text(self._config.ollama_model)} "
            f"visual={'enabled' if self._config.visual_console_enabled else 'off'} "
            f"console={_clip_diagnostic_text(self._visual_console_base_url())} "
            f"bridge={_clip_diagnostic_text(self._visual_bridge_base_url())}",
            flush=True,
        )
        print(
            "[JARVIS_OLLAMA] "
            f"status={_clip_diagnostic_text(ollama_status['status'])} "
            f"model={_clip_diagnostic_text(self._config.ollama_model)} "
            f"installed={_clip_diagnostic_text(ollama_status['installed'])} "
            f"loaded={_clip_diagnostic_text(ollama_status['loaded'])} "
            f"message={_clip_diagnostic_text(ollama_status['message'])}",
            flush=True,
        )

    def _probe_ollama_readiness(self) -> dict[str, object]:
        base_url = self._config.ollama_base_url.rstrip("/")
        timeout_seconds = self._config.ollama_readiness_probe_timeout_seconds
        tags = self._probe_http_json(
            f"{base_url}/api/tags",
            timeout_seconds=timeout_seconds,
        )
        if tags is None:
            return {
                "status": "unreachable",
                "installed": False,
                "loaded": False,
                "message": "Ollama service did not answer the fast probe.",
            }

        models = tags.get("models", ())
        installed = _ollama_model_list_contains(
            models,
            self._config.ollama_model,
        )

        ps = self._probe_http_json(
            f"{base_url}/api/ps",
            timeout_seconds=timeout_seconds,
        )
        loaded = False
        if ps is not None:
            loaded = _ollama_model_list_contains(
                ps.get("models", ()),
                self._config.ollama_model,
            )

        if not installed:
            return {
                "status": "model_missing",
                "installed": False,
                "loaded": loaded,
                "message": "Ollama answered, but the configured model was not listed.",
            }
        if loaded:
            return {
                "status": "ready",
                "installed": True,
                "loaded": True,
                "message": "Ollama model is already loaded.",
            }
        return {
            "status": "cold",
            "installed": True,
            "loaded": False,
            "message": "Ollama is reachable; first cognition call may load the model.",
        }

    def _run_preflight(self) -> PureJarvisPreflightReport:
        return PureJarvisPreflightRuntime(
            config=PureJarvisPreflightConfig(
                bindings_path=self._config.bindings_path,
                piper_executable_path=self._config.piper_executable_path,
                piper_model_path=self._config.piper_model_path,
                piper_config_path=self._config.piper_config_path,
                source_fingerprint_paths=self._config.source_fingerprint_paths,
                ollama_base_url=self._config.ollama_base_url,
                ollama_model=self._config.ollama_model,
                require_ollama=self._config.require_ollama,
                metadata={"entrypoint": "pure_connected_jarvis"},
            )
        ).run()

    def _print_preflight_report(
        self,
        report: PureJarvisPreflightReport,
    ) -> None:
        print(
            "[JARVIS_PREFLIGHT] "
            f"status={report.status.value} "
            f"fingerprint={report.source_fingerprint[:16]} "
            f"passed={report.passed_count} "
            f"warnings={report.warning_count} "
            f"failed={report.failed_count}",
            flush=True,
        )
        for check in report.checks:
            if check.status == PureJarvisPreflightStatus.PASSED:
                continue
            print(
                "[JARVIS_PREFLIGHT_CHECK] "
                f"kind={check.kind.value} "
                f"status={check.status.value} "
                f"message={check.message} "
                f"metadata={check.metadata}",
                flush=True,
            )

    def _print_health_if_needed(self, result: JarvisStartControlResult) -> None:
        mode = self._config.diagnostic_mode
        status_changed = result.status != self._last_health_status
        periodic = (
            self._health_checks % self._config.diagnostic_every_health_checks
        ) == 0
        unhealthy = result.status in {
            JarvisStartControlStatus.DEGRADED,
            JarvisStartControlStatus.FAILED,
        }

        if mode == "off" and not unhealthy:
            if self._config.live_voice_events_enabled:
                voice_snapshot = self._voice_snapshot_for_diagnostics(
                    compact=True
                )
                if (
                    self._voice_event_changed(voice_snapshot)
                    and self._voice_event_is_important(voice_snapshot)
                ):
                    self._print_voice_snapshot(
                        compact=True,
                        snapshot=voice_snapshot,
                    )
            self._last_health_status = result.status
            return

        show_periodic = mode in {"normal", "verbose"} and periodic
        show_health = (mode != "off" and status_changed) or show_periodic or unhealthy
        voice_snapshot = self._voice_snapshot_for_diagnostics(
            compact=mode in {"off", "quiet"}
        )
        voice_changed = (
            mode != "off" and self._voice_event_changed(voice_snapshot)
        )
        show_voice = (
            (unhealthy and mode != "off")
            or (mode == "verbose" and (status_changed or periodic))
            or (mode == "normal" and status_changed)
            or (
                mode == "quiet"
                and voice_changed
                and self._voice_event_is_important(voice_snapshot)
            )
        )

        if show_health:
            print(
                "[JARVIS_DIAGNOSTIC] "
                f"health_status={result.status.value} "
                f"organs={len(result.health)} "
                f"reason={result.reason}",
                flush=True,
            )
            self._print_non_running_organs(result)
        if show_voice:
            self._print_voice_snapshot(
                compact=mode == "quiet",
                snapshot=voice_snapshot,
            )

        self._last_health_status = result.status

    def _voice_snapshot_for_diagnostics(
        self,
        *,
        compact: bool,
    ) -> VoiceRuntimeLauncherSnapshot | None:
        if self._voice_launcher is None:
            return None
        if compact:
            live_snapshot = getattr(self._voice_launcher, "live_snapshot", None)
            if callable(live_snapshot):
                snapshot = live_snapshot()
                if isinstance(snapshot, VoiceRuntimeLauncherSnapshot):
                    return snapshot
        return self._voice_launcher.snapshot()

    def _voice_event_is_important(
        self,
        snapshot: VoiceRuntimeLauncherSnapshot | None,
    ) -> bool:
        if snapshot is None:
            return False
        session = snapshot.session_snapshot
        if session is None:
            return False

        event = getattr(session.last_event, "value", None)
        status = session.status.value
        gate_reason = session.metadata.get("last_gate_reason")
        cognitive_route_action = session.metadata.get("cognitive_route_action")
        perception_intent_state = session.metadata.get("perception_intent_state")
        return (
            status in {"failed", "degraded", "interrupted", "stopped"}
            or event
            in {
                "speech_started",
                "partial_transcript",
                "final_transcript",
                "response_ready",
                "tts_ready",
                "playback_finished",
                "barge_in_interrupted",
                "stopped",
                "error",
            }
            or gate_reason == "requires_wake_or_active_attention"
            or perception_intent_state in {"interruption", "ready_for_routing"}
            or cognitive_route_action in {"wait_for_stability", "clarify"}
        )

    def _print_non_running_organs(self, result: JarvisStartControlResult) -> None:
        for health in result.health:
            if health.status.value == "running":
                continue
            print(
                "[JARVIS_ORGAN] "
                f"kind={health.kind.value} "
                f"status={health.status.value} "
                f"name={health.name} "
                f"message={health.message} "
                f"metadata={health.metadata}",
                flush=True,
            )

    def _voice_event_changed(
        self,
        snapshot: VoiceRuntimeLauncherSnapshot | None,
    ) -> bool:
        if snapshot is None:
            return False
        session = snapshot.session_snapshot
        if session is None:
            return False
        signature = (
            session.status.value,
            session.assistant_speaking,
            session.final_transcripts,
            session.responses,
            session.interruptions,
            getattr(session.last_event, "value", None),
            session.last_transcript_text,
            session.last_response_text,
        )
        if signature == self._last_voice_event_signature:
            return False
        self._last_voice_event_signature = signature
        return True

    def _print_voice_snapshot(
        self,
        *,
        compact: bool = False,
        snapshot: VoiceRuntimeLauncherSnapshot | None = None,
    ) -> None:
        if snapshot is None:
            if self._voice_launcher is None:
                return
            snapshot = self._voice_launcher.snapshot()
        session = snapshot.session_snapshot
        if session is None:
            return
        live_spine = snapshot.metadata.get("live_spine")
        live_spine_status = None
        live_spine_message = None
        live_spine_fsm_violations = None
        if isinstance(live_spine, dict):
            live_spine_status = live_spine.get("status")
            live_spine_message = live_spine.get("message")
            live_spine_fsm_violations = live_spine.get("fsm_violations")
        last_message = session.metadata.get("last_result_message")
        gate_reason = session.metadata.get("last_gate_reason")
        gate_text = session.metadata.get("last_gate_text")
        cognitive_route_action = session.metadata.get("cognitive_route_action")
        cognitive_route_state = session.metadata.get("cognitive_route_state")
        cognitive_route_reason = session.metadata.get("cognitive_route_reason")
        perception_intent_state = session.metadata.get("perception_intent_state")
        perception_reason = session.metadata.get("perception_reason")
        perception_confidence = session.metadata.get("perception_confidence")
        perception_stability = session.metadata.get("perception_stability")
        cognition_status = session.metadata.get("last_cognition_status")
        cognition_message = session.metadata.get("last_cognition_message")
        cognition_safety = session.metadata.get("last_cognition_safety")
        runner_status = session.metadata.get("last_runner_status")
        runner_reason = session.metadata.get("last_runner_reason")
        wake_decision = session.metadata.get("last_wake_decision")
        wake_reason = session.metadata.get("last_wake_reason")
        playback_status = session.metadata.get("playback_status")
        playback_current_status = session.metadata.get("playback_current_status")
        playback_queued = session.metadata.get("playback_queued")
        playback_stopped_count = session.metadata.get("playback_stopped_count")
        playback_last_latency_ms = session.metadata.get("playback_last_latency_ms")
        playback_first_audio_latency_ms = session.metadata.get(
            "playback_first_audio_latency_ms"
        )
        microphone_device_name = session.metadata.get("microphone_device_name")
        microphone_device_index = session.metadata.get("microphone_device_index")
        vad_last_energy = session.metadata.get("vad_last_energy")
        vad_noise_floor = session.metadata.get("vad_noise_floor")
        stt_last_text = session.metadata.get("stt_last_text")
        stt_empty_results = session.metadata.get("stt_empty_results")
        stt_low_confidence_results = session.metadata.get(
            "stt_low_confidence_results"
        )
        reflex_responses = session.metadata.get("reflex_responses")
        last_reflex_kind = session.metadata.get("last_reflex_kind")
        last_reflex_reason = session.metadata.get("last_reflex_reason")
        if compact:
            print(
                "[JARVIS_VOICE_EVENT] "
                f"status={session.status.value} "
                f"finals={session.final_transcripts} "
                f"responses={session.responses} "
                f"reflexes={_clip_diagnostic_text(reflex_responses)} "
                f"interruptions={session.interruptions} "
                f"last_event={getattr(session.last_event, 'value', None)} "
                f"assistant_speaking={session.assistant_speaking} "
                f"last_text={_clip_diagnostic_text(session.last_transcript_text)} "
                f"last_response={_clip_diagnostic_text(session.last_response_text)} "
                f"gate_reason={_clip_diagnostic_text(gate_reason)} "
                f"perception={_clip_diagnostic_text(perception_intent_state)} "
                f"perception_stability="
                f"{_clip_diagnostic_number(perception_stability)} "
                f"spine={_clip_diagnostic_text(live_spine_status)} "
                f"fsm_violations={_clip_diagnostic_text(live_spine_fsm_violations)} "
                f"route={_clip_diagnostic_text(cognitive_route_action)} "
                f"route_state={_clip_diagnostic_text(cognitive_route_state)} "
                f"reflex={_clip_diagnostic_text(last_reflex_kind)} "
                f"reflex_reason={_clip_diagnostic_text(last_reflex_reason)} "
                f"latency_ms={_clip_diagnostic_number(session.last_latency_ms)} "
                f"playback={_clip_diagnostic_text(playback_status)} "
                f"first_audio={_clip_diagnostic_number(playback_first_audio_latency_ms)}",
                flush=True,
            )
            return
        print(
            "[JARVIS_VOICE] "
            f"status={session.status.value} "
            f"running={session.running} "
            f"frames={session.captured_frames} "
            f"segments={session.speech_segments} "
            f"finals={session.final_transcripts} "
            f"responses={session.responses} "
            f"last_event={getattr(session.last_event, 'value', None)} "
            f"assistant_speaking={session.assistant_speaking} "
            f"buffered={session.buffered_segment_frames} "
            f"tts={session.tts_outputs} "
            f"played={session.played_outputs} "
            f"interruptions={session.interruptions} "
            f"last_error={session.last_error} "
            f"last_text={_clip_diagnostic_text(session.last_transcript_text)} "
            f"last_response={_clip_diagnostic_text(session.last_response_text)} "
            f"last_message={_clip_diagnostic_text(last_message)} "
            f"gate_reason={_clip_diagnostic_text(gate_reason)} "
            f"gate_text={_clip_diagnostic_text(gate_text)} "
            f"perception={_clip_diagnostic_text(perception_intent_state)} "
            f"perception_reason={_clip_diagnostic_text(perception_reason)} "
            f"perception_confidence={_clip_diagnostic_number(perception_confidence)} "
            f"perception_stability={_clip_diagnostic_number(perception_stability)} "
            f"spine={_clip_diagnostic_text(live_spine_status)} "
            f"spine_message={_clip_diagnostic_text(live_spine_message)} "
            f"fsm_violations={_clip_diagnostic_text(live_spine_fsm_violations)} "
            f"route={_clip_diagnostic_text(cognitive_route_action)} "
            f"route_state={_clip_diagnostic_text(cognitive_route_state)} "
            f"route_reason={_clip_diagnostic_text(cognitive_route_reason)} "
            f"cognition_status={_clip_diagnostic_text(cognition_status)} "
            f"cognition_message={_clip_diagnostic_text(cognition_message)} "
            f"cognition_safety={_clip_diagnostic_text(cognition_safety)} "
            f"runner_status={_clip_diagnostic_text(runner_status)} "
            f"runner_reason={_clip_diagnostic_text(runner_reason)} "
            f"wake_decision={_clip_diagnostic_text(wake_decision)} "
            f"wake_reason={_clip_diagnostic_text(wake_reason)} "
            f"reflexes={_clip_diagnostic_text(reflex_responses)} "
            f"reflex={_clip_diagnostic_text(last_reflex_kind)} "
            f"reflex_reason={_clip_diagnostic_text(last_reflex_reason)} "
            f"mic={_clip_diagnostic_text(microphone_device_name)} "
            f"mic_index={_clip_diagnostic_text(microphone_device_index)} "
            f"vad_energy={_clip_diagnostic_number(vad_last_energy)} "
            f"vad_noise={_clip_diagnostic_number(vad_noise_floor)} "
            f"stt_last={_clip_diagnostic_text(stt_last_text)} "
            f"stt_empty={_clip_diagnostic_text(stt_empty_results)} "
            f"stt_low_conf={_clip_diagnostic_text(stt_low_confidence_results)} "
            f"latency_ms={_clip_diagnostic_number(session.last_latency_ms)} "
            f"playback={_clip_diagnostic_text(playback_status)} "
            f"playback_current={_clip_diagnostic_text(playback_current_status)} "
            f"playback_queued={_clip_diagnostic_text(playback_queued)} "
            f"playback_stops={_clip_diagnostic_text(playback_stopped_count)} "
            f"playback_latency={_clip_diagnostic_number(playback_last_latency_ms)} "
            f"first_audio={_clip_diagnostic_number(playback_first_audio_latency_ms)}",
            flush=True,
        )


def _env_int(name: str) -> int | None:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return None
    try:
        return int(raw.strip())
    except ValueError as exc:
        raise RuntimeError(f"{name} must be an integer device index") from exc


def _env_int_default(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        return int(raw.strip())
    except ValueError as exc:
        raise RuntimeError(f"{name} must be an integer") from exc


def _env_float_default(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        return float(raw.strip())
    except ValueError as exc:
        raise RuntimeError(f"{name} must be a number") from exc


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    normalized = raw.strip().casefold()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise RuntimeError(f"{name} must be true/false")


def _env_text(name: str, default: str) -> str:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    return raw.strip()


def _env_startup_self_test_mode(default: str) -> str:
    raw = os.environ.get("JARVIS_STARTUP_SELF_TEST")
    if raw is None or not raw.strip():
        return default

    normalized = raw.strip().casefold()
    if normalized in {"0", "false", "no", "off", "skip"}:
        return "off"
    if normalized in {"1", "true", "yes", "on", "fast"}:
        return "fast"
    if normalized == "deep":
        return "deep"
    raise RuntimeError("JARVIS_STARTUP_SELF_TEST must be fast, deep, or off")


def _clip_diagnostic_text(value: object, *, limit: int = 96) -> str:
    if value is None:
        return "None"
    text = str(value).replace("\n", " ").replace("\r", " ").strip()
    if not text:
        return "None"
    if len(text) <= limit:
        return repr(text)
    return repr(text[: limit - 3] + "...")


def _clip_diagnostic_number(value: object) -> str:
    if value is None:
        return "None"
    try:
        return f"{float(str(value)):.1f}"
    except (TypeError, ValueError):
        return _clip_diagnostic_text(value)


def _result_status_value(result: object) -> str:
    status = getattr(result, "status", None)
    if status is None:
        return "unknown"
    return str(getattr(status, "value", status))


def _result_message(result: object) -> str:
    return str(getattr(result, "message", "") or "")


def _result_succeeded(result: object) -> bool:
    return bool(getattr(result, "succeeded", False))


def _ollama_model_list_contains(models: object, wanted_model: str) -> bool:
    wanted = wanted_model.strip().casefold()
    if not isinstance(models, list):
        return False
    for item in models:
        if not isinstance(item, dict):
            continue
        for key in ("name", "model"):
            value = item.get(key)
            if isinstance(value, str) and value.strip().casefold() == wanted:
                return True
    return False


def _visual_port_candidates(start_port: int) -> tuple[int, ...]:
    return tuple(
        port
        for port in range(start_port, min(65_535, start_port + 20) + 1)
        if 0 < port < 65_536
    )


def _tcp_port_open(host: str, port: int) -> bool:
    try:
        with socket.create_connection((host, port), timeout=0.2):
            return True
    except OSError:
        pass

    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            probe.bind((host, port))
            return False
    except OSError:
        return True


def _load_voice_profile_values(path: Path) -> dict[str, object]:
    try:
        profile = load_voice_capture_profile(path)
    except Exception as exc:
        print(
            "[JARVIS_VOICE_PROFILE] "
            f"status=ignored path={_clip_diagnostic_text(path)} "
            f"error={_clip_diagnostic_text(exc)}",
            flush=True,
        )
        return {}

    if profile is None:
        return {}

    print(
        "[JARVIS_VOICE_PROFILE] "
        f"status=loaded path={_clip_diagnostic_text(path)} "
        f"vad_min_energy={profile.vad_min_energy:.1f} "
        f"vad_ratio={profile.vad_speech_start_ratio:.1f} "
        f"final_confidence={profile.stt_min_final_confidence:.2f}",
        flush=True,
    )
    return profile.to_dict()


def _profile_float_default(
    values: dict[str, object],
    key: str,
    default: float,
) -> float:
    value = values.get(key)
    if isinstance(value, int | float):
        return float(value)
    return default


def _profile_int_default(
    values: dict[str, object],
    key: str,
    default: int,
) -> int:
    value = values.get(key)
    if isinstance(value, int):
        return value
    return default


def main() -> int:
    startup_self_test_mode = _env_startup_self_test_mode("off")
    voice_profile_enabled = _env_bool("JARVIS_VOICE_PROFILE", True)
    voice_profile_path = Path(
        _env_text(
            "JARVIS_VOICE_PROFILE_PATH",
            str(PROJECT_ROOT / "config" / "voice_profile.json"),
        )
    )
    voice_profile = (
        _load_voice_profile_values(voice_profile_path)
        if voice_profile_enabled
        else {}
    )
    launcher = PureConnectedJarvisLauncher(
        config=PureConnectedJarvisLauncherConfig(
            verify_factory_dry_run=_env_bool(
                "JARVIS_VERIFY_FACTORY_DRY_RUN",
                False,
            ),
            diagnostic_mode=_env_text("JARVIS_DIAGNOSTIC_MODE", "off").casefold(),
            run_startup_self_test=startup_self_test_mode != "off",
            startup_self_test_mode=startup_self_test_mode,
            play_startup_audio_probe=_env_bool(
                "JARVIS_STARTUP_AUDIO_PROBE",
                False,
            ),
            synthesize_startup_tts_probe=_env_bool(
                "JARVIS_STARTUP_TTS_SYNTHESIS_PROBE",
                False,
            ),
            speak_ready_on_start=_env_bool(
                "JARVIS_READY_VOICE",
                False,
            ),
            live_voice_events_enabled=_env_bool(
                "JARVIS_LIVE_VOICE_EVENTS",
                True,
            ),
            voice_profile_enabled=voice_profile_enabled,
            voice_profile_path=voice_profile_path,
            reflex_responses_enabled=_env_bool(
                "JARVIS_REFLEX_RESPONSES",
                True,
            ),
            reflex_min_confidence=_env_float_default(
                "JARVIS_REFLEX_MIN_CONFIDENCE",
                0.70,
            ),
            require_wake_word_for_companion_speech=_env_bool(
                "JARVIS_REQUIRE_WAKE_WORD",
                True,
            ),
            background_ollama_warmup=_env_bool(
                "JARVIS_BACKGROUND_OLLAMA_WARMUP",
                True,
            ),
            ollama_readiness_probe_timeout_seconds=_env_float_default(
                "JARVIS_OLLAMA_READINESS_PROBE_TIMEOUT_SECONDS",
                0.75,
            ),
            background_stt_warmup=_env_bool(
                "JARVIS_BACKGROUND_STT_WARMUP",
                False,
            ),
            background_tts_warmup=_env_bool(
                "JARVIS_BACKGROUND_TTS_WARMUP",
                False,
            ),
            ollama_prewarm_on_start=_env_bool(
                "JARVIS_OLLAMA_PREWARM_ON_START",
                False,
            ),
            stt_prewarm_on_prepare=_env_bool(
                "JARVIS_STT_PREWARM_ON_START",
                True,
            ),
            stt_partial_model_name=_env_text("JARVIS_STT_PARTIAL_MODEL", "base.en"),
            stt_final_model_name=_env_text("JARVIS_STT_FINAL_MODEL", "small"),
            ollama_stream_response=_env_bool(
                "JARVIS_OLLAMA_STREAM_RESPONSE",
                True,
            ),
            stt_partial_beam_size=_env_int_default(
                "JARVIS_STT_PARTIAL_BEAM_SIZE",
                1,
            ),
            stt_final_beam_size=_env_int_default(
                "JARVIS_STT_FINAL_BEAM_SIZE",
                1,
            ),
            stt_min_partial_confidence=_env_float_default(
                "JARVIS_STT_MIN_PARTIAL_CONFIDENCE",
                _profile_float_default(
                    voice_profile,
                    "stt_min_partial_confidence",
                    0.40,
                ),
            ),
            stt_min_final_confidence=_env_float_default(
                "JARVIS_STT_MIN_FINAL_CONFIDENCE",
                _profile_float_default(
                    voice_profile,
                    "stt_min_final_confidence",
                    0.70,
                ),
            ),
            stt_min_transcript_chars=_env_int_default(
                "JARVIS_STT_MIN_TRANSCRIPT_CHARS",
                _profile_int_default(
                    voice_profile,
                    "stt_min_transcript_chars",
                    3,
                ),
            ),
            stt_max_no_speech_prob=_env_float_default(
                "JARVIS_STT_MAX_NO_SPEECH_PROB",
                _profile_float_default(
                    voice_profile,
                    "stt_max_no_speech_prob",
                    0.55,
                ),
            ),
            stt_max_compression_ratio=_env_float_default(
                "JARVIS_STT_MAX_COMPRESSION_RATIO",
                _profile_float_default(
                    voice_profile,
                    "stt_max_compression_ratio",
                    2.35,
                ),
            ),
            stt_min_avg_logprob=_env_float_default(
                "JARVIS_STT_MIN_AVG_LOGPROB",
                _profile_float_default(
                    voice_profile,
                    "stt_min_avg_logprob",
                    -1.15,
                ),
            ),
            ollama_timeout_seconds=_env_int_default(
                "JARVIS_OLLAMA_TIMEOUT_SECONDS",
                8,
            ),
            ollama_max_sentences=_env_int_default(
                "JARVIS_OLLAMA_MAX_SENTENCES",
                2,
            ),
            ollama_num_predict=_env_int_default(
                "JARVIS_OLLAMA_NUM_PREDICT",
                96,
            ),
            voice_max_silence_ms=_env_int_default(
                "JARVIS_VOICE_MAX_SILENCE_MS",
                900,
            ),
            partial_transcript_every_frames=_env_int_default(
                "JARVIS_PARTIAL_TRANSCRIPT_EVERY_FRAMES",
                6,
            ),
            vad_min_speech_ms=_env_int_default(
                "JARVIS_VAD_MIN_SPEECH_MS",
                _profile_int_default(voice_profile, "vad_min_speech_ms", 400),
            ),
            vad_end_silence_ms=_env_int_default(
                "JARVIS_VAD_END_SILENCE_MS",
                _profile_int_default(voice_profile, "vad_end_silence_ms", 650),
            ),
            vad_min_energy=_env_float_default(
                "JARVIS_VAD_MIN_ENERGY",
                _profile_float_default(voice_profile, "vad_min_energy", 1_100.0),
            ),
            vad_speech_start_ratio=_env_float_default(
                "JARVIS_VAD_SPEECH_START_RATIO",
                _profile_float_default(
                    voice_profile,
                    "vad_speech_start_ratio",
                    5.0,
                ),
            ),
            vad_start_trigger_frames=_env_int_default(
                "JARVIS_VAD_START_TRIGGER_FRAMES",
                _profile_int_default(
                    voice_profile,
                    "vad_start_trigger_frames",
                    5,
                ),
            ),
            audio_preprocessor=_env_text(
                "JARVIS_AUDIO_PREPROCESSOR",
                "webrtcvad",
            ).casefold(),
            audio_preprocessor_drop_non_speech=_env_bool(
                "JARVIS_AUDIO_PREPROCESSOR_DROP_NON_SPEECH",
                False,
            ),
            require_echo_cancellation=_env_bool(
                "JARVIS_REQUIRE_AEC",
                False,
            ),
            require_noise_suppression=_env_bool(
                "JARVIS_REQUIRE_NS",
                False,
            ),
            require_auto_gain_control=_env_bool(
                "JARVIS_REQUIRE_AGC",
                False,
            ),
            webrtc_vad_aggressiveness=_env_int_default(
                "JARVIS_WEBRTC_VAD_AGGRESSIVENESS",
                2,
            ),
            tts_max_chars_per_chunk=_env_int_default(
                "JARVIS_TTS_MAX_CHARS_PER_CHUNK",
                90,
            ),
            tts_max_total_chars=_env_int_default(
                "JARVIS_TTS_MAX_TOTAL_CHARS",
                720,
            ),
            tts_target_first_chunk_ms=_env_int_default(
                "JARVIS_TTS_TARGET_FIRST_CHUNK_MS",
                350,
            ),
            visual_console_enabled=_env_bool(
                "JARVIS_VISUAL_CONSOLE",
                True,
            ),
            visual_bridge_enabled=_env_bool(
                "JARVIS_VISUAL_BRIDGE",
                True,
            ),
            visual_console_dir=Path(
                _env_text(
                    "JARVIS_VISUAL_CONSOLE_DIR",
                    str(DEFAULT_VISUAL_CONSOLE_DIR),
                )
            ),
            visual_console_node_path=Path(
                _env_text(
                    "JARVIS_VISUAL_NODE_PATH",
                    str(DEFAULT_VISUAL_NODE_PATH),
                )
            ),
            visual_console_auto_open=_env_bool(
                "JARVIS_VISUAL_AUTO_OPEN",
                False,
            ),
            visual_console_url_marker_path=Path(
                _env_text(
                    "JARVIS_VISUAL_CONSOLE_URL_FILE",
                    str(DEFAULT_VISUAL_CONSOLE_DIR / "visual_console.url"),
                )
            ),
            visual_console_host=_env_text(
                "JARVIS_VISUAL_CONSOLE_HOST",
                "127.0.0.1",
            ),
            visual_console_port=_env_int_default(
                "JARVIS_VISUAL_CONSOLE_PORT",
                3000,
            ),
            visual_bridge_host=_env_text(
                "JARVIS_VISUAL_BRIDGE_HOST",
                "127.0.0.1",
            ),
            visual_bridge_port=_env_int_default(
                "JARVIS_VISUAL_BRIDGE_PORT",
                8765,
            ),
            visual_bridge_start_timeout_seconds=_env_float_default(
                "JARVIS_VISUAL_BRIDGE_START_TIMEOUT_SECONDS",
                4.0,
            ),
            visual_console_start_timeout_seconds=_env_float_default(
                "JARVIS_VISUAL_CONSOLE_START_TIMEOUT_SECONDS",
                2.0,
            ),
            wait_for_voice_listening_before_ready=_env_bool(
                "JARVIS_WAIT_FOR_VOICE_READY",
                True,
            ),
            voice_ready_timeout_seconds=_env_float_default(
                "JARVIS_VOICE_READY_TIMEOUT_SECONDS",
                45.0,
            ),
            run_ten_turn_spine_gate=_env_bool(
                "JARVIS_TEN_TURN_SPINE_GATE",
                False,
            ),
        )
    )

    def _handle_signal(signum: int, frame: FrameType | None) -> None:
        del signum, frame
        launcher.request_stop()

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    return launcher.run()


if __name__ == "__main__":
    raise SystemExit(main())
