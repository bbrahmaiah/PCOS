from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

from jarvis.runtime import JarvisStartControlStatus
from jarvis.voice import (
    VoiceRuntimeLauncherSnapshot,
    VoiceRuntimeLauncherStatus,
    VoiceSessionLoopEvent,
    VoiceSessionLoopSnapshot,
    VoiceSessionLoopStatus,
    utc_now,
)
from scripts.run_connected_jarvis import (
    PureConnectedJarvisLauncher,
    PureConnectedJarvisLauncherConfig,
    _profile_float_default,
    _profile_int_default,
)


def test_connected_launcher_module_imports_without_cognitive_cycle() -> None:
    import scripts.run_connected_jarvis as launcher_module

    assert launcher_module.PureConnectedJarvisLauncher is PureConnectedJarvisLauncher


@dataclass(frozen=True, slots=True)
class FakeStatus:
    value: str


@dataclass(frozen=True, slots=True)
class FakeResult:
    status: FakeStatus
    message: str
    succeeded: bool = True
    metadata: dict[str, object] = field(default_factory=dict)
    frame: object | None = None
    device: object | None = None


class FakeMicrophone:
    def __init__(self, *, fail: str | None = None) -> None:
        self._fail = fail
        self.stopped = False

    def start(self) -> FakeResult:
        if self._fail == "microphone_start":
            return FakeResult(
                status=FakeStatus("failed"),
                message="microphone failed",
                succeeded=False,
            )
        return FakeResult(
            status=FakeStatus("capturing"),
            message="microphone started",
            device=SimpleNamespace(name="Fake microphone"),
        )

    def capture_once(self) -> FakeResult:
        if self._fail == "microphone_capture":
            return FakeResult(
                status=FakeStatus("failed"),
                message="capture failed",
                succeeded=False,
            )
        return FakeResult(
            status=FakeStatus("capturing"),
            message="frame captured",
            frame=SimpleNamespace(data=b"audio"),
        )

    def stop(self) -> FakeResult:
        self.stopped = True
        return FakeResult(status=FakeStatus("stopped"), message="stopped")


class FakeSTT:
    def __init__(self, *, fail: bool = False) -> None:
        self._fail = fail

    def prepare(self) -> FakeResult:
        if self._fail:
            return FakeResult(
                status=FakeStatus("failed"),
                message="stt failed",
                succeeded=False,
                metadata={"error": "missing model"},
            )
        return FakeResult(status=FakeStatus("ready"), message="stt ready")


class FakeCognition:
    def __init__(self, *, fail: bool = False) -> None:
        self._fail = fail
        self.calls = 0

    def prepare(self, *, user_label: str, assistant_name: str) -> None:
        assert user_label == "Tony"
        assert assistant_name == "JARVIS"

    def think_from_transcript(self, request: object) -> object:
        del request
        self.calls += 1
        response = None if self._fail else SimpleNamespace(text="generated response")
        return SimpleNamespace(
            response=response,
            message="cognition ready",
            status=FakeStatus("thinking"),
            safety=FakeStatus("safe_for_dialogue"),
        )


class FakeTTS:
    def __init__(self, *, fail_prepare: bool = False, fail_synth: bool = False) -> None:
        self._fail_prepare = fail_prepare
        self._fail_synth = fail_synth
        self.synths = 0

    def prepare(self) -> FakeResult:
        if self._fail_prepare:
            return FakeResult(
                status=FakeStatus("failed"),
                message="tts prepare failed",
                succeeded=False,
                metadata={"error": "piper missing"},
            )
        return FakeResult(status=FakeStatus("ready"), message="tts ready")

    def synthesize_text(
        self,
        *,
        text: str,
        session_id: object,
        metadata: dict[str, object],
    ) -> object:
        self.synths += 1
        assert text
        assert session_id
        assert metadata["source"] == "startup_self_test"
        chunks = () if self._fail_synth else (SimpleNamespace(audio=b"RIFF"),)
        return SimpleNamespace(
            status=FakeStatus("synthesizing"),
            message="tts synthesized",
            chunks=chunks,
        )


class FakePlayback:
    def __init__(self, *, fail: bool = False) -> None:
        self._fail = fail
        self.played = 0

    def prepare(self) -> FakeResult:
        return self._result("playback prepared")

    def enqueue_chunks(self, chunks: tuple[object, ...]) -> FakeResult:
        assert chunks
        return self._result("playback enqueued", status="queued")

    def play_all(self) -> FakeResult:
        self.played += 1
        return self._result("playback played")

    def _result(self, message: str, *, status: str = "ready") -> FakeResult:
        if self._fail:
            return FakeResult(
                status=FakeStatus("failed"),
                message=message,
                succeeded=False,
            )
        return FakeResult(status=FakeStatus(status), message=message)


class FakeVisualBridgeServer:
    def __init__(self) -> None:
        self.daemon_threads = False
        self.served = False
        self.shutdown_called = False
        self.server_close_called = False

    def serve_forever(self, *, poll_interval: float = 0.5) -> None:
        assert poll_interval > 0
        self.served = True

    def shutdown(self) -> None:
        self.shutdown_called = True

    def server_close(self) -> None:
        self.server_close_called = True


@dataclass
class FakeHealthResult:
    status: JarvisStartControlStatus
    reason: str = "healthy"
    health: tuple[object, ...] = ()


class FakeDiagnosticVoiceLauncher:
    def __init__(self) -> None:
        self.snapshot_calls = 0
        self.live_snapshot_calls = 0

    def snapshot(self) -> VoiceRuntimeLauncherSnapshot:
        self.snapshot_calls += 1
        return self._snapshot()

    def live_snapshot(self) -> VoiceRuntimeLauncherSnapshot:
        self.live_snapshot_calls += 1
        return self._snapshot()

    def _snapshot(self) -> VoiceRuntimeLauncherSnapshot:
        session = VoiceSessionLoopSnapshot(
            status=VoiceSessionLoopStatus.LISTENING,
            running=True,
            assistant_speaking=False,
            cycles=1,
            captured_frames=1,
            speech_segments=1,
            partial_transcripts=0,
            final_transcripts=1,
            responses=1,
            tts_outputs=1,
            played_outputs=1,
            interruptions=0,
            recoveries=0,
            consecutive_failures=0,
            buffered_segment_frames=0,
            last_event=VoiceSessionLoopEvent.PLAYBACK_FINISHED,
            last_transcript_text="jarvis status",
            last_response_text="ready",
            last_latency_ms=1.0,
            last_error=None,
            created_at=utc_now(),
            metadata={
                "perception_intent_state": "ready_for_routing",
                "fsm_violations": 0,
                "playback_status": "ready",
            },
        )
        return VoiceRuntimeLauncherSnapshot(
            status=VoiceRuntimeLauncherStatus.RUNNING,
            booted=True,
            running=True,
            stop_requested=False,
            boot_count=1,
            run_cycles=1,
            stop_count=0,
            last_event=None,
            last_error=None,
            last_latency_ms=1.0,
            session_snapshot=session,
            created_at=utc_now(),
            metadata={},
        )


class FakeSelfTestLauncher(PureConnectedJarvisLauncher):
    def __init__(
        self,
        *,
        fail: str | None = None,
        startup_self_test_mode: str = "deep",
        play_startup_audio_probe: bool = True,
    ) -> None:
        super().__init__(
            config=PureConnectedJarvisLauncherConfig(
                startup_self_test_mode=startup_self_test_mode,
                play_startup_audio_probe=play_startup_audio_probe,
            )
        )
        self._fail = fail
        self.cognition = FakeCognition(
            fail=self._fail == "ollama_cognition_probe"
        )
        self.playback = FakePlayback(fail=self._fail == "windows_playback_play")
        self.tts = FakeTTS(
            fail_prepare=self._fail == "piper_prepare",
            fail_synth=self._fail == "piper_synthesize",
        )

    def _build_microphone_runtime(
        self,
        *,
        voice_config: object,
        microphone_device_index: int | None,
    ) -> Any:
        del voice_config, microphone_device_index
        return FakeMicrophone(fail=self._fail)

    def _build_stt_runtime(self, *, voice_config: object) -> Any:
        del voice_config
        return FakeSTT(fail=self._fail == "stt_prepare")

    def _build_companion_cognition_runtime(self) -> Any:
        return self.cognition

    def _build_real_piper_tts_runtime(self) -> Any:
        return self.tts

    def _build_startup_playback_runtime(self) -> Any:
        return self.playback


def test_connected_launcher_startup_self_test_passes_fake_runtime() -> None:
    launcher = FakeSelfTestLauncher()

    launcher._run_startup_self_test()


def test_connected_launcher_startup_self_test_fails_exact_check() -> None:
    launcher = FakeSelfTestLauncher(fail="stt_prepare")

    with pytest.raises(RuntimeError, match="stt_prepare"):
        launcher._run_startup_self_test()


def test_connected_launcher_fast_self_test_skips_blocking_cognition() -> None:
    launcher = FakeSelfTestLauncher(
        startup_self_test_mode="fast",
        play_startup_audio_probe=False,
    )

    launcher._run_startup_self_test()

    assert launcher.cognition.calls == 0
    assert launcher.tts.synths == 0
    assert launcher.playback.played == 0


def test_connected_launcher_default_startup_path_is_fast_daily_driver() -> None:
    config = PureConnectedJarvisLauncherConfig()

    assert config.verify_factory_dry_run is False
    assert config.run_startup_self_test is False
    assert config.startup_self_test_mode == "off"
    assert config.play_startup_audio_probe is False
    assert config.synthesize_startup_tts_probe is False
    assert config.speak_ready_on_start is False
    assert config.live_voice_events_enabled is True
    assert config.voice_profile_enabled is True
    assert config.voice_profile_path.name == "voice_profile.json"
    assert config.ollama_prewarm_on_start is False
    assert config.stt_prewarm_on_prepare is True
    assert config.background_stt_warmup is False
    assert config.background_tts_warmup is False
    assert config.background_ollama_warmup is True
    assert config.ollama_readiness_probe_timeout_seconds == 0.75
    assert config.diagnostic_mode == "off"
    assert config.reflex_responses_enabled is True
    assert config.require_wake_word_for_companion_speech is True
    assert config.ollama_stream_response is True
    assert config.ollama_timeout_seconds == 8
    assert config.ollama_max_sentences == 2
    assert config.ollama_num_predict == 96
    assert config.stt_partial_model_name == "base.en"
    assert config.stt_final_model_name == "small"
    assert config.stt_min_partial_confidence == 0.40
    assert config.stt_min_final_confidence == 0.70
    assert config.stt_min_transcript_chars == 3
    assert config.stt_max_no_speech_prob == 0.55
    assert config.stt_max_compression_ratio == 2.35
    assert config.stt_min_avg_logprob == -1.15
    assert config.voice_max_silence_ms == 900
    assert config.partial_transcript_every_frames == 6
    assert config.vad_min_energy == 1_100.0
    assert config.vad_speech_start_ratio == 5.0
    assert config.vad_start_trigger_frames == 5
    assert config.vad_min_speech_ms == 400
    assert config.vad_end_silence_ms == 650
    assert config.audio_preprocessor == "webrtcvad"
    assert config.audio_preprocessor_drop_non_speech is False
    assert config.require_echo_cancellation is False
    assert config.require_noise_suppression is False
    assert config.require_auto_gain_control is False
    assert config.webrtc_vad_aggressiveness == 2
    assert config.tts_max_chars_per_chunk == 90
    assert config.tts_max_total_chars == 720
    assert config.tts_target_first_chunk_ms == 350
    assert config.visual_console_enabled is True
    assert config.visual_bridge_enabled is True
    assert config.visual_console_auto_open is False
    assert config.visual_console_port == 3000
    assert config.visual_bridge_port == 8765
    assert config.visual_console_start_timeout_seconds == 2.0
    assert config.wait_for_voice_listening_before_ready is True
    assert config.run_ten_turn_spine_gate is False


def test_connected_launcher_profile_defaults_are_type_safe() -> None:
    profile: dict[str, object] = {
        "vad_min_energy": 640.0,
        "vad_start_trigger_frames": 6,
        "bad_int": 4.5,
        "bad_float": "nope",
    }

    assert _profile_float_default(profile, "vad_min_energy", 1_100.0) == 640.0
    assert _profile_int_default(profile, "vad_start_trigger_frames", 5) == 6
    assert _profile_int_default(profile, "bad_int", 5) == 5
    assert _profile_float_default(profile, "bad_float", 0.55) == 0.55


def test_connected_launcher_diagnostics_off_skips_healthy_voice_snapshot(
    capsys: pytest.CaptureFixture[str],
) -> None:
    launcher = PureConnectedJarvisLauncher(
        config=PureConnectedJarvisLauncherConfig(
            diagnostic_mode="off",
            live_voice_events_enabled=False,
        )
    )
    voice = FakeDiagnosticVoiceLauncher()
    launcher._voice_launcher = cast(Any, voice)
    launcher._health_checks = 1

    launcher._print_health_if_needed(
        cast(Any, FakeHealthResult(status=JarvisStartControlStatus.RUNNING))
    )

    assert capsys.readouterr().out == ""
    assert voice.live_snapshot_calls == 0
    assert voice.snapshot_calls == 0


def test_connected_launcher_diagnostics_off_prints_compact_live_voice_events(
    capsys: pytest.CaptureFixture[str],
) -> None:
    launcher = PureConnectedJarvisLauncher(
        config=PureConnectedJarvisLauncherConfig(diagnostic_mode="off")
    )
    voice = FakeDiagnosticVoiceLauncher()
    launcher._voice_launcher = cast(Any, voice)
    launcher._health_checks = 1

    launcher._print_health_if_needed(
        cast(Any, FakeHealthResult(status=JarvisStartControlStatus.RUNNING))
    )

    output = capsys.readouterr().out
    assert "[JARVIS_VOICE_EVENT]" in output
    assert "last_text='jarvis status'" in output
    assert voice.live_snapshot_calls == 1
    assert voice.snapshot_calls == 0


def test_connected_launcher_reports_ollama_ready_when_model_loaded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    launcher = PureConnectedJarvisLauncher(config=PureConnectedJarvisLauncherConfig())

    def fake_probe(url: str, *, timeout_seconds: float) -> dict[str, object] | None:
        del timeout_seconds
        if url.endswith("/api/tags"):
            return {"models": [{"name": "llama3.2:3b"}]}
        if url.endswith("/api/ps"):
            return {"models": [{"model": "llama3.2:3b"}]}
        return None

    monkeypatch.setattr(launcher, "_probe_http_json", fake_probe)

    status = launcher._probe_ollama_readiness()

    assert status["status"] == "ready"
    assert status["installed"] is True
    assert status["loaded"] is True


def test_connected_launcher_reports_ollama_cold_when_model_not_loaded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    launcher = PureConnectedJarvisLauncher(config=PureConnectedJarvisLauncherConfig())

    def fake_probe(url: str, *, timeout_seconds: float) -> dict[str, object] | None:
        del timeout_seconds
        if url.endswith("/api/tags"):
            return {"models": [{"name": "llama3.2:3b"}]}
        if url.endswith("/api/ps"):
            return {"models": []}
        return None

    monkeypatch.setattr(launcher, "_probe_http_json", fake_probe)

    status = launcher._probe_ollama_readiness()

    assert status["status"] == "cold"
    assert status["installed"] is True
    assert status["loaded"] is False


def test_connected_launcher_reports_ollama_unreachable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    launcher = PureConnectedJarvisLauncher(config=PureConnectedJarvisLauncherConfig())
    monkeypatch.setattr(launcher, "_probe_http_json", lambda *args, **kwargs: None)

    status = launcher._probe_ollama_readiness()

    assert status["status"] == "unreachable"
    assert status["installed"] is False
    assert status["loaded"] is False


def test_connected_launcher_starts_and_stops_visual_bridge(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    servers: list[FakeVisualBridgeServer] = []

    def fake_create_jarvis_http_server(
        *,
        config: object | None = None,
        runtime: object | None = None,
    ) -> FakeVisualBridgeServer:
        del config, runtime
        server = FakeVisualBridgeServer()
        servers.append(server)
        return server

    monkeypatch.setattr(
        "scripts.run_connected_jarvis.create_jarvis_http_server",
        fake_create_jarvis_http_server,
    )
    launcher = PureConnectedJarvisLauncher(
        config=PureConnectedJarvisLauncherConfig(
            visual_console_enabled=False,
            visual_bridge_enabled=True,
        )
    )

    launcher._start_visual_console_stack()

    assert servers
    assert servers[0].daemon_threads is True
    deadline = time.perf_counter() + 1.0
    while not servers[0].served and time.perf_counter() < deadline:
        time.sleep(0.01)
    assert servers[0].served is True

    launcher._stop_visual_console_stack()

    assert servers[0].shutdown_called is True
    assert servers[0].server_close_called is True


def test_connected_launcher_missing_visual_console_build_degrades(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    launcher = PureConnectedJarvisLauncher(
        config=PureConnectedJarvisLauncherConfig(
            visual_console_enabled=True,
            visual_bridge_enabled=False,
            visual_console_dir=tmp_path / "missing-console",
        )
    )
    monkeypatch.setattr(launcher, "_probe_http_json", lambda *args, **kwargs: None)

    launcher._start_visual_console_stack()

    assert "console=disabled" in capsys.readouterr().out


def test_connected_launcher_visual_console_uses_next_free_port(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    console_dir = tmp_path / "console"
    server_entry = console_dir / "dist" / "server.cjs"
    server_entry.parent.mkdir(parents=True)
    server_entry.write_text("console.log('fake console');", encoding="utf-8")
    started_ports: list[str] = []
    opened_urls: list[str] = []

    class FakeConsoleProcess:
        def poll(self) -> int | None:
            return None

        def terminate(self) -> None:
            return None

        def wait(self, timeout: float | None = None) -> int:
            del timeout
            return 0

        def kill(self) -> None:
            return None

    def fake_popen(*args: object, **kwargs: object) -> FakeConsoleProcess:
        del args
        env = kwargs.get("env")
        assert isinstance(env, dict)
        started_ports.append(str(env["PORT"]))
        return FakeConsoleProcess()

    def fake_probe(url: str, *, timeout_seconds: float) -> dict[str, object] | None:
        del timeout_seconds
        if started_ports and ":3001/" in url:
            return {"fakeFallbackEnabled": False}
        return None

    def fake_browser_open(
        url: str,
        new: int = 0,
        autoraise: bool = True,
    ) -> bool:
        del new, autoraise
        opened_urls.append(str(url))
        return True

    monkeypatch.setattr(
        "scripts.run_connected_jarvis._tcp_port_open",
        lambda host, port: port == 3000,
    )
    monkeypatch.setattr(
        "scripts.run_connected_jarvis.subprocess.Popen",
        fake_popen,
    )
    monkeypatch.setattr(
        "scripts.run_connected_jarvis.webbrowser.open",
        fake_browser_open,
    )
    launcher = PureConnectedJarvisLauncher(
        config=PureConnectedJarvisLauncherConfig(
            visual_console_enabled=True,
            visual_bridge_enabled=False,
            visual_console_dir=console_dir,
            visual_console_url_marker_path=tmp_path / "visual_console.url",
            visual_console_port=3000,
        )
    )
    monkeypatch.setattr(launcher, "_probe_http_json", fake_probe)

    launcher._start_visual_console_stack()

    output = capsys.readouterr().out
    assert "console=port_busy port=3000" in output
    assert "console=ready url=http://127.0.0.1:3001" in output
    assert "console_browser=opened" not in output
    assert started_ports == ["3001"]
    assert opened_urls == []
    url_marker_lines = (
        tmp_path / "visual_console.url"
    ).read_text(encoding="utf-8").splitlines()
    assert url_marker_lines == [
        "http://127.0.0.1:3001",
        "state=ready",
    ]


def test_connected_launcher_quiet_diagnostics_use_live_voice_snapshot(
    capsys: pytest.CaptureFixture[str],
) -> None:
    launcher = PureConnectedJarvisLauncher(
        config=PureConnectedJarvisLauncherConfig(diagnostic_mode="quiet")
    )
    voice = FakeDiagnosticVoiceLauncher()
    launcher._voice_launcher = cast(Any, voice)
    launcher._health_checks = 1

    launcher._print_health_if_needed(
        cast(Any, FakeHealthResult(status=JarvisStartControlStatus.RUNNING))
    )

    output = capsys.readouterr().out
    assert "[JARVIS_VOICE_EVENT]" in output
    assert voice.live_snapshot_calls == 1
    assert voice.snapshot_calls == 0
