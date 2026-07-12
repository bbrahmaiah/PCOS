from __future__ import annotations

from pathlib import Path

from jarvis.system import (
    PureJarvisPreflightCheckKind,
    PureJarvisPreflightConfig,
    PureJarvisPreflightRuntime,
    PureJarvisPreflightStatus,
    summarize_pure_jarvis_preflight,
)


class FakeOllamaProbe:
    def __init__(self, models: tuple[str, ...]) -> None:
        self._models = models

    def list_models(self) -> tuple[str, ...]:
        return self._models


class FailingOllamaProbe:
    def list_models(self) -> tuple[str, ...]:
        raise RuntimeError("offline")


def _write_file(path: Path, text: str = "content") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def _config(
    tmp_path: Path,
    *,
    require_ollama: bool = True,
) -> PureJarvisPreflightConfig:
    source = _write_file(tmp_path / "source.py", "print('source')\n")
    return PureJarvisPreflightConfig(
        bindings_path=_write_file(tmp_path / "runtime_bindings.env", "x=y\n"),
        piper_executable_path=_write_file(tmp_path / "piper.exe", "exe"),
        piper_model_path=_write_file(tmp_path / "voice.onnx", "model"),
        piper_config_path=_write_file(tmp_path / "voice.onnx.json", "{}"),
        source_fingerprint_paths=(source,),
        ollama_model="llama3.2:3b",
        require_ollama=require_ollama,
    )


def test_pure_jarvis_preflight_passes_with_real_assets_and_model(
    tmp_path: Path,
) -> None:
    runtime = PureJarvisPreflightRuntime(
        config=_config(tmp_path),
        ollama_probe=FakeOllamaProbe(("llama3.2:3b",)),
    )

    report = runtime.run()
    kinds = {check.kind for check in report.checks}

    assert report.status == PureJarvisPreflightStatus.PASSED
    assert report.failed_count == 0
    assert report.source_fingerprint != "unavailable"
    assert PureJarvisPreflightCheckKind.PURE_MANIFEST in kinds
    assert PureJarvisPreflightCheckKind.REALTIME_EXPECTATION in kinds
    assert PureJarvisPreflightCheckKind.SOURCE_FINGERPRINT in kinds


def test_pure_jarvis_preflight_declares_true_ms_answer_limit(
    tmp_path: Path,
) -> None:
    runtime = PureJarvisPreflightRuntime(
        config=_config(tmp_path),
        ollama_probe=FakeOllamaProbe(("llama3.2:3b",)),
    )

    report = runtime.run()
    realtime = next(
        check
        for check in report.checks
        if check.kind == PureJarvisPreflightCheckKind.REALTIME_EXPECTATION
    )

    assert realtime.passed is True
    assert realtime.metadata["complete_answer_true_milliseconds_available"] is False
    assert realtime.metadata["millisecond_interruption_target_available"] is True
    physics_limited = realtime.metadata["physics_limited_requirements"]
    assert isinstance(physics_limited, tuple)
    assert "true milliseconds for complete movie-like answers" in (
        physics_limited
    )


def test_pure_jarvis_preflight_fails_missing_voice_model(tmp_path: Path) -> None:
    config = _config(tmp_path)
    missing_config = PureJarvisPreflightConfig(
        bindings_path=config.bindings_path,
        piper_executable_path=config.piper_executable_path,
        piper_model_path=tmp_path / "missing.onnx",
        piper_config_path=config.piper_config_path,
        source_fingerprint_paths=config.source_fingerprint_paths,
        ollama_model=config.ollama_model,
    )
    runtime = PureJarvisPreflightRuntime(
        config=missing_config,
        ollama_probe=FakeOllamaProbe(("llama3.2:3b",)),
    )

    report = runtime.run()

    assert report.status == PureJarvisPreflightStatus.FAILED
    assert any(
        check.kind == PureJarvisPreflightCheckKind.REAL_TTS_MODEL
        and check.failed
        for check in report.checks
    )


def test_pure_jarvis_preflight_fails_missing_piper_executable(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    missing_config = PureJarvisPreflightConfig(
        bindings_path=config.bindings_path,
        piper_executable_path=tmp_path / "missing-piper.exe",
        piper_model_path=config.piper_model_path,
        piper_config_path=config.piper_config_path,
        source_fingerprint_paths=config.source_fingerprint_paths,
        ollama_model=config.ollama_model,
    )
    runtime = PureJarvisPreflightRuntime(
        config=missing_config,
        ollama_probe=FakeOllamaProbe(("llama3.2:3b",)),
    )

    report = runtime.run()

    assert report.status == PureJarvisPreflightStatus.FAILED
    assert any(
        check.kind == PureJarvisPreflightCheckKind.REAL_TTS_EXECUTABLE
        and check.failed
        for check in report.checks
    )


def test_pure_jarvis_preflight_fails_missing_required_ollama_model(
    tmp_path: Path,
) -> None:
    runtime = PureJarvisPreflightRuntime(
        config=_config(tmp_path),
        ollama_probe=FakeOllamaProbe(("other-model:latest",)),
    )

    report = runtime.run()

    assert report.status == PureJarvisPreflightStatus.FAILED
    assert any(
        check.kind == PureJarvisPreflightCheckKind.OLLAMA_MODEL and check.failed
        for check in report.checks
    )


def test_pure_jarvis_preflight_warns_when_optional_ollama_probe_fails(
    tmp_path: Path,
) -> None:
    runtime = PureJarvisPreflightRuntime(
        config=_config(tmp_path, require_ollama=False),
        ollama_probe=FailingOllamaProbe(),
    )

    report = runtime.run()
    summary = summarize_pure_jarvis_preflight(report)

    assert report.status == PureJarvisPreflightStatus.WARNING
    assert report.warning_count == 1
    assert "ollama_model" in summary
