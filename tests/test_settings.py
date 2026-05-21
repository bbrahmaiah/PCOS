from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from jarvis.runtime.config.settings import RuntimeSettings, get_settings
from jarvis.runtime.shared.enums import RuntimeEnvironment


def test_settings_loads_from_yaml() -> None:
    settings = RuntimeSettings.from_yaml()

    assert settings.runtime.system_name == "JARVIS_OS"
    assert settings.runtime.environment == RuntimeEnvironment.DEVELOPMENT
    assert settings.workers.max_concurrent_tasks == 100


def test_settings_paths_are_absolute() -> None:
    settings = RuntimeSettings.from_yaml()

    assert settings.paths.logs_dir.is_absolute()
    assert settings.paths.data_dir.is_absolute()
    assert settings.paths.cache_dir.is_absolute()
    assert settings.paths.models_dir.is_absolute()


def test_get_settings_returns_cached_singleton() -> None:
    first = get_settings()
    second = get_settings()

    assert first is second


def test_missing_yaml_fails_fast(tmp_path: Path) -> None:
    missing_file = tmp_path / "missing_runtime.yaml"

    with pytest.raises(FileNotFoundError):
        RuntimeSettings.from_yaml(missing_file)


def test_invalid_worker_value_fails(tmp_path: Path) -> None:
    bad_yaml = tmp_path / "runtime.yaml"
    bad_yaml.write_text(
        """
runtime:
  system_name: JARVIS_OS
  environment: development
  debug: true
  version: 0.1.0

workers:
  max_concurrent_tasks: 0
""",
        encoding="utf-8",
    )

    with pytest.raises(ValidationError):
        RuntimeSettings.from_yaml(bad_yaml)