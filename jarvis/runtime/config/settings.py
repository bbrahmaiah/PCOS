from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from jarvis.runtime.shared.constants import (
    DEFAULT_EVENT_QUEUE_SIZE,
    DEFAULT_HEALTH_CHECK_INTERVAL_SECONDS,
    DEFAULT_LOG_LEVEL,
    DEFAULT_RETRY_COUNT,
    DEFAULT_WORKER_TIMEOUT_SECONDS,
    JARVIS_SYSTEM_NAME,
    MAX_CONCURRENT_TASKS,
    RUNTIME_VERSION,
)
from jarvis.runtime.shared.enums import RuntimeEnvironment

PROJECT_ROOT = Path(__file__).resolve().parents[3]
CONFIG_DIR = Path(__file__).resolve().parent
RUNTIME_YAML = CONFIG_DIR / "runtime.yaml"
PERMISSIONS_YAML = CONFIG_DIR / "permissions.yaml"


class RuntimeSection(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    system_name: str = JARVIS_SYSTEM_NAME
    environment: RuntimeEnvironment = RuntimeEnvironment.DEVELOPMENT
    debug: bool = True
    version: str = RUNTIME_VERSION


class LoggingSection(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    log_level: str = DEFAULT_LOG_LEVEL
    enable_console_logging: bool = True
    enable_file_logging: bool = True


class WorkerSection(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    max_concurrent_tasks: int = Field(default=MAX_CONCURRENT_TASKS, ge=1, le=10_000)
    default_worker_timeout_seconds: int = Field(
        default=DEFAULT_WORKER_TIMEOUT_SECONDS,
        ge=1,
        le=3600,
    )
    event_queue_size: int = Field(default=DEFAULT_EVENT_QUEUE_SIZE, ge=100, le=100_000)
    retry_count: int = Field(default=DEFAULT_RETRY_COUNT, ge=0, le=100)


class SecuritySection(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    enable_permissions: bool = True
    enable_audit_logging: bool = True
    require_action_confirmation: bool = True
    deny_unknown_actions: bool = True


class ObservabilitySection(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    enable_metrics: bool = True
    enable_tracing: bool = True
    enable_performance_monitoring: bool = True
    health_check_interval_seconds: int = Field(
        default=DEFAULT_HEALTH_CHECK_INTERVAL_SECONDS,
        ge=1,
        le=300,
    )


class PathsSection(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    logs_dir: Path = PROJECT_ROOT / "logs"
    data_dir: Path = PROJECT_ROOT / "data"
    cache_dir: Path = PROJECT_ROOT / "data" / "cache"
    models_dir: Path = PROJECT_ROOT / "data" / "models"

    @field_validator("logs_dir", "data_dir", "cache_dir", "models_dir", mode="before")
    @classmethod
    def normalize_path(cls, value: Any) -> Path:
        path = Path(value).expanduser()

        if path.is_absolute():
            return path.resolve()

        return (PROJECT_ROOT / path).resolve()


class RuntimeSettings(BaseSettings):
    """
    Strict runtime settings for JARVIS_OS.

    Load order:
    1. Defaults from typed models
    2. runtime.yaml
    3. .env / environment variables when explicitly passed by pydantic-settings

    This class is intentionally strict. Bad config should fail at startup.
    """

    model_config = SettingsConfigDict(
        env_file=str(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
        case_sensitive=False,
        extra="forbid",
        validate_default=True,
    )

    runtime: RuntimeSection = Field(default_factory=RuntimeSection)
    logging: LoggingSection = Field(default_factory=LoggingSection)
    workers: WorkerSection = Field(default_factory=WorkerSection)
    security: SecuritySection = Field(default_factory=SecuritySection)
    observability: ObservabilitySection = Field(default_factory=ObservabilitySection)
    paths: PathsSection = Field(default_factory=PathsSection)

    @classmethod
    def load_yaml(cls, yaml_path: str | Path = RUNTIME_YAML) -> dict[str, Any]:
        path = Path(yaml_path)

        if not path.is_absolute():
            path = CONFIG_DIR / path

        if not path.exists():
            raise FileNotFoundError(f"Missing required runtime config: {path}")

        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}

        if not isinstance(data, dict):
            raise TypeError("runtime.yaml must contain a top-level mapping/object.")

        return data

    @classmethod
    def from_yaml(cls, yaml_path: str | Path = RUNTIME_YAML) -> RuntimeSettings:
        yaml_data = cls.load_yaml(yaml_path)
        return cls.model_validate(yaml_data)

    @property
    def is_development(self) -> bool:
        return self.runtime.environment == RuntimeEnvironment.DEVELOPMENT

    @property
    def is_testing(self) -> bool:
        return self.runtime.environment == RuntimeEnvironment.TESTING

    @property
    def is_staging(self) -> bool:
        return self.runtime.environment == RuntimeEnvironment.STAGING

    @property
    def is_production(self) -> bool:
        return self.runtime.environment == RuntimeEnvironment.PRODUCTION

    def ensure_directories(self) -> None:
        self.paths.logs_dir.mkdir(parents=True, exist_ok=True)
        self.paths.data_dir.mkdir(parents=True, exist_ok=True)
        self.paths.cache_dir.mkdir(parents=True, exist_ok=True)
        self.paths.models_dir.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> RuntimeSettings:
    settings = RuntimeSettings.from_yaml(RUNTIME_YAML)
    settings.ensure_directories()
    return settings