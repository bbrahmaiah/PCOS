from enum import StrEnum


class RuntimeEnvironment(StrEnum):
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"


class WorkerState(StrEnum):
    IDLE = "idle"
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    STOPPED = "stopped"
    FAILED = "failed"


class EventPriority(StrEnum):
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"


class EventCategory(StrEnum):
    SYSTEM = "system"
    VOICE = "voice"
    MEMORY = "memory"
    ACTION = "action"
    SECURITY = "security"
    COGNITION = "cognition"


class PermissionLevel(StrEnum):
    ALLOWED = "allowed"
    DENIED = "denied"
    REQUIRE_CONFIRMATION = "require_confirmation"