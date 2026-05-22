from __future__ import annotations

from enum import StrEnum


class RuntimeEnvironment(StrEnum):
    DEVELOPMENT = "development"
    TESTING = "testing"
    STAGING = "staging"
    PRODUCTION = "production"


class RuntimeStatus(StrEnum):
    CREATED = "created"
    STARTING = "starting"
    RUNNING = "running"
    PAUSING = "pausing"
    PAUSED = "paused"
    STOPPING = "stopping"
    STOPPED = "stopped"
    FAILED = "failed"


class WorkerStatus(StrEnum):
    CREATED = "created"
    STARTING = "starting"
    RUNNING = "running"
    IDLE = "idle"
    BUSY = "busy"
    STOPPING = "stopping"
    STOPPED = "stopped"
    FAILED = "failed"


class SystemMode(StrEnum):
    PASSIVE = "passive"
    ACTIVE = "active"
    LISTENING = "listening"
    THINKING = "thinking"
    SPEAKING = "speaking"
    EXECUTING = "executing"
    WAITING_PERMISSION = "waiting_permission"
    INTERRUPTED = "interrupted"


class RiskLevel(StrEnum):
    SAFE = "safe"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"
    BLOCKED = "blocked"


class PermissionDecision(StrEnum):
    ALLOW = "allow"
    DENY = "deny"
    REQUIRE_CONFIRMATION = "require_confirmation"
    REQUIRE_DOUBLE_CONFIRMATION = "require_double_confirmation"
    BLOCK = "block"


class EventPriority(StrEnum):
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"


class EventCategory(StrEnum):
    RUNTIME = "runtime"
    WORKER = "worker"
    STATE = "state"
    SECURITY = "security"
    OBSERVABILITY = "observability"

    PRESENCE = "presence"
    AWARENESS = "awareness"
    MEMORY = "memory"
    ROUTER = "router"
    COGNITION = "cognition"
    ACTION = "action"
    DIALOGUE = "dialogue"
    OPERATIONS = "operations"


class EventType(StrEnum):
    # Runtime lifecycle
    RUNTIME_STARTING = "runtime.starting"
    RUNTIME_STARTED = "runtime.started"
    RUNTIME_STOPPING = "runtime.stopping"
    RUNTIME_STOPPED = "runtime.stopped"
    RUNTIME_FAILED = "runtime.failed"
    RUNTIME_TICK = "runtime.tick"

    # Worker lifecycle
    WORKER_REGISTERED = "worker.registered"
    WORKER_STARTING = "worker.starting"
    WORKER_STARTED = "worker.started"
    WORKER_STOPPING = "worker.stopping"
    WORKER_STOPPED = "worker.stopped"
    WORKER_FAILED = "worker.failed"
    WORKER_HEALTH_UPDATED = "worker.health_updated"

    # State / context
    STATE_UPDATED = "state.updated"
    CONTEXT_UPDATED = "state.context_updated"
    SESSION_STARTED = "state.session_started"
    SESSION_ENDED = "state.session_ended"

    # Security
    PERMISSION_REQUESTED = "security.permission_requested"
    PERMISSION_GRANTED = "security.permission_granted"
    PERMISSION_DENIED = "security.permission_denied"
    POLICY_EVALUATED = "security.policy_evaluated"
    AUDIT_RECORDED = "security.audit_recorded"

    # Presence - legacy/simple contracts.
    # Keep these for backward compatibility with Phase 1 tests and early runtime flows.
    WAKE_WORD_DETECTED = "presence.wake_word_detected"
    USER_SPOKE = "presence.user_spoke"
    SPEECH_STARTED = "presence.speech_started"
    SPEECH_ENDED = "presence.speech_ended"
    INTERRUPT_REQUESTED = "presence.interrupt_requested"
    
    

    # Presence lifecycle
    PRESENCE_STARTED = "presence.started"
    PRESENCE_STOPPED = "presence.stopped"
    PRESENCE_STATE_CHANGED = "presence.state_changed"

    # Presence listening and wake flow
    PRESENCE_WAKE_DETECTED = "presence.wake_detected"
    PRESENCE_SLEEP_REQUESTED = "presence.sleep_requested"
    PRESENCE_LISTEN_STARTED = "presence.listen_started"
    PRESENCE_LISTEN_STOPPED = "presence.listen_stopped"
    PRESENCE_LISTEN_TIMEOUT = "presence.listen_timeout"

    # Presence user speech flow
    PRESENCE_USER_STARTED_SPEAKING = "presence.user_started_speaking"
    PRESENCE_USER_STOPPED_SPEAKING = "presence.user_stopped_speaking"
    PRESENCE_USER_INTERRUPTED = "presence.user_interrupted"

    # Presence transcript flow
    PRESENCE_TRANSCRIPT_PARTIAL = "presence.transcript_partial"
    PRESENCE_TRANSCRIPT_FINAL = "presence.transcript_final"
    PRESENCE_TRANSCRIPT_REJECTED = "presence.transcript_rejected"

    # Presence assistant speech flow
    PRESENCE_ASSISTANT_SPEAKING_STARTED = "presence.assistant_speaking_started"
    PRESENCE_ASSISTANT_SPEAKING_STOPPED = "presence.assistant_speaking_stopped"
    PRESENCE_ASSISTANT_SPEECH_CANCELLED = "presence.assistant_speech_cancelled"

    # Presence audio lifecycle.
    # These are categorized as EventCategory.PRESENCE for now.
    AUDIO_FRAME_CAPTURED = "audio.frame_captured"
    AUDIO_SPEECH_SEGMENT_STARTED = "audio.speech_segment_started"
    AUDIO_SPEECH_SEGMENT_COMPLETED = "audio.speech_segment_completed"
    AUDIO_PLAYBACK_STARTED = "audio.playback_started"
    AUDIO_PLAYBACK_COMPLETED = "audio.playback_completed"
    AUDIO_PLAYBACK_STOPPED = "audio.playback_stopped"
    AUDIO_PLAYBACK_FAILED = "audio.playback_failed"

    # Awareness
    WINDOW_CHANGED = "awareness.window_changed"
    SCREEN_UPDATED = "awareness.screen_updated"
    CLIPBOARD_CHANGED = "awareness.clipboard_changed"
    TERMINAL_OUTPUT_DETECTED = "awareness.terminal_output_detected"
    TERMINAL_ERROR_DETECTED = "awareness.terminal_error_detected"

    # Memory
    MEMORY_WRITE_REQUESTED = "memory.write_requested"
    MEMORY_WRITTEN = "memory.written"
    MEMORY_QUERY_REQUESTED = "memory.query_requested"
    MEMORY_QUERY_RESULT = "memory.query_result"

    # Router
    INTENT_DETECTED = "router.intent_detected"
    ROUTE_SELECTED = "router.route_selected"

    # Cognition
    COGNITION_REQUESTED = "cognition.requested"
    COGNITION_STARTED = "cognition.started"
    COGNITION_COMPLETED = "cognition.completed"
    PLAN_CREATED = "cognition.plan_created"

    # Actions
    ACTION_REQUESTED = "action.requested"
    ACTION_STARTED = "action.started"
    ACTION_COMPLETED = "action.completed"
    ACTION_FAILED = "action.failed"
    ACTION_CANCELLED = "action.cancelled"

    # Dialogue
    ASSISTANT_RESPONSE_REQUESTED = "dialogue.response_requested"
    ASSISTANT_RESPONSE_READY = "dialogue.response_ready"
    ASSISTANT_SPEAKING_STARTED = "dialogue.speaking_started"
    ASSISTANT_SPEAKING_STOPPED = "dialogue.speaking_stopped"
    
    # TTS / generated speech audio
    TTS_SYNTHESIS_STARTED = "tts.synthesis_started"
    TTS_SYNTHESIS_COMPLETED = "tts.synthesis_completed"
    TTS_SYNTHESIS_FAILED = "tts.synthesis_failed"
    AUDIO_SPEECH_CHUNK_READY = "audio.speech_chunk_ready"

    # Operations / observability
    METRIC_RECORDED = "operations.metric_recorded"
    LATENCY_RECORDED = "operations.latency_recorded"
    DIAGNOSTIC_REPORTED = "operations.diagnostic_reported"