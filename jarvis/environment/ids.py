from __future__ import annotations

from uuid import uuid4

ENVIRONMENT_ID_PREFIX = "env"
DISPLAY_ID_PREFIX = "display"
WINDOW_ID_PREFIX = "window"
APP_ID_PREFIX = "app"
PROCESS_ID_PREFIX = "process"
REGION_ID_PREFIX = "region"
ELEMENT_ID_PREFIX = "element"
SNAPSHOT_ID_PREFIX = "snapshot"
DELTA_ID_PREFIX = "delta"
EVENT_ID_PREFIX = "event"
WORKFLOW_ID_PREFIX = "workflow"
INTENT_ID_PREFIX = "intent"
INTERACTION_ID_PREFIX = "interaction"
VERIFICATION_ID_PREFIX = "verify"
RECOVERY_ID_PREFIX = "recovery"
SIMULATION_ID_PREFIX = "simulation"
TRUST_ID_PREFIX = "trust"
MEMORY_ID_PREFIX = "envmem"


def new_environment_id() -> str:
    return _new_id(ENVIRONMENT_ID_PREFIX)


def new_display_id() -> str:
    return _new_id(DISPLAY_ID_PREFIX)


def new_window_id() -> str:
    return _new_id(WINDOW_ID_PREFIX)


def new_app_id() -> str:
    return _new_id(APP_ID_PREFIX)


def new_process_id() -> str:
    return _new_id(PROCESS_ID_PREFIX)


def new_region_id() -> str:
    return _new_id(REGION_ID_PREFIX)


def new_element_id() -> str:
    return _new_id(ELEMENT_ID_PREFIX)


def new_snapshot_id() -> str:
    return _new_id(SNAPSHOT_ID_PREFIX)


def new_delta_id() -> str:
    return _new_id(DELTA_ID_PREFIX)


def new_event_id() -> str:
    return _new_id(EVENT_ID_PREFIX)


def new_workflow_id() -> str:
    return _new_id(WORKFLOW_ID_PREFIX)


def new_intent_id() -> str:
    return _new_id(INTENT_ID_PREFIX)


def new_interaction_id() -> str:
    return _new_id(INTERACTION_ID_PREFIX)


def new_verification_id() -> str:
    return _new_id(VERIFICATION_ID_PREFIX)


def new_recovery_id() -> str:
    return _new_id(RECOVERY_ID_PREFIX)


def new_simulation_id() -> str:
    return _new_id(SIMULATION_ID_PREFIX)


def new_trust_id() -> str:
    return _new_id(TRUST_ID_PREFIX)


def new_environment_memory_id() -> str:
    return _new_id(MEMORY_ID_PREFIX)


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid4().hex}"