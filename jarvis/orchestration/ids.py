from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

TaskId = str
JobId = str
WorkerId = str
OrchestrationId = str


_TASK_PREFIX = "task"
_JOB_PREFIX = "job"
_WORKER_PREFIX = "worker"
_ORCHESTRATION_PREFIX = "orch"


def utc_now() -> datetime:
    """
    Return timezone-aware UTC time for orchestration contracts.
    """

    return datetime.now(UTC)


def new_task_id() -> TaskId:
    """
    Create a new orchestration task id.
    """

    return _new_prefixed_id(_TASK_PREFIX)


def new_job_id() -> JobId:
    """
    Create a new orchestration job id.
    """

    return _new_prefixed_id(_JOB_PREFIX)


def new_worker_id() -> WorkerId:
    """
    Create a new orchestration worker id.
    """

    return _new_prefixed_id(_WORKER_PREFIX)


def new_orchestration_id() -> OrchestrationId:
    """
    Create a new orchestration run id.
    """

    return _new_prefixed_id(_ORCHESTRATION_PREFIX)


def validate_task_id(value: str) -> TaskId:
    """
    Validate a task id boundary.
    """

    return _validate_prefixed_id(value=value, prefix=_TASK_PREFIX)


def validate_job_id(value: str) -> JobId:
    """
    Validate a job id boundary.
    """

    return _validate_prefixed_id(value=value, prefix=_JOB_PREFIX)


def validate_worker_id(value: str) -> WorkerId:
    """
    Validate a worker id boundary.
    """

    return _validate_prefixed_id(value=value, prefix=_WORKER_PREFIX)


def validate_orchestration_id(value: str) -> OrchestrationId:
    """
    Validate an orchestration id boundary.
    """

    return _validate_prefixed_id(value=value, prefix=_ORCHESTRATION_PREFIX)


def _new_prefixed_id(prefix: str) -> str:
    return f"{prefix}_{uuid4().hex}"


def _validate_prefixed_id(*, value: str, prefix: str) -> str:
    cleaned = value.strip()

    if not cleaned:
        raise ValueError("id cannot be empty.")

    required_prefix = f"{prefix}_"

    if not cleaned.startswith(required_prefix):
        raise ValueError(f"id must start with '{required_prefix}'.")

    return cleaned