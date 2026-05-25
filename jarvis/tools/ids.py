from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4


def utc_now() -> datetime:
    """
    Return timezone-aware UTC timestamp.

    Tool/action runtime timestamps must always be timezone-aware so audit logs,
    cancellation, retries, and scheduling remain deterministic.
    """

    return datetime.now(UTC)


def new_tool_id() -> str:
    """
    Create a unique tool identifier.
    """

    return f"tool_{uuid4().hex}"


def new_action_id() -> str:
    """
    Create a unique action identifier.
    """

    return f"action_{uuid4().hex}"


def new_action_step_id() -> str:
    """
    Create a unique action-step identifier.
    """

    return f"step_{uuid4().hex}"


def new_action_plan_id() -> str:
    """
    Create a unique action-plan identifier.
    """

    return f"plan_{uuid4().hex}"


def new_action_result_id() -> str:
    """
    Create a unique action-result identifier.
    """

    return f"result_{uuid4().hex}"


def new_action_error_id() -> str:
    """
    Create a unique action-error identifier.
    """

    return f"error_{uuid4().hex}"


def new_policy_decision_id() -> str:
    """
    Create a unique policy-decision identifier.
    """

    return f"policy_{uuid4().hex}"