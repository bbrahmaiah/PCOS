from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from fnmatch import fnmatch
from typing import Any
from uuid import uuid4

from jarvis.runtime.security.identity_manager import SecurityIdentity
from jarvis.runtime.shared.enums import PermissionDecision, RiskLevel


def utc_now() -> datetime:
    return datetime.now(UTC)


def new_request_id() -> str:
    return uuid4().hex


@dataclass(frozen=True, slots=True)
class PermissionRequest:
    """
    Security request for one action.

    Every future OS/browser/shell action should become a PermissionRequest
    before execution.
    """

    action: str
    requested_by: str
    payload: dict[str, Any] = field(default_factory=dict)
    request_id: str = field(default_factory=new_request_id)
    correlation_id: str | None = None
    created_at: datetime = field(default_factory=utc_now)

    def __post_init__(self) -> None:
        if not self.action.strip():
            raise ValueError("permission request action cannot be empty.")

        if not self.requested_by.strip():
            raise ValueError("permission request requested_by cannot be empty.")


@dataclass(frozen=True, slots=True)
class PermissionPolicy:
    """
    One policy rule.

    action_pattern supports exact names and wildcards:
    - browser.open_url
    - browser.*
    - system.*
    """

    action_pattern: str
    decision: PermissionDecision
    risk: RiskLevel
    reason: str
    require_authenticated: bool = True

    def __post_init__(self) -> None:
        if not self.action_pattern.strip():
            raise ValueError("policy action_pattern cannot be empty.")

        if not self.reason.strip():
            raise ValueError("policy reason cannot be empty.")

    def matches(self, action: str) -> bool:
        return fnmatch(action.strip(), self.action_pattern.strip())


@dataclass(frozen=True, slots=True)
class PermissionResult:
    """
    Final permission decision.
    """

    request: PermissionRequest
    decision: PermissionDecision
    risk: RiskLevel
    reason: str
    identity_id: str | None = None
    evaluated_at: datetime = field(default_factory=utc_now)

    @property
    def allowed(self) -> bool:
        return self.decision == PermissionDecision.ALLOW

    @property
    def denied(self) -> bool:
        return self.decision in {
            PermissionDecision.DENY,
            PermissionDecision.BLOCK,
        }

    @property
    def requires_confirmation(self) -> bool:
        return self.decision in {
            PermissionDecision.REQUIRE_CONFIRMATION,
            PermissionDecision.REQUIRE_DOUBLE_CONFIRMATION,
        }


class PolicyEngine:
    """
    Deterministic permission policy engine.

    Important rule:
    unknown actions are denied by default.
    """

    def __init__(self, *, deny_unknown_actions: bool = True) -> None:
        self.deny_unknown_actions = deny_unknown_actions
        self._policies: list[PermissionPolicy] = []
        self._load_default_policies()

    def add_policy(self, policy: PermissionPolicy) -> None:
        if not isinstance(policy, PermissionPolicy):
            raise TypeError("policy must be a PermissionPolicy.")

        self._policies.append(policy)

    def policies(self) -> tuple[PermissionPolicy, ...]:
        return tuple(self._policies)

    def evaluate(
        self,
        request: PermissionRequest,
        *,
        identity: SecurityIdentity | None = None,
    ) -> PermissionResult:
        if not isinstance(request, PermissionRequest):
            raise TypeError("request must be a PermissionRequest.")

        policy = self._find_policy(request.action)

        if policy is None:
            if self.deny_unknown_actions:
                return PermissionResult(
                    request=request,
                    decision=PermissionDecision.DENY,
                    risk=RiskLevel.HIGH,
                    reason="Unknown action denied by default.",
                    identity_id=identity.identity_id if identity else None,
                )

            return PermissionResult(
                request=request,
                decision=PermissionDecision.REQUIRE_CONFIRMATION,
                risk=RiskLevel.MEDIUM,
                reason="Unknown action requires confirmation.",
                identity_id=identity.identity_id if identity else None,
            )

        if policy.require_authenticated and identity is None:
            return PermissionResult(
                request=request,
                decision=PermissionDecision.DENY,
                risk=RiskLevel.HIGH,
                reason="Authenticated identity required.",
            )

        return PermissionResult(
            request=request,
            decision=policy.decision,
            risk=policy.risk,
            reason=policy.reason,
            identity_id=identity.identity_id if identity else None,
        )

    def _find_policy(self, action: str) -> PermissionPolicy | None:
        for policy in self._policies:
            if policy.matches(action):
                return policy

        return None

    def _load_default_policies(self) -> None:
        """
        Safe defaults for early JARVIS runtime.

        Dangerous actions are denied or require confirmation until we build
        a stronger confirmation pipeline.
        """

        self._policies.extend(
            [
                PermissionPolicy(
                    action_pattern="browser.open_url",
                    decision=PermissionDecision.ALLOW,
                    risk=RiskLevel.SAFE,
                    reason=(
                        "Opening browser URLs is allowed "
                        "for authenticated users."
                    ),
                ),
                PermissionPolicy(
                    action_pattern="browser.search",
                    decision=PermissionDecision.ALLOW,
                    risk=RiskLevel.SAFE,
                    reason=(
                        "Browser search is allowed "
                        "for authenticated users."
                    ),
                ),
                PermissionPolicy(
                    action_pattern="system.open_app",
                    decision=PermissionDecision.ALLOW,
                    risk=RiskLevel.LOW,
                    reason=(
                        "Opening local applications is allowed "
                        "for authenticated users."
                    ),
                ),
                PermissionPolicy(
                    action_pattern="media.*",
                    decision=PermissionDecision.ALLOW,
                    risk=RiskLevel.SAFE,
                    reason=(
                        "Media control is allowed "
                        "for authenticated users."
                    ),
                ),
                PermissionPolicy(
                    action_pattern="filesystem.delete",
                    decision=PermissionDecision.REQUIRE_DOUBLE_CONFIRMATION,
                    risk=RiskLevel.CRITICAL,
                    reason="Deleting files requires double confirmation.",
                ),
                PermissionPolicy(
                    action_pattern="shell.execute",
                    decision=PermissionDecision.REQUIRE_CONFIRMATION,
                    risk=RiskLevel.HIGH,
                    reason="Shell execution requires confirmation.",
                ),
                PermissionPolicy(
                    action_pattern="network.download",
                    decision=PermissionDecision.REQUIRE_CONFIRMATION,
                    risk=RiskLevel.MEDIUM,
                    reason="Downloads require confirmation.",
                ),
            ]
        )