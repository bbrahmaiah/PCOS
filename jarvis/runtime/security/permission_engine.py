from __future__ import annotations

from typing import Any

from jarvis.runtime.events import EventBus, RuntimeEvent
from jarvis.runtime.observability.performance_monitor import get_performance_monitor
from jarvis.runtime.observability.structured_logger import get_logger
from jarvis.runtime.security.audit_logger import AuditLogger
from jarvis.runtime.security.identity_manager import IdentityManager, SecurityIdentity
from jarvis.runtime.security.policy_engine import (
    PermissionRequest,
    PermissionResult,
    PolicyEngine,
)
from jarvis.runtime.shared.enums import EventCategory, EventType


class PermissionEngine:
    """
    Security boundary for all runtime actions.

    Future execution flow:
    Action request
        -> PermissionEngine
        -> PolicyEngine
        -> AuditLogger
        -> permission granted / denied event
        -> ActionExecutor only runs if allowed
    """

    def __init__(
        self,
        *,
        event_bus: EventBus,
        identity_manager: IdentityManager | None = None,
        policy_engine: PolicyEngine | None = None,
        audit_logger: AuditLogger | None = None,
    ) -> None:
        self.event_bus = event_bus
        self.identity_manager = identity_manager or IdentityManager()
        self.policy_engine = policy_engine or PolicyEngine()
        self.audit_logger = audit_logger or AuditLogger()

        self._logger = get_logger("security.permission_engine")
        self._performance = get_performance_monitor()

    def authenticate_local_user(
        self,
        *,
        user_id: str = "bala",
        display_name: str = "Bala",
    ) -> SecurityIdentity:
        identity = self.identity_manager.authenticate_local_user(
            user_id=user_id,
            display_name=display_name,
        )

        self._logger.info(
            "identity_authenticated",
            identity_id=identity.identity_id,
            user_id=identity.user_id,
            display_name=identity.display_name,
        )

        return identity

    def request_permission(
        self,
        *,
        action: str,
        requested_by: str,
        payload: dict[str, Any] | None = None,
        correlation_id: str | None = None,
    ) -> PermissionResult:
        with self._performance.measure(
            "security.request_permission",
            correlation_id=correlation_id,
        ):
            request = PermissionRequest(
                action=action,
                requested_by=requested_by,
                payload=payload or {},
                correlation_id=correlation_id,
            )

            self._emit_security_event(
                EventType.PERMISSION_REQUESTED,
                payload={
                    "request_id": request.request_id,
                    "action": request.action,
                    "requested_by": request.requested_by,
                    "correlation_id": request.correlation_id,
                },
            )

            identity = self.identity_manager.current_identity()

            result = self.policy_engine.evaluate(
                request,
                identity=identity,
            )

            self._emit_security_event(
                EventType.POLICY_EVALUATED,
                payload={
                    "request_id": request.request_id,
                    "action": request.action,
                    "decision": result.decision.value,
                    "risk": result.risk.value,
                    "reason": result.reason,
                    "identity_id": result.identity_id,
                },
            )

            audit_record = self.audit_logger.record_result(result)

            self._emit_security_event(
                EventType.AUDIT_RECORDED,
                payload={
                    "audit_id": audit_record.audit_id,
                    "request_id": request.request_id,
                    "action": request.action,
                    "decision": result.decision.value,
                },
            )

            if result.allowed:
                self._emit_security_event(
                    EventType.PERMISSION_GRANTED,
                    payload={
                        "request_id": request.request_id,
                        "action": request.action,
                        "risk": result.risk.value,
                    },
                )
            else:
                self._emit_security_event(
                    EventType.PERMISSION_DENIED,
                    payload={
                        "request_id": request.request_id,
                        "action": request.action,
                        "decision": result.decision.value,
                        "risk": result.risk.value,
                        "reason": result.reason,
                    },
                )

            self._logger.info(
                "permission_evaluated",
                request_id=request.request_id,
                action=request.action,
                decision=result.decision.value,
                risk=result.risk.value,
                allowed=result.allowed,
            )

            return result

    def _emit_security_event(
        self,
        event_type: EventType,
        *,
        payload: dict[str, object],
    ) -> None:
        event = RuntimeEvent(
            event_type=event_type,
            category=EventCategory.SECURITY,
            source="permission_engine",
            payload=payload,
        )

        self.event_bus.publish(event)