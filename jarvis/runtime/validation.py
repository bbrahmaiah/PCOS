from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from time import perf_counter
from typing import Any

from jarvis.runtime.kernel import RuntimeKernel
from jarvis.runtime.observability.structured_logger import get_logger
from jarvis.runtime.shared.enums import (
    EventType,
    PermissionDecision,
    RuntimeStatus,
    SystemMode,
)


def utc_now() -> datetime:
    return datetime.now(UTC)


@dataclass(frozen=True, slots=True)
class ValidationCheck:
    """
    Result of one runtime integration validation check.
    """

    name: str
    passed: bool
    details: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    checked_at: datetime = field(default_factory=utc_now)


@dataclass(frozen=True, slots=True)
class ValidationReport:
    """
    Full integration validation report.
    """

    passed: bool
    checks: tuple[ValidationCheck, ...]
    started_at: datetime
    finished_at: datetime
    duration_ms: float

    @property
    def failed_checks(self) -> tuple[ValidationCheck, ...]:
        return tuple(check for check in self.checks if not check.passed)

    @property
    def passed_count(self) -> int:
        return sum(1 for check in self.checks if check.passed)

    @property
    def failed_count(self) -> int:
        return sum(1 for check in self.checks if not check.passed)


KernelFactory = Callable[[], RuntimeKernel]


class RuntimeIntegrationValidator:
    """
    End-to-end validator for the Phase 1 runtime foundation.

    This is not a unit-test replacement. It is a runtime-level proof that:
    - kernel boots
    - subsystems are wired together
    - state works
    - security works
    - scheduler works
    - cancellation works
    - health works
    - event flow works
    - shutdown works
    """

    def __init__(
        self,
        *,
        kernel_factory: KernelFactory = RuntimeKernel,
    ) -> None:
        self.kernel_factory = kernel_factory
        self._logger = get_logger("runtime.validation")

    def run(self) -> ValidationReport:
        started_at = utc_now()
        started_perf = perf_counter()
        checks: list[ValidationCheck] = []

        kernel = self.kernel_factory()

        try:
            kernel.start()

            start_snapshot = kernel.snapshot()

            self._record_check(
                checks,
                name="kernel_started",
                passed=(
                    start_snapshot.state.runtime.status == RuntimeStatus.RUNNING
                    and start_snapshot.event_bus.running
                ),
                details={
                    "runtime_status": start_snapshot.state.runtime.status.value,
                    "event_bus_running": start_snapshot.event_bus.running,
                    "worker_count": start_snapshot.workers.worker_count,
                },
            )

            self._validate_state_engine(kernel, checks)
            self._validate_security_flow(kernel, checks)
            self._validate_cancellation(kernel, checks)
            self._validate_scheduler(kernel, checks)
            self._validate_health(kernel, checks)
            self._validate_event_flow(kernel, checks)

        except Exception as exc:
            self._record_check(
                checks,
                name="integration_exception",
                passed=False,
                error=f"{type(exc).__name__}: {exc}",
            )

        finally:
            try:
                kernel.stop()
                stopped_snapshot = kernel.snapshot()

                self._record_check(
                    checks,
                    name="kernel_stopped",
                    passed=(
                        stopped_snapshot.state.runtime.status == RuntimeStatus.STOPPED
                        and not stopped_snapshot.event_bus.running
                    ),
                    details={
                        "runtime_status": stopped_snapshot.state.runtime.status.value,
                        "event_bus_running": stopped_snapshot.event_bus.running,
                    },
                )

            except Exception as exc:
                self._record_check(
                    checks,
                    name="kernel_stop_failed",
                    passed=False,
                    error=f"{type(exc).__name__}: {exc}",
                )

        finished_at = utc_now()
        duration_ms = (perf_counter() - started_perf) * 1000
        passed = all(check.passed for check in checks)

        report = ValidationReport(
            passed=passed,
            checks=tuple(checks),
            started_at=started_at,
            finished_at=finished_at,
            duration_ms=duration_ms,
        )

        self._logger.info(
            "runtime_integration_validation_completed",
            passed=report.passed,
            passed_count=report.passed_count,
            failed_count=report.failed_count,
            duration_ms=round(report.duration_ms, 3),
        )

        return report

    def _validate_state_engine(
        self,
        kernel: RuntimeKernel,
        checks: list[ValidationCheck],
    ) -> None:
        session = kernel.state_engine.start_session(
            user_id="integration",
            active_goal="Validate runtime foundation",
            active_topic="Phase 1 integration",
        )

        kernel.state_engine.set_system_mode(SystemMode.ACTIVE)
        kernel.state_engine.set_context(
            "integration.active_window",
            {"title": "JARVIS Runtime Validation"},
        )

        snapshot = kernel.state_engine.snapshot()
        context_value = snapshot.context.values.get("integration.active_window", {})

        self._record_check(
            checks,
            name="state_engine_updates",
            passed=(
                snapshot.runtime.mode == SystemMode.ACTIVE
                and snapshot.session is not None
                and snapshot.session.session_id == session.session_id
                and context_value.get("title") == "JARVIS Runtime Validation"
            ),
            details={
                "session_id": session.session_id,
                "mode": snapshot.runtime.mode.value,
                "context_size": snapshot.context.size,
            },
        )

    def _validate_security_flow(
        self,
        kernel: RuntimeKernel,
        checks: list[ValidationCheck],
    ) -> None:
        identity = kernel.permission_engine.authenticate_local_user(
            user_id="integration",
            display_name="Integration Test",
        )

        result = kernel.permission_engine.request_permission(
            action="browser.search",
            requested_by="integration_validator",
            payload={"query": "jarvis runtime validation"},
        )

        self._record_check(
            checks,
            name="security_permission_flow",
            passed=(
                identity.authenticated
                and result.allowed
                and result.decision == PermissionDecision.ALLOW
            ),
            details={
                "identity_id": identity.identity_id,
                "decision": result.decision.value,
                "risk": result.risk.value,
            },
        )

    def _validate_cancellation(
        self,
        kernel: RuntimeKernel,
        checks: list[ValidationCheck],
    ) -> None:
        token = kernel.cancellation_manager.create_token(
            token_id="integration-token",
            metadata={"source": "integration_validator"},
        )

        token.cancel("integration cancellation test")
        snapshot = kernel.cancellation_manager.snapshot()

        self._record_check(
            checks,
            name="cancellation_manager",
            passed=token.cancelled and snapshot.cancelled_count == 1,
            details={
                "token_id": token.token_id,
                "cancelled": token.cancelled,
                "cancelled_count": snapshot.cancelled_count,
            },
        )

    def _validate_scheduler(
        self,
        kernel: RuntimeKernel,
        checks: list[ValidationCheck],
    ) -> None:
        task_completed = False

        def mark_task_completed() -> None:
            nonlocal task_completed
            task_completed = True

        kernel.scheduler.register_task(
            name="integration_scheduler_task",
            callback=mark_task_completed,
            interval_seconds=0.001,
            run_once=True,
            start_immediately=True,
        )

        executed = kernel.scheduler.run_due_tasks()
        scheduler_snapshot = kernel.scheduler.scheduler_snapshot()

        self._record_check(
            checks,
            name="scheduler_executes_task",
            passed=executed == 1 and task_completed,
            details={
                "executed": executed,
                "task_completed": task_completed,
                "task_count": scheduler_snapshot.task_count,
            },
        )

    def _validate_health(
        self,
        kernel: RuntimeKernel,
        checks: list[ValidationCheck],
    ) -> None:
        snapshot = kernel.snapshot()

        self._record_check(
            checks,
            name="runtime_health_check",
            passed=snapshot.health.healthy,
            details={
                "healthy": snapshot.health.healthy,
                "reasons": list(snapshot.health.reasons),
                "worker_count": snapshot.health.worker_count,
                "running_workers": snapshot.health.running_workers,
                "failed_workers": snapshot.health.failed_workers,
            },
        )

    def _validate_event_flow(
        self,
        kernel: RuntimeKernel,
        checks: list[ValidationCheck],
    ) -> None:
        kernel.event_bus.drain(timeout_seconds=2.0)

        history = kernel.event_bus.history()
        event_types = {event.event_type for event in history}

        required_events = {
            EventType.RUNTIME_STARTED,
            EventType.STATE_UPDATED,
            EventType.PERMISSION_REQUESTED,
            EventType.PERMISSION_GRANTED,
            EventType.WORKER_STARTED,
        }

        missing = required_events - event_types

        self._record_check(
            checks,
            name="runtime_event_flow",
            passed=not missing,
            details={
                "required_events": sorted(event.value for event in required_events),
                "missing_events": sorted(event.value for event in missing),
                "history_size": len(history),
            },
        )

    @staticmethod
    def _record_check(
        checks: list[ValidationCheck],
        *,
        name: str,
        passed: bool,
        details: dict[str, Any] | None = None,
        error: str | None = None,
    ) -> None:
        checks.append(
            ValidationCheck(
                name=name,
                passed=passed,
                details=details or {},
                error=error,
            )
        )