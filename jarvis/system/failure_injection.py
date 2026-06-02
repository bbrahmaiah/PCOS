from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum

from jarvis.conversation.models import ConversationMode, TurnInputSource
from jarvis.conversation.runtime import RealConversationInput
from jarvis.system.bootstrap import JarvisBootstrapStatus
from jarvis.system.contracts import (
    JarvisPipelineStatus,
    utc_now,
)
from jarvis.system.live_wiring import LiveDependencyWiring


class FailureInjectionStatus(StrEnum):
    PASSED = "passed"
    FAILED = "failed"


class FailureInjectionScenarioKind(StrEnum):
    MEMORY_FACTORY_FAILURE = "memory_factory_failure"
    COGNITION_RUNTIME_FAILURE = "cognition_runtime_failure"
    MEMORY_WRITE_FAILURE = "memory_write_failure"
    PRESENCE_OUTPUT_FAILURE = "presence_output_failure"
    ORCHESTRATION_START_FAILURE = "orchestration_start_failure"
    INTERRUPTION_DURING_FAILURE = "interruption_during_failure"
    SHUTDOWN_AFTER_FAILURE = "shutdown_after_failure"


class FailureInjectionOutcome(StrEnum):
    DETECTED = "detected"
    CONTAINED = "contained"
    RECOVERED = "recovered"
    DEGRADED = "degraded"
    CLEAN_SHUTDOWN = "clean_shutdown"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class FailureInjectionScenario:
    kind: FailureInjectionScenarioKind
    name: str
    description: str
    runner: Callable[[LiveDependencyWiring], FailureInjectionScenarioResult]

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("failure scenario name cannot be empty.")
        if not self.description.strip():
            raise ValueError("failure scenario description cannot be empty.")


@dataclass(frozen=True, slots=True)
class FailureInjectionScenarioResult:
    kind: FailureInjectionScenarioKind
    passed: bool
    outcome: FailureInjectionOutcome
    message: str
    created_at: datetime
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class FailureInjectionReport:
    status: FailureInjectionStatus
    results: tuple[FailureInjectionScenarioResult, ...]
    started_at: datetime
    finished_at: datetime
    error: str | None = None
    metadata: dict[str, object] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return self.status == FailureInjectionStatus.PASSED

    @property
    def passed_count(self) -> int:
        return sum(1 for result in self.results if result.passed)

    @property
    def failed_count(self) -> int:
        return sum(1 for result in self.results if not result.passed)


@dataclass(frozen=True, slots=True)
class FailureInjectionConfig:
    session_id: str = "failure_injection"
    normal_prompt: str = "What am I building?"
    memory_prompt: str = "Remember that failure injection is important."
    interruption_prompt: str = "stop"
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.session_id.strip():
            raise ValueError("failure injection session_id cannot be empty.")
        if not self.normal_prompt.strip():
            raise ValueError("failure injection normal_prompt cannot be empty.")
        if not self.memory_prompt.strip():
            raise ValueError("failure injection memory_prompt cannot be empty.")
        if not self.interruption_prompt.strip():
            raise ValueError(
                "failure injection interruption_prompt cannot be empty."
            )


class FailureInjectionReview:
    """
    Step 46D.5 failure injection review.

    This review deliberately injects controlled failures and verifies that
    JARVIS detects, contains, degrades, recovers, or shuts down cleanly.

    It does not execute desktop tools or autonomous actions.
    """

    def __init__(
        self,
        *,
        config: FailureInjectionConfig,
        wiring: LiveDependencyWiring,
        scenarios: tuple[FailureInjectionScenario, ...] | None = None,
    ) -> None:
        self._config = config
        self._wiring = wiring
        self._scenarios = scenarios or default_failure_scenarios(config)

    def run(self) -> FailureInjectionReport:
        started_at = utc_now()
        results: list[FailureInjectionScenarioResult] = []

        try:
            for scenario in self._scenarios:
                try:
                    results.append(scenario.runner(self._wiring))
                except Exception as exc:
                    results.append(
                        FailureInjectionScenarioResult(
                            kind=scenario.kind,
                            passed=False,
                            outcome=FailureInjectionOutcome.FAILED,
                            message=(
                                "failure scenario crashed instead of "
                                "returning a controlled result"
                            ),
                            created_at=utc_now(),
                            metadata={
                                "scenario": scenario.name,
                                "error": f"{type(exc).__name__}: {exc}",
                            },
                        )
                    )

            return _report(
                results=results,
                started_at=started_at,
                error=None,
                metadata=self._config.metadata,
            )

        except Exception as exc:
            return _report(
                results=results,
                started_at=started_at,
                error=f"{type(exc).__name__}: {exc}",
                metadata=self._config.metadata,
            )


def default_failure_scenarios(
    config: FailureInjectionConfig,
) -> tuple[FailureInjectionScenario, ...]:
    return (
        FailureInjectionScenario(
            kind=FailureInjectionScenarioKind.MEMORY_FACTORY_FAILURE,
            name="memory_factory_failure",
            description="memory factory fails before boot",
            runner=_memory_factory_failure,
        ),
        FailureInjectionScenario(
            kind=FailureInjectionScenarioKind.COGNITION_RUNTIME_FAILURE,
            name="cognition_runtime_failure",
            description="cognition pipeline fails during request",
            runner=lambda wiring: _cognition_runtime_failure(
                wiring=wiring,
                config=config,
            ),
        ),
        FailureInjectionScenario(
            kind=FailureInjectionScenarioKind.MEMORY_WRITE_FAILURE,
            name="memory_write_failure",
            description="memory write fails during governed write",
            runner=lambda wiring: _memory_write_failure(
                wiring=wiring,
                config=config,
            ),
        ),
        FailureInjectionScenario(
            kind=FailureInjectionScenarioKind.PRESENCE_OUTPUT_FAILURE,
            name="presence_output_failure",
            description="presence response publication fails safely",
            runner=lambda wiring: _presence_output_failure(
                wiring=wiring,
                config=config,
            ),
        ),
        FailureInjectionScenario(
            kind=FailureInjectionScenarioKind.ORCHESTRATION_START_FAILURE,
            name="orchestration_start_failure",
            description="orchestration runtime fails during start",
            runner=_orchestration_start_failure,
        ),
        FailureInjectionScenario(
            kind=FailureInjectionScenarioKind.INTERRUPTION_DURING_FAILURE,
            name="interruption_during_failure",
            description="interruption still cancels without cognition execution",
            runner=lambda wiring: _interruption_during_failure(
                wiring=wiring,
                config=config,
            ),
        ),
        FailureInjectionScenario(
            kind=FailureInjectionScenarioKind.SHUTDOWN_AFTER_FAILURE,
            name="shutdown_after_failure",
            description="runtime shuts down cleanly after handled failure",
            runner=lambda wiring: _shutdown_after_failure(
                wiring=wiring,
                config=config,
            ),
        ),
    )


def _memory_factory_failure(
    wiring: LiveDependencyWiring,
) -> FailureInjectionScenarioResult:
    report = wiring.validate()

    return FailureInjectionScenarioResult(
        kind=FailureInjectionScenarioKind.MEMORY_FACTORY_FAILURE,
        passed=not report.succeeded or report.error is not None,
        outcome=(
            FailureInjectionOutcome.DETECTED
            if not report.succeeded or report.error is not None
            else FailureInjectionOutcome.FAILED
        ),
        message="memory factory failure was detected before safe boot",
        created_at=utc_now(),
        metadata={
            "wiring_status": report.status.value,
            "error": report.error,
        },
    )


def _cognition_runtime_failure(
    *,
    wiring: LiveDependencyWiring,
    config: FailureInjectionConfig,
) -> FailureInjectionScenarioResult:
    bootstrap = wiring.build_bootstrap()
    boot = bootstrap.start()

    try:
        system = bootstrap.system
        if system is None:
            raise RuntimeError("bootstrap did not expose JarvisSystem.")

        result = system.process_user_utterance(
            text=config.normal_prompt,
            session_id=config.session_id,
        )

        passed = (
            result.status == JarvisPipelineStatus.FAILED
            or (
                result.response is not None
                and not result.response.succeeded
            )
        )

        return FailureInjectionScenarioResult(
            kind=FailureInjectionScenarioKind.COGNITION_RUNTIME_FAILURE,
            passed=passed,
            outcome=(
                FailureInjectionOutcome.CONTAINED
                if passed
                else FailureInjectionOutcome.FAILED
            ),
            message="cognition runtime failure was contained by pipeline",
            created_at=utc_now(),
            metadata={
                "boot_status": boot.status.value,
                "pipeline_status": result.status.value,
                "reason": result.reason,
                "event_count": len(result.events),
            },
        )

    finally:
        bootstrap.stop()


def _memory_write_failure(
    *,
    wiring: LiveDependencyWiring,
    config: FailureInjectionConfig,
) -> FailureInjectionScenarioResult:
    bootstrap = wiring.build_bootstrap()
    boot = bootstrap.start()

    try:
        system = bootstrap.system
        if system is None:
            raise RuntimeError("bootstrap did not expose JarvisSystem.")

        result = system.process_user_utterance(
            text=config.memory_prompt,
            session_id=config.session_id,
        )

        response = result.response
        passed = (
            result.status == JarvisPipelineStatus.COMPLETED
            and response is not None
            and not response.wrote_memory
            and response.memory_write.status.value in {
                "blocked",
                "failed",
            }
        )

        return FailureInjectionScenarioResult(
            kind=FailureInjectionScenarioKind.MEMORY_WRITE_FAILURE,
            passed=passed,
            outcome=(
                FailureInjectionOutcome.CONTAINED
                if passed
                else FailureInjectionOutcome.FAILED
            ),
            message="memory write failure was surfaced without crashing",
            created_at=utc_now(),
            metadata={
                "boot_status": boot.status.value,
                "pipeline_status": result.status.value,
                "memory_write_status": (
                    response.memory_write.status.value
                    if response is not None
                    else None
                ),
            },
        )

    finally:
        bootstrap.stop()


def _presence_output_failure(
    *,
    wiring: LiveDependencyWiring,
    config: FailureInjectionConfig,
) -> FailureInjectionScenarioResult:
    bootstrap = wiring.build_bootstrap()
    boot = bootstrap.start()

    try:
        system = bootstrap.system
        if system is None:
            raise RuntimeError("bootstrap did not expose JarvisSystem.")

        try:
            system.publish_presence_response_ready(
                text="Failure injection presence output."
            )
        except Exception as exc:
            return FailureInjectionScenarioResult(
                kind=FailureInjectionScenarioKind.PRESENCE_OUTPUT_FAILURE,
                passed=True,
                outcome=FailureInjectionOutcome.CONTAINED,
                message="presence output failure was detected and contained",
                created_at=utc_now(),
                metadata={
                    "boot_status": boot.status.value,
                    "error": f"{type(exc).__name__}: {exc}",
                    "session_id": config.session_id,
                },
            )

        return FailureInjectionScenarioResult(
            kind=FailureInjectionScenarioKind.PRESENCE_OUTPUT_FAILURE,
            passed=False,
            outcome=FailureInjectionOutcome.FAILED,
            message="presence output failure was not triggered",
            created_at=utc_now(),
            metadata={"boot_status": boot.status.value},
        )

    finally:
        bootstrap.stop()


def _orchestration_start_failure(
    wiring: LiveDependencyWiring,
) -> FailureInjectionScenarioResult:
    bootstrap = wiring.build_bootstrap()
    result = bootstrap.start()

    try:
        passed = (
            result.status == JarvisBootstrapStatus.FAILED
            and result.error is not None
        )

        return FailureInjectionScenarioResult(
            kind=FailureInjectionScenarioKind.ORCHESTRATION_START_FAILURE,
            passed=passed,
            outcome=(
                FailureInjectionOutcome.DETECTED
                if passed
                else FailureInjectionOutcome.FAILED
            ),
            message="orchestration startup failure was detected by bootstrap",
            created_at=utc_now(),
            metadata={
                "bootstrap_status": result.status.value,
                "error": result.error,
            },
        )

    finally:
        bootstrap.stop()


def _interruption_during_failure(
    *,
    wiring: LiveDependencyWiring,
    config: FailureInjectionConfig,
) -> FailureInjectionScenarioResult:
    bootstrap = wiring.build_bootstrap()
    boot = bootstrap.start()

    try:
        system = bootstrap.system
        if system is None:
            raise RuntimeError("bootstrap did not expose JarvisSystem.")

        result = system.process_user_utterance(
            text=config.interruption_prompt,
            session_id=config.session_id,
            signal=_interruption_signal(config.interruption_prompt),
        )

        passed = (
            result.status == JarvisPipelineStatus.CANCELLED
            and result.cancelled
            and result.response is None
        )

        return FailureInjectionScenarioResult(
            kind=FailureInjectionScenarioKind.INTERRUPTION_DURING_FAILURE,
            passed=passed,
            outcome=(
                FailureInjectionOutcome.CONTAINED
                if passed
                else FailureInjectionOutcome.FAILED
            ),
            message="interruption path cancelled safely under failure review",
            created_at=utc_now(),
            metadata={
                "boot_status": boot.status.value,
                "pipeline_status": result.status.value,
                "cancelled": result.cancelled,
            },
        )

    finally:
        bootstrap.stop()


def _shutdown_after_failure(
    *,
    wiring: LiveDependencyWiring,
    config: FailureInjectionConfig,
) -> FailureInjectionScenarioResult:
    bootstrap = wiring.build_bootstrap()
    boot = bootstrap.start()

    try:
        system = bootstrap.system
        if system is None:
            raise RuntimeError("bootstrap did not expose JarvisSystem.")

        system.process_user_utterance(
            text=config.normal_prompt,
            session_id=config.session_id,
        )
        stop = bootstrap.stop()

        passed = (
            stop.status == JarvisBootstrapStatus.STOPPED
            and stop.error is None
        )

        return FailureInjectionScenarioResult(
            kind=FailureInjectionScenarioKind.SHUTDOWN_AFTER_FAILURE,
            passed=passed,
            outcome=(
                FailureInjectionOutcome.CLEAN_SHUTDOWN
                if passed
                else FailureInjectionOutcome.FAILED
            ),
            message="shutdown completed cleanly after failure scenario",
            created_at=utc_now(),
            metadata={
                "boot_status": boot.status.value,
                "shutdown_status": stop.status.value,
                "shutdown_error": stop.error,
            },
        )

    finally:
        bootstrap.stop()


def _report(
    *,
    results: list[FailureInjectionScenarioResult],
    started_at: datetime,
    error: str | None,
    metadata: dict[str, object],
) -> FailureInjectionReport:
    status = (
        FailureInjectionStatus.PASSED
        if results and all(result.passed for result in results) and error is None
        else FailureInjectionStatus.FAILED
    )
    return FailureInjectionReport(
        status=status,
        results=tuple(results),
        started_at=started_at,
        finished_at=utc_now(),
        error=error,
        metadata=metadata,
    )


def _interruption_signal(text: str) -> RealConversationInput:
    return RealConversationInput(
        transcript=text,
        source=TurnInputSource.INTERRUPTION_WORKER,
        is_speech_active=True,
        is_assistant_speaking=True,
        silence_ms=0,
        speech_ms=250,
        vad_confidence=0.99,
        transcript_stability=1.0,
        conversation_mode=ConversationMode.COMMAND,
    )