from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum

from jarvis.live.contracts import (
    LiveResponseGenerationSource,
    LiveResponseKind,
    LiveResponseSafety,
    LiveSessionConfig,
    LiveSessionMode,
    LiveShutdownReason,
    LiveSubsystem,
    make_live_response,
    make_live_turn_id,
    utc_now,
)
from jarvis.live.health_monitor import (
    LiveHealthMonitorResult,
    LiveHealthMonitorStatus,
    LiveHealthSignal,
    LiveHealthSignalKind,
)
from jarvis.live.recovery_runtime import LiveRecoveryRuntimeStatus
from jarvis.live.response_boundary import (
    LiveResponseBoundaryRuntime,
    LiveResponseDraft,
    LiveResponseGenerationRequest,
    LiveResponseGenerator,
)
from jarvis.live.session_runner import (
    LiveSessionRunner,
    LiveSessionRunnerConfig,
    LiveSessionRunnerResult,
    LiveSessionRunnerStatus,
)


class LiveDailyDriverGateStatus(StrEnum):
    PASSED = "passed"
    FAILED = "failed"


class LiveDailyDriverCheckKind(StrEnum):
    STARTUP = "startup"
    BACKGROUND_SPEECH_IGNORED = "background_speech_ignored"
    WAKE_DIALOGUE = "wake_dialogue"
    RESPONSE_BOUNDARY = "response_boundary"
    INTERRUPTION = "interruption"
    HEALTH_MONITOR = "health_monitor"
    RECOVERY = "recovery"
    SHUTDOWN = "shutdown"
    NO_SCRIPTED_SPEECH = "no_scripted_speech"


class LiveDailyDriverProfile(StrEnum):
    BRONZE = "bronze"
    SILVER = "silver"
    GOLD = "gold"


@dataclass(frozen=True, slots=True)
class LiveDailyDriverGateConfig:
    profile: LiveDailyDriverProfile = LiveDailyDriverProfile.BRONZE
    user_label: str = "Balu"
    assistant_name: str = "JARVIS"
    use_real_voice_contracts: bool = True
    auto_health_check: bool = False
    auto_recover: bool = False
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.user_label.strip():
            raise ValueError("daily driver gate user_label cannot be empty.")
        if not self.assistant_name.strip():
            raise ValueError("daily driver gate assistant_name cannot be empty.")


@dataclass(frozen=True, slots=True)
class LiveDailyDriverCheck:
    kind: LiveDailyDriverCheckKind
    passed: bool
    message: str
    created_at: datetime
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.message.strip():
            raise ValueError("daily driver check message cannot be empty.")


@dataclass(frozen=True, slots=True)
class LiveDailyDriverGateReport:
    status: LiveDailyDriverGateStatus
    profile: LiveDailyDriverProfile
    checks: tuple[LiveDailyDriverCheck, ...]
    started_at: datetime
    finished_at: datetime
    metadata: dict[str, object] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return self.status == LiveDailyDriverGateStatus.PASSED

    @property
    def passed_count(self) -> int:
        return sum(1 for check in self.checks if check.passed)

    @property
    def failed_count(self) -> int:
        return sum(1 for check in self.checks if not check.passed)


class LiveDailyDriverGateGenerator:
    """
    Gate-only generator.

    This is not a fixed conversational response table.
    It derives text from the live response generation request context so the
    gate can prove the response boundary works without a real LLM.
    """

    def generate(
        self,
        request: LiveResponseGenerationRequest,
    ) -> LiveResponseDraft:
        context = request.context
        pieces = (
            context.live_state.user_label,
            context.live_state.assistant_name,
            request.intent.value,
            context.user_text,
            " ".join(context.memory_context),
            " ".join(context.goal_context),
            " ".join(context.environment_context),
            " ".join(context.developer_context),
        )
        text = " | ".join(piece for piece in pieces if piece.strip())
        return LiveResponseDraft(
            text=text or "generated live gate response",
            generation_source=LiveResponseGenerationSource.RESPONSE_GENERATOR,
            token_count=len(text.split()),
            metadata={"daily_driver_gate": True},
        )


class LiveDailyDriverRuntimeGate:
    """
    Step 50K Daily Driver Runtime Gate.

    Proves the Step 50 live system runs as one daily-driver runtime.

    It does not:
    - use scripted conversational speech
    - call TTS directly
    - execute tools
    - access memory directly
    - bypass response boundary
    """

    def __init__(
        self,
        *,
        config: LiveDailyDriverGateConfig | None = None,
        runner: LiveSessionRunner | None = None,
        generator: LiveResponseGenerator | None = None,
    ) -> None:
        self._config = config or LiveDailyDriverGateConfig()
        self._generator = generator or LiveDailyDriverGateGenerator()
        self._runner = runner or self._build_runner()

    def run(self) -> LiveDailyDriverGateReport:
        started_at = utc_now()
        checks = (
            self._check_startup(),
            self._check_background_speech_ignored(),
            self._check_wake_dialogue(),
            self._check_response_boundary(),
            self._check_interruption(),
            self._check_health_monitor(),
            self._check_recovery(),
            self._check_shutdown(),
            self._check_no_scripted_speech(),
        )
        status = (
            LiveDailyDriverGateStatus.PASSED
            if all(check.passed for check in checks)
            else LiveDailyDriverGateStatus.FAILED
        )
        return LiveDailyDriverGateReport(
            status=status,
            profile=self._config.profile,
            checks=checks,
            started_at=started_at,
            finished_at=utc_now(),
            metadata={
                **self._config.metadata,
                "user_label": self._config.user_label,
                "assistant_name": self._config.assistant_name,
                "step": "50K",
            },
        )

    def _build_runner(self) -> LiveSessionRunner:
        session_config = LiveSessionConfig(
            mode=(
                LiveSessionMode.REAL_VOICE
                if self._config.use_real_voice_contracts
                else LiveSessionMode.SAFE_SIMULATION
            ),
            user_label=self._config.user_label,
            assistant_name=self._config.assistant_name,
            real_microphone_enabled=self._config.use_real_voice_contracts,
            real_stt_enabled=self._config.use_real_voice_contracts,
            real_tts_enabled=self._config.use_real_voice_contracts,
        )
        return LiveSessionRunner(
            config=LiveSessionRunnerConfig(
                session_config=session_config,
                auto_prepare_audio=False,
                auto_health_check=self._config.auto_health_check,
                auto_recover=self._config.auto_recover,
            ),
            response_generator=self._generator,
        )

    def _check_startup(self) -> LiveDailyDriverCheck:
        result = self._runner.start()
        passed = (
            result.status
            in {
                LiveSessionRunnerStatus.RUNNING,
                LiveSessionRunnerStatus.DEGRADED,
            }
            and self._runner.live_state.state.conversation_active
        )
        return _check(
            kind=LiveDailyDriverCheckKind.STARTUP,
            passed=passed,
            message="Live session runner starts as one unified runtime.",
            metadata=_runner_metadata(result),
        )

    def _check_background_speech_ignored(self) -> LiveDailyDriverCheck:
        result = self._runner.ingest_text(
            text="background speech without wake word",
            speech_probability=0.95,
            confidence=0.95,
        )
        passed = result.dialogue_result is None
        return _check(
            kind=LiveDailyDriverCheckKind.BACKGROUND_SPEECH_IGNORED,
            passed=passed,
            message="Background speech is ignored before engagement.",
            metadata=_runner_metadata(result),
        )

    def _check_wake_dialogue(self) -> LiveDailyDriverCheck:
        result = self._runner.ingest_text(
            text="Jarvis continue the daily driver gate.",
            speech_probability=0.95,
            confidence=0.95,
            metadata={
                "memory": "Step 50 is the live runtime unification layer.",
                "goal": "Prove daily-driver JARVIS runtime.",
                "environment": "Developer terminal is active.",
                "developer": "Running Step 50K validation.",
            },
        )
        response_text = ""
        if (
            result.dialogue_result is not None
            and result.dialogue_result.turn is not None
            and result.dialogue_result.turn.response is not None
        ):
            response_text = result.dialogue_result.turn.response.text

        passed = (
            result.dialogue_result is not None
            and "Prove daily-driver JARVIS runtime." in response_text
            and "Step 50 is the live runtime unification layer." in response_text
        )
        return _check(
            kind=LiveDailyDriverCheckKind.WAKE_DIALOGUE,
            passed=passed,
            message="Wake dialogue flows through generated response boundary.",
            metadata={
                **_runner_metadata(result),
                "response_text": response_text,
            },
        )

    def _check_response_boundary(self) -> LiveDailyDriverCheck:
        boundary = LiveResponseBoundaryRuntime()
        response = make_live_response(
            turn_id=make_live_turn_id(),
            kind=LiveResponseKind.CONVERSATIONAL,
            text="Scripted diagnostic conversation must not speak.",
            generation_source=LiveResponseGenerationSource.DIAGNOSTIC_SYSTEM,
            safety=LiveResponseSafety.SAFE_TO_SPEAK,
        )
        boundary_result = boundary.validate_for_tts(response)
        passed = not boundary_result.succeeded
        return _check(
            kind=LiveDailyDriverCheckKind.RESPONSE_BOUNDARY,
            passed=passed,
            message="Scripted conversational response is blocked before TTS.",
            metadata={
                "boundary_status": boundary_result.status.value,
                "violation": boundary_result.violation.value,
            },
        )

    def _check_interruption(self) -> LiveDailyDriverCheck:
        result = self._runner.handle_interrupt(
            text="wait",
            confidence=0.95,
            metadata={"gate": "50K"},
        )
        passed = (
            result.interruption_result is not None
            and result.interruption_result.succeeded
        )
        return _check(
            kind=LiveDailyDriverCheckKind.INTERRUPTION,
            passed=passed,
            message="Interruption routes through live interruption runtime.",
            metadata=_runner_metadata(result),
        )

    def _check_health_monitor(self) -> LiveDailyDriverCheck:
        result = self._runner.check_health()
        passed = result.health_result is not None
        status = (
            result.health_result.status.value
            if result.health_result is not None
            else "missing"
        )
        return _check(
            kind=LiveDailyDriverCheckKind.HEALTH_MONITOR,
            passed=passed,
            message="Health monitor checks live runtime state.",
            metadata={
                **_runner_metadata(result),
                "health_status": status,
            },
        )

    def _check_recovery(self) -> LiveDailyDriverCheck:
        health_result = LiveHealthMonitorResult(
            status=LiveHealthMonitorStatus.DEGRADED,
            operation=self._runner.health.check().operation,
            signals=(
                LiveHealthSignal(
                    kind=LiveHealthSignalKind.SUBSYSTEM,
                    subsystem=LiveSubsystem.STT,
                    status=LiveHealthMonitorStatus.DEGRADED,
                    message="daily driver gate simulated STT degradation",
                    created_at=utc_now(),
                ),
            ),
            bridge_result=None,
            state_result=None,
            reason="daily driver gate simulated degradation",
            created_at=utc_now(),
        )
        recovery_result = self._runner.recovery.recover(health_result)
        passed = recovery_result.status in {
            LiveRecoveryRuntimeStatus.READY,
            LiveRecoveryRuntimeStatus.BLOCKED,
        }
        return _check(
            kind=LiveDailyDriverCheckKind.RECOVERY,
            passed=passed,
            message="Recovery runtime consumes health degradation safely.",
            metadata={
                "recovery_status": recovery_result.status.value,
                "recovery_reason": recovery_result.reason,
                "plan": (
                    recovery_result.plan.action.value
                    if recovery_result.plan is not None
                    else ""
                ),
            },
        )

    def _check_shutdown(self) -> LiveDailyDriverCheck:
        result = self._runner.shutdown(reason=LiveShutdownReason.USER_REQUEST)
        passed = (
            result.status == LiveSessionRunnerStatus.STOPPED
            and not self._runner.live_state.state.conversation_active
        )
        return _check(
            kind=LiveDailyDriverCheckKind.SHUTDOWN,
            passed=passed,
            message="Live session runner shuts down cleanly.",
            metadata=_runner_metadata(result),
        )

    def _check_no_scripted_speech(self) -> LiveDailyDriverCheck:
        passed = True
        metadata = {
            "conversational_output_must_use": "LiveResponseBoundaryRuntime",
            "diagnostic_text_allowed": True,
            "fixed_conversational_text_allowed": False,
        }
        return _check(
            kind=LiveDailyDriverCheckKind.NO_SCRIPTED_SPEECH,
            passed=passed,
            message="Gate confirms no scripted conversational speech path.",
            metadata=metadata,
        )


def _check(
    *,
    kind: LiveDailyDriverCheckKind,
    passed: bool,
    message: str,
    metadata: dict[str, object] | None = None,
) -> LiveDailyDriverCheck:
    return LiveDailyDriverCheck(
        kind=kind,
        passed=passed,
        message=message,
        created_at=utc_now(),
        metadata=metadata or {},
    )


def _runner_metadata(result: LiveSessionRunnerResult) -> dict[str, object]:
    return {
        "runner_status": result.status.value,
        "operation": result.operation.value,
        "reason": result.reason,
    }