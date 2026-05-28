from __future__ import annotations

import statistics
import time
from dataclasses import dataclass
from enum import StrEnum
from threading import RLock
from uuid import uuid4

from pydantic import Field, field_validator, model_validator

from jarvis.orchestration.ids import utc_now
from jarvis.orchestration.models import OrchestrationModel


class LatencyRegressionMetric(StrEnum):
    """
    Phase 7 latency regression metric.

    These metrics are CI contracts. They fail when latency gets worse beyond
    agreed p95 targets.
    """

    VOICE_FIRST_WORD = "voice_first_word"
    MEMORY_RETRIEVAL = "memory_retrieval"
    INTERRUPTION_RECOVERY = "interruption_recovery"
    STREAMING_FIRST_TOKEN = "streaming_first_token"


class LatencyRegressionMachineProfile(StrEnum):
    """
    Runtime profile used for regression validation.
    """

    FAST_MACHINE = "fast_machine"
    SLOW_MACHINE = "slow_machine"
    HIGH_LOAD = "high_load"


class LatencyRegressionStatus(StrEnum):
    """
    Latency regression evaluation status.
    """

    CREATED = "created"
    RUNNING = "running"
    PASSED = "passed"
    WARNING = "warning"
    FAILED = "failed"
    CANCELLED = "cancelled"


class LatencyRegressionReason(StrEnum):
    """
    Machine-readable regression reasons.
    """

    SESSION_CREATED = "session_created"
    SESSION_STARTED = "session_started"
    SAMPLE_RECORDED = "sample_recorded"
    CONTRACT_PASSED = "contract_passed"
    CONTRACT_FAILED_P95 = "contract_failed_p95"
    CONTRACT_FAILED_CORRECTNESS = "contract_failed_correctness"
    CONTRACT_FAILED_MEMORY = "contract_failed_memory"
    CONTRACT_FAILED_CPU = "contract_failed_cpu"
    REPORT_BUILT = "report_built"
    SESSION_CANCELLED = "session_cancelled"
    SESSION_NOT_FOUND = "session_not_found"
    SESSION_NOT_RUNNING = "session_not_running"
    RUNTIME_RESET = "runtime_reset"


class LatencyRegressionEventKind(StrEnum):
    """
    Regression runtime event kind.
    """

    SESSION_CREATED = "session_created"
    SESSION_STARTED = "session_started"
    SAMPLE_RECORDED = "sample_recorded"
    CONTRACT_EVALUATED = "contract_evaluated"
    REPORT_BUILT = "report_built"
    SESSION_CANCELLED = "session_cancelled"


class LatencyRegressionContract(OrchestrationModel):
    """
    Regression contract for one metric under one machine profile.
    """

    contract_id: str = Field(default_factory=lambda: uuid4().hex)
    metric: LatencyRegressionMetric
    machine_profile: LatencyRegressionMachineProfile
    p95_target_ms: float = Field(gt=0)
    correctness_required: bool = True
    memory_overhead_limit_ratio: float = Field(default=0.20, ge=0)
    cpu_spike_limit_ratio: float = Field(default=0.30, ge=0)
    min_samples: int = Field(default=5, ge=1)
    created_at: object = Field(default_factory=utc_now)
    metadata: dict[str, object] = Field(default_factory=dict)

    @field_validator("contract_id")
    @classmethod
    def _required_id(cls, value: str) -> str:
        cleaned = value.strip()

        if not cleaned:
            raise ValueError("contract_id cannot be empty.")

        return cleaned


class LatencyRegressionSample(OrchestrationModel):
    """
    One measured sample.

    correctness_ok is evaluated separately from latency because a fast but
    wrong response is still a failed optimization.
    """

    sample_id: str = Field(default_factory=lambda: uuid4().hex)
    metric: LatencyRegressionMetric
    machine_profile: LatencyRegressionMachineProfile
    latency_ms: float = Field(ge=0)
    correctness_ok: bool = True
    memory_overhead_ratio: float = Field(default=0.0, ge=0)
    cpu_spike_ratio: float = Field(default=0.0, ge=0)
    created_at_ns: int = Field(default_factory=time.perf_counter_ns, ge=0)
    created_at: object = Field(default_factory=utc_now)
    metadata: dict[str, object] = Field(default_factory=dict)

    @field_validator("sample_id")
    @classmethod
    def _required_id(cls, value: str) -> str:
        cleaned = value.strip()

        if not cleaned:
            raise ValueError("sample_id cannot be empty.")

        return cleaned


class LatencyRegressionEvaluation(OrchestrationModel):
    """
    Evaluation result for one contract.
    """

    evaluation_id: str = Field(default_factory=lambda: uuid4().hex)
    contract: LatencyRegressionContract
    status: LatencyRegressionStatus
    reason: LatencyRegressionReason
    sample_count: int = Field(ge=0)
    p50_ms: float = Field(default=0.0, ge=0)
    p95_ms: float = Field(default=0.0, ge=0)
    max_ms: float = Field(default=0.0, ge=0)
    max_memory_overhead_ratio: float = Field(default=0.0, ge=0)
    max_cpu_spike_ratio: float = Field(default=0.0, ge=0)
    correctness_failures: int = Field(default=0, ge=0)
    created_at: object = Field(default_factory=utc_now)
    metadata: dict[str, object] = Field(default_factory=dict)

    @field_validator("evaluation_id")
    @classmethod
    def _required_id(cls, value: str) -> str:
        cleaned = value.strip()

        if not cleaned:
            raise ValueError("evaluation_id cannot be empty.")

        return cleaned


class LatencyRegressionEvent(OrchestrationModel):
    """
    Typed event for regression observability.
    """

    event_id: str = Field(default_factory=lambda: uuid4().hex)
    session_id: str
    kind: LatencyRegressionEventKind
    reason: LatencyRegressionReason
    metric: LatencyRegressionMetric | None = None
    machine_profile: LatencyRegressionMachineProfile | None = None
    latency_ms: float | None = None
    created_at: object = Field(default_factory=utc_now)
    metadata: dict[str, object] = Field(default_factory=dict)

    @field_validator("event_id", "session_id")
    @classmethod
    def _required_text(cls, value: str) -> str:
        cleaned = value.strip()

        if not cleaned:
            raise ValueError("field cannot be empty.")

        return cleaned


class LatencyRegressionSessionState(OrchestrationModel):
    """
    State for one latency regression run.
    """

    session_id: str = Field(default_factory=lambda: uuid4().hex)
    trace_id: str = Field(default_factory=lambda: uuid4().hex)
    status: LatencyRegressionStatus = LatencyRegressionStatus.CREATED
    started_at_ns: int | None = None
    completed_at_ns: int | None = None
    cancelled_at_ns: int | None = None
    contract_count: int = Field(default=0, ge=0)
    sample_count: int = Field(default=0, ge=0)
    failed_count: int = Field(default=0, ge=0)
    created_at: object = Field(default_factory=utc_now)
    metadata: dict[str, object] = Field(default_factory=dict)

    @field_validator("session_id", "trace_id")
    @classmethod
    def _required_ids(cls, value: str) -> str:
        cleaned = value.strip()

        if not cleaned:
            raise ValueError("field cannot be empty.")

        return cleaned


class LatencyRegressionResult(OrchestrationModel):
    """
    Result from latency regression runtime operation.
    """

    success: bool
    reason: LatencyRegressionReason
    session_id: str
    status: LatencyRegressionStatus
    event: LatencyRegressionEvent | None = None
    state: LatencyRegressionSessionState | None = None
    message: str
    created_at: object = Field(default_factory=utc_now)

    @field_validator("session_id", "message")
    @classmethod
    def _required_text(cls, value: str) -> str:
        cleaned = value.strip()

        if not cleaned:
            raise ValueError("field cannot be empty.")

        return cleaned


class LatencyRegressionReport(OrchestrationModel):
    """
    Final CI-friendly regression report.
    """

    session_id: str
    trace_id: str
    status: LatencyRegressionStatus
    contract_count: int = Field(ge=0)
    sample_count: int = Field(ge=0)
    passed_count: int = Field(ge=0)
    failed_count: int = Field(ge=0)
    evaluations: tuple[LatencyRegressionEvaluation, ...]
    events: tuple[LatencyRegressionEvent, ...]
    created_at: object = Field(default_factory=utc_now)

    @field_validator("session_id", "trace_id")
    @classmethod
    def _required_ids(cls, value: str) -> str:
        cleaned = value.strip()

        if not cleaned:
            raise ValueError("field cannot be empty.")

        return cleaned

    @model_validator(mode="after")
    def _counts_match(self) -> LatencyRegressionReport:
        if self.passed_count + self.failed_count != len(self.evaluations):
            raise ValueError("passed_count + failed_count must match evaluations.")

        return self


class LatencyRegressionRuntimeSnapshot(OrchestrationModel):
    """
    Runtime diagnostics for Phase 7 Step 17.
    """

    name: str
    session_count: int = Field(ge=0)
    running_count: int = Field(ge=0)
    passed_count: int = Field(ge=0)
    failed_count: int = Field(ge=0)
    cancelled_count: int = Field(ge=0)
    report_count: int = Field(ge=0)
    last_reason: LatencyRegressionReason | None = None
    created_at: object = Field(default_factory=utc_now)

    @field_validator("name")
    @classmethod
    def _required_name(cls, value: str) -> str:
        cleaned = value.strip()

        if not cleaned:
            raise ValueError("name cannot be empty.")

        return cleaned


@dataclass(frozen=True, slots=True)
class LatencyRegressionRuntimeConfig:
    """
    Phase 7 Step 17 regression configuration.
    """

    name: str = "latency_regression_test_suite"
    voice_first_word_p95_ms: float = 800.0
    memory_retrieval_p95_ms: float = 150.0
    interruption_recovery_p95_ms: float = 300.0
    streaming_first_token_p95_ms: float = 400.0
    memory_overhead_limit_ratio: float = 0.20
    cpu_spike_limit_ratio: float = 0.30
    min_samples: int = 5

    def validate(self) -> None:
        if not self.name.strip():
            raise ValueError("name cannot be empty.")

        for field_name, value in (
            ("voice_first_word_p95_ms", self.voice_first_word_p95_ms),
            ("memory_retrieval_p95_ms", self.memory_retrieval_p95_ms),
            ("interruption_recovery_p95_ms", self.interruption_recovery_p95_ms),
            ("streaming_first_token_p95_ms", self.streaming_first_token_p95_ms),
        ):
            if value <= 0:
                raise ValueError(f"{field_name} must be positive.")

        if self.memory_overhead_limit_ratio < 0:
            raise ValueError("memory_overhead_limit_ratio cannot be negative.")

        if self.cpu_spike_limit_ratio < 0:
            raise ValueError("cpu_spike_limit_ratio cannot be negative.")

        if self.min_samples < 1:
            raise ValueError("min_samples must be positive.")


class LatencyRegressionRuntime:
    """
    Phase 7 Step 17 Latency Regression Test Suite.

    Responsibilities:
    - enforce p95 latency contracts
    - enforce correctness guardrails
    - enforce memory overhead guardrails
    - enforce CPU spike guardrails
    - run across fast, slow, and high-load profiles
    - produce CI-friendly reports

    Non-responsibilities:
    - no real model calls
    - no real TTS synthesis
    - no real action execution
    - no benchmark cheating through hidden warm state
    """

    def __init__(
        self,
        *,
        config: LatencyRegressionRuntimeConfig | None = None,
    ) -> None:
        self._config = config or LatencyRegressionRuntimeConfig()
        self._config.validate()

        self._states: dict[str, LatencyRegressionSessionState] = {}
        self._contracts: dict[str, list[LatencyRegressionContract]] = {}
        self._samples: dict[str, list[LatencyRegressionSample]] = {}
        self._events: dict[str, list[LatencyRegressionEvent]] = {}
        self._reports: list[LatencyRegressionReport] = []
        self._lock = RLock()
        self._last_reason: LatencyRegressionReason | None = None

    @property
    def name(self) -> str:
        return self._config.name

    def create_session(
        self,
        *,
        trace_id: str | None = None,
        metadata: dict[str, object] | None = None,
    ) -> LatencyRegressionSessionState:
        contracts = self.default_contracts()
        state = LatencyRegressionSessionState(
            trace_id=trace_id or uuid4().hex,
            contract_count=len(contracts),
            metadata=metadata or {},
        )
        event = self._event(
            session_id=state.session_id,
            kind=LatencyRegressionEventKind.SESSION_CREATED,
            reason=LatencyRegressionReason.SESSION_CREATED,
        )

        with self._lock:
            self._states[state.session_id] = state
            self._contracts[state.session_id] = list(contracts)
            self._samples[state.session_id] = []
            self._events[state.session_id] = [event]
            self._last_reason = LatencyRegressionReason.SESSION_CREATED

        return state

    def start_session(self, session_id: str) -> LatencyRegressionResult:
        with self._lock:
            state = self._states.get(session_id)

            if state is None:
                return self._missing_session(session_id)

            if state.status != LatencyRegressionStatus.CREATED:
                return self._failure(
                    session_id=session_id,
                    reason=LatencyRegressionReason.SESSION_NOT_RUNNING,
                    status=state.status,
                    message="regression session cannot start from current state",
                    state=state,
                )

            started = state.model_copy(
                update={
                    "status": LatencyRegressionStatus.RUNNING,
                    "started_at_ns": time.perf_counter_ns(),
                }
            )
            self._states[session_id] = started
            event = self._event(
                session_id=session_id,
                kind=LatencyRegressionEventKind.SESSION_STARTED,
                reason=LatencyRegressionReason.SESSION_STARTED,
            )
            self._events[session_id].append(event)
            self._last_reason = LatencyRegressionReason.SESSION_STARTED

        return LatencyRegressionResult(
            success=True,
            reason=LatencyRegressionReason.SESSION_STARTED,
            session_id=session_id,
            status=LatencyRegressionStatus.RUNNING,
            event=event,
            state=started,
            message="latency regression session started",
        )

    def record_sample(
        self,
        *,
        session_id: str,
        sample: LatencyRegressionSample,
    ) -> LatencyRegressionResult:
        state = self.state_for(session_id)

        if state is None:
            return self._missing_session(session_id)

        if state.status != LatencyRegressionStatus.RUNNING:
            return self._failure(
                session_id=session_id,
                reason=LatencyRegressionReason.SESSION_NOT_RUNNING,
                status=state.status,
                message="regression session is not running",
                state=state,
            )

        with self._lock:
            self._samples[session_id].append(sample)
            current = self._states[session_id]
            updated = current.model_copy(
                update={"sample_count": current.sample_count + 1}
            )
            self._states[session_id] = updated
            event = self._event(
                session_id=session_id,
                kind=LatencyRegressionEventKind.SAMPLE_RECORDED,
                reason=LatencyRegressionReason.SAMPLE_RECORDED,
                metric=sample.metric,
                machine_profile=sample.machine_profile,
                latency_ms=sample.latency_ms,
            )
            self._events[session_id].append(event)
            self._last_reason = LatencyRegressionReason.SAMPLE_RECORDED

        return LatencyRegressionResult(
            success=True,
            reason=LatencyRegressionReason.SAMPLE_RECORDED,
            session_id=session_id,
            status=LatencyRegressionStatus.RUNNING,
            event=event,
            state=updated,
            message="latency regression sample recorded",
        )

    def run_simulated_suite(
        self,
        *,
        session_id: str,
        failing: bool = False,
    ) -> LatencyRegressionReport:
        state = self.state_for(session_id)

        if state is None:
            raise ValueError(f"latency regression session not found: {session_id}")

        if state.status != LatencyRegressionStatus.RUNNING:
            raise ValueError("latency regression session is not running")

        for contract in self.contracts_for(session_id):
            samples = self._simulated_samples_for(contract=contract, failing=failing)

            for sample in samples:
                self.record_sample(session_id=session_id, sample=sample)

        return self.build_report(session_id)

    def build_report(self, session_id: str) -> LatencyRegressionReport:
        state = self.state_for(session_id)

        if state is None:
            raise ValueError(f"latency regression session not found: {session_id}")

        evaluations = tuple(
            self._evaluate_contract(
                contract=contract,
                samples=self._samples_for_contract(
                    session_id=session_id,
                    contract=contract,
                ),
            )
            for contract in self.contracts_for(session_id)
        )
        failed_count = sum(
            1
            for evaluation in evaluations
            if evaluation.status == LatencyRegressionStatus.FAILED
        )
        passed_count = len(evaluations) - failed_count
        final_status = (
            LatencyRegressionStatus.FAILED
            if failed_count > 0
            else LatencyRegressionStatus.PASSED
        )

        with self._lock:
            current = self._states[session_id]
            completed = current.model_copy(
                update={
                    "status": final_status,
                    "completed_at_ns": time.perf_counter_ns(),
                    "failed_count": failed_count,
                }
            )
            self._states[session_id] = completed

            for evaluation in evaluations:
                self._events[session_id].append(
                    self._event(
                        session_id=session_id,
                        kind=LatencyRegressionEventKind.CONTRACT_EVALUATED,
                        reason=evaluation.reason,
                        metric=evaluation.contract.metric,
                        machine_profile=evaluation.contract.machine_profile,
                        latency_ms=evaluation.p95_ms,
                    )
                )

            self._events[session_id].append(
                self._event(
                    session_id=session_id,
                    kind=LatencyRegressionEventKind.REPORT_BUILT,
                    reason=LatencyRegressionReason.REPORT_BUILT,
                )
            )
            self._last_reason = LatencyRegressionReason.REPORT_BUILT

        report = LatencyRegressionReport(
            session_id=session_id,
            trace_id=state.trace_id,
            status=final_status,
            contract_count=len(evaluations),
            sample_count=len(self.samples_for(session_id)),
            passed_count=passed_count,
            failed_count=failed_count,
            evaluations=evaluations,
            events=self.events_for(session_id),
        )

        with self._lock:
            self._reports.append(report)

        return report

    def cancel_session(self, session_id: str) -> LatencyRegressionResult:
        with self._lock:
            state = self._states.get(session_id)

            if state is None:
                return self._missing_session(session_id)

            cancelled = state.model_copy(
                update={
                    "status": LatencyRegressionStatus.CANCELLED,
                    "cancelled_at_ns": time.perf_counter_ns(),
                }
            )
            self._states[session_id] = cancelled
            event = self._event(
                session_id=session_id,
                kind=LatencyRegressionEventKind.SESSION_CANCELLED,
                reason=LatencyRegressionReason.SESSION_CANCELLED,
            )
            self._events[session_id].append(event)
            self._last_reason = LatencyRegressionReason.SESSION_CANCELLED

        return LatencyRegressionResult(
            success=True,
            reason=LatencyRegressionReason.SESSION_CANCELLED,
            session_id=session_id,
            status=LatencyRegressionStatus.CANCELLED,
            event=event,
            state=cancelled,
            message="latency regression session cancelled",
        )

    def default_contracts(self) -> tuple[LatencyRegressionContract, ...]:
        contracts: list[LatencyRegressionContract] = []

        targets = {
            LatencyRegressionMetric.VOICE_FIRST_WORD: (
                self._config.voice_first_word_p95_ms
            ),
            LatencyRegressionMetric.MEMORY_RETRIEVAL: (
                self._config.memory_retrieval_p95_ms
            ),
            LatencyRegressionMetric.INTERRUPTION_RECOVERY: (
                self._config.interruption_recovery_p95_ms
            ),
            LatencyRegressionMetric.STREAMING_FIRST_TOKEN: (
                self._config.streaming_first_token_p95_ms
            ),
        }

        for profile in LatencyRegressionMachineProfile:
            for metric, target_ms in targets.items():
                contracts.append(
                    LatencyRegressionContract(
                        metric=metric,
                        machine_profile=profile,
                        p95_target_ms=target_ms,
                        memory_overhead_limit_ratio=(
                            self._config.memory_overhead_limit_ratio
                        ),
                        cpu_spike_limit_ratio=self._config.cpu_spike_limit_ratio,
                        min_samples=self._config.min_samples,
                    )
                )

        return tuple(contracts)

    def state_for(self, session_id: str) -> LatencyRegressionSessionState | None:
        with self._lock:
            return self._states.get(session_id)

    def contracts_for(
        self,
        session_id: str,
    ) -> tuple[LatencyRegressionContract, ...]:
        with self._lock:
            return tuple(self._contracts.get(session_id, ()))

    def samples_for(
        self,
        session_id: str,
    ) -> tuple[LatencyRegressionSample, ...]:
        with self._lock:
            return tuple(self._samples.get(session_id, ()))

    def events_for(
        self,
        session_id: str,
    ) -> tuple[LatencyRegressionEvent, ...]:
        with self._lock:
            return tuple(self._events.get(session_id, ()))

    def reports(self) -> tuple[LatencyRegressionReport, ...]:
        with self._lock:
            return tuple(self._reports)

    def latest_report(self) -> LatencyRegressionReport | None:
        with self._lock:
            if not self._reports:
                return None

            return self._reports[-1]

    def snapshot(self) -> LatencyRegressionRuntimeSnapshot:
        with self._lock:
            states = tuple(self._states.values())

            return LatencyRegressionRuntimeSnapshot(
                name=self.name,
                session_count=len(states),
                running_count=sum(
                    1
                    for state in states
                    if state.status == LatencyRegressionStatus.RUNNING
                ),
                passed_count=sum(
                    1
                    for state in states
                    if state.status == LatencyRegressionStatus.PASSED
                ),
                failed_count=sum(
                    1
                    for state in states
                    if state.status == LatencyRegressionStatus.FAILED
                ),
                cancelled_count=sum(
                    1
                    for state in states
                    if state.status == LatencyRegressionStatus.CANCELLED
                ),
                report_count=len(self._reports),
                last_reason=self._last_reason,
            )

    def reset(self) -> None:
        with self._lock:
            self._states.clear()
            self._contracts.clear()
            self._samples.clear()
            self._events.clear()
            self._reports.clear()
            self._last_reason = LatencyRegressionReason.RUNTIME_RESET

    @staticmethod
    def percentile(values: tuple[float, ...], percentile: float) -> float:
        if not values:
            return 0.0

        if len(values) == 1:
            return values[0]

        sorted_values = sorted(values)
        rank = (percentile / 100.0) * (len(sorted_values) - 1)
        lower_index = int(rank)
        upper_index = min(lower_index + 1, len(sorted_values) - 1)
        fraction = rank - lower_index

        return (
            sorted_values[lower_index] * (1.0 - fraction)
            + sorted_values[upper_index] * fraction
        )

    def _samples_for_contract(
        self,
        *,
        session_id: str,
        contract: LatencyRegressionContract,
    ) -> tuple[LatencyRegressionSample, ...]:
        return tuple(
            sample
            for sample in self.samples_for(session_id)
            if sample.metric == contract.metric
            and sample.machine_profile == contract.machine_profile
        )

    def _evaluate_contract(
        self,
        *,
        contract: LatencyRegressionContract,
        samples: tuple[LatencyRegressionSample, ...],
    ) -> LatencyRegressionEvaluation:
        latencies = tuple(sample.latency_ms for sample in samples)
        p50 = statistics.median(latencies) if latencies else 0.0
        p95 = self.percentile(latencies, 95.0)
        max_latency = max(latencies) if latencies else 0.0
        max_memory = max(
            (sample.memory_overhead_ratio for sample in samples),
            default=0.0,
        )
        max_cpu = max((sample.cpu_spike_ratio for sample in samples), default=0.0)
        correctness_failures = sum(
            1 for sample in samples if not sample.correctness_ok
        )

        status = LatencyRegressionStatus.PASSED
        reason = LatencyRegressionReason.CONTRACT_PASSED

        if len(samples) < contract.min_samples:
            status = LatencyRegressionStatus.FAILED
            reason = LatencyRegressionReason.CONTRACT_FAILED_P95
        elif p95 > contract.p95_target_ms:
            status = LatencyRegressionStatus.FAILED
            reason = LatencyRegressionReason.CONTRACT_FAILED_P95
        elif contract.correctness_required and correctness_failures > 0:
            status = LatencyRegressionStatus.FAILED
            reason = LatencyRegressionReason.CONTRACT_FAILED_CORRECTNESS
        elif max_memory > contract.memory_overhead_limit_ratio:
            status = LatencyRegressionStatus.FAILED
            reason = LatencyRegressionReason.CONTRACT_FAILED_MEMORY
        elif max_cpu > contract.cpu_spike_limit_ratio:
            status = LatencyRegressionStatus.FAILED
            reason = LatencyRegressionReason.CONTRACT_FAILED_CPU

        return LatencyRegressionEvaluation(
            contract=contract,
            status=status,
            reason=reason,
            sample_count=len(samples),
            p50_ms=p50,
            p95_ms=p95,
            max_ms=max_latency,
            max_memory_overhead_ratio=max_memory,
            max_cpu_spike_ratio=max_cpu,
            correctness_failures=correctness_failures,
        )

    def _simulated_samples_for(
        self,
        *,
        contract: LatencyRegressionContract,
        failing: bool,
    ) -> tuple[LatencyRegressionSample, ...]:
        base = contract.p95_target_ms * 0.70
        profile_multiplier = {
            LatencyRegressionMachineProfile.FAST_MACHINE: 0.70,
            LatencyRegressionMachineProfile.SLOW_MACHINE: 0.90,
            LatencyRegressionMachineProfile.HIGH_LOAD: 0.96,
        }[contract.machine_profile]

        if failing:
            base = contract.p95_target_ms * 1.15
            profile_multiplier = 1.0

        latencies = tuple(
            base * profile_multiplier * multiplier
            for multiplier in (0.85, 0.90, 0.95, 1.00, 1.05)
        )

        return tuple(
            LatencyRegressionSample(
                metric=contract.metric,
                machine_profile=contract.machine_profile,
                latency_ms=latency,
                correctness_ok=not failing,
                memory_overhead_ratio=0.10 if not failing else 0.25,
                cpu_spike_ratio=0.12 if not failing else 0.35,
            )
            for latency in latencies
        )

    @staticmethod
    def _event(
        *,
        session_id: str,
        kind: LatencyRegressionEventKind,
        reason: LatencyRegressionReason,
        metric: LatencyRegressionMetric | None = None,
        machine_profile: LatencyRegressionMachineProfile | None = None,
        latency_ms: float | None = None,
        metadata: dict[str, object] | None = None,
    ) -> LatencyRegressionEvent:
        return LatencyRegressionEvent(
            session_id=session_id,
            kind=kind,
            reason=reason,
            metric=metric,
            machine_profile=machine_profile,
            latency_ms=latency_ms,
            metadata=metadata or {},
        )

    @staticmethod
    def _missing_session(session_id: str) -> LatencyRegressionResult:
        return LatencyRegressionResult(
            success=False,
            reason=LatencyRegressionReason.SESSION_NOT_FOUND,
            session_id=session_id,
            status=LatencyRegressionStatus.FAILED,
            message="latency regression session not found",
        )

    @staticmethod
    def _failure(
        *,
        session_id: str,
        reason: LatencyRegressionReason,
        status: LatencyRegressionStatus,
        message: str,
        state: LatencyRegressionSessionState | None = None,
    ) -> LatencyRegressionResult:
        return LatencyRegressionResult(
            success=False,
            reason=reason,
            session_id=session_id,
            status=status,
            state=state,
            message=message,
        )