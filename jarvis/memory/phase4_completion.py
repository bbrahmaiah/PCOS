from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from threading import RLock

from pydantic import Field, field_validator

from jarvis.memory.models import MemoryModel
from jarvis.memory.safety_audit import (
    MemorySafetyAuditStatus,
    audit_phase4_memory_safety,
)
from jarvis.memory.validation import (
    MemoryPhase4ValidationStatus,
    validate_phase4_memory,
)
from jarvis.runtime.observability.structured_logger import get_logger


class Phase4CompletionStatus(StrEnum):
    """
    Final Phase 4 completion status.
    """

    PASSED = "passed"
    FAILED = "failed"


class Phase4CompletionCheckKind(StrEnum):
    """
    Type of Phase 4 completion check.
    """

    VALIDATION = "validation"
    SAFETY_AUDIT = "safety_audit"
    SMOKE_READY = "smoke_ready"
    COMPLETION = "completion"


class Phase4CompletionCheck(MemoryModel):
    """
    One final Phase 4 completion check.
    """

    name: str
    kind: Phase4CompletionCheckKind
    passed: bool
    detail: str
    metadata: dict[str, object] = Field(default_factory=dict)

    @field_validator("name", "detail")
    @classmethod
    def _required_text(cls, value: str) -> str:
        cleaned = value.strip()

        if not cleaned:
            raise ValueError("field cannot be empty.")

        return cleaned


class Phase4CompletionResult(MemoryModel):
    """
    Final Phase 4 completion result.
    """

    status: Phase4CompletionStatus
    checks: tuple[Phase4CompletionCheck, ...]
    metadata: dict[str, object] = Field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return self.status == Phase4CompletionStatus.PASSED

    @property
    def check_count(self) -> int:
        return len(self.checks)

    @property
    def passed_count(self) -> int:
        return sum(1 for check in self.checks if check.passed)

    @property
    def failed_count(self) -> int:
        return sum(1 for check in self.checks if not check.passed)


@dataclass(frozen=True, slots=True)
class Phase4CompletionGateConfig:
    """
    Configuration for the final Phase 4 completion gate.
    """

    name: str = "phase4_memory_completion_gate"
    sqlite_path: Path = Path(".jarvis_memory_completion/memory.db")
    include_sqlite: bool = True
    include_vector: bool = True
    include_safety_audit: bool = True

    def validate(self) -> None:
        if not self.name.strip():
            raise ValueError("name cannot be empty.")


@dataclass(frozen=True, slots=True)
class Phase4CompletionGateSnapshot:
    """
    Observable diagnostics for the Phase 4 completion gate.
    """

    name: str
    run_count: int
    last_status: Phase4CompletionStatus | None
    last_passed_count: int
    last_failed_count: int
    last_error: str | None


class Phase4CompletionGate:
    """
    Final completion gate for Phase 4 Memory Runtime.

    Responsibilities:
    - run formal Phase 4 memory validation
    - run formal memory safety audit
    - verify required smoke scripts exist
    - produce one final completion result
    - expose diagnostics

    Non-responsibilities:
    - no direct LLM calls
    - no production vector DB calls
    - no mutation of global runtime state
    - no hidden cognition-store shortcuts
    """

    def __init__(
        self,
        *,
        config: Phase4CompletionGateConfig | None = None,
    ) -> None:
        self._config = config or Phase4CompletionGateConfig()
        self._config.validate()

        self._lock = RLock()
        self._logger = get_logger("memory.phase4_completion")

        self._run_count = 0
        self._last_status: Phase4CompletionStatus | None = None
        self._last_passed_count = 0
        self._last_failed_count = 0
        self._last_error: str | None = None

    @property
    def name(self) -> str:
        return self._config.name

    def run(self) -> Phase4CompletionResult:
        """
        Run the final Phase 4 completion gate.
        """

        checks = [
            self._check_validation(),
            self._check_smoke_scripts_exist(),
        ]

        if self._config.include_safety_audit:
            checks.append(self._check_safety_audit())

        checks.append(self._check_completion_contract(checks=tuple(checks)))

        status = (
            Phase4CompletionStatus.PASSED
            if all(check.passed for check in checks)
            else Phase4CompletionStatus.FAILED
        )
        result = Phase4CompletionResult(
            status=status,
            checks=tuple(checks),
            metadata={
                "gate": self.name,
                "include_sqlite": self._config.include_sqlite,
                "include_vector": self._config.include_vector,
                "include_safety_audit": self._config.include_safety_audit,
                "phase": "phase4_memory_runtime",
            },
        )
        self._record_result(result)

        self._logger.info(
            "phase4_memory_completion_gate_completed",
            gate=self.name,
            status=result.status.value,
            passed_count=result.passed_count,
            failed_count=result.failed_count,
        )

        return result

    def snapshot(self) -> Phase4CompletionGateSnapshot:
        """
        Return completion gate diagnostics.
        """

        with self._lock:
            return Phase4CompletionGateSnapshot(
                name=self.name,
                run_count=self._run_count,
                last_status=self._last_status,
                last_passed_count=self._last_passed_count,
                last_failed_count=self._last_failed_count,
                last_error=self._last_error,
            )

    def reset(self) -> None:
        """
        Reset completion gate diagnostics.
        """

        with self._lock:
            self._run_count = 0
            self._last_status = None
            self._last_passed_count = 0
            self._last_failed_count = 0
            self._last_error = None

        self._logger.info("phase4_memory_completion_gate_reset", gate=self.name)

    def _check_validation(self) -> Phase4CompletionCheck:
        validation = validate_phase4_memory(
            sqlite_path=self._config.sqlite_path,
            include_sqlite=self._config.include_sqlite,
            include_vector=self._config.include_vector,
        )
        passed = validation.status == MemoryPhase4ValidationStatus.PASSED

        return Phase4CompletionCheck(
            name="phase4_memory_validation",
            kind=Phase4CompletionCheckKind.VALIDATION,
            passed=passed,
            detail="formal Phase 4 memory integration validation completed",
            metadata={
                "status": validation.status.value,
                "passed_count": validation.passed_count,
                "failed_count": validation.failed_count,
                "check_count": validation.check_count,
            },
        )

    def _check_safety_audit(self) -> Phase4CompletionCheck:
        safety = audit_phase4_memory_safety()
        passed = safety.status == MemorySafetyAuditStatus.PASSED

        return Phase4CompletionCheck(
            name="phase4_memory_safety_audit",
            kind=Phase4CompletionCheckKind.SAFETY_AUDIT,
            passed=passed,
            detail="formal Phase 4 memory safety audit completed",
            metadata={
                "status": safety.status.value,
                "passed_count": safety.passed_count,
                "failed_count": safety.failed_count,
                "check_count": safety.check_count,
            },
        )

    def _check_smoke_scripts_exist(self) -> Phase4CompletionCheck:
        required = (
            Path("scripts/smoke_memory_search.py"),
            Path("scripts/validate_memory_phase4.py"),
            Path("scripts/smoke_cognition_memory.py"),
            Path("scripts/audit_memory_safety.py"),
        )
        missing = tuple(str(path) for path in required if not path.exists())
        passed = not missing

        return Phase4CompletionCheck(
            name="phase4_memory_smoke_scripts",
            kind=Phase4CompletionCheckKind.SMOKE_READY,
            passed=passed,
            detail=(
                "required Phase 4 smoke and validation scripts exist"
                if passed
                else "missing required Phase 4 smoke or validation scripts"
            ),
            metadata={
                "required_scripts": tuple(str(path) for path in required),
                "missing_scripts": missing,
            },
        )

    def _check_completion_contract(
        self,
        *,
        checks: tuple[Phase4CompletionCheck, ...],
    ) -> Phase4CompletionCheck:
        failed = tuple(check.name for check in checks if not check.passed)
        passed = not failed

        return Phase4CompletionCheck(
            name="phase4_memory_completion_contract",
            kind=Phase4CompletionCheckKind.COMPLETION,
            passed=passed,
            detail=(
                "Phase 4 Memory Runtime is complete and ready for Phase 5"
                if passed
                else "Phase 4 Memory Runtime has failed prerequisite checks"
            ),
            metadata={
                "failed_prerequisites": failed,
                "validated_capabilities": (
                    "typed_memory_contracts",
                    "gateway_controlled_memory",
                    "write_policy",
                    "privacy_policy",
                    "lifecycle_policy",
                    "episodic_memory",
                    "semantic_memory",
                    "user_profile_memory",
                    "summarization_contracts",
                    "context_builder",
                    "cognition_memory_bridge",
                    "diagnostics",
                    "sqlite_persistence",
                    "vector_boundary",
                    "memory_search_smoke",
                    "cognition_memory_smoke",
                    "integration_validation",
                    "safety_audit",
                ),
            },
        )

    def _record_result(self, result: Phase4CompletionResult) -> None:
        with self._lock:
            self._run_count += 1
            self._last_status = result.status
            self._last_passed_count = result.passed_count
            self._last_failed_count = result.failed_count
            self._last_error = (
                None
                if result.passed
                else f"phase4 completion failed: {result.failed_count} checks"
            )


def complete_phase4_memory(
    *,
    sqlite_path: Path | None = None,
    include_sqlite: bool = True,
    include_vector: bool = True,
    include_safety_audit: bool = True,
) -> Phase4CompletionResult:
    """
    Convenience function for scripts and tests.
    """

    gate = Phase4CompletionGate(
        config=Phase4CompletionGateConfig(
            sqlite_path=sqlite_path or Path(".jarvis_memory_completion/memory.db"),
            include_sqlite=include_sqlite,
            include_vector=include_vector,
            include_safety_audit=include_safety_audit,
        )
    )

    return gate.run()