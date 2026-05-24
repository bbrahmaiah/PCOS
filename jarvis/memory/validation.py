from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from threading import RLock

from pydantic import Field, field_validator

from jarvis.memory.cognition_integration import MemoryCognitionBridge
from jarvis.memory.context import MemoryContextBuilder
from jarvis.memory.diagnostics import (
    MemoryDiagnosticsCollector,
    MemoryDiagnosticStatus,
)
from jarvis.memory.gateway import GovernedMemoryGateway
from jarvis.memory.models import (
    MemoryImportance,
    MemoryKind,
    MemoryModel,
    MemoryQuery,
    MemoryRecord,
    MemorySensitivity,
    MemoryWriteRequest,
)
from jarvis.memory.sqlite_store import SQLiteMemoryStore, SQLiteMemoryStoreConfig
from jarvis.memory.store import InMemoryMemoryStore
from jarvis.memory.summarization import ExtractiveMemorySummarizer
from jarvis.memory.vector import (
    DeterministicEmbeddingProvider,
    InMemoryVectorIndex,
    MemoryVectorDocument,
    MemoryVectorSearchQuery,
)
from jarvis.runtime.observability.structured_logger import get_logger


class MemoryPhase4ValidationStatus(StrEnum):
    """
    Status for Phase 4 memory validation.
    """

    PASSED = "passed"
    FAILED = "failed"


class MemoryPhase4ValidationCheck(MemoryModel):
    """
    One Phase 4 memory validation check.
    """

    name: str
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


class MemoryPhase4ValidationResult(MemoryModel):
    """
    Full Phase 4 memory validation result.
    """

    status: MemoryPhase4ValidationStatus
    checks: tuple[MemoryPhase4ValidationCheck, ...]
    metadata: dict[str, object] = Field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return self.status == MemoryPhase4ValidationStatus.PASSED

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
class MemoryPhase4ValidatorConfig:
    """
    Configuration for Phase 4 memory validation.
    """

    name: str = "memory_phase4_validator"
    sqlite_path: Path = Path(".jarvis_memory_validation/memory.db")
    include_sqlite: bool = True
    include_vector: bool = True

    def validate(self) -> None:
        if not self.name.strip():
            raise ValueError("name cannot be empty.")


@dataclass(frozen=True, slots=True)
class MemoryPhase4ValidatorSnapshot:
    """
    Observable diagnostics for the Phase 4 validator.
    """

    name: str
    validation_count: int
    last_status: MemoryPhase4ValidationStatus | None
    last_passed_count: int
    last_failed_count: int
    last_error: str | None


class MemoryPhase4Validator:
    """
    Formal Phase 4 memory integration validator.

    Responsibilities:
    - validate memory contracts
    - validate gateway-controlled write/retrieve path
    - validate retrieval explainability
    - validate cognition-memory context bridge
    - validate summarization/context path
    - validate diagnostics audit
    - validate SQLite persistence when enabled
    - validate vector boundary when enabled

    Non-responsibilities:
    - no real LLM calls
    - no production vector database
    - no direct cognition-store shortcut
    - no memory policy mutation
    """

    def __init__(
        self,
        *,
        config: MemoryPhase4ValidatorConfig | None = None,
    ) -> None:
        self._config = config or MemoryPhase4ValidatorConfig()
        self._config.validate()

        self._lock = RLock()
        self._logger = get_logger("memory.phase4_validator")

        self._validation_count = 0
        self._last_status: MemoryPhase4ValidationStatus | None = None
        self._last_passed_count = 0
        self._last_failed_count = 0
        self._last_error: str | None = None

    @property
    def name(self) -> str:
        return self._config.name

    def validate(self) -> MemoryPhase4ValidationResult:
        """
        Run complete Phase 4 memory validation.
        """

        checks = [
            self._safe_check(
                name="core_contracts",
                fn=self._validate_core_contracts,
            ),
            self._safe_check(
                name="gateway_write_retrieve",
                fn=self._validate_gateway_write_retrieve,
            ),
            self._safe_check(
                name="retrieval_explainability",
                fn=self._validate_retrieval_explainability,
            ),
            self._safe_check(
                name="cognition_context_bridge",
                fn=self._validate_cognition_context_bridge,
            ),
            self._safe_check(
                name="summary_context_path",
                fn=self._validate_summary_context_path,
            ),
            self._safe_check(
                name="diagnostics_audit",
                fn=self._validate_diagnostics_audit,
            ),
        ]

        if self._config.include_sqlite:
            checks.append(
                self._safe_check(
                    name="sqlite_persistence",
                    fn=self._validate_sqlite_persistence,
                )
            )

        if self._config.include_vector:
            checks.append(
                self._safe_check(
                    name="vector_boundary",
                    fn=self._validate_vector_boundary,
                )
            )

        status = (
            MemoryPhase4ValidationStatus.PASSED
            if all(check.passed for check in checks)
            else MemoryPhase4ValidationStatus.FAILED
        )
        result = MemoryPhase4ValidationResult(
            status=status,
            checks=tuple(checks),
            metadata={
                "validator": self.name,
                "include_sqlite": self._config.include_sqlite,
                "include_vector": self._config.include_vector,
            },
        )
        self._record_result(result)

        self._logger.info(
            "memory_phase4_validation_completed",
            validator=self.name,
            status=result.status.value,
            passed_count=result.passed_count,
            failed_count=result.failed_count,
        )

        return result

    def snapshot(self) -> MemoryPhase4ValidatorSnapshot:
        """
        Return validator diagnostics.
        """

        with self._lock:
            return MemoryPhase4ValidatorSnapshot(
                name=self.name,
                validation_count=self._validation_count,
                last_status=self._last_status,
                last_passed_count=self._last_passed_count,
                last_failed_count=self._last_failed_count,
                last_error=self._last_error,
            )

    def reset(self) -> None:
        """
        Reset validator diagnostics.
        """

        with self._lock:
            self._validation_count = 0
            self._last_status = None
            self._last_passed_count = 0
            self._last_failed_count = 0
            self._last_error = None

        self._logger.info("memory_phase4_validator_reset", validator=self.name)

    def _safe_check(
        self,
        *,
        name: str,
        fn: Callable[[], MemoryPhase4ValidationCheck],
    ) -> MemoryPhase4ValidationCheck:
        try:
            return fn()

        except Exception as exc:
            return MemoryPhase4ValidationCheck(
                name=name,
                passed=False,
                detail=f"{type(exc).__name__}: {exc}",
                metadata={
                    "validator": self.name,
                },
            )

    def _validate_core_contracts(self) -> MemoryPhase4ValidationCheck:
        record = MemoryRecord(
            kind=MemoryKind.PROJECT,
            text="Phase 4 memory contracts are typed and strict.",
            importance=MemoryImportance.HIGH,
            tags=("phase4", "contracts"),
        )
        query = MemoryQuery(text="memory contracts", kinds=(MemoryKind.PROJECT,))

        passed = (
            record.memory_id != ""
            and query.text == "memory contracts"
            and query.kinds == (MemoryKind.PROJECT,)
        )

        return MemoryPhase4ValidationCheck(
            name="core_contracts",
            passed=passed,
            detail="memory core contracts validated",
            metadata={
                "memory_id": record.memory_id,
                "query_id": query.query_id,
            },
        )

    def _validate_gateway_write_retrieve(self) -> MemoryPhase4ValidationCheck:
        gateway = self._seed_gateway()
        retrieval = gateway.retrieve(MemoryQuery(text="memory gateway"))

        passed = retrieval.allowed and retrieval.result_count > 0

        return MemoryPhase4ValidationCheck(
            name="gateway_write_retrieve",
            passed=passed,
            detail="gateway-controlled write and retrieval validated",
            metadata={
                "result_count": retrieval.result_count,
                "gateway": gateway.name,
            },
        )

    def _validate_retrieval_explainability(self) -> MemoryPhase4ValidationCheck:
        gateway = self._seed_gateway()
        retrieval = gateway.retrieve(MemoryQuery(text="cognition memory"))

        explanations_complete = all(
            result.explanation.reason
            and result.explanation.source
            and result.explanation.retrieved_at.tzinfo is not None
            and 0.0 <= result.explanation.confidence <= 1.0
            and result.explanation.policy_classification
            for result in retrieval.results
        )

        passed = retrieval.result_count > 0 and explanations_complete

        return MemoryPhase4ValidationCheck(
            name="retrieval_explainability",
            passed=passed,
            detail="retrieval explanation fields validated",
            metadata={
                "result_count": retrieval.result_count,
            },
        )

    def _validate_cognition_context_bridge(self) -> MemoryPhase4ValidationCheck:
        gateway = self._seed_gateway()
        bridge = MemoryCognitionBridge(gateway=gateway)

        result = bridge.build_context_from_text("memory gateway")

        passed = (
            result.allowed
            and not result.blocked
            and result.context.item_count > 0
            and result.context.total_chars > 0
        )

        return MemoryPhase4ValidationCheck(
            name="cognition_context_bridge",
            passed=passed,
            detail="cognition-memory bridge built bounded context",
            metadata={
                "context_id": result.context.context_id,
                "item_count": result.context.item_count,
                "total_chars": result.context.total_chars,
            },
        )

    def _validate_summary_context_path(self) -> MemoryPhase4ValidationCheck:
        gateway = self._seed_gateway()
        bridge = MemoryCognitionBridge(
            gateway=gateway,
            context_builder=MemoryContextBuilder(),
            summarizer=ExtractiveMemorySummarizer(),
        )
        result = bridge.build_context_from_text(
            "memory",
            include_summary=True,
            max_context_items=4,
        )

        passed = (
            result.allowed
            and result.summary_result is not None
            and result.summary_result.succeeded
            and result.context.item_count > 0
        )

        return MemoryPhase4ValidationCheck(
            name="summary_context_path",
            passed=passed,
            detail="summary-to-context path validated",
            metadata={
                "summary_used": result.summary_result is not None,
                "context_item_count": result.context.item_count,
            },
        )

    def _validate_diagnostics_audit(self) -> MemoryPhase4ValidationCheck:
        gateway = self._seed_gateway()
        bridge = MemoryCognitionBridge(gateway=gateway)
        context_result = bridge.build_context_from_text("memory gateway")

        diagnostics = MemoryDiagnosticsCollector(
            components={
                "gateway": gateway,
                "cognition_bridge": bridge,
            }
        )
        report = diagnostics.collect()
        retrieval_audit = diagnostics.audit_retrieval(context_result.retrieval)
        context_audit = diagnostics.audit_context(context_result.context)

        passed = (
            report.status != MemoryDiagnosticStatus.FAILED
            and retrieval_audit.status != MemoryDiagnosticStatus.FAILED
            and context_audit.status != MemoryDiagnosticStatus.FAILED
        )

        return MemoryPhase4ValidationCheck(
            name="diagnostics_audit",
            passed=passed,
            detail="diagnostics collection and audits validated",
            metadata={
                "diagnostics_status": report.status.value,
                "retrieval_audit": retrieval_audit.status.value,
                "context_audit": context_audit.status.value,
            },
        )

    def _validate_sqlite_persistence(self) -> MemoryPhase4ValidationCheck:
        sqlite_path = self._config.sqlite_path
        sqlite_path.parent.mkdir(parents=True, exist_ok=True)

        store = SQLiteMemoryStore(
            config=SQLiteMemoryStoreConfig(path=sqlite_path)
        )
        store.clear()

        gateway = GovernedMemoryGateway(store=store)
        write = gateway.remember(
            MemoryWriteRequest(
                kind=MemoryKind.PROJECT,
                text="SQLite memory validation persists records.",
                importance=MemoryImportance.HIGH,
                tags=("sqlite", "validation"),
            )
        )

        recreated = SQLiteMemoryStore(
            config=SQLiteMemoryStoreConfig(path=sqlite_path)
        )
        retrieval = GovernedMemoryGateway(store=recreated).retrieve(
            MemoryQuery(text="sqlite validation")
        )

        passed = (
            write.allowed
            and write.record is not None
            and retrieval.allowed
            and retrieval.result_count > 0
        )

        return MemoryPhase4ValidationCheck(
            name="sqlite_persistence",
            passed=passed,
            detail="SQLite persistent memory store validated",
            metadata={
                "path": str(sqlite_path),
                "result_count": retrieval.result_count,
            },
        )

    def _validate_vector_boundary(self) -> MemoryPhase4ValidationCheck:
        gateway = self._seed_gateway()
        retrieval = gateway.retrieve(MemoryQuery(max_results=50))

        provider = DeterministicEmbeddingProvider()
        index = InMemoryVectorIndex()

        for record in retrieval.records:
            embedding = provider.embed_text(record.text)
            document = MemoryVectorDocument.from_record(
                record=record,
                embedding=embedding,
            )
            index.upsert(document)

        query_embedding = provider.embed_text("memory gateway")
        response = index.search(
            MemoryVectorSearchQuery(
                text="memory gateway",
                embedding=query_embedding,
                max_results=8,
            )
        )

        passed = response.result_count > 0 and index.snapshot().document_count > 0

        return MemoryPhase4ValidationCheck(
            name="vector_boundary",
            passed=passed,
            detail="embedding provider and vector index boundary validated",
            metadata={
                "vector_results": response.result_count,
                "document_count": index.snapshot().document_count,
            },
        )

    @staticmethod
    def _seed_gateway() -> GovernedMemoryGateway:
        gateway = GovernedMemoryGateway(store=InMemoryMemoryStore())

        requests = (
            MemoryWriteRequest(
                kind=MemoryKind.PROJECT,
                text=(
                    "JARVIS memory gateway is the only cognition-facing "
                    "memory entry point."
                ),
                importance=MemoryImportance.CRITICAL,
                tags=("jarvis", "memory", "gateway"),
            ),
            MemoryWriteRequest(
                kind=MemoryKind.SEMANTIC,
                text=(
                    "Memory context builder prepares bounded "
                    "cognition-ready context."
                ),
                importance=MemoryImportance.HIGH,
                tags=("jarvis", "memory", "context"),
            ),
            MemoryWriteRequest(
                kind=MemoryKind.USER_PROFILE,
                text=(
                    "User is building a personal cognition OS with "
                    "safe governed memory."
                ),
                importance=MemoryImportance.HIGH,
                tags=("jarvis", "profile", "cognition"),
            ),
            MemoryWriteRequest(
                kind=MemoryKind.SYSTEM,
                text=(
                    "Sensitive memories are filtered unless policy "
                    "explicitly allows retrieval."
                ),
                importance=MemoryImportance.HIGH,
                sensitivity=MemorySensitivity.PRIVATE,
                tags=("jarvis", "policy", "safety"),
            ),
        )

        for request in requests:
            result = gateway.remember(request)

            if not result.allowed:
                raise RuntimeError(f"seed memory write blocked: {result.reason}")

        return gateway

    def _record_result(self, result: MemoryPhase4ValidationResult) -> None:
        with self._lock:
            self._validation_count += 1
            self._last_status = result.status
            self._last_passed_count = result.passed_count
            self._last_failed_count = result.failed_count
            self._last_error = (
                None
                if result.passed
                else f"memory validation failed: {result.failed_count} checks"
            )


def validate_phase4_memory(
    *,
    sqlite_path: Path | None = None,
    include_sqlite: bool = True,
    include_vector: bool = True,
) -> MemoryPhase4ValidationResult:
    """
    Convenience function for scripts and tests.
    """

    config = MemoryPhase4ValidatorConfig(
        sqlite_path=sqlite_path or Path(".jarvis_memory_validation/memory.db"),
        include_sqlite=include_sqlite,
        include_vector=include_vector,
    )
    validator = MemoryPhase4Validator(config=config)

    return validator.validate()