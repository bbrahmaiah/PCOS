from __future__ import annotations

import argparse
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from jarvis.memory import (  # noqa: E402
    DeterministicEmbeddingProvider,
    GovernedMemoryGateway,
    InMemoryMemoryStore,
    InMemoryVectorIndex,
    MemoryCognitionBridge,
    MemoryCognitionContextResult,
    MemoryDiagnosticsCollector,
    MemoryDiagnosticStatus,
    MemoryImportance,
    MemoryKind,
    MemoryQuery,
    MemorySensitivity,
    MemoryVectorDocument,
    MemoryVectorSearchQuery,
    MemoryVectorSearchResponse,
    MemoryWriteRequest,
    SQLiteMemoryStore,
    SQLiteMemoryStoreConfig,
)

MemorySmokeBackend = Literal["in-memory", "sqlite"]


@dataclass(frozen=True, slots=True)
class MemorySearchSmokeConfig:
    """
    Configuration for the Phase 4 memory search smoke script.
    """

    query_text: str = "memory gateway"
    backend: MemorySmokeBackend = "in-memory"
    sqlite_path: Path | None = None
    include_vector: bool = True
    parallel: bool = True
    max_results: int = 8
    max_context_items: int = 8
    max_context_chars: int = 2_000


@dataclass(frozen=True, slots=True)
class VectorSearchSmokeResult:
    """
    Result of the optional vector-boundary smoke path.
    """

    enabled: bool
    result_count: int
    passed: bool

    @staticmethod
    def skipped() -> VectorSearchSmokeResult:
        return VectorSearchSmokeResult(
            enabled=False,
            result_count=0,
            passed=True,
        )


@dataclass(frozen=True, slots=True)
class MemorySearchSmokeReport:
    """
    Final smoke report.
    """

    passed: bool
    backend: MemorySmokeBackend
    query_text: str
    lexical_result_count: int
    context_item_count: int
    context_total_chars: int
    vector_enabled: bool
    vector_result_count: int
    diagnostics_status: MemoryDiagnosticStatus
    retrieval_audit_status: MemoryDiagnosticStatus
    context_audit_status: MemoryDiagnosticStatus

    def as_lines(self) -> tuple[str, ...]:
        return (
            "JARVIS Phase 4 Memory Search Smoke",
            "----------------------------------",
            f"Passed: {self.passed}",
            f"Backend: {self.backend}",
            f"Query: {self.query_text}",
            f"Lexical results: {self.lexical_result_count}",
            f"Context items: {self.context_item_count}",
            f"Context chars: {self.context_total_chars}",
            f"Vector enabled: {self.vector_enabled}",
            f"Vector results: {self.vector_result_count}",
            f"Diagnostics: {self.diagnostics_status.value}",
            f"Retrieval audit: {self.retrieval_audit_status.value}",
            f"Context audit: {self.context_audit_status.value}",
        )


def run_memory_search_smoke(
    config: MemorySearchSmokeConfig,
) -> MemorySearchSmokeReport:
    """
    Run the Phase 4 memory search smoke.

    This proves memory can be seeded, retrieved, converted into cognition-ready
    context, audited, and optionally searched through the vector boundary.
    """

    gateway = _build_gateway(config)
    _seed_memory(gateway)

    bridge = MemoryCognitionBridge(gateway=gateway)

    if config.parallel:
        context_result, vector_result = _run_parallel(
            bridge=bridge,
            gateway=gateway,
            config=config,
        )
    else:
        context_result = _build_context(bridge=bridge, config=config)
        vector_result = (
            _run_vector_search(gateway=gateway, config=config)
            if config.include_vector
            else VectorSearchSmokeResult.skipped()
        )

    diagnostics = MemoryDiagnosticsCollector(
        components={
            "gateway": gateway,
            "cognition_bridge": bridge,
        }
    )
    diagnostics_report = diagnostics.collect()
    retrieval_audit = diagnostics.audit_retrieval(context_result.retrieval)
    context_audit = diagnostics.audit_context(context_result.context)

    passed = (
        context_result.allowed
        and not context_result.blocked
        and context_result.retrieval.result_count > 0
        and context_result.context.item_count > 0
        and vector_result.passed
        and diagnostics_report.status != MemoryDiagnosticStatus.FAILED
        and retrieval_audit.status != MemoryDiagnosticStatus.FAILED
        and context_audit.status != MemoryDiagnosticStatus.FAILED
    )

    return MemorySearchSmokeReport(
        passed=passed,
        backend=config.backend,
        query_text=config.query_text,
        lexical_result_count=context_result.retrieval.result_count,
        context_item_count=context_result.context.item_count,
        context_total_chars=context_result.context.total_chars,
        vector_enabled=vector_result.enabled,
        vector_result_count=vector_result.result_count,
        diagnostics_status=diagnostics_report.status,
        retrieval_audit_status=retrieval_audit.status,
        context_audit_status=context_audit.status,
    )


def _run_parallel(
    *,
    bridge: MemoryCognitionBridge,
    gateway: GovernedMemoryGateway,
    config: MemorySearchSmokeConfig,
) -> tuple[MemoryCognitionContextResult, VectorSearchSmokeResult]:
    with ThreadPoolExecutor(max_workers=2) as executor:
        context_future = executor.submit(
            _build_context,
            bridge=bridge,
            config=config,
        )
        vector_future = (
            executor.submit(
                _run_vector_search,
                gateway=gateway,
                config=config,
            )
            if config.include_vector
            else None
        )

        context_result = context_future.result()
        vector_result = (
            vector_future.result()
            if vector_future is not None
            else VectorSearchSmokeResult.skipped()
        )

    return context_result, vector_result


def _build_context(
    *,
    bridge: MemoryCognitionBridge,
    config: MemorySearchSmokeConfig,
) -> MemoryCognitionContextResult:
    return bridge.build_context_from_text(
        config.query_text,
        max_results=config.max_results,
        max_context_items=config.max_context_items,
        max_context_chars=config.max_context_chars,
    )


def _run_vector_search(
    *,
    gateway: GovernedMemoryGateway,
    config: MemorySearchSmokeConfig,
) -> VectorSearchSmokeResult:
    provider = DeterministicEmbeddingProvider()
    index = InMemoryVectorIndex()

    retrieval = gateway.retrieve(
        MemoryQuery(
            max_results=50,
            include_sensitive=False,
        )
    )

    for record in retrieval.records:
        embedding = provider.embed_text(record.text)
        document = MemoryVectorDocument.from_record(
            record=record,
            embedding=embedding,
        )
        index.upsert(document)

    query_embedding = provider.embed_text(config.query_text)
    response: MemoryVectorSearchResponse = index.search(
        MemoryVectorSearchQuery(
            text=config.query_text,
            embedding=query_embedding,
            max_results=config.max_results,
        )
    )

    return VectorSearchSmokeResult(
        enabled=True,
        result_count=response.result_count,
        passed=response.result_count > 0,
    )


def _build_gateway(config: MemorySearchSmokeConfig) -> GovernedMemoryGateway:
    if config.backend == "sqlite":
        sqlite_path = config.sqlite_path or (
            PROJECT_ROOT / ".jarvis_memory_smoke" / "memory.db"
        )
        store = SQLiteMemoryStore(
            config=SQLiteMemoryStoreConfig(path=sqlite_path)
        )

        store.clear()

        return GovernedMemoryGateway(store=store)

    return GovernedMemoryGateway(store=InMemoryMemoryStore())


def _seed_memory(gateway: GovernedMemoryGateway) -> None:
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
            text="Memory context builder prepares bounded cognition-ready context.",
            importance=MemoryImportance.HIGH,
            tags=("jarvis", "memory", "context"),
        ),
        MemoryWriteRequest(
            kind=MemoryKind.USER_PROFILE,
            text="User is building a personal cognition OS with safe governed memory.",
            importance=MemoryImportance.HIGH,
            tags=("jarvis", "profile", "cognition"),
        ),
        MemoryWriteRequest(
            kind=MemoryKind.PROJECT,
            text=(
                "Vector search must stay behind typed embedding and "
                "vector index boundaries."
            ),
            importance=MemoryImportance.NORMAL,
            tags=("jarvis", "vector", "boundary"),
        ),
        MemoryWriteRequest(
            kind=MemoryKind.SYSTEM,
            text=(
                "Sensitive memories are filtered unless policy explicitly "
                "allows retrieval."
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


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Phase 4 memory search smoke."
    )
    parser.add_argument(
        "--query",
        default="memory gateway",
        help="Memory search query text.",
    )
    parser.add_argument(
        "--backend",
        choices=("in-memory", "sqlite"),
        default="in-memory",
        help="Memory store backend for smoke.",
    )
    parser.add_argument(
        "--sqlite-path",
        type=Path,
        default=None,
        help="SQLite memory DB path when backend=sqlite.",
    )
    parser.add_argument(
        "--no-vector",
        action="store_true",
        help="Disable vector-boundary smoke.",
    )
    parser.add_argument(
        "--serial",
        action="store_true",
        help="Run smoke sequentially instead of parallel.",
    )
    parser.add_argument(
        "--max-results",
        type=int,
        default=8,
        help="Maximum memory retrieval results.",
    )

    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    config = MemorySearchSmokeConfig(
        query_text=args.query,
        backend=args.backend,
        sqlite_path=args.sqlite_path,
        include_vector=not args.no_vector,
        parallel=not args.serial,
        max_results=args.max_results,
    )
    report = run_memory_search_smoke(config)

    print()
    for line in report.as_lines():
        print(line)

    return 0 if report.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())