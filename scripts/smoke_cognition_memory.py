from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from jarvis.cognition import (  # noqa: E402
    CognitionRequest,
    FakeCognitionAdapter,
    FakeCognitionConfig,
)
from jarvis.memory import (  # noqa: E402
    EpisodicMemoryActor,
    EpisodicMemoryEventKind,
    EpisodicMemoryRuntime,
    ExtractiveMemorySummarizer,
    GovernedMemoryGateway,
    InMemoryMemoryStore,
    MemoryCognitionBridge,
    MemoryCognitionContextResult,
    MemoryDiagnosticsCollector,
    MemoryDiagnosticStatus,
    MemoryImportance,
    MemoryKind,
    MemoryPhase4ValidationStatus,
    MemorySensitivity,
    MemorySource,
    MemoryWriteRequest,
    SQLiteMemoryStore,
    SQLiteMemoryStoreConfig,
    UserProfileMemoryCategory,
    UserProfileMemoryRuntime,
    validate_phase4_memory,
)

MemorySmokeBackend = Literal["in-memory", "sqlite"]


@dataclass(frozen=True, slots=True)
class CognitionMemorySmokeConfig:
    """
    Configuration for the real cognition + memory smoke.

    The adapter remains fake by default so this smoke is deterministic, fast,
    and safe in CI. The architecture path is real.
    """

    prompt: str = "What should JARVIS remember about my project?"
    backend: MemorySmokeBackend = "in-memory"
    sqlite_path: Path | None = None
    include_summary: bool = True
    validate_phase4: bool = True
    capture_response: bool = True


@dataclass(frozen=True, slots=True)
class CognitionMemorySmokeReport:
    """
    Final smoke report.
    """

    passed: bool
    backend: MemorySmokeBackend
    prompt: str
    response_text: str
    memory_context_items: int
    memory_context_chars: int
    retrieval_results: int
    captured_response: bool
    diagnostics_status: MemoryDiagnosticStatus
    retrieval_audit_status: MemoryDiagnosticStatus
    context_audit_status: MemoryDiagnosticStatus
    phase4_validation_status: MemoryPhase4ValidationStatus | None

    def as_lines(self) -> tuple[str, ...]:
        validation_status = (
            self.phase4_validation_status.value
            if self.phase4_validation_status is not None
            else "skipped"
        )

        return (
            "JARVIS Phase 4 Cognition + Memory Smoke",
            "---------------------------------------",
            f"Passed: {self.passed}",
            f"Backend: {self.backend}",
            f"Prompt: {self.prompt}",
            f"Response: {self.response_text}",
            f"Memory context items: {self.memory_context_items}",
            f"Memory context chars: {self.memory_context_chars}",
            f"Retrieval results: {self.retrieval_results}",
            f"Captured response: {self.captured_response}",
            f"Diagnostics: {self.diagnostics_status.value}",
            f"Retrieval audit: {self.retrieval_audit_status.value}",
            f"Context audit: {self.context_audit_status.value}",
            f"Phase 4 validation: {validation_status}",
        )


def run_cognition_memory_smoke(
    config: CognitionMemorySmokeConfig,
) -> CognitionMemorySmokeReport:
    """
    Run the Phase 4 real cognition + memory smoke.

    This proves cognition can receive memory context through the MemoryGateway
    boundary and produce a response with memory metadata available.
    """

    gateway = _build_gateway(config)
    _seed_memory(gateway)

    memory_bridge = MemoryCognitionBridge(
        gateway=gateway,
        summarizer=ExtractiveMemorySummarizer(),
    )
    episodic_memory = EpisodicMemoryRuntime(gateway=gateway)
    profile_memory = UserProfileMemoryRuntime(gateway=gateway)

    profile_memory.save_text(
        "User wants JARVIS to preserve strict boundaries, typed contracts, "
        "gateway architecture, policy-first safety, observability, "
        "fake-first validation, cancellation support, streaming readiness, "
        "runtime isolation, and no direct LLM control.",
        profile_id="smoke-profile-architecture-principles",
        category=UserProfileMemoryCategory.SYSTEM_PREFERENCE,
        importance=MemoryImportance.CRITICAL,
        source=MemorySource.USER_EXPLICIT,
        tags=("jarvis", "architecture", "safety"),
    )

    memory_context_result = memory_bridge.build_context_from_text(
        config.prompt,
        include_summary=config.include_summary,
        max_results=8,
        max_context_items=8,
        max_context_chars=2_000,
    )

    cognition_response = _run_cognition(
        prompt=config.prompt,
        memory_context_result=memory_context_result,
    )

    captured_response = False

    if config.capture_response:
        captured = episodic_memory.capture_text(
            cognition_response,
            event_id="smoke-cognition-response",
            kind=EpisodicMemoryEventKind.ASSISTANT_RESPONSE,
            actor=EpisodicMemoryActor.ASSISTANT,
            importance=MemoryImportance.NORMAL,
            sensitivity=MemorySensitivity.PRIVATE,
            tags=("smoke", "cognition", "memory"),
            metadata={
                "prompt": config.prompt,
                "memory_context_id": memory_context_result.context.context_id,
            },
        )
        captured_response = captured.allowed

    diagnostics = MemoryDiagnosticsCollector(
        components={
            "gateway": gateway,
            "memory_bridge": memory_bridge,
        }
    )
    diagnostics_report = diagnostics.collect()
    retrieval_audit = diagnostics.audit_retrieval(memory_context_result.retrieval)
    context_audit = diagnostics.audit_context(memory_context_result.context)

    phase4_validation_status: MemoryPhase4ValidationStatus | None = None

    if config.validate_phase4:
        validation_result = validate_phase4_memory(
            sqlite_path=_validation_sqlite_path(config),
            include_sqlite=True,
            include_vector=True,
        )
        phase4_validation_status = validation_result.status

    passed = (
        memory_context_result.allowed
        and not memory_context_result.blocked
        and memory_context_result.retrieval.result_count > 0
        and memory_context_result.context.item_count > 0
        and cognition_response.strip() != ""
        and (captured_response or not config.capture_response)
        and diagnostics_report.status != MemoryDiagnosticStatus.FAILED
        and retrieval_audit.status != MemoryDiagnosticStatus.FAILED
        and context_audit.status != MemoryDiagnosticStatus.FAILED
        and (
            phase4_validation_status is None
            or phase4_validation_status == MemoryPhase4ValidationStatus.PASSED
        )
    )

    return CognitionMemorySmokeReport(
        passed=passed,
        backend=config.backend,
        prompt=config.prompt,
        response_text=cognition_response,
        memory_context_items=memory_context_result.context.item_count,
        memory_context_chars=memory_context_result.context.total_chars,
        retrieval_results=memory_context_result.retrieval.result_count,
        captured_response=captured_response,
        diagnostics_status=diagnostics_report.status,
        retrieval_audit_status=retrieval_audit.status,
        context_audit_status=context_audit.status,
        phase4_validation_status=phase4_validation_status,
    )


def _run_cognition(
    *,
    prompt: str,
    memory_context_result: MemoryCognitionContextResult,
) -> str:
    adapter = FakeCognitionAdapter(
        config=FakeCognitionConfig(
            default_response=(
                "Memory is active, sir. I have project context, profile "
                "principles, and governed retrieval ready."
            )
        )
    )
    request = CognitionRequest(
        request_id="smoke-cognition-memory-request",
        text=prompt,
        metadata={
            **memory_context_result.as_cognition_metadata(),
            "memory_context_block": memory_context_result.context.as_text_block(),
            "smoke": "phase4_cognition_memory",
        },
    )

    result = adapter.generate(request)

    if not result.succeeded or result.response is None:
        raise RuntimeError("cognition adapter failed")

    return result.response.text


def _build_gateway(config: CognitionMemorySmokeConfig) -> GovernedMemoryGateway:
    if config.backend == "sqlite":
        sqlite_path = config.sqlite_path or (
            PROJECT_ROOT / ".jarvis_memory_smoke" / "cognition_memory.db"
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
                "JARVIS is a local personal cognition OS with governed "
                "memory, safe tools, voice runtime, and cognition runtime."
            ),
            importance=MemoryImportance.CRITICAL,
            tags=("jarvis", "project", "cognition", "memory"),
        ),
        MemoryWriteRequest(
            kind=MemoryKind.SEMANTIC,
            text=(
                "MemoryGateway is the only approved entry point from "
                "cognition into memory."
            ),
            importance=MemoryImportance.CRITICAL,
            tags=("jarvis", "gateway", "memory", "safety"),
        ),
        MemoryWriteRequest(
            kind=MemoryKind.SEMANTIC,
            text=(
                "Every memory retrieval must include source, reason, "
                "confidence, timestamp, and policy classification."
            ),
            importance=MemoryImportance.CRITICAL,
            tags=("jarvis", "memory", "explainability"),
        ),
        MemoryWriteRequest(
            kind=MemoryKind.USER_PROFILE,
            text=(
                "User wants a real-time personal cognitive OS that supports "
                "education, debugging, research, engineering, and safe laptop "
                "control through governed actions."
            ),
            importance=MemoryImportance.HIGH,
            tags=("jarvis", "profile", "goals"),
        ),
    )

    for request in requests:
        result = gateway.remember(request)

        if not result.allowed:
            raise RuntimeError(f"seed memory blocked: {result.reason}")


def _validation_sqlite_path(config: CognitionMemorySmokeConfig) -> Path:
    if config.sqlite_path is not None:
        return config.sqlite_path.with_name("validation_memory.db")

    return PROJECT_ROOT / ".jarvis_memory_validation" / "cognition_memory.db"


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Phase 4 cognition + memory smoke."
    )
    parser.add_argument(
        "--prompt",
        default="What should JARVIS remember about my project?",
        help="Prompt to send through cognition-memory smoke.",
    )
    parser.add_argument(
        "--backend",
        choices=("in-memory", "sqlite"),
        default="in-memory",
        help="Memory backend.",
    )
    parser.add_argument(
        "--sqlite-path",
        type=Path,
        default=None,
        help="SQLite DB path when backend=sqlite.",
    )
    parser.add_argument(
        "--no-summary",
        action="store_true",
        help="Disable memory summary path.",
    )
    parser.add_argument(
        "--no-validation",
        action="store_true",
        help="Skip Phase 4 validation inside this smoke.",
    )
    parser.add_argument(
        "--no-capture",
        action="store_true",
        help="Do not capture assistant response into episodic memory.",
    )

    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    config = CognitionMemorySmokeConfig(
        prompt=args.prompt,
        backend=args.backend,
        sqlite_path=args.sqlite_path,
        include_summary=not args.no_summary,
        validate_phase4=not args.no_validation,
        capture_response=not args.no_capture,
    )
    report = run_cognition_memory_smoke(config)

    print()
    for line in report.as_lines():
        print(line)

    return 0 if report.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())