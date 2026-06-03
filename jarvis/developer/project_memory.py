from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any, Protocol
from uuid import uuid4

from jarvis.developer.code_context import CodeContextSnapshot
from jarvis.developer.error_intelligence import ErrorIntelligenceReport
from jarvis.developer.fix_suggestion import FixSuggestionReport
from jarvis.developer.test_runner import TestRunResult


def utc_now() -> datetime:
    return datetime.now(UTC)


class ProjectMemoryStatus(StrEnum):
    STORED = "stored"
    LOADED = "loaded"
    CLEARED = "cleared"
    BLOCKED = "blocked"


class ProjectMemoryKind(StrEnum):
    CODE_CONTEXT = "code_context"
    TEST_RESULT = "test_result"
    ERROR_DIAGNOSIS = "error_diagnosis"
    FIX_SUGGESTION = "fix_suggestion"
    TODO = "todo"
    NOTE = "note"
    SESSION_STATE = "session_state"


class ProjectMemoryImportance(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass(frozen=True, slots=True)
class ProjectMemoryRecord:
    record_id: str
    project_id: str
    kind: ProjectMemoryKind
    title: str
    summary: str
    details: str
    importance: ProjectMemoryImportance
    tags: tuple[str, ...]
    source: str
    created_at: datetime
    updated_at: datetime
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.record_id.strip():
            raise ValueError("project memory record_id cannot be empty.")
        if not self.project_id.strip():
            raise ValueError("project memory project_id cannot be empty.")
        if not self.title.strip():
            raise ValueError("project memory title cannot be empty.")
        if not self.summary.strip():
            raise ValueError("project memory summary cannot be empty.")
        if not self.source.strip():
            raise ValueError("project memory source cannot be empty.")


@dataclass(frozen=True, slots=True)
class ProjectMemorySnapshot:
    project_id: str
    project_root: str
    records: tuple[ProjectMemoryRecord, ...]
    active_files: tuple[str, ...]
    recent_errors: tuple[str, ...]
    recent_fixes: tuple[str, ...]
    todos: tuple[str, ...]
    last_updated_at: datetime
    metadata: dict[str, object] = field(default_factory=dict)

    @property
    def record_count(self) -> int:
        return len(self.records)


@dataclass(frozen=True, slots=True)
class ProjectMemoryStoreRequest:
    project_root: Path
    kind: ProjectMemoryKind
    title: str
    summary: str
    details: str = ""
    importance: ProjectMemoryImportance = ProjectMemoryImportance.MEDIUM
    tags: tuple[str, ...] = ()
    source: str = "developer"
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.title.strip():
            raise ValueError("project memory store title cannot be empty.")
        if not self.summary.strip():
            raise ValueError("project memory store summary cannot be empty.")
        if not self.source.strip():
            raise ValueError("project memory store source cannot be empty.")


@dataclass(frozen=True, slots=True)
class ProjectMemoryRecallRequest:
    project_root: Path
    query: str = ""
    kinds: tuple[ProjectMemoryKind, ...] = ()
    tags: tuple[str, ...] = ()
    limit: int = 10
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.limit < 1:
            raise ValueError("project memory recall limit must be at least 1.")


@dataclass(frozen=True, slots=True)
class ProjectMemoryResult:
    status: ProjectMemoryStatus
    reason: str
    snapshot: ProjectMemorySnapshot | None
    record: ProjectMemoryRecord | None = None
    records: tuple[ProjectMemoryRecord, ...] = ()
    created_at: datetime = field(default_factory=utc_now)
    metadata: dict[str, object] = field(default_factory=dict)

    @property
    def succeeded(self) -> bool:
        return self.status in {
            ProjectMemoryStatus.STORED,
            ProjectMemoryStatus.LOADED,
            ProjectMemoryStatus.CLEARED,
        }


class ProjectMemoryStore(Protocol):
    def load(self, project_root: Path) -> ProjectMemorySnapshot:
        ...

    def save(self, snapshot: ProjectMemorySnapshot) -> None:
        ...

    def clear(self, project_root: Path) -> None:
        ...


class JsonProjectMemoryStore:
    """
    JSON-backed project memory store.

    It writes only to the explicit project memory file:
    .jarvis/project_memory.json

    It does not mutate source files.
    """

    def memory_path(self, project_root: Path) -> Path:
        return project_root.expanduser().resolve() / ".jarvis" / (
            "project_memory.json"
        )

    def load(self, project_root: Path) -> ProjectMemorySnapshot:
        root = project_root.expanduser().resolve()
        path = self.memory_path(root)

        if not path.exists():
            return _empty_snapshot(project_root=root)

        data = json.loads(path.read_text(encoding="utf-8"))
        return _snapshot_from_json(data=data, project_root=root)

    def save(self, snapshot: ProjectMemorySnapshot) -> None:
        path = self.memory_path(Path(snapshot.project_root))
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(_snapshot_to_json(snapshot), indent=2, sort_keys=True),
            encoding="utf-8",
        )

    def clear(self, project_root: Path) -> None:
        path = self.memory_path(project_root)
        if path.exists():
            path.unlink()


class ProjectMemoryEngine:
    """
    Step 47E Project Memory Engine.

    This gives Developer Pack continuity across sessions.

    It remembers:
    - code context summaries
    - test results
    - error diagnoses
    - fix suggestions
    - TODOs and notes
    - active files and project state

    It never edits source files.
    It never runs commands.
    It only writes explicit project-scoped memory.
    """

    def __init__(
        self,
        *,
        store: ProjectMemoryStore | None = None,
    ) -> None:
        self._store = store or JsonProjectMemoryStore()

    def snapshot(self, project_root: Path) -> ProjectMemoryResult:
        root_result = _validate_project_root(project_root)
        if root_result is not None:
            return root_result

        snapshot = self._store.load(project_root)
        return ProjectMemoryResult(
            status=ProjectMemoryStatus.LOADED,
            reason="project memory snapshot loaded",
            snapshot=snapshot,
            records=snapshot.records,
        )

    def store(self, request: ProjectMemoryStoreRequest) -> ProjectMemoryResult:
        root_result = _validate_project_root(request.project_root)
        if root_result is not None:
            return root_result

        snapshot = self._store.load(request.project_root)
        now = utc_now()
        record = ProjectMemoryRecord(
            record_id=f"pm_{uuid4().hex}",
            project_id=snapshot.project_id,
            kind=request.kind,
            title=request.title,
            summary=request.summary,
            details=request.details,
            importance=request.importance,
            tags=_clean_tags(request.tags),
            source=request.source,
            created_at=now,
            updated_at=now,
            metadata=request.metadata,
        )
        updated_snapshot = _with_record(snapshot=snapshot, record=record)
        self._store.save(updated_snapshot)

        return ProjectMemoryResult(
            status=ProjectMemoryStatus.STORED,
            reason="project memory record stored",
            snapshot=updated_snapshot,
            record=record,
            records=updated_snapshot.records,
        )

    def recall(
        self,
        request: ProjectMemoryRecallRequest,
    ) -> ProjectMemoryResult:
        root_result = _validate_project_root(request.project_root)
        if root_result is not None:
            return root_result

        snapshot = self._store.load(request.project_root)
        records = _filter_records(
            records=snapshot.records,
            query=request.query,
            kinds=request.kinds,
            tags=request.tags,
            limit=request.limit,
        )

        return ProjectMemoryResult(
            status=ProjectMemoryStatus.LOADED,
            reason="project memory records recalled",
            snapshot=snapshot,
            records=records,
            metadata={
                **request.metadata,
                "query": request.query,
                "limit": request.limit,
            },
        )

    def clear(self, project_root: Path) -> ProjectMemoryResult:
        root_result = _validate_project_root(project_root)
        if root_result is not None:
            return root_result

        self._store.clear(project_root)
        snapshot = _empty_snapshot(project_root=project_root.resolve())

        return ProjectMemoryResult(
            status=ProjectMemoryStatus.CLEARED,
            reason="project memory cleared",
            snapshot=snapshot,
            records=(),
        )

    def remember_code_context(
        self,
        *,
        project_root: Path,
        code_context: CodeContextSnapshot,
    ) -> ProjectMemoryResult:
        if code_context.summary is None:
            return ProjectMemoryResult(
                status=ProjectMemoryStatus.BLOCKED,
                reason="code context has no summary to remember",
                snapshot=None,
            )

        summary = code_context.summary
        return self.store(
            ProjectMemoryStoreRequest(
                project_root=project_root,
                kind=ProjectMemoryKind.CODE_CONTEXT,
                title="Code context snapshot",
                summary=(
                    f"{summary.project_name}: "
                    f"{summary.file_count} files, "
                    f"{summary.primary_language.value} primary language"
                ),
                details=(
                    f"Active file: {summary.active_file}\n"
                    f"Recent files: {', '.join(summary.recent_files[:10])}\n"
                    f"Important files: {', '.join(summary.important_files)}"
                ),
                importance=ProjectMemoryImportance.MEDIUM,
                tags=("code-context", summary.primary_language.value),
                source="code_context_engine",
                metadata={
                    "active_file": summary.active_file or "",
                    "file_count": summary.file_count,
                    "test_count": summary.test_count,
                    "config_count": summary.config_count,
                    "risks": summary.risks,
                },
            )
        )

    def remember_test_result(
        self,
        *,
        project_root: Path,
        test_result: TestRunResult,
    ) -> ProjectMemoryResult:
        importance = (
            ProjectMemoryImportance.HIGH
            if not test_result.passed
            else ProjectMemoryImportance.LOW
        )
        return self.store(
            ProjectMemoryStoreRequest(
                project_root=project_root,
                kind=ProjectMemoryKind.TEST_RESULT,
                title=f"Test run {test_result.status.value}",
                summary=test_result.summary,
                details=(
                    f"Command: {' '.join(test_result.plan.command)}\n"
                    f"Exit code: {test_result.exit_code}\n"
                    f"Duration ms: {test_result.duration_ms}\n"
                    f"Stdout excerpt: {test_result.stdout[:1000]}\n"
                    f"Stderr excerpt: {test_result.stderr[:1000]}"
                ),
                importance=importance,
                tags=("test-result", test_result.status.value),
                source="test_runner_engine",
                metadata={
                    "status": test_result.status.value,
                    "exit_code": test_result.exit_code,
                    "duration_ms": test_result.duration_ms,
                },
            )
        )

    def remember_error_report(
        self,
        *,
        project_root: Path,
        error_report: ErrorIntelligenceReport,
    ) -> ProjectMemoryResult:
        if error_report.diagnosis is None:
            return ProjectMemoryResult(
                status=ProjectMemoryStatus.BLOCKED,
                reason="error report has no diagnosis to remember",
                snapshot=None,
            )

        diagnosis = error_report.diagnosis
        importance = (
            ProjectMemoryImportance.HIGH
            if error_report.has_errors
            else ProjectMemoryImportance.LOW
        )
        return self.store(
            ProjectMemoryStoreRequest(
                project_root=project_root,
                kind=ProjectMemoryKind.ERROR_DIAGNOSIS,
                title=diagnosis.title,
                summary=diagnosis.summary,
                details=(
                    f"Likely cause: {diagnosis.likely_cause}\n"
                    f"Next action: {diagnosis.next_action}\n"
                    f"Affected files: {', '.join(diagnosis.affected_files)}"
                ),
                importance=importance,
                tags=("error", diagnosis.primary_category.value),
                source="error_intelligence_engine",
                metadata={
                    "category": diagnosis.primary_category.value,
                    "severity": diagnosis.severity.value,
                    "confidence": diagnosis.confidence,
                    "affected_files": diagnosis.affected_files,
                },
            )
        )

    def remember_fix_report(
        self,
        *,
        project_root: Path,
        fix_report: FixSuggestionReport,
    ) -> ProjectMemoryResult:
        if not fix_report.suggestions:
            return ProjectMemoryResult(
                status=ProjectMemoryStatus.BLOCKED,
                reason="fix report has no suggestions to remember",
                snapshot=None,
            )

        first = fix_report.suggestions[0]
        return self.store(
            ProjectMemoryStoreRequest(
                project_root=project_root,
                kind=ProjectMemoryKind.FIX_SUGGESTION,
                title=first.title,
                summary=first.proposed_change,
                details=(
                    f"Rationale: {first.rationale}\n"
                    f"Risk: {first.risk.value}\n"
                    f"Affected files: {', '.join(first.affected_files)}"
                ),
                importance=ProjectMemoryImportance.MEDIUM,
                tags=("fix-suggestion", first.kind.value),
                source="fix_suggestion_engine",
                metadata={
                    "kind": first.kind.value,
                    "risk": first.risk.value,
                    "confidence": first.confidence,
                    "suggestion_count": len(fix_report.suggestions),
                },
            )
        )


def _validate_project_root(project_root: Path) -> ProjectMemoryResult | None:
    root = project_root.expanduser().resolve()

    if not root.exists():
        return ProjectMemoryResult(
            status=ProjectMemoryStatus.BLOCKED,
            reason=f"project root does not exist: {root}",
            snapshot=None,
        )

    if not root.is_dir():
        return ProjectMemoryResult(
            status=ProjectMemoryStatus.BLOCKED,
            reason=f"project root is not a directory: {root}",
            snapshot=None,
        )

    return None


def _empty_snapshot(project_root: Path) -> ProjectMemorySnapshot:
    root = project_root.expanduser().resolve()
    return ProjectMemorySnapshot(
        project_id=_project_id(root),
        project_root=str(root),
        records=(),
        active_files=(),
        recent_errors=(),
        recent_fixes=(),
        todos=(),
        last_updated_at=utc_now(),
    )


def _with_record(
    *,
    snapshot: ProjectMemorySnapshot,
    record: ProjectMemoryRecord,
) -> ProjectMemorySnapshot:
    records = (record, *snapshot.records)
    return ProjectMemorySnapshot(
        project_id=snapshot.project_id,
        project_root=snapshot.project_root,
        records=records,
        active_files=_active_files(records),
        recent_errors=_recent_by_kind(
            records=records,
            kind=ProjectMemoryKind.ERROR_DIAGNOSIS,
        ),
        recent_fixes=_recent_by_kind(
            records=records,
            kind=ProjectMemoryKind.FIX_SUGGESTION,
        ),
        todos=_recent_by_kind(records=records, kind=ProjectMemoryKind.TODO),
        last_updated_at=utc_now(),
        metadata=snapshot.metadata,
    )


def _filter_records(
    *,
    records: tuple[ProjectMemoryRecord, ...],
    query: str,
    kinds: tuple[ProjectMemoryKind, ...],
    tags: tuple[str, ...],
    limit: int,
) -> tuple[ProjectMemoryRecord, ...]:
    query_text = query.strip().lower()
    wanted_tags = {tag.lower() for tag in tags}
    filtered: list[ProjectMemoryRecord] = []

    for record in records:
        if kinds and record.kind not in kinds:
            continue

        if wanted_tags and not wanted_tags.intersection(
            {tag.lower() for tag in record.tags}
        ):
            continue

        if query_text and query_text not in _search_text(record):
            continue

        filtered.append(record)

    return tuple(filtered[:limit])


def _search_text(record: ProjectMemoryRecord) -> str:
    return " ".join(
        (
            record.title,
            record.summary,
            record.details,
            " ".join(record.tags),
            record.kind.value,
        )
    ).lower()


def _active_files(
    records: tuple[ProjectMemoryRecord, ...],
) -> tuple[str, ...]:
    files: list[str] = []

    for record in records:
        active_file = record.metadata.get("active_file")
        if isinstance(active_file, str) and active_file and active_file not in files:
            files.append(active_file)

    return tuple(files[:20])


def _recent_by_kind(
    *,
    records: tuple[ProjectMemoryRecord, ...],
    kind: ProjectMemoryKind,
) -> tuple[str, ...]:
    return tuple(
        record.summary
        for record in records
        if record.kind == kind
    )[:20]


def _clean_tags(tags: tuple[str, ...]) -> tuple[str, ...]:
    cleaned: list[str] = []

    for tag in tags:
        value = tag.strip().lower()
        if value and value not in cleaned:
            cleaned.append(value)

    return tuple(cleaned)


def _project_id(root: Path) -> str:
    safe = root.name.strip().lower().replace(" ", "_")
    return safe or "project"


def _snapshot_to_json(snapshot: ProjectMemorySnapshot) -> dict[str, Any]:
    return {
        "project_id": snapshot.project_id,
        "project_root": snapshot.project_root,
        "records": [_record_to_json(record) for record in snapshot.records],
        "active_files": list(snapshot.active_files),
        "recent_errors": list(snapshot.recent_errors),
        "recent_fixes": list(snapshot.recent_fixes),
        "todos": list(snapshot.todos),
        "last_updated_at": snapshot.last_updated_at.isoformat(),
        "metadata": snapshot.metadata,
    }


def _snapshot_from_json(
    *,
    data: dict[str, Any],
    project_root: Path,
) -> ProjectMemorySnapshot:
    records = tuple(
        _record_from_json(item)
        for item in data.get("records", [])
        if isinstance(item, dict)
    )
    return ProjectMemorySnapshot(
        project_id=str(data.get("project_id") or _project_id(project_root)),
        project_root=str(data.get("project_root") or project_root),
        records=records,
        active_files=tuple(data.get("active_files", [])),
        recent_errors=tuple(data.get("recent_errors", [])),
        recent_fixes=tuple(data.get("recent_fixes", [])),
        todos=tuple(data.get("todos", [])),
        last_updated_at=_datetime_from_json(
            data.get("last_updated_at"),
            fallback=utc_now(),
        ),
        metadata=dict(data.get("metadata", {})),
    )


def _record_to_json(record: ProjectMemoryRecord) -> dict[str, Any]:
    return {
        "record_id": record.record_id,
        "project_id": record.project_id,
        "kind": record.kind.value,
        "title": record.title,
        "summary": record.summary,
        "details": record.details,
        "importance": record.importance.value,
        "tags": list(record.tags),
        "source": record.source,
        "created_at": record.created_at.isoformat(),
        "updated_at": record.updated_at.isoformat(),
        "metadata": record.metadata,
    }


def _record_from_json(data: dict[str, Any]) -> ProjectMemoryRecord:
    return ProjectMemoryRecord(
        record_id=str(data["record_id"]),
        project_id=str(data["project_id"]),
        kind=ProjectMemoryKind(str(data["kind"])),
        title=str(data["title"]),
        summary=str(data["summary"]),
        details=str(data.get("details", "")),
        importance=ProjectMemoryImportance(str(data["importance"])),
        tags=tuple(data.get("tags", [])),
        source=str(data["source"]),
        created_at=_datetime_from_json(data.get("created_at")),
        updated_at=_datetime_from_json(data.get("updated_at")),
        metadata=dict(data.get("metadata", {})),
    )


def _datetime_from_json(
    value: object,
    *,
    fallback: datetime | None = None,
) -> datetime:
    if isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(value)
            if parsed.tzinfo is None:
                return parsed.replace(tzinfo=UTC)
            return parsed
        except ValueError:
            pass

    if fallback is not None:
        return fallback

    return utc_now()