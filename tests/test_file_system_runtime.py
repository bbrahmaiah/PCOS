from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from jarvis.tools import (
    ActionStatus,
    FileOperationDecision,
    FileOperationKind,
    FileOperationReason,
    FileOperationRequest,
    FileSystemRuntime,
    FileSystemRuntimeConfig,
)


def runtime(tmp_path: Path) -> FileSystemRuntime:
    return FileSystemRuntime(
        config=FileSystemRuntimeConfig(workspace_root=str(tmp_path))
    )


def test_config_rejects_invalid_values() -> None:
    with pytest.raises(ValueError):
        FileSystemRuntimeConfig(name=" ").validate()

    with pytest.raises(ValueError):
        FileSystemRuntimeConfig(max_read_chars=0).validate()

    with pytest.raises(ValueError):
        FileSystemRuntimeConfig(max_search_results=0).validate()


def test_request_requires_content_for_write() -> None:
    with pytest.raises(ValidationError):
        FileOperationRequest(
            kind=FileOperationKind.WRITE_FILE,
            path="notes.txt",
        )


def test_request_requires_patch_payload() -> None:
    with pytest.raises(ValidationError):
        FileOperationRequest(
            kind=FileOperationKind.PATCH_FILE,
            path="notes.txt",
        )


def test_request_requires_destination_for_copy() -> None:
    with pytest.raises(ValidationError):
        FileOperationRequest(
            kind=FileOperationKind.COPY_FILE,
            path="source.txt",
        )


def test_read_file(tmp_path: Path) -> None:
    (tmp_path / "README.md").write_text("hello jarvis", encoding="utf-8")
    fs = runtime(tmp_path)

    result = fs.execute(
        FileOperationRequest(
            kind=FileOperationKind.READ_FILE,
            path="README.md",
        )
    )

    assert result.success is True
    assert result.status == ActionStatus.SUCCEEDED
    assert result.content == "hello jarvis"
    assert result.output == "read 12 characters"


def test_list_directory(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("a", encoding="utf-8")
    (tmp_path / "b.txt").write_text("b", encoding="utf-8")
    fs = runtime(tmp_path)

    result = fs.execute(
        FileOperationRequest(
            kind=FileOperationKind.LIST_DIRECTORY,
            path=".",
        )
    )

    assert result.success is True
    assert "a.txt" in result.output
    assert "b.txt" in result.output


def test_search_files(tmp_path: Path) -> None:
    (tmp_path / "a.py").write_text("print('a')", encoding="utf-8")
    (tmp_path / "b.txt").write_text("b", encoding="utf-8")
    fs = runtime(tmp_path)

    result = fs.execute(
        FileOperationRequest(
            kind=FileOperationKind.SEARCH_FILES,
            path=".",
            pattern="*.py",
        )
    )

    assert result.success is True
    assert "a.py" in result.output
    assert "b.txt" not in result.output


def test_create_draft_allowed(tmp_path: Path) -> None:
    fs = runtime(tmp_path)

    result = fs.execute(
        FileOperationRequest(
            kind=FileOperationKind.CREATE_DRAFT,
            path="draft.txt",
            content="draft",
        )
    )

    assert result.success is True
    assert (tmp_path / "draft.txt").read_text(encoding="utf-8") == "draft"


def test_create_draft_blocks_existing_without_overwrite(tmp_path: Path) -> None:
    (tmp_path / "draft.txt").write_text("old", encoding="utf-8")
    fs = runtime(tmp_path)

    result = fs.execute(
        FileOperationRequest(
            kind=FileOperationKind.CREATE_DRAFT,
            path="draft.txt",
            content="new",
        )
    )

    assert result.success is False
    assert result.metadata["reason"] == FileOperationReason.DESTINATION_EXISTS.value


def test_write_requires_confirmation(tmp_path: Path) -> None:
    fs = runtime(tmp_path)

    result = fs.execute(
        FileOperationRequest(
            kind=FileOperationKind.WRITE_FILE,
            path="notes.txt",
            content="hello",
        )
    )

    assert result.success is False
    assert result.status == ActionStatus.BLOCKED
    assert result.policy_result.decision == FileOperationDecision.REQUIRE_CONFIRMATION


def test_write_file_with_backup_and_diff(tmp_path: Path) -> None:
    target = tmp_path / "notes.txt"
    target.write_text("old", encoding="utf-8")
    fs = runtime(tmp_path)

    result = fs.execute(
        FileOperationRequest(
            kind=FileOperationKind.WRITE_FILE,
            path="notes.txt",
            content="new",
            confirmed=True,
        )
    )

    assert result.success is True
    assert target.read_text(encoding="utf-8") == "new"
    assert result.backup_path is not None
    assert result.rollback_supported is True
    assert "-old" in (result.diff or "")
    assert "+new" in (result.diff or "")


def test_patch_file_with_backup_and_diff(tmp_path: Path) -> None:
    target = tmp_path / "notes.txt"
    target.write_text("hello old world", encoding="utf-8")
    fs = runtime(tmp_path)

    result = fs.execute(
        FileOperationRequest(
            kind=FileOperationKind.PATCH_FILE,
            path="notes.txt",
            old_text="old",
            new_text="new",
            confirmed=True,
        )
    )

    assert result.success is True
    assert target.read_text(encoding="utf-8") == "hello new world"
    assert result.backup_path is not None
    assert "-hello old world" in (result.diff or "")
    assert "+hello new world" in (result.diff or "")


def test_copy_requires_confirmation(tmp_path: Path) -> None:
    (tmp_path / "source.txt").write_text("source", encoding="utf-8")
    fs = runtime(tmp_path)

    result = fs.execute(
        FileOperationRequest(
            kind=FileOperationKind.COPY_FILE,
            path="source.txt",
            destination_path="copy.txt",
        )
    )

    assert result.success is False
    assert result.policy_result.decision == FileOperationDecision.REQUIRE_CONFIRMATION


def test_copy_file_with_confirmation(tmp_path: Path) -> None:
    (tmp_path / "source.txt").write_text("source", encoding="utf-8")
    fs = runtime(tmp_path)

    result = fs.execute(
        FileOperationRequest(
            kind=FileOperationKind.COPY_FILE,
            path="source.txt",
            destination_path="copy.txt",
            confirmed=True,
        )
    )

    assert result.success is True
    assert (tmp_path / "copy.txt").read_text(encoding="utf-8") == "source"


def test_move_file_with_confirmation(tmp_path: Path) -> None:
    (tmp_path / "source.txt").write_text("source", encoding="utf-8")
    fs = runtime(tmp_path)

    result = fs.execute(
        FileOperationRequest(
            kind=FileOperationKind.MOVE_FILE,
            path="source.txt",
            destination_path="moved.txt",
            confirmed=True,
        )
    )

    assert result.success is True
    assert not (tmp_path / "source.txt").exists()
    assert (tmp_path / "moved.txt").exists()
    assert result.backup_path is not None


def test_delete_requires_approval(tmp_path: Path) -> None:
    (tmp_path / "delete.txt").write_text("delete", encoding="utf-8")
    fs = runtime(tmp_path)

    result = fs.execute(
        FileOperationRequest(
            kind=FileOperationKind.DELETE_FILE,
            path="delete.txt",
        )
    )

    assert result.success is False
    assert result.policy_result.decision == FileOperationDecision.REQUIRE_APPROVAL
    assert (tmp_path / "delete.txt").exists()


def test_delete_with_approval(tmp_path: Path) -> None:
    (tmp_path / "delete.txt").write_text("delete", encoding="utf-8")
    fs = runtime(tmp_path)

    result = fs.execute(
        FileOperationRequest(
            kind=FileOperationKind.DELETE_FILE,
            path="delete.txt",
            approved=True,
        )
    )

    assert result.success is True
    assert not (tmp_path / "delete.txt").exists()
    assert result.backup_path is not None
    assert result.rollback_supported is True


def test_path_traversal_is_blocked(tmp_path: Path) -> None:
    fs = runtime(tmp_path)

    result = fs.execute(
        FileOperationRequest(
            kind=FileOperationKind.READ_FILE,
            path="../secret.txt",
        )
    )

    assert result.success is False
    assert result.status == ActionStatus.FAILED


def test_absolute_path_is_blocked(tmp_path: Path) -> None:
    fs = runtime(tmp_path)

    result = fs.execute(
        FileOperationRequest(
            kind=FileOperationKind.READ_FILE,
            path=str(tmp_path.parent / "secret.txt"),
        )
    )

    assert result.success is False
    assert result.status == ActionStatus.FAILED


def test_read_missing_file_fails(tmp_path: Path) -> None:
    fs = runtime(tmp_path)

    result = fs.execute(
        FileOperationRequest(
            kind=FileOperationKind.READ_FILE,
            path="missing.txt",
        )
    )

    assert result.success is False
    assert result.metadata["reason"] == FileOperationReason.SOURCE_NOT_FOUND.value


def test_output_truncation(tmp_path: Path) -> None:
    (tmp_path / "long.txt").write_text("abcdef", encoding="utf-8")
    fs = FileSystemRuntime(
        config=FileSystemRuntimeConfig(
            workspace_root=str(tmp_path),
            max_read_chars=3,
        )
    )

    result = fs.execute(
        FileOperationRequest(
            kind=FileOperationKind.READ_FILE,
            path="long.txt",
        )
    )

    assert result.content == "abc"


def test_snapshot_and_reset(tmp_path: Path) -> None:
    (tmp_path / "README.md").write_text("hello", encoding="utf-8")
    fs = runtime(tmp_path)

    fs.execute(
        FileOperationRequest(
            kind=FileOperationKind.READ_FILE,
            path="README.md",
        )
    )
    snapshot = fs.snapshot()

    assert snapshot.operation_count == 1
    assert snapshot.success_count == 1

    fs.reset()
    reset_snapshot = fs.snapshot()

    assert reset_snapshot.operation_count == 0
    assert reset_snapshot.last_status is None


def test_enum_values_are_stable() -> None:
    assert FileOperationKind.READ_FILE.value == "read_file"
    assert FileOperationDecision.REQUIRE_APPROVAL.value == "require_approval"
    assert FileOperationReason.DELETE_REQUIRES_APPROVAL.value == (
        "delete_requires_approval"
    )