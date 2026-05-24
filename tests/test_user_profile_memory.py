from __future__ import annotations

import pytest
from pydantic import ValidationError

from jarvis.memory import (
    GovernedMemoryGateway,
    InMemoryMemoryStore,
    MemoryImportance,
    MemoryKind,
    MemoryRetention,
    MemoryScope,
    MemorySensitivity,
    MemorySource,
    UserProfileMemoryCategory,
    UserProfileMemoryConfidence,
    UserProfileMemoryFact,
    UserProfileMemoryQuery,
    UserProfileMemoryRuntime,
    UserProfileMemoryRuntimeConfig,
)


def make_runtime() -> UserProfileMemoryRuntime:
    return UserProfileMemoryRuntime(
        gateway=GovernedMemoryGateway(store=InMemoryMemoryStore()),
    )


def make_fact(
    *,
    profile_id: str = "profile-1",
    text: str = "User prefers direct, elite engineering guidance.",
    category: UserProfileMemoryCategory = (
        UserProfileMemoryCategory.COMMUNICATION_STYLE
    ),
    sensitivity: MemorySensitivity = MemorySensitivity.PRIVATE,
) -> UserProfileMemoryFact:
    return UserProfileMemoryFact(
        profile_id=profile_id,
        text=text,
        category=category,
        sensitivity=sensitivity,
        tags=("jarvis", "style"),
    )


def test_user_profile_runtime_config_rejects_invalid_values() -> None:
    with pytest.raises(ValueError):
        UserProfileMemoryRuntimeConfig(name=" ").validate()

    with pytest.raises(ValueError):
        UserProfileMemoryRuntimeConfig(default_confidence=-0.1).validate()

    with pytest.raises(ValueError):
        UserProfileMemoryRuntimeConfig(default_confidence=1.1).validate()


def test_user_profile_fact_rejects_empty_fields() -> None:
    with pytest.raises(ValidationError):
        UserProfileMemoryFact(
            profile_id=" ",
            text="valid",
            category=UserProfileMemoryCategory.PREFERENCE,
        )

    with pytest.raises(ValidationError):
        UserProfileMemoryFact(
            profile_id="profile-1",
            text=" ",
            category=UserProfileMemoryCategory.PREFERENCE,
        )


def test_user_profile_fact_cleans_tags() -> None:
    fact = UserProfileMemoryFact(
        profile_id="profile-1",
        text="User likes concise replies.",
        category=UserProfileMemoryCategory.PREFERENCE,
        tags=(" Style ", "style", " Voice "),
    )

    assert fact.tags == ("style", "voice")


def test_user_profile_fact_to_write_request() -> None:
    fact = make_fact()
    request = fact.to_write_request()

    assert request.kind == MemoryKind.USER_PROFILE
    assert request.scope == MemoryScope.USER
    assert request.text == fact.text
    assert request.source == MemorySource.USER_EXPLICIT
    assert request.importance == MemoryImportance.HIGH
    assert request.retention == MemoryRetention.PERSISTENT
    assert request.confidence == 0.9
    assert "profile" in request.tags
    assert "communication_style" in request.tags
    assert "high" in request.tags
    assert request.metadata["profile_id"] == "profile-1"
    assert request.metadata["profile_category"] == "communication_style"


def test_user_profile_query_to_memory_query() -> None:
    query = UserProfileMemoryQuery(
        text="engineering guidance",
        categories=(UserProfileMemoryCategory.COMMUNICATION_STYLE,),
        confidence_labels=(UserProfileMemoryConfidence.HIGH,),
        tags=("jarvis",),
        max_results=5,
    )
    memory_query = query.to_memory_query()

    assert memory_query.text == "engineering guidance"
    assert memory_query.kinds == (MemoryKind.USER_PROFILE,)
    assert memory_query.scopes == (MemoryScope.USER,)
    assert memory_query.max_results == 5
    assert "profile" in memory_query.tags
    assert "communication_style" in memory_query.tags
    assert "high" in memory_query.tags
    assert "jarvis" in memory_query.tags


def test_user_profile_runtime_saves_fact() -> None:
    runtime = make_runtime()

    result = runtime.save(make_fact())
    snapshot = runtime.snapshot()

    assert result.allowed is True
    assert result.record is not None
    assert result.record.kind == MemoryKind.USER_PROFILE
    assert result.record.importance == MemoryImportance.HIGH
    assert snapshot.saved_count == 1
    assert snapshot.saved_allowed_count == 1
    assert snapshot.last_profile_id == "profile-1"


def test_user_profile_runtime_blocks_sensitive_fact_by_gateway_policy() -> None:
    runtime = make_runtime()

    result = runtime.save(
        make_fact(
            text="Sensitive profile memory.",
            sensitivity=MemorySensitivity.SENSITIVE,
        )
    )
    snapshot = runtime.snapshot()

    assert result.allowed is False
    assert result.blocked is True
    assert snapshot.saved_count == 1
    assert snapshot.saved_blocked_count == 1
    assert snapshot.last_error == result.reason


def test_user_profile_runtime_save_text() -> None:
    runtime = make_runtime()

    result = runtime.save_text(
        "User is building JARVIS for education, debugging, and research.",
        profile_id="profile-2",
        category=UserProfileMemoryCategory.GOAL,
        confidence_label=UserProfileMemoryConfidence.VERIFIED,
        importance=MemoryImportance.CRITICAL,
        retention=MemoryRetention.PERSISTENT,
        source=MemorySource.USER_EXPLICIT,
        tags=("goal", "jarvis"),
    )

    assert result.allowed is True
    assert result.record is not None
    assert "goal" in result.record.tags
    assert "verified" in result.record.tags
    assert result.record.importance == MemoryImportance.CRITICAL


def test_user_profile_runtime_retrieves_facts() -> None:
    runtime = make_runtime()

    runtime.save(make_fact())
    runtime.save_text(
        "Unrelated profile memory.",
        profile_id="profile-2",
        category=UserProfileMemoryCategory.PROJECT,
    )

    result = runtime.retrieve(
        UserProfileMemoryQuery(
            text="engineering guidance",
            categories=(UserProfileMemoryCategory.COMMUNICATION_STYLE,),
            tags=("jarvis",),
        )
    )

    assert result.allowed is True
    assert result.result_count == 1
    assert result.records[0].text == (
        "User prefers direct, elite engineering guidance."
    )
    assert result.results[0].explanation.reason


def test_user_profile_runtime_snapshot_and_reset() -> None:
    runtime = make_runtime()

    runtime.save(make_fact())
    runtime.retrieve(UserProfileMemoryQuery(text="engineering"))

    snapshot = runtime.snapshot()

    assert snapshot.saved_count == 1
    assert snapshot.retrieved_count == 1

    runtime.reset()
    reset_snapshot = runtime.snapshot()

    assert reset_snapshot.saved_count == 0
    assert reset_snapshot.retrieved_count == 0
    assert reset_snapshot.last_profile_id is None


def test_user_profile_enum_values_are_stable() -> None:
    assert UserProfileMemoryCategory.PREFERENCE.value == "preference"
    assert UserProfileMemoryCategory.COMMUNICATION_STYLE.value == (
        "communication_style"
    )
    assert UserProfileMemoryCategory.SYSTEM_PREFERENCE.value == (
        "system_preference"
    )
    assert UserProfileMemoryConfidence.LOW.value == "low"
    assert UserProfileMemoryConfidence.VERIFIED.value == "verified"