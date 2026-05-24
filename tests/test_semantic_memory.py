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
    SemanticMemoryDomain,
    SemanticMemoryFact,
    SemanticMemoryFactType,
    SemanticMemoryQuery,
    SemanticMemoryRuntime,
    SemanticMemoryRuntimeConfig,
)


def make_runtime() -> SemanticMemoryRuntime:
    return SemanticMemoryRuntime(
        gateway=GovernedMemoryGateway(store=InMemoryMemoryStore()),
    )


def make_fact(
    *,
    fact_id: str = "fact-1",
    text: str = "Memory lifecycle policy manages expiration and retention.",
    domain: SemanticMemoryDomain = SemanticMemoryDomain.ENGINEERING,
    fact_type: SemanticMemoryFactType = SemanticMemoryFactType.CONCEPT,
    sensitivity: MemorySensitivity = MemorySensitivity.PRIVATE,
) -> SemanticMemoryFact:
    return SemanticMemoryFact(
        fact_id=fact_id,
        text=text,
        domain=domain,
        fact_type=fact_type,
        sensitivity=sensitivity,
        importance=MemoryImportance.HIGH,
        tags=("memory", "lifecycle"),
    )


def test_semantic_memory_runtime_config_rejects_invalid_name() -> None:
    with pytest.raises(ValueError):
        SemanticMemoryRuntimeConfig(name=" ").validate()


def test_semantic_fact_rejects_empty_fields() -> None:
    with pytest.raises(ValidationError):
        SemanticMemoryFact(
            fact_id=" ",
            text="valid",
        )

    with pytest.raises(ValidationError):
        SemanticMemoryFact(
            fact_id="fact-1",
            text=" ",
        )


def test_semantic_fact_cleans_tags() -> None:
    fact = SemanticMemoryFact(
        fact_id="fact-1",
        text="Semantic memory stores stable knowledge.",
        tags=(" Memory ", "memory", " Semantic "),
    )

    assert fact.tags == ("memory", "semantic")


def test_semantic_fact_to_write_request() -> None:
    fact = make_fact()
    request = fact.to_write_request()

    assert request.kind == MemoryKind.SEMANTIC
    assert request.scope == MemoryScope.USER
    assert request.text == fact.text
    assert request.importance == MemoryImportance.HIGH
    assert request.sensitivity == fact.sensitivity
    assert "semantic" in request.tags
    assert "engineering" in request.tags
    assert "concept" in request.tags
    assert request.metadata["fact_id"] == "fact-1"
    assert request.metadata["semantic_domain"] == "engineering"
    assert request.metadata["fact_type"] == "concept"


def test_semantic_fact_project_domain_maps_to_project_scope() -> None:
    fact = make_fact(domain=SemanticMemoryDomain.PROJECT)
    request = fact.to_write_request()

    assert request.scope == MemoryScope.PROJECT


def test_semantic_fact_system_domain_maps_to_system_scope() -> None:
    fact = make_fact(domain=SemanticMemoryDomain.SYSTEM)
    request = fact.to_write_request()

    assert request.scope == MemoryScope.SYSTEM


def test_semantic_query_to_memory_query() -> None:
    query = SemanticMemoryQuery(
        text="memory lifecycle",
        domains=(SemanticMemoryDomain.ENGINEERING,),
        fact_types=(SemanticMemoryFactType.CONCEPT,),
        tags=("memory",),
        max_results=5,
    )
    memory_query = query.to_memory_query()

    assert memory_query.text == "memory lifecycle"
    assert memory_query.kinds == (MemoryKind.SEMANTIC,)
    assert memory_query.max_results == 5
    assert "semantic" in memory_query.tags
    assert "engineering" in memory_query.tags
    assert "concept" in memory_query.tags
    assert "memory" in memory_query.tags


def test_semantic_runtime_learns_fact() -> None:
    runtime = make_runtime()

    result = runtime.learn(make_fact())
    snapshot = runtime.snapshot()

    assert result.allowed is True
    assert result.record is not None
    assert result.record.kind == MemoryKind.SEMANTIC
    assert result.record.importance == MemoryImportance.HIGH
    assert snapshot.learned_count == 1
    assert snapshot.learned_allowed_count == 1
    assert snapshot.last_fact_id == "fact-1"


def test_semantic_runtime_blocks_sensitive_fact_by_gateway_policy() -> None:
    runtime = make_runtime()

    result = runtime.learn(
        make_fact(
            text="Sensitive semantic fact.",
            sensitivity=MemorySensitivity.SENSITIVE,
        )
    )
    snapshot = runtime.snapshot()

    assert result.allowed is False
    assert result.blocked is True
    assert snapshot.learned_count == 1
    assert snapshot.learned_blocked_count == 1
    assert snapshot.last_error == result.reason


def test_semantic_runtime_learn_text() -> None:
    runtime = make_runtime()

    result = runtime.learn_text(
        "A memory gateway is the only cognition-facing memory entry point.",
        fact_id="fact-2",
        domain=SemanticMemoryDomain.ENGINEERING,
        fact_type=SemanticMemoryFactType.RULE,
        importance=MemoryImportance.CRITICAL,
        retention=MemoryRetention.PERSISTENT,
        source=MemorySource.COGNITION,
        tags=("gateway",),
    )

    assert result.allowed is True
    assert result.record is not None
    assert "rule" in result.record.tags
    assert "engineering" in result.record.tags
    assert "gateway" in result.record.tags
    assert result.record.importance == MemoryImportance.CRITICAL


def test_semantic_runtime_retrieves_facts() -> None:
    runtime = make_runtime()

    runtime.learn(make_fact())
    runtime.learn_text(
        "Unrelated semantic memory.",
        fact_id="fact-2",
        domain=SemanticMemoryDomain.GENERAL,
        fact_type=SemanticMemoryFactType.FACT,
    )

    result = runtime.retrieve(
        SemanticMemoryQuery(
            text="lifecycle policy",
            domains=(SemanticMemoryDomain.ENGINEERING,),
            fact_types=(SemanticMemoryFactType.CONCEPT,),
            tags=("memory",),
        )
    )

    assert result.allowed is True
    assert result.result_count == 1
    assert result.records[0].text == (
        "Memory lifecycle policy manages expiration and retention."
    )
    assert result.results[0].explanation.reason


def test_semantic_runtime_snapshot_and_reset() -> None:
    runtime = make_runtime()

    runtime.learn(make_fact())
    runtime.retrieve(SemanticMemoryQuery(text="lifecycle"))

    snapshot = runtime.snapshot()

    assert snapshot.learned_count == 1
    assert snapshot.retrieved_count == 1

    runtime.reset()
    reset_snapshot = runtime.snapshot()

    assert reset_snapshot.learned_count == 0
    assert reset_snapshot.retrieved_count == 0
    assert reset_snapshot.last_fact_id is None


def test_semantic_enum_values_are_stable() -> None:
    assert SemanticMemoryDomain.ENGINEERING.value == "engineering"
    assert SemanticMemoryDomain.RESEARCH.value == "research"
    assert SemanticMemoryFactType.CONCEPT.value == "concept"
    assert SemanticMemoryFactType.RULE.value == "rule"
    assert SemanticMemoryFactType.CONSTRAINT.value == "constraint"