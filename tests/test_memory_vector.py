from __future__ import annotations

import pytest
from pydantic import ValidationError

from jarvis.memory import (
    DeterministicEmbeddingProvider,
    DeterministicEmbeddingProviderConfig,
    InMemoryVectorIndex,
    InMemoryVectorIndexConfig,
    MemoryEmbedding,
    MemoryEmbeddingProvider,
    MemoryKind,
    MemoryPolicyClassification,
    MemoryRecord,
    MemorySensitivity,
    MemoryVectorDocument,
    MemoryVectorIndex,
    MemoryVectorSearchQuery,
)


def make_record(
    *,
    text: str = "JARVIS memory vector boundary is typed and replaceable.",
    sensitivity: MemorySensitivity = MemorySensitivity.PRIVATE,
) -> MemoryRecord:
    return MemoryRecord(
        kind=MemoryKind.SEMANTIC,
        text=text,
        sensitivity=sensitivity,
        tags=("jarvis", "vector"),
    )


def make_document(
    *,
    text: str = "JARVIS memory vector boundary is typed and replaceable.",
    sensitivity: MemorySensitivity = MemorySensitivity.PRIVATE,
) -> MemoryVectorDocument:
    provider = DeterministicEmbeddingProvider()
    record = make_record(text=text, sensitivity=sensitivity)
    embedding = provider.embed_text(record.text)

    return MemoryVectorDocument.from_record(
        record=record,
        embedding=embedding,
    )


def test_embedding_provider_config_rejects_invalid_values() -> None:
    with pytest.raises(ValueError):
        DeterministicEmbeddingProviderConfig(name=" ").validate()

    with pytest.raises(ValueError):
        DeterministicEmbeddingProviderConfig(dimensions=0).validate()


def test_vector_index_config_rejects_invalid_values() -> None:
    with pytest.raises(ValueError):
        InMemoryVectorIndexConfig(name=" ").validate()


def test_memory_embedding_rejects_invalid_fields() -> None:
    with pytest.raises(ValidationError):
        MemoryEmbedding(
            text="valid",
            vector=(),
            dimensions=0,
        )

    with pytest.raises(ValidationError):
        MemoryEmbedding(
            text=" ",
            vector=(1.0,),
            dimensions=1,
        )


def test_deterministic_embedding_provider_satisfies_protocol() -> None:
    provider: MemoryEmbeddingProvider = DeterministicEmbeddingProvider()

    embedding = provider.embed_text("hello memory")
    snapshot = provider.snapshot()

    assert embedding.text == "hello memory"
    assert embedding.dimensions == 32
    assert len(embedding.vector) == 32
    assert snapshot.embed_count == 1
    assert snapshot.last_dimensions == 32


def test_deterministic_embedding_provider_is_stable() -> None:
    provider = DeterministicEmbeddingProvider()

    first = provider.embed_text("same text")
    second = provider.embed_text("same text")

    assert first.vector == second.vector


def test_deterministic_embedding_provider_rejects_empty_text() -> None:
    provider = DeterministicEmbeddingProvider()

    with pytest.raises(ValueError):
        provider.embed_text(" ")

    assert provider.snapshot().last_error == "text cannot be empty"


def test_vector_document_from_record() -> None:
    provider = DeterministicEmbeddingProvider()
    record = make_record()
    embedding = provider.embed_text(record.text)

    document = MemoryVectorDocument.from_record(
        record=record,
        embedding=embedding,
    )

    assert document.memory_id == record.memory_id
    assert document.text == record.text
    assert document.embedding == embedding
    assert document.policy_classification == MemoryPolicyClassification.ALLOWED
    assert document.metadata["kind"] == "semantic"
    assert document.metadata["tags"] == ("jarvis", "vector")


def test_vector_document_marks_sensitive_as_restricted() -> None:
    document = make_document(sensitivity=MemorySensitivity.SENSITIVE)

    assert document.policy_classification == MemoryPolicyClassification.RESTRICTED


def test_vector_search_query_requires_text() -> None:
    provider = DeterministicEmbeddingProvider()
    embedding = provider.embed_text("query")

    with pytest.raises(ValidationError):
        MemoryVectorSearchQuery(text=" ", embedding=embedding)


def test_vector_index_satisfies_protocol() -> None:
    index: MemoryVectorIndex = InMemoryVectorIndex()
    document = make_document()

    stored = index.upsert(document)

    assert stored == document
    assert index.snapshot().document_count == 1


def test_vector_index_searches_documents() -> None:
    provider = DeterministicEmbeddingProvider()
    index = InMemoryVectorIndex()

    target = make_document(text="memory gateway vector search")
    other = make_document(text="weather forecast unrelated")

    index.upsert(target)
    index.upsert(other)

    query_embedding = provider.embed_text("memory gateway vector")
    response = index.search(
        MemoryVectorSearchQuery(
            text="memory gateway vector",
            embedding=query_embedding,
            max_results=2,
        )
    )

    assert response.result_count == 2
    assert response.results[0].score >= response.results[1].score
    assert response.results[0].reason == "cosine similarity vector match"


def test_vector_index_respects_min_score() -> None:
    provider = DeterministicEmbeddingProvider()
    index = InMemoryVectorIndex()
    index.upsert(make_document(text="memory gateway vector search"))

    response = index.search(
        MemoryVectorSearchQuery(
            text="unrelated",
            embedding=provider.embed_text("unrelated"),
            min_score=0.99,
        )
    )

    assert response.result_count <= 1


def test_vector_index_filters_restricted_by_default() -> None:
    provider = DeterministicEmbeddingProvider()
    index = InMemoryVectorIndex()

    restricted = make_document(
        text="sensitive vector memory",
        sensitivity=MemorySensitivity.SENSITIVE,
    )
    index.upsert(restricted)

    response = index.search(
        MemoryVectorSearchQuery(
            text="sensitive vector",
            embedding=provider.embed_text("sensitive vector"),
        )
    )

    assert response.result_count == 0


def test_vector_index_can_include_restricted() -> None:
    provider = DeterministicEmbeddingProvider()
    index = InMemoryVectorIndex()

    restricted = make_document(
        text="sensitive vector memory",
        sensitivity=MemorySensitivity.SENSITIVE,
    )
    index.upsert(restricted)

    response = index.search(
        MemoryVectorSearchQuery(
            text="sensitive vector",
            embedding=provider.embed_text("sensitive vector"),
            include_restricted=True,
        )
    )

    assert response.result_count == 1
    assert response.results[0].document.policy_classification == (
        MemoryPolicyClassification.RESTRICTED
    )


def test_vector_index_delete_and_clear() -> None:
    index = InMemoryVectorIndex()
    document = make_document()

    index.upsert(document)

    assert index.delete(document.document_id) is True
    assert index.delete(document.document_id) is False

    index.upsert(document)
    index.clear()

    snapshot = index.snapshot()

    assert snapshot.document_count == 0
    assert snapshot.delete_count == 1
    assert snapshot.clear_count == 1


def test_vector_boundary_enum_values_are_stable() -> None:
    assert MemoryPolicyClassification.ALLOWED.value == "allowed"
    assert MemoryPolicyClassification.RESTRICTED.value == "restricted"