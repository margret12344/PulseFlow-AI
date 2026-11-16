
import pytest

from pulseflow.knowledge_base.vector_store import (
    COLLECTION_NAME,
    DEFAULT_VECTOR_STORE_DIR,
    get_vector_collection,
    search_chunks,
    store_chunks,
)


@pytest.fixture
def collection(tmp_path):
    """Create an isolated ChromaDB collection for each test."""

    return get_vector_collection(
        persist_directory=tmp_path / "chroma_test"
    )


@pytest.fixture
def sample_chunks():
    return [
        {
            "chunk_id": "chunk-001",
            "document_id": "KB-001",
            "chunk_index": 0,
            "title": "Network Troubleshooting",
            "document_type": "procedure",
            "department": "IT",
            "version": "1.0",
            "source_type": "synthetic",
            "language": "en",
            "content": "How to troubleshoot network connectivity.",
            "source_path": "network_troubleshooting.md",
            "embedding": [1.0, 0.0, 0.0],
        },
        {
            "chunk_id": "chunk-002",
            "document_id": "KB-002",
            "chunk_index": 0,
            "title": "Hardware Troubleshooting",
            "document_type": "guide",
            "department": "IT",
            "version": "1.0",
            "source_type": "synthetic",
            "language": "en",
            "content": "How to troubleshoot hardware problems.",
            "source_path": "hardware_troubleshooting.md",
            "embedding": [0.0, 1.0, 0.0],
        },
    ]


def test_default_vector_store_directory():
    assert DEFAULT_VECTOR_STORE_DIR.name == "chroma_db"
    assert DEFAULT_VECTOR_STORE_DIR.parent.name == "processed"


def test_get_vector_collection_creates_directory(tmp_path):
    directory = tmp_path / "new_chroma_directory"

    collection = get_vector_collection(directory)

    assert directory.exists()
    assert collection.name == COLLECTION_NAME


def test_get_vector_collection_uses_cosine_similarity(
    collection,
):
    assert collection.metadata["hnsw:space"] == "cosine"


def test_get_vector_collection_reuses_existing_collection(
    tmp_path,
):
    directory = tmp_path / "persistent_chroma"

    first_collection = get_vector_collection(directory)
    first_collection.upsert(
        ids=["existing-chunk"],
        documents=["Existing document"],
        embeddings=[[1.0, 0.0, 0.0]],
    )

    second_collection = get_vector_collection(directory)

    assert second_collection.name == first_collection.name
    assert second_collection.count() == 1


def test_store_chunks_saves_all_chunks(
    collection,
    sample_chunks,
):
    stored_count = store_chunks(
        sample_chunks,
        collection=collection,
    )

    assert stored_count == 2
    assert collection.count() == 2


def test_store_chunks_saves_correct_content(
    collection,
    sample_chunks,
):
    store_chunks(sample_chunks, collection=collection)

    result = collection.get(
        ids=["chunk-001"],
        include=["documents", "metadatas"],
    )

    assert result["documents"][0] == (
        "How to troubleshoot network connectivity."
    )


def test_store_chunks_saves_metadata(
    collection,
    sample_chunks,
):
    store_chunks(sample_chunks, collection=collection)

    result = collection.get(
        ids=["chunk-001"],
        include=["metadatas"],
    )

    metadata = result["metadatas"][0]

    assert metadata["document_id"] == "KB-001"
    assert metadata["title"] == "Network Troubleshooting"
    assert metadata["department"] == "IT"
    assert metadata["language"] == "en"


def test_store_chunks_rejects_empty_list(collection):
    with pytest.raises(ValueError, match="No chunks provided"):
        store_chunks([], collection=collection)


def test_store_chunks_rejects_missing_fields(collection):
    chunks = [
        {
            "chunk_id": "chunk-001",
            "content": "Test content",
        }
    ]

    with pytest.raises(ValueError, match="Missing chunk fields"):
        store_chunks(chunks, collection=collection)


def test_store_chunks_rejects_empty_content(collection):
    chunks = [
        {
            "chunk_id": "chunk-001",
            "document_id": "KB-001",
            "chunk_index": 0,
            "content": "   ",
            "embedding": [1.0, 0.0, 0.0],
        }
    ]

    with pytest.raises(ValueError, match="content cannot be empty"):
        store_chunks(chunks, collection=collection)


def test_store_chunks_rejects_empty_embedding(collection):
    chunks = [
        {
            "chunk_id": "chunk-001",
            "document_id": "KB-001",
            "chunk_index": 0,
            "content": "Test content",
            "embedding": [],
        }
    ]

    with pytest.raises(ValueError, match="embedding cannot be empty"):
        store_chunks(chunks, collection=collection)


def test_store_chunks_rejects_duplicate_ids(collection):
    chunks = [
        {
            "chunk_id": "chunk-001",
            "document_id": "KB-001",
            "chunk_index": 0,
            "content": "First content",
            "embedding": [1.0, 0.0, 0.0],
        },
        {
            "chunk_id": "chunk-001",
            "document_id": "KB-001",
            "chunk_index": 1,
            "content": "Second content",
            "embedding": [0.0, 1.0, 0.0],
        },
    ]

    with pytest.raises(ValueError, match="Duplicate chunk IDs"):
        store_chunks(chunks, collection=collection)


def test_store_chunks_updates_existing_chunk(
    collection,
    sample_chunks,
):
    store_chunks(
        sample_chunks[:1],
        collection=collection,
    )

    updated_chunk = {
        **sample_chunks[0],
        "content": "Updated network troubleshooting content.",
    }

    store_chunks(
        [updated_chunk],
        collection=collection,
    )

    result = collection.get(
        ids=["chunk-001"],
        include=["documents"],
    )

    assert collection.count() == 1
    assert result["documents"][0] == (
        "Updated network troubleshooting content."
    )


def test_search_chunks_returns_similar_results(
    collection,
    sample_chunks,
):
    store_chunks(sample_chunks, collection=collection)

    results = search_chunks(
        query_embedding=[1.0, 0.0, 0.0],
        top_k=2,
        collection=collection,
    )

    assert len(results) == 2
    assert results[0]["chunk_id"] == "chunk-001"
    assert results[0]["content"] == (
        "How to troubleshoot network connectivity."
    )
    assert results[0]["similarity_score"] == pytest.approx(1.0)


def test_search_chunks_returns_expected_result_structure(
    collection,
    sample_chunks,
):
    store_chunks(sample_chunks, collection=collection)

    results = search_chunks(
        query_embedding=[1.0, 0.0, 0.0],
        collection=collection,
    )

    result = results[0]

    assert {
        "chunk_id",
        "content",
        "metadata",
        "distance",
        "similarity_score",
    }.issubset(result.keys())


def test_search_chunks_limits_top_k(
    collection,
    sample_chunks,
):
    store_chunks(sample_chunks, collection=collection)

    results = search_chunks(
        query_embedding=[1.0, 0.0, 0.0],
        top_k=1,
        collection=collection,
    )

    assert len(results) == 1


def test_search_chunks_caps_top_k_to_collection_size(
    collection,
    sample_chunks,
):
    store_chunks(sample_chunks, collection=collection)

    results = search_chunks(
        query_embedding=[1.0, 0.0, 0.0],
        top_k=10,
        collection=collection,
    )

    assert len(results) == 2


def test_search_chunks_returns_empty_list_for_empty_collection(
    collection,
):
    results = search_chunks(
        query_embedding=[1.0, 0.0, 0.0],
        collection=collection,
    )

    assert results == []


def test_search_chunks_rejects_empty_query_embedding():
    with pytest.raises(
        ValueError,
        match="Query embedding cannot be empty",
    ):
        search_chunks(query_embedding=[])


@pytest.mark.parametrize("top_k", [0, -1])
def test_search_chunks_rejects_invalid_top_k(top_k):
    with pytest.raises(ValueError, match="top_k must be at least 1"):
        search_chunks(
            query_embedding=[1.0, 0.0, 0.0],
            top_k=top_k,
        )
