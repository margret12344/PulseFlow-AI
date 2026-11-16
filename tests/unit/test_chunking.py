
import pytest

from pulseflow.knowledge_base.chunking import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    chunk_document,
    chunk_documents,
    generate_chunk_id,
)


@pytest.fixture
def sample_document():
    return {
        "document_id": "KB-TEST-001",
        "title": "Test Document",
        "document_type": "procedure",
        "department": "IT",
        "version": "1.0",
        "source_type": "synthetic",
        "language": "en",
        "content": (
            "# Network Troubleshooting\n\n"
            + "This is a network troubleshooting instruction. " * 30
        ),
        "source_path": "data/knowledge_base/test.md",
    }


def test_chunk_document_returns_chunks(sample_document):
    chunks = chunk_document(sample_document)

    assert len(chunks) > 1
    assert all(chunk["content"].strip() for chunk in chunks)


def test_chunks_respect_maximum_size(sample_document):
    chunks = chunk_document(sample_document)

    assert all(len(chunk["content"]) <= CHUNK_SIZE for chunk in chunks)


def test_chunk_configuration():
    assert CHUNK_SIZE == 500
    assert CHUNK_OVERLAP == 100
    assert CHUNK_OVERLAP < CHUNK_SIZE


def test_chunk_metadata_is_preserved(sample_document):
    chunks = chunk_document(sample_document)

    for chunk in chunks:
        assert chunk["document_id"] == sample_document["document_id"]
        assert chunk["title"] == sample_document["title"]
        assert chunk["document_type"] == sample_document["document_type"]
        assert chunk["department"] == sample_document["department"]
        assert chunk["version"] == sample_document["version"]
        assert chunk["source_type"] == sample_document["source_type"]
        assert chunk["language"] == sample_document["language"]
        assert chunk["source_path"] == sample_document["source_path"]


def test_chunk_indexes_are_sequential(sample_document):
    chunks = chunk_document(sample_document)

    assert [chunk["chunk_index"] for chunk in chunks] == list(
        range(len(chunks))
    )


def test_chunk_ids_are_deterministic(sample_document):
    first_run = chunk_document(sample_document)
    second_run = chunk_document(sample_document)

    first_ids = [chunk["chunk_id"] for chunk in first_run]
    second_ids = [chunk["chunk_id"] for chunk in second_run]

    assert first_ids == second_ids


def test_chunk_ids_are_unique(sample_document):
    chunks = chunk_document(sample_document)

    chunk_ids = [chunk["chunk_id"] for chunk in chunks]

    assert len(chunk_ids) == len(set(chunk_ids))


def test_chunk_id_changes_when_version_changes():
    first_id = generate_chunk_id(
        document_id="KB-TEST-001",
        version="1.0",
        chunk_index=0,
        content="Test content",
    )

    second_id = generate_chunk_id(
        document_id="KB-TEST-001",
        version="2.0",
        chunk_index=0,
        content="Test content",
    )

    assert first_id != second_id


def test_chunk_id_changes_when_content_changes():
    first_id = generate_chunk_id(
        document_id="KB-TEST-001",
        version="1.0",
        chunk_index=0,
        content="First content",
    )

    second_id = generate_chunk_id(
        document_id="KB-TEST-001",
        version="1.0",
        chunk_index=0,
        content="Second content",
    )

    assert first_id != second_id


def test_chunk_document_rejects_missing_fields(sample_document):
    del sample_document["title"]

    with pytest.raises(ValueError, match="Missing document fields"):
        chunk_document(sample_document)


def test_chunk_document_rejects_empty_content(sample_document):
    sample_document["content"] = "   \n  "

    with pytest.raises(ValueError, match="empty content"):
        chunk_document(sample_document)


def test_chunk_documents_returns_chunks(sample_document):
    second_document = sample_document.copy()
    second_document["document_id"] = "KB-TEST-002"

    chunks = chunk_documents([sample_document, second_document])

    document_ids = {chunk["document_id"] for chunk in chunks}

    assert len(chunks) > 0
    assert document_ids == {"KB-TEST-001", "KB-TEST-002"}


def test_chunk_documents_rejects_empty_list():
    with pytest.raises(ValueError, match="No documents provided"):
        chunk_documents([])


def test_chunk_documents_rejects_duplicate_document_ids(
    sample_document,
):
    duplicate_document = sample_document.copy()

    with pytest.raises(ValueError, match="Duplicate document ID"):
        chunk_documents([sample_document, duplicate_document])


def test_chunk_ids_are_unique_across_documents(sample_document):
    second_document = sample_document.copy()
    second_document["document_id"] = "KB-TEST-002"

    chunks = chunk_documents([sample_document, second_document])

    first_document_ids = {
        chunk["chunk_id"]
        for chunk in chunks
        if chunk["document_id"] == "KB-TEST-001"
    }

    second_document_ids = {
        chunk["chunk_id"]
        for chunk in chunks
        if chunk["document_id"] == "KB-TEST-002"
    }

    assert first_document_ids
    assert second_document_ids
    assert first_document_ids.isdisjoint(second_document_ids)
