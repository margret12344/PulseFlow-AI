
import pytest

from pulseflow.knowledge_base import retrieval


@pytest.fixture
def sample_results():
    return [
        {
            "chunk_id": "chunk-001",
            "content": "Check the network cable and connection.",
            "metadata": {
                "title": "Network Troubleshooting",
                "department": "IT",
            },
            "distance": 0.1,
            "similarity_score": 0.9,
        },
        {
            "chunk_id": "chunk-002",
            "content": "Restart the network adapter if needed.",
            "metadata": {
                "title": "Hardware Troubleshooting",
                "department": "Technical Support",
            },
            "distance": 0.2,
            "similarity_score": 0.8,
        },
    ]


def test_default_top_k():
    assert retrieval.DEFAULT_TOP_K == 5


def test_retrieve_relevant_chunks(
    monkeypatch,
    sample_results,
):
    captured = {}

    def fake_generate_embeddings(texts, input_type):
        captured["texts"] = texts
        captured["input_type"] = input_type
        return [[0.1, 0.2, 0.3]]

    def fake_search_chunks(query_embedding, top_k):
        captured["query_embedding"] = query_embedding
        captured["top_k"] = top_k
        return sample_results

    monkeypatch.setattr(
        retrieval,
        "generate_embeddings",
        fake_generate_embeddings,
    )
    monkeypatch.setattr(
        retrieval,
        "search_chunks",
        fake_search_chunks,
    )

    results = retrieval.retrieve_relevant_chunks(
        "How to fix network issues?"
    )

    assert results == sample_results
    assert captured["texts"] == ["How to fix network issues?"]
    assert captured["input_type"] == "query"
    assert captured["query_embedding"] == [0.1, 0.2, 0.3]
    assert captured["top_k"] == retrieval.DEFAULT_TOP_K


def test_retrieve_relevant_chunks_uses_custom_top_k(
    monkeypatch,
):
    captured = {}

    monkeypatch.setattr(
        retrieval,
        "generate_embeddings",
        lambda texts, input_type: [[0.1, 0.2, 0.3]],
    )

    def fake_search_chunks(query_embedding, top_k):
        captured["top_k"] = top_k
        return []

    monkeypatch.setattr(
        retrieval,
        "search_chunks",
        fake_search_chunks,
    )

    retrieval.retrieve_relevant_chunks(
        "Network problem",
        top_k=3,
    )

    assert captured["top_k"] == 3


def test_retrieve_relevant_chunks_strips_query(
    monkeypatch,
):
    captured = {}

    def fake_generate_embeddings(texts, input_type):
        captured["texts"] = texts
        return [[0.1, 0.2, 0.3]]

    monkeypatch.setattr(
        retrieval,
        "generate_embeddings",
        fake_generate_embeddings,
    )
    monkeypatch.setattr(
        retrieval,
        "search_chunks",
        lambda query_embedding, top_k: [],
    )

    retrieval.retrieve_relevant_chunks("  Network problem  ")

    assert captured["texts"] == ["Network problem"]


@pytest.mark.parametrize(
    "query",
    [
        "",
        "   ",
        None,
        123,
    ],
)
def test_retrieve_relevant_chunks_rejects_invalid_query(query):
    with pytest.raises(
        ValueError,
        match="Query must be a non-empty string",
    ):
        retrieval.retrieve_relevant_chunks(query)


@pytest.mark.parametrize("top_k", [0, -1])
def test_retrieve_relevant_chunks_rejects_invalid_top_k(top_k):
    with pytest.raises(
        ValueError,
        match="top_k must be at least 1",
    ):
        retrieval.retrieve_relevant_chunks(
            "Network problem",
            top_k=top_k,
        )


def test_retrieve_relevant_chunks_wraps_embedding_error(
    monkeypatch,
):
    def fake_generate_embeddings(texts, input_type):
        raise RuntimeError("Embedding model failed")

    monkeypatch.setattr(
        retrieval,
        "generate_embeddings",
        fake_generate_embeddings,
    )

    with pytest.raises(
        retrieval.RetrievalError,
        match="Knowledge retrieval failed",
    ) as error:
        retrieval.retrieve_relevant_chunks("Network problem")

    assert isinstance(error.value.__cause__, RuntimeError)
    assert "Embedding model failed" in str(error.value)


def test_retrieve_relevant_chunks_wraps_search_error(
    monkeypatch,
):
    monkeypatch.setattr(
        retrieval,
        "generate_embeddings",
        lambda texts, input_type: [[0.1, 0.2, 0.3]],
    )

    def fake_search_chunks(query_embedding, top_k):
        raise RuntimeError("Vector search failed")

    monkeypatch.setattr(
        retrieval,
        "search_chunks",
        fake_search_chunks,
    )

    with pytest.raises(
        retrieval.RetrievalError,
        match="Knowledge retrieval failed",
    ) as error:
        retrieval.retrieve_relevant_chunks("Network problem")

    assert isinstance(error.value.__cause__, RuntimeError)
    assert "Vector search failed" in str(error.value)


def test_retrieve_relevant_chunks_returns_empty_results(
    monkeypatch,
):
    monkeypatch.setattr(
        retrieval,
        "generate_embeddings",
        lambda texts, input_type: [[0.1, 0.2, 0.3]],
    )
    monkeypatch.setattr(
        retrieval,
        "search_chunks",
        lambda query_embedding, top_k: [],
    )

    results = retrieval.retrieve_relevant_chunks("Unknown topic")

    assert results == []


def test_format_retrieved_context(sample_results):
    context = retrieval.format_retrieved_context(sample_results)

    assert "[Source 1]" in context
    assert "[Source 2]" in context
    assert "Network Troubleshooting" in context
    assert "Hardware Troubleshooting" in context
    assert "Department: IT" in context
    assert "Similarity: 0.9000" in context
    assert "Similarity: 0.8000" in context
    assert "Check the network cable and connection." in context
    assert "Restart the network adapter if needed." in context
    assert "\n\n---\n\n" in context


def test_format_retrieved_context_uses_unknown_metadata():
    results = [
        {
            "content": "Example content",
            "metadata": {},
            "similarity_score": 0.75,
        }
    ]

    context = retrieval.format_retrieved_context(results)

    assert "Document: Unknown" in context
    assert "Department: Unknown" in context
    assert "Similarity: 0.7500" in context


def test_format_retrieved_context_handles_empty_results():
    context = retrieval.format_retrieved_context([])

    assert context == (
        "No relevant knowledge documents were found."
    )

