
import numpy as np
import pytest

from pulseflow.knowledge_base import embeddings


class FakeEmbeddingModel:
    """Fake model for testing without downloading the real model."""

    def __init__(self):
        self.received_texts = None
        self.encode_kwargs = None

    def encode(self, texts, **kwargs):
        self.received_texts = texts
        self.encode_kwargs = kwargs

        return np.array(
            [
                [0.6, 0.8, 0.0]
                for _ in texts
            ],
            dtype=float,
        )


@pytest.fixture
def fake_model(monkeypatch):
    model = FakeEmbeddingModel()

    monkeypatch.setattr(
        embeddings,
        "get_embedding_model",
        lambda: model,
    )

    return model


@pytest.fixture
def sample_chunks():
    return [
        {
            "chunk_id": "chunk-001",
            "document_id": "KB-001",
            "chunk_index": 0,
            "content": "Network troubleshooting procedure.",
        },
        {
            "chunk_id": "chunk-002",
            "document_id": "KB-001",
            "chunk_index": 1,
            "content": "Hardware troubleshooting procedure.",
        },
    ]


def test_model_name_and_batch_size():
    assert embeddings.MODEL_NAME == "intfloat/multilingual-e5-small"
    assert embeddings.BATCH_SIZE == 16


def test_generate_document_embeddings(fake_model):
    result = embeddings.generate_embeddings(
        ["Network troubleshooting"],
        input_type="document",
    )

    assert len(result) == 1
    assert result[0] == [0.6, 0.8, 0.0]
    assert fake_model.received_texts == [
        "passage: Network troubleshooting"
    ]


def test_generate_query_embeddings(fake_model):
    result = embeddings.generate_embeddings(
        ["How to fix network issues?"],
        input_type="query",
    )

    assert len(result) == 1
    assert fake_model.received_texts == [
        "query: How to fix network issues?"
    ]


def test_generate_embeddings_strips_whitespace(fake_model):
    embeddings.generate_embeddings(
        ["  Network troubleshooting  "],
    )

    assert fake_model.received_texts == [
        "passage: Network troubleshooting"
    ]


def test_generate_embeddings_multiple_texts(fake_model):
    texts = [
        "First document",
        "Second document",
        "Third document",
    ]

    result = embeddings.generate_embeddings(texts)

    assert len(result) == 3
    assert len(fake_model.received_texts) == 3


def test_generate_embeddings_uses_expected_encode_settings(
    fake_model,
):
    embeddings.generate_embeddings(["Test document"])

    assert fake_model.encode_kwargs == {
        "batch_size": embeddings.BATCH_SIZE,
        "show_progress_bar": False,
        "normalize_embeddings": True,
        "convert_to_numpy": True,
    }


def test_generate_embeddings_rejects_empty_list():
    with pytest.raises(ValueError, match="No texts provided"):
        embeddings.generate_embeddings([])


def test_generate_embeddings_rejects_invalid_input_type():
    with pytest.raises(ValueError, match="input_type"):
        embeddings.generate_embeddings(
            ["Test document"],
            input_type="invalid",
        )


@pytest.mark.parametrize(
    "texts",
    [
        [""],
        ["   "],
        ["valid text", ""],
        [None],
        [123],
    ],
)
def test_generate_embeddings_rejects_invalid_texts(texts):
    with pytest.raises(ValueError, match="non-empty strings"):
        embeddings.generate_embeddings(texts)


def test_embed_chunks_returns_embedded_chunks(
    fake_model,
    sample_chunks,
):
    result = embeddings.embed_chunks(sample_chunks)

    assert len(result) == len(sample_chunks)

    for original, embedded in zip(
        sample_chunks,
        result,
        strict=True,
    ):
        assert embedded["chunk_id"] == original["chunk_id"]
        assert embedded["content"] == original["content"]
        assert embedded["embedding"] == [0.6, 0.8, 0.0]


def test_embed_chunks_preserves_original_data(
    fake_model,
    sample_chunks,
):
    result = embeddings.embed_chunks(sample_chunks)

    for original, embedded in zip(
        sample_chunks,
        result,
        strict=True,
    ):
        for key, value in original.items():
            assert embedded[key] == value


def test_embed_chunks_rejects_empty_list():
    with pytest.raises(ValueError, match="No chunks provided"):
        embeddings.embed_chunks([])


def test_embed_chunks_rejects_missing_content():
    chunks = [{"chunk_id": "chunk-001"}]

    with pytest.raises(KeyError):
        embeddings.embed_chunks(chunks)


def test_embed_chunks_rejects_embedding_count_mismatch(
    monkeypatch,
    sample_chunks,
):
    monkeypatch.setattr(
        embeddings,
        "generate_embeddings",
        lambda texts, input_type: [[0.1, 0.2, 0.3]],
    )

    with pytest.raises(
        RuntimeError,
        match="Embedding count does not match chunk count",
    ):
        embeddings.embed_chunks(sample_chunks)
