
from functools import lru_cache

from sentence_transformers import SentenceTransformer

MODEL_NAME = "intfloat/multilingual-e5-small"
BATCH_SIZE = 16


@lru_cache(maxsize=1)
def get_embedding_model() -> SentenceTransformer:
    """Load and cache the multilingual embedding model."""

    return SentenceTransformer(MODEL_NAME)


def generate_embeddings(
    texts: list[str],
    input_type: str = "document",
) -> list[list[float]]:
    """Generate normalized embeddings for documents or queries."""

    if not texts:
        raise ValueError("No texts provided for embedding.")

    if input_type not in {"document", "query"}:
        raise ValueError(
            "input_type must be either 'document' or 'query'."
        )

    if any(not isinstance(text, str) or not text.strip() for text in texts):
        raise ValueError("All input texts must be non-empty strings.")

    prefix = "passage: " if input_type == "document" else "query: "

    prepared_texts = [
        f"{prefix}{text.strip()}"
        for text in texts
    ]

    model = get_embedding_model()

    embeddings = model.encode(
        prepared_texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=False,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )

    return embeddings.tolist()


def embed_chunks(
    chunks: list[dict],
) -> list[dict]:
    """Generate embeddings and attach them to their corresponding chunks."""

    if not chunks:
        raise ValueError("No chunks provided for embedding.")

    texts = [
        chunk["content"]
        for chunk in chunks
    ]

    embeddings = generate_embeddings(
        texts=texts,
        input_type="document",
    )

    if len(embeddings) != len(chunks):
        raise RuntimeError(
            "Embedding count does not match chunk count."
        )

    embedded_chunks = []

    for chunk, embedding in zip(chunks, embeddings, strict=True):
        embedded_chunks.append(
            {
                **chunk,
                "embedding": embedding,
            }
        )

    return embedded_chunks
