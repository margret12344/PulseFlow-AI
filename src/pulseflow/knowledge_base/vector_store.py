
from pathlib import Path
from typing import Any

import chromadb
from chromadb.api.models.Collection import Collection

DEFAULT_VECTOR_STORE_DIR = (
    Path(__file__).resolve().parents[3]
    / "data"
    / "knowledge_base"
    / "processed"
    / "chroma_db"
)

COLLECTION_NAME = "pulseflow_knowledge_base"


def get_vector_collection(
    persist_directory: Path | str = DEFAULT_VECTOR_STORE_DIR,
) -> Collection:
    """Create or retrieve the persistent ChromaDB collection."""

    directory = Path(persist_directory)
    directory.mkdir(parents=True, exist_ok=True)

    client = chromadb.PersistentClient(
        path=str(directory)
    )

    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={
            "description": "PulseFlow synthetic hospital IT knowledge base",
            "hnsw:space": "cosine",
        },
    )


def store_chunks(
    chunks: list[dict[str, Any]],
    collection: Collection | None = None,
) -> int:
    """Store embedded chunks in ChromaDB using deterministic IDs."""

    if not chunks:
        raise ValueError("No chunks provided for storage.")

    if collection is None:
        collection = get_vector_collection()

    ids = []
    documents = []
    embeddings = []
    metadatas = []

    for chunk in chunks:
        required_fields = {
            "chunk_id",
            "document_id",
            "chunk_index",
            "content",
            "embedding",
        }

        missing_fields = required_fields - chunk.keys()

        if missing_fields:
            raise ValueError(
                f"Missing chunk fields: {sorted(missing_fields)}"
            )

        if not chunk["content"].strip():
            raise ValueError("Chunk content cannot be empty.")

        embedding = chunk["embedding"]

        if not embedding:
            raise ValueError("Chunk embedding cannot be empty.")

        ids.append(chunk["chunk_id"])
        documents.append(chunk["content"])
        embeddings.append(embedding)

        metadata = {
            key: chunk[key]
            for key in (
                "document_id",
                "chunk_index",
                "title",
                "document_type",
                "department",
                "version",
                "source_type",
                "language",
                "source_path",
            )
            if chunk.get(key) is not None
        }

        metadatas.append(metadata)

    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate chunk IDs detected.")

    collection.upsert(
        ids=ids,
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas,
    )

    return len(ids)


def search_chunks(
    query_embedding: list[float],
    top_k: int = 5,
    collection: Collection | None = None,
) -> list[dict[str, Any]]:
    """Retrieve the most similar chunks from ChromaDB."""

    if not query_embedding:
        raise ValueError("Query embedding cannot be empty.")

    if top_k < 1:
        raise ValueError("top_k must be at least 1.")

    if collection is None:
        collection = get_vector_collection()

    total_chunks = collection.count()

    if total_chunks == 0:
        return []

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(top_k, total_chunks),
        include=[
            "documents",
            "metadatas",
            "distances",
        ],
    )

    retrieved_chunks = []

    for index in range(len(results["ids"][0])):
        distance = results["distances"][0][index]

        retrieved_chunks.append(
            {
                "chunk_id": results["ids"][0][index],
                "content": results["documents"][0][index],
                "metadata": results["metadatas"][0][index],
                "distance": distance,
                "similarity_score": max(
                    0.0,
                    min(1.0, 1.0 - distance),
                ),
            }
        )

    return retrieved_chunks
