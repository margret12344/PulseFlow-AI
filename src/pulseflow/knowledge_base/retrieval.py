
from typing import Any

from pulseflow.knowledge_base.embeddings import generate_embeddings
from pulseflow.knowledge_base.vector_store import search_chunks

DEFAULT_TOP_K = 5


class RetrievalError(Exception):
    """Raised when knowledge retrieval fails."""


def retrieve_relevant_chunks(
    query: str,
    top_k: int = DEFAULT_TOP_K,
) -> list[dict[str, Any]]:
    """Retrieve relevant knowledge chunks for a user query."""

    if not isinstance(query, str) or not query.strip():
        raise ValueError("Query must be a non-empty string.")

    if top_k < 1:
        raise ValueError("top_k must be at least 1.")

    try:
        query_embedding = generate_embeddings(
            texts=[query.strip()],
            input_type="query",
        )[0]

        results = search_chunks(
            query_embedding=query_embedding,
            top_k=top_k,
        )

        return results

    except Exception as exc:
        raise RetrievalError(
            f"Knowledge retrieval failed: {exc}"
        ) from exc


def format_retrieved_context(
    results: list[dict[str, Any]],
) -> str:
    """Format retrieved chunks into context for a future LLM prompt."""

    if not results:
        return "No relevant knowledge documents were found."

    context_parts = []

    for index, result in enumerate(results, start=1):
        metadata = result["metadata"]

        context_parts.append(
            f"[Source {index}]\n"
            f"Document: {metadata.get('title', 'Unknown')}\n"
            f"Department: {metadata.get('department', 'Unknown')}\n"
            f"Similarity: {result['similarity_score']:.4f}\n"
            f"Content:\n{result['content']}"
        )

    return "\n\n---\n\n".join(context_parts)
