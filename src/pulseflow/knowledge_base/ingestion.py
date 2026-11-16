
from typing import Any

from pulseflow.knowledge_base.chunking import chunk_documents
from pulseflow.knowledge_base.document_loader import load_documents
from pulseflow.knowledge_base.embeddings import embed_chunks
from pulseflow.knowledge_base.persistence import persist_knowledge_base
from pulseflow.knowledge_base.vector_store import store_chunks


def run_ingestion_pipeline() -> dict[str, Any]:
    """Load, process, embed, and persist knowledge base documents."""

    # 1. Load Markdown documents.
    documents = load_documents()

    # 2. Split documents into chunks.
    chunks = chunk_documents(documents)

    # 3. Generate embeddings for all chunks.
    embedded_chunks = embed_chunks(chunks)

    # 4. Store embeddings and metadata in ChromaDB.
    stored_chunks = store_chunks(embedded_chunks)

    # 5. Persist canonical documents and chunks in PostgreSQL.
    persistence_result = persist_knowledge_base(
        documents=documents,
        chunks=embedded_chunks,
    )

    return {
        "documents_loaded": len(documents),
        "chunks_created": len(chunks),
        "chunks_stored_in_chroma": stored_chunks,
        "postgresql": persistence_result,
    }
