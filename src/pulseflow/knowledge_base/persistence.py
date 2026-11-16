
import math
import uuid
from collections import defaultdict
from typing import Any

from sqlalchemy import select
from sqlalchemy.engine import Engine
from sqlalchemy.orm import sessionmaker

from pulseflow.core.database import get_engine
from pulseflow.database.models.ai_processing import TicketAnalysis
from pulseflow.database.models.knowledge_base import (
    DocumentChunk,
    KnowledgeDocument,
    RetrievedChunk,
)


def persist_knowledge_base(
    documents: list[dict[str, Any]],
    chunks: list[dict[str, Any]],
    engine: Engine | None = None,
) -> dict[str, int]:
    """Persist knowledge documents and chunks in PostgreSQL."""

    if not documents:
        raise ValueError("No documents provided for persistence.")

    if not chunks:
        raise ValueError("No chunks provided for persistence.")

    chunks_by_document: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for chunk in chunks:
        document_id = chunk.get("document_id")

        if not document_id:
            raise ValueError("A chunk is missing its document ID.")

        chunks_by_document[document_id].append(chunk)

    document_ids = [doc.get("document_id") for doc in documents]

    if None in document_ids or len(document_ids) != len(set(document_ids)):
        raise ValueError("Missing or duplicate document IDs detected.")

    unknown_documents = set(chunks_by_document) - set(document_ids)

    if unknown_documents:
        raise ValueError(
            f"Chunks reference unknown documents: {sorted(unknown_documents)}"
        )

    created_engine = engine is None

    if engine is None:
        engine = get_engine()

    SessionFactory = sessionmaker(bind=engine, expire_on_commit=False)

    inserted_documents = 0
    inserted_chunks = 0
    skipped_documents = 0

    try:
        with SessionFactory() as session:
            with session.begin():
                for document in documents:
                    external_id = document["document_id"]
                    title = document["title"]
                    version = document["version"]
                    source_path = document.get("source_path")

                    if not source_path:
                        raise ValueError(
                            f"Document {external_id} has no source path."
                        )

                    if len(source_path) > 500:
                        raise ValueError(
                            f"Source path is too long for document {external_id}."
                        )

                    document_chunks = chunks_by_document.get(external_id, [])

                    if not document_chunks:
                        raise ValueError(
                            f"Document {external_id} has no chunks."
                        )

                    existing_document = session.scalar(
                        select(KnowledgeDocument).where(
                            KnowledgeDocument.source == source_path,
                            KnowledgeDocument.version == version,
                        )
                    )

                    if existing_document is not None:
                        skipped_documents += 1
                        continue

                    db_document = KnowledgeDocument(
                        title=title,
                        source=source_path,
                        version=version,
                        processing_status="processed",
                    )

                    session.add(db_document)
                    session.flush()

                    for chunk in document_chunks:
                        chunk_id = uuid.UUID(chunk["chunk_id"])

                        if not chunk.get("content", "").strip():
                            raise ValueError(
                                f"Empty chunk detected in document {external_id}."
                            )

                        db_chunk = DocumentChunk(
                            id=chunk_id,
                            document_id=db_document.id,
                            chunk_index=chunk["chunk_index"],
                            content=chunk["content"],
                        )

                        session.add(db_chunk)
                        inserted_chunks += 1

                    inserted_documents += 1

        return {
            "inserted_documents": inserted_documents,
            "inserted_chunks": inserted_chunks,
            "skipped_documents": skipped_documents,
        }

    finally:
        if created_engine:
            engine.dispose()


def persist_retrieved_chunks(
    analysis_id: uuid.UUID | str,
    results: list[dict[str, Any]],
    engine: Engine | None = None,
) -> dict[str, int]:
    """Persist retrieved chunks linked to a ticket analysis."""

    if not results:
        raise ValueError("No retrieved chunks provided.")

    try:
        analysis_uuid = uuid.UUID(str(analysis_id))
    except (ValueError, TypeError, AttributeError) as exc:
        raise ValueError("Invalid analysis ID.") from exc

    prepared_results = []
    seen_chunk_ids = set()

    for result in results:
        try:
            chunk_id = uuid.UUID(result["chunk_id"])
            score = float(result["similarity_score"])
        except (KeyError, ValueError, TypeError, AttributeError) as exc:
            raise ValueError("Invalid retrieved chunk data.") from exc

        if chunk_id in seen_chunk_ids:
            raise ValueError(f"Duplicate retrieved chunk ID: {chunk_id}")

        if not math.isfinite(score) or not 0.0 <= score <= 1.0:
            raise ValueError("Similarity score must be between 0 and 1.")

        seen_chunk_ids.add(chunk_id)
        prepared_results.append((chunk_id, score))

    created_engine = engine is None

    if engine is None:
        engine = get_engine()

    SessionFactory = sessionmaker(bind=engine, expire_on_commit=False)

    inserted_count = 0
    updated_count = 0

    try:
        with SessionFactory() as session:
            with session.begin():
                analysis = session.get(TicketAnalysis, analysis_uuid)

                if analysis is None:
                    raise ValueError(
                        f"Ticket analysis {analysis_uuid} does not exist."
                    )

                chunk_ids = [chunk_id for chunk_id, _ in prepared_results]

                existing_chunk_ids = set(
                    session.scalars(
                        select(DocumentChunk.id).where(
                            DocumentChunk.id.in_(chunk_ids)
                        )
                    ).all()
                )

                missing_chunk_ids = set(chunk_ids) - existing_chunk_ids

                if missing_chunk_ids:
                    raise ValueError(
                        "Unknown document chunk IDs: "
                        f"{sorted(str(item) for item in missing_chunk_ids)}"
                    )

                existing_records = {
                    record.chunk_id: record
                    for record in session.scalars(
                        select(RetrievedChunk).where(
                            RetrievedChunk.analysis_id == analysis_uuid,
                            RetrievedChunk.chunk_id.in_(chunk_ids),
                        )
                    ).all()
                }

                for chunk_id, score in prepared_results:
                    existing_record = existing_records.get(chunk_id)

                    if existing_record is not None:
                        existing_record.similarity_score = score
                        updated_count += 1
                    else:
                        session.add(
                            RetrievedChunk(
                                analysis_id=analysis_uuid,
                                chunk_id=chunk_id,
                                similarity_score=score,
                            )
                        )
                        inserted_count += 1

        return {
            "inserted": inserted_count,
            "updated": updated_count,
        }

    finally:
        if created_engine:
            engine.dispose()
