
import uuid
from typing import Any

from langchain_text_splitters import RecursiveCharacterTextSplitter

CHUNK_SIZE = 500
CHUNK_OVERLAP = 100

CHUNK_NAMESPACE = uuid.UUID("8c1d9e65-2a47-4d83-9f12-6b5e4a0c7d31")

SEPARATORS = [
    "\n## ",
    "\n# ",
    "\n\n",
    "\n",
    ". ",
    " ",
    "",
]


def create_text_splitter() -> RecursiveCharacterTextSplitter:
    """Create the configured Markdown-aware text splitter."""

    return RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=SEPARATORS,
        length_function=len,
        strip_whitespace=True,
    )


def generate_chunk_id(
    document_id: str,
    version: str,
    chunk_index: int,
    content: str,
) -> str:
    """Generate a deterministic UUID for a document chunk."""

    unique_value = (
        f"pulseflow:{document_id}:{version}:"
        f"{chunk_index}:{content}"
    )

    return str(
        uuid.uuid5(CHUNK_NAMESPACE, unique_value)
    )


def chunk_document(
    document: dict[str, Any],
    splitter: RecursiveCharacterTextSplitter | None = None,
) -> list[dict[str, Any]]:
    """Split one document into chunks while preserving metadata."""

    required_fields = {
        "document_id",
        "title",
        "version",
        "content",
    }

    missing_fields = required_fields - document.keys()

    if missing_fields:
        raise ValueError(
            f"Missing document fields: {sorted(missing_fields)}"
        )

    if not document["content"].strip():
        raise ValueError(
            f"Document {document['document_id']} has empty content."
        )

    if splitter is None:
        splitter = create_text_splitter()

    text_chunks = splitter.split_text(document["content"])

    chunks = []

    for index, content in enumerate(text_chunks):
        if not content.strip():
            continue

        chunk = {
            "chunk_id": generate_chunk_id(
                document_id=document["document_id"],
                version=document["version"],
                chunk_index=index,
                content=content,
            ),
            "document_id": document["document_id"],
            "chunk_index": index,
            "title": document["title"],
            "document_type": document.get("document_type"),
            "department": document.get("department"),
            "version": document["version"],
            "source_type": document.get("source_type"),
            "language": document.get("language"),
            "content": content,
            "source_path": document.get("source_path"),
        }

        chunks.append(chunk)

    return chunks


def chunk_documents(
    documents: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Split multiple documents into chunks."""

    if not documents:
        raise ValueError("No documents provided for chunking.")

    splitter = create_text_splitter()

    all_chunks = []
    seen_document_ids = set()

    for document in documents:
        document_id = document.get("document_id")

        if document_id in seen_document_ids:
            raise ValueError(
                f"Duplicate document ID: {document_id}"
            )

        seen_document_ids.add(document_id)

        document_chunks = chunk_document(
            document=document,
            splitter=splitter,
        )

        all_chunks.extend(document_chunks)

    chunk_ids = [
        chunk["chunk_id"]
        for chunk in all_chunks
    ]

    if len(chunk_ids) != len(set(chunk_ids)):
        raise ValueError("Duplicate chunk IDs detected.")

    return all_chunks
