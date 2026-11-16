
from pathlib import Path
from typing import Any

import yaml

DEFAULT_DOCUMENTS_DIR = (
    Path(__file__).resolve().parents[3]
    / "data"
    / "knowledge_base"
    / "raw"
    / "documents"
)

REQUIRED_METADATA_FIELDS = {
    "document_id",
    "title",
    "document_type",
    "department",
    "version",
    "source_type",
    "language",
    "status",
}

ALLOWED_SOURCE_TYPES = {"synthetic"}


class DocumentLoaderError(Exception):
    """Raised when a knowledge document cannot be loaded."""


def extract_front_matter(
    content: str,
    file_path: Path,
) -> tuple[dict[str, Any], str]:
    """Extract YAML metadata and Markdown body."""

    lines = content.splitlines()

    if not lines or lines[0].strip() != "---":
        raise DocumentLoaderError(
            f"{file_path.name}: Missing YAML front matter."
        )

    try:
        closing_index = lines.index("---", 1)
    except ValueError as exc:
        raise DocumentLoaderError(
            f"{file_path.name}: Closing YAML delimiter not found."
        ) from exc

    yaml_content = "\n".join(lines[1:closing_index])
    markdown_content = "\n".join(lines[closing_index + 1 :]).strip()

    try:
        metadata = yaml.safe_load(yaml_content)
    except yaml.YAMLError as exc:
        raise DocumentLoaderError(
            f"{file_path.name}: Invalid YAML metadata: {exc}"
        ) from exc

    if not isinstance(metadata, dict):
        raise DocumentLoaderError(
            f"{file_path.name}: YAML metadata must be a mapping."
        )

    return metadata, markdown_content


def validate_metadata(
    metadata: dict[str, Any],
    file_path: Path,
) -> None:
    """Validate required metadata fields."""

    missing_fields = REQUIRED_METADATA_FIELDS - metadata.keys()

    if missing_fields:
        missing = ", ".join(sorted(missing_fields))
        raise DocumentLoaderError(
            f"{file_path.name}: Missing metadata fields: {missing}"
        )

    for field in REQUIRED_METADATA_FIELDS:
        value = metadata[field]

        if not isinstance(value, str) or not value.strip():
            raise DocumentLoaderError(
                f"{file_path.name}: Invalid value for '{field}'."
            )

    if metadata["source_type"] not in ALLOWED_SOURCE_TYPES:
        raise DocumentLoaderError(
            f"{file_path.name}: Unsupported source_type."
        )

    if metadata["status"] != "active":
        raise DocumentLoaderError(
            f"{file_path.name}: Document status must be active."
        )


def load_document(file_path: Path) -> dict[str, Any]:
    """Load and validate a single Markdown document."""

    try:
        content = file_path.read_text(encoding="utf-8-sig")
    except OSError as exc:
        raise DocumentLoaderError(
            f"Cannot read {file_path.name}: {exc}"
        ) from exc

    if not content.strip():
        raise DocumentLoaderError(
            f"{file_path.name}: Document is empty."
        )

    metadata, markdown_content = extract_front_matter(
        content,
        file_path,
    )

    validate_metadata(metadata, file_path)

    if not markdown_content:
        raise DocumentLoaderError(
            f"{file_path.name}: Markdown body is empty."
        )

    return {
        "document_id": metadata["document_id"],
        "title": metadata["title"],
        "document_type": metadata["document_type"],
        "department": metadata["department"],
        "version": metadata["version"],
        "source_type": metadata["source_type"],
        "language": metadata["language"],
        "status": metadata["status"],
        "content": markdown_content,
        "source_path": str(file_path.resolve()),
    }


def load_documents(
    documents_dir: Path | str = DEFAULT_DOCUMENTS_DIR,
) -> list[dict[str, Any]]:
    """Load all Markdown documents from a directory."""

    directory = Path(documents_dir)

    if not directory.exists():
        raise DocumentLoaderError(
            f"Documents directory does not exist: {directory}"
        )

    if not directory.is_dir():
        raise DocumentLoaderError(
            f"Expected a directory, received: {directory}"
        )

    files = sorted(directory.glob("*.md"))

    if not files:
        raise DocumentLoaderError(
            f"No Markdown documents found in: {directory}"
        )

    documents = []
    errors = []

    for file_path in files:
        try:
            document = load_document(file_path)
            documents.append(document)

        except DocumentLoaderError as exc:
            errors.append(str(exc))

    document_ids = [
        document["document_id"]
        for document in documents
    ]

    if len(document_ids) != len(set(document_ids)):
        raise DocumentLoaderError(
            "Duplicate document_id values detected."
        )

    if errors:
        error_details = "\n".join(errors)
        raise DocumentLoaderError(
            f"Document loading failed:\n{error_details}"
        )

    return documents
