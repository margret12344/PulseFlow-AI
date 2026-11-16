
from pathlib import Path

import pytest

from pulseflow.knowledge_base.document_loader import (
    DocumentLoaderError,
    load_document,
    load_documents,
)

VALID_DOCUMENT = """---
document_id: KB-TEST-001
title: Test Document
document_type: procedure
department: IT
version: "1.0"
source_type: synthetic
language: en
status: active
---

# Test Procedure

This is a test knowledge document.
"""


def create_document(tmp_path: Path, content: str, name: str = "test.md") -> Path:
    document_path = tmp_path / name
    document_path.write_text(content, encoding="utf-8")
    return document_path


def test_load_valid_document(tmp_path):
    path = create_document(tmp_path, VALID_DOCUMENT)

    document = load_document(path)

    assert document["document_id"] == "KB-TEST-001"
    assert document["title"] == "Test Document"
    assert document["content"].startswith("# Test Procedure")
    assert document["source_path"] == str(path.resolve())


def test_load_document_without_front_matter(tmp_path):
    path = create_document(tmp_path, "# Document without metadata")

    with pytest.raises(DocumentLoaderError, match="Missing YAML front matter"):
        load_document(path)


def test_load_document_with_invalid_yaml(tmp_path):
    content = "---\ntitle: [invalid yaml\n---\n# Body"
    path = create_document(tmp_path, content)

    with pytest.raises(DocumentLoaderError, match="Invalid YAML metadata"):
        load_document(path)


def test_load_document_with_missing_metadata(tmp_path):
    content = """---
title: Test Document
department: IT
---

# Test
"""
    path = create_document(tmp_path, content)

    with pytest.raises(DocumentLoaderError, match="Missing metadata fields"):
        load_document(path)


def test_load_document_with_invalid_source_type(tmp_path):
    content = VALID_DOCUMENT.replace(
        "source_type: synthetic",
        "source_type: external",
    )
    path = create_document(tmp_path, content)

    with pytest.raises(DocumentLoaderError, match="Unsupported source_type"):
        load_document(path)


def test_load_document_with_inactive_status(tmp_path):
    content = VALID_DOCUMENT.replace("status: active", "status: inactive")
    path = create_document(tmp_path, content)

    with pytest.raises(DocumentLoaderError, match="status must be active"):
        load_document(path)


def test_load_empty_document(tmp_path):
    path = create_document(tmp_path, "")

    with pytest.raises(DocumentLoaderError, match="Document is empty"):
        load_document(path)


def test_load_document_with_empty_markdown_body(tmp_path):
    content = VALID_DOCUMENT.split("# Test Procedure")[0]
    path = create_document(tmp_path, content)

    with pytest.raises(DocumentLoaderError, match="Markdown body is empty"):
        load_document(path)


def test_load_documents_from_directory(tmp_path):
    create_document(tmp_path, VALID_DOCUMENT)

    documents = load_documents(tmp_path)

    assert len(documents) == 1
    assert documents[0]["document_id"] == "KB-TEST-001"


def test_load_documents_from_empty_directory(tmp_path):
    with pytest.raises(DocumentLoaderError, match="No Markdown documents found"):
        load_documents(tmp_path)


def test_load_documents_detects_duplicate_ids(tmp_path):
    create_document(tmp_path, VALID_DOCUMENT, "first.md")
    create_document(tmp_path, VALID_DOCUMENT, "second.md")

    with pytest.raises(DocumentLoaderError, match="Duplicate document_id"):
        load_documents(tmp_path)


def test_load_documents_rejects_invalid_document(tmp_path):
    create_document(tmp_path, VALID_DOCUMENT, "valid.md")
    create_document(tmp_path, "# Invalid document", "invalid.md")

    with pytest.raises(DocumentLoaderError, match="Document loading failed"):
        load_documents(tmp_path)
