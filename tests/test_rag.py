"""Tests for the RAG ingestion and retrieval pipeline."""

import numpy as np
import pytest

from agentic_finance_research.rag.ingestion import (
    build_index,
    chunk_text,
    create_embeddings,
    load_document,
)
from agentic_finance_research.rag.retrieval import search

SAMPLE_TEXT = (
    "Revenue for the fiscal year was $94.9 billion. "
    "The company reported net income of $23.1 billion. "
    "Long-term debt was $42.5 billion with a weighted average interest rate of 3.8%. "
    "The company is exposed to foreign currency risk, interest rate risk, and commodity price risk."
)


def test_load_document_reads_file(tmp_path):
    """Test that load_document reads a file and returns its content."""
    test_file = tmp_path / "test.txt"
    test_file.write_text("Hello, this is a test document.")

    result = load_document(str(test_file))

    assert result == "Hello, this is a test document."


def test_load_document_raises_on_missing_file():
    """Test that load_document raises FileNotFoundError for a missing file."""
    with pytest.raises(FileNotFoundError):
        load_document("nonexistent/path/fake.txt")


def test_chunk_text_returns_chunks():
    """Test that chunk_text_ splits text into a non-empty list of strings."""
    chunks = chunk_text(SAMPLE_TEXT)

    assert isinstance(chunks, list)
    assert len(chunks) > 0
    assert all(isinstance(chunk, str) for chunk in chunks)


def test_create_embeddings_shape():
    """Test that embeddings have the correct shape and dtype."""
    chunks = chunk_text(SAMPLE_TEXT)
    embeddings = create_embeddings(chunks)

    assert isinstance(embeddings, np.ndarray)
    assert embeddings.dtype == np.float32
    assert embeddings.shape[0] == len(chunks)
    assert embeddings.shape[1] == 384


def test_build_index_contains_vectors():
    """Test that the FAISS index contains the correct number of vectors."""
    chunks = chunk_text(SAMPLE_TEXT)
    embeddings = create_embeddings(chunks)
    index = build_index(embeddings)

    assert index.ntotal == len(chunks)


def test_search_returns_relevant_chunks():
    """Test that search returns results relevant to the query."""
    chunks = chunk_text(SAMPLE_TEXT)
    embeddings = create_embeddings(chunks)
    index = build_index(embeddings)

    results = search("What was the total revenue?", index, chunks)

    assert len(results) > 0
    assert len(results) <= 3
    assert "revenue" in results[0]["text"].lower()


def test_search_result_structure():
    """Test that each search result has the expected keys and types."""
    chunks = chunk_text(SAMPLE_TEXT)
    embeddings = create_embeddings(chunks)
    index = build_index(embeddings)

    results = search("debt", index, chunks)
    result = results[0]

    assert "rank" in result
    assert "chunk_index" in result
    assert "distance" in result
    assert "text" in result
    assert isinstance(result["rank"], int)
    assert isinstance(result["distance"], float)
    assert isinstance(result["text"], str)
