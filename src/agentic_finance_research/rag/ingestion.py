"""Document ingestion pipeline - load, chunk, embed, and store in FAISS."""

from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

import faiss
import numpy as np

# --- Configuration ---
CHUNK_SIZE = 512
CHUNK_OVERLAP = 64
EMBEDDING_MODEL = "all-MiniLM-L6-v2"


def load_document(file_path: str) -> str:
    """Read a text file and return its content as a string."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Document not found: {file_path}")
    return path.read_text(encoding="utf-8")


def chunk_text(text: str) -> list[str]:
    """Split text into overlapping chunks using recursive character splitting."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    documents = splitter.create_documents([text])
    return [doc.page_content for doc in documents]


def create_embeddings(chunks: list[str]) -> np.ndarray:
    """Convert text chunks into numerical vectors using a sentence transformer model."""
    model = SentenceTransformer(EMBEDDING_MODEL)
    embeddings = model.encode(chunks, show_progress_bar=True)
    return np.array(embeddings, dtype=np.float32)


def build_index(embeddings: np.ndarray) -> faiss.IndexFlatL2:
    """Build a FAISS vector index from embeddings for similarity search."""
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return index


def ingest(file_path: str) -> tuple[faiss.IndexFlatL2, list[str]]:
    """Run the full ingestion pipeline: load -> chunk -> embed -> index."""
    print(f"Loading document: {file_path}")
    text = load_document(file_path)
    print(f"Document loaded: {len(text)} characters")

    print("Chunking text...")
    chunks = chunk_text(text)
    print(f"Created {len(chunks)} chunks")

    print("Creating embeddings...")
    embeddings = create_embeddings(chunks)
    print(f"Embeddings shape: {embeddings.shape}")

    print("Building FAISS index...")
    index = build_index(embeddings)
    print(f"Index built: {index.ntotal} vectors")

    return index, chunks


if __name__ == "__main__":
    index, chunks = ingest("data/raw/sample_10k.txt")
    print(f"\nDone! {len(chunks)} chunks indexed and ready for search.")

    

