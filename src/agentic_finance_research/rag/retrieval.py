"""" Retrieval layer - embed a query and find the most relevant chunks in the FAISS index."""

from sentence_transformers import SentenceTransformer

import faiss
import numpy as np

from agentic_finance_research.rag.ingestion import EMBEDDING_MODEL


def search(
    query: str,
    index: faiss.IndexFlatL2,
    chunks: list[str],
    top_k: int = 3,
) -> list[dict]:
    """Embed a query and return the top-k most relevant chunks from the index."""
    model = SentenceTransformer(EMBEDDING_MODEL)
    query_embedding = model.encode([query])
    query_embedding = np.array(query_embedding, dtype=np.float32)
    
    distances, indices = index.search(query_embedding, top_k)

    results = []
    for rank, (dist, idx) in enumerate(zip(distances[0], indices[0])):
        results.append({
            "rank": rank + 1,
            "chunk_index": int(idx),
            "distance": float(dist),
            "text": chunks[idx],
        })

    return results


if __name__ == "__main__":
    from agentic_finance_research.rag.ingestion import ingest

    index, chunks = ingest("data/raw/sample_10k.txt")

    queries = [
        "What was the total revenue?",
        "How much long-term debt does the company have?",
        "What are the foreign currency risks?",
    ]

    for query in queries:
        print(f"\n{'='*60}")
        print(f"Query: {query}")
        print(f"{'='*60}")
        results = search(query, index, chunks)
        for result in results:
            print(f"\n Rank {result['rank']} (distance: {result['distance']:.4f})")
            print(f" {result['text'][:150]}...")