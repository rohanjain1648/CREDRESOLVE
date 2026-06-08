"""
Part 5: Knowledge & RAG — Retriever
Queries ChromaDB and returns grounded policy/FAQ context.
"""
from .knowledge_base import get_collection
from backend.monitoring.metrics import RAG_LATENCY, RAG_RETRIEVED_DOCS
import time


def retrieve(
    query: str,
    n_results: int = 3,
    category_filter: str = None,
) -> dict:
    """
    Query the knowledge base and return relevant chunks with sources.

    Args:
        query: Natural language query (Hindi or English)
        n_results: Number of top results to return
        category_filter: Optional filter by category (e.g., 'settlement', 'compliance')

    Returns:
        dict with 'documents', 'sources', 'relevance_scores'
    """
    start = time.time()
    collection = get_collection()

    where_clause = {"category": category_filter} if category_filter else None

    results = collection.query(
        query_texts=[query],
        n_results=n_results,
        where=where_clause,
        include=["documents", "metadatas", "distances"],
    )

    latency = (time.time() - start) * 1000
    RAG_LATENCY.observe(latency)

    docs = results["documents"][0] if results["documents"] else []
    metas = results["metadatas"][0] if results["metadatas"] else []
    distances = results["distances"][0] if results["distances"] else []

    # Cosine distance → relevance score
    relevance_scores = [round(1 - d, 4) for d in distances]

    RAG_RETRIEVED_DOCS.observe(len(docs))

    formatted = []
    sources = []
    for doc, meta, score in zip(docs, metas, relevance_scores):
        if score > 0.3:  # relevance threshold
            formatted.append(f"[{meta.get('source', 'unknown')}] (relevance: {score:.2f})\n{doc}")
            sources.append(meta.get("source", "unknown"))

    return {
        "documents": formatted,
        "sources": sources,
        "raw_docs": docs,
        "relevance_scores": relevance_scores,
        "query": query,
        "latency_ms": latency,
    }


def retrieve_for_intent(intent: str, customer_context: str = "") -> dict:
    """Build a targeted query based on detected intent."""
    intent_queries = {
        "settlement": "settlement discount waiver one-time payment policy",
        "delay": "PTP promise to pay extension policy guidelines",
        "dispute": "payment dispute resolution UTR verification process",
        "angry": "RBI fair practices code harassment escalation borrower rights",
        "refusal": "restructuring EMI hardship moratorium options",
        "cooperative": "payment options methods online portal UPI NEFT",
    }
    base_query = intent_queries.get(intent, "debt collection policy guidelines")
    full_query = f"{base_query} {customer_context}".strip()
    return retrieve(full_query, n_results=3)
