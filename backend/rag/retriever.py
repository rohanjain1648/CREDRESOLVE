"""
Part 5: Knowledge & RAG — Retriever
Queries the in-process numpy vector store and returns grounded policy/FAQ context.
"""
from .knowledge_base import search
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
    hits = search(query, n=n_results)

    if category_filter:
        hits = [(d, m, s) for d, m, s in hits if m.get("category") == category_filter]

    latency = (time.time() - start) * 1000
    RAG_LATENCY.observe(latency)
    RAG_RETRIEVED_DOCS.observe(len(hits))

    formatted, sources, relevance_scores = [], [], []
    for doc, meta, score in hits:
        relevance_scores.append(round(score, 4))
        if score > 0.3:
            formatted.append(f"[{meta.get('source', 'unknown')}] (relevance: {score:.2f})\n{doc}")
            sources.append(meta.get("source", "unknown"))

    return {
        "documents": formatted,
        "sources": sources,
        "raw_docs": [d for d, _, _ in hits],
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
