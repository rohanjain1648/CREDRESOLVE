"""RetrievalAgent — handles KNOWLEDGE_RETRIEVAL.

Responsibilities:
  1. Build an intent-aware query string from DecisionAgent's intent output.
  2. Query ChromaDB with cosine similarity, threshold 0.3, top-3 results.
  3. Format retrieved chunks into a policy context string.
  4. Pass policies downstream to DecisionAgent (negotiation phase).

Agent-to-agent communication:
  Receives: intent, customer_data from state (set by DecisionAgent/ContextAgent).
  Emits:    AgentMessage with retrieved chunk summaries + source IDs.
  Routes:   next_agent → "decision" (DecisionAgent will generate Hindi response).
"""

import time

from backend.rag.retriever import retrieve_for_intent
from backend.monitoring.metrics import AGENT_HANDOFFS, RAG_LATENCY
from .state import MultiAgentState, AgentMessage

RELEVANCE_THRESHOLD = 0.3


def retrieval_agent(state: MultiAgentState) -> dict:
    """LangGraph node function — returns a partial state update."""
    t0 = time.time()

    intent = state.get("intent", "unknown")
    customer_data = state.get("customer_data", {})

    context_snippet = (
        f"DPD={customer_data.get('days_past_due', 0)} "
        f"outstanding=₹{customer_data.get('outstanding_amount', 0)}"
    )

    # ── RAG query ─────────────────────────────────────────────────────────────
    try:
        raw_results = retrieve_for_intent(intent=intent, customer_context=context_snippet)
    except Exception as exc:
        agent_msgs = [AgentMessage(
            agent="retrieval",
            role="error",
            content=f"ChromaDB query failed: {exc}. Continuing with empty policies.",
            metadata={"intent": intent, "error": str(exc)},
        )]
        AGENT_HANDOFFS.labels(from_agent="retrieval", to_agent="decision").inc()
        return {
            "retrieved_policies": [],
            "retrieval_query": intent,
            "agent_messages": agent_msgs,
            "next_agent": "decision",
        }

    latency_ms = (time.time() - t0) * 1000
    RAG_LATENCY.observe(latency_ms / 1000)

    # ── Filter by relevance threshold ─────────────────────────────────────────
    filtered = [r for r in raw_results if r.get("relevance", 0) >= RELEVANCE_THRESHOLD]

    # ── Build human-readable summary for the agent bus ────────────────────────
    summaries = []
    for r in filtered:
        doc_id = r.get("id", "?")
        text_preview = r.get("text", "")[:80].replace("\n", " ")
        rel = round(r.get("relevance", 0), 3)
        summaries.append(f"[{doc_id}] (rel={rel}) {text_preview}…")

    handoff_content = (
        f"Retrieved {len(filtered)}/{len(raw_results)} chunks above threshold {RELEVANCE_THRESHOLD} "
        f"for intent='{intent}'. Sources: " + ", ".join(r.get("id", "?") for r in filtered)
        if filtered else f"No chunks above threshold for intent='{intent}'."
    )

    AGENT_HANDOFFS.labels(from_agent="retrieval", to_agent="decision").inc()

    return {
        "retrieved_policies": filtered,
        "retrieval_query": context_snippet,
        "agent_messages": [AgentMessage(
            agent="retrieval",
            role="handoff",
            content=handoff_content,
            metadata={
                "next": "decision",
                "intent": intent,
                "results_total": len(raw_results),
                "results_above_threshold": len(filtered),
                "latency_ms": round(latency_ms, 1),
                "summaries": summaries,
            },
        )],
        "next_agent": "decision",
    }
