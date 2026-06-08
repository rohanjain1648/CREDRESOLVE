"""
Part 8: Observability & Monitoring — Prometheus metrics
Tracks latency, token usage, cost, resolution/escalation rates, hallucinations.
"""
from prometheus_client import (
    Counter, Histogram, Gauge, Summary,
    generate_latest, CONTENT_TYPE_LATEST, CollectorRegistry
)

REGISTRY = CollectorRegistry()

# ── Agent Metrics ─────────────────────────────────────────────────────────────
CONVERSATION_TOTAL = Counter(
    "dco_conversations_total",
    "Total conversations started",
    ["channel", "language"],
    registry=REGISTRY,
)

CONVERSATION_DURATION = Histogram(
    "dco_conversation_duration_seconds",
    "Duration of full conversation",
    buckets=[10, 30, 60, 120, 300, 600],
    registry=REGISTRY,
)

TOKEN_USAGE = Counter(
    "dco_token_usage_total",
    "Total LLM tokens consumed",
    ["model", "direction"],  # direction: input/output
    registry=REGISTRY,
)

COST_USD = Counter(
    "dco_cost_usd_total",
    "Estimated LLM cost in USD",
    ["model"],
    registry=REGISTRY,
)

LLM_LATENCY = Histogram(
    "dco_llm_latency_ms",
    "LLM response latency in milliseconds",
    ["model", "node"],
    buckets=[100, 500, 1000, 2000, 5000, 10000],
    registry=REGISTRY,
)

# ── Business Metrics ──────────────────────────────────────────────────────────
RESOLUTION_OUTCOME = Counter(
    "dco_resolution_outcome_total",
    "Count of resolution outcomes",
    ["outcome"],   # ptp_logged, settlement_agreed, escalated, dispute_raised, callback_scheduled
    registry=REGISTRY,
)

ESCALATION_RATE = Counter(
    "dco_escalations_total",
    "Total escalations triggered",
    ["reason"],
    registry=REGISTRY,
)

PTP_AMOUNT = Histogram(
    "dco_ptp_amount_inr",
    "Promise-to-Pay amounts in INR",
    buckets=[1000, 5000, 10000, 25000, 50000, 100000, 500000],
    registry=REGISTRY,
)

INTENT_DETECTED = Counter(
    "dco_intent_detected_total",
    "Intent classifications",
    ["intent"],
    registry=REGISTRY,
)

# ── RAG / Prompt Metrics ──────────────────────────────────────────────────────
RAG_LATENCY = Histogram(
    "dco_rag_latency_ms",
    "ChromaDB retrieval latency in milliseconds",
    buckets=[10, 50, 100, 500, 1000],
    registry=REGISTRY,
)

RAG_RETRIEVED_DOCS = Histogram(
    "dco_rag_retrieved_docs",
    "Number of documents retrieved per query",
    buckets=[0, 1, 2, 3, 5, 10],
    registry=REGISTRY,
)

HALLUCINATION_DETECTED = Counter(
    "dco_hallucination_total",
    "Hallucinations detected by evaluation prompt",
    registry=REGISTRY,
)

# Alias used by multi-agent QAAgent (per-model label)
HALLUCINATION_FLAGS = Counter(
    "dco_hallucination_flags_total",
    "Hallucinations flagged by QAAgent per model",
    ["model"],
    registry=REGISTRY,
)

# Agent-to-agent handoff counter
AGENT_HANDOFFS = Counter(
    "dco_agent_handoffs_total",
    "Multi-agent handoffs between specialist agents",
    ["from_agent", "to_agent"],
    registry=REGISTRY,
)

TOOL_CALLS = Counter(
    "dco_tool_calls_total",
    "Total tool calls made by agent",
    ["tool_name", "status"],  # status: success/error
    registry=REGISTRY,
)

STATE_TRANSITIONS = Counter(
    "dco_state_transitions_total",
    "LangGraph state transitions",
    ["from_state", "to_state"],
    registry=REGISTRY,
)

# ── Active Sessions ───────────────────────────────────────────────────────────
ACTIVE_SESSIONS = Gauge(
    "dco_active_sessions",
    "Currently active conversation sessions",
    registry=REGISTRY,
)


def get_metrics_output() -> bytes:
    return generate_latest(REGISTRY)


# ── Cost estimation (per 1K tokens) ──────────────────────────────────────────
# Gemini free-tier models — cost is $0 under free quota
# Paid rates shown for post-quota tracking
COST_PER_1K = {
    "gemini-2.5-flash-lite": {"input": 0.0, "output": 0.0},  # free tier: fastest, low latency
    "gemini-2.5-flash": {"input": 0.0, "output": 0.0},       # free tier: best price/performance + thinking
    "gemini-2.5-pro": {"input": 0.00125, "output": 0.01},    # free tier: 5 RPD, then paid
}


def record_llm_call(
    model: str,
    input_tokens: int,
    output_tokens: int,
    latency_ms: float,
):
    TOKEN_USAGE.labels(model=model, direction="input").inc(input_tokens)
    TOKEN_USAGE.labels(model=model, direction="output").inc(output_tokens)
    LLM_LATENCY.labels(model=model, node="single_agent").observe(latency_ms)

    cost_info = COST_PER_1K.get(model, {"input": 0.003, "output": 0.015})
    cost = (input_tokens / 1000) * cost_info["input"] + (output_tokens / 1000) * cost_info["output"]
    COST_USD.labels(model=model).inc(cost)
