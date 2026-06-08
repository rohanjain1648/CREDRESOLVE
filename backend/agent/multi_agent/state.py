"""Shared state TypedDict for the multi-agent system.

Every agent reads from this dict and returns a partial update.
The `agent_messages` list is the inter-agent communication bus —
each agent appends one AgentMessage so the full reasoning trace
is visible at the end of a session.
"""

import operator
from typing import Annotated, List
from typing_extensions import TypedDict


class AgentMessage(TypedDict):
    """One message on the agent-to-agent communication bus."""
    agent: str       # "context" | "retrieval" | "decision" | "execution" | "qa"
    role: str        # "output" | "handoff" | "error"
    content: str     # human-readable summary of what this agent did/found
    metadata: dict   # agent-specific payload (amounts, ticket_ids, scores …)


class MultiAgentState(TypedDict):
    # ── Session identifiers ──────────────────────────────────────────────────
    session_id: str
    loan_id: str
    customer_message: str
    language: str

    # ── Inter-agent communication bus (append-only) ──────────────────────────
    agent_messages: Annotated[List[AgentMessage], operator.add]

    # ── Routing ──────────────────────────────────────────────────────────────
    # Supervisor reads this after every agent to decide where to route next.
    next_agent: str   # "context" | "retrieval" | "decision" | "execution" | "qa" | "__end__"

    # ── ContextAgent outputs ─────────────────────────────────────────────────
    authenticated: bool
    auth_attempts: int
    customer_data: dict       # raw CRM record
    memory_summary: str       # SQLite cross-session summary string

    # ── RetrievalAgent outputs ───────────────────────────────────────────────
    retrieved_policies: list  # list of {id, text, relevance} dicts
    retrieval_query: str      # the intent-aware query that was run

    # ── DecisionAgent outputs ────────────────────────────────────────────────
    intent: str               # delay | settlement | dispute | angry | cooperative | refusal | unknown
    intent_confidence: float
    negotiation_round: int
    proposed_amount: float
    proposed_date: str
    escalation_required: bool
    negotiation_response: str # raw Hindi pitch from DecisionAgent (pre-QA)

    # ── ExecutionAgent outputs ───────────────────────────────────────────────
    tool_results: dict        # aggregated results from all tool calls
    ptp_logged: bool
    ptp_id: str
    ticket_id: str
    sms_sent: bool

    # ── QAAgent outputs ──────────────────────────────────────────────────────
    final_response: str       # validated, safe Hindi response
    evaluation_passed: bool
    compliance_violation: bool
    hallucination_detected: bool
    accuracy_score: float
    tone_score: float

    # ── Full turn history (appended by each agent for LLM context) ───────────
    messages: Annotated[List[dict], operator.add]
