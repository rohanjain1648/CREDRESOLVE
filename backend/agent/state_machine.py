"""
Part 1: State Machine — all states, transitions, entry/exit criteria
"""
from enum import Enum
from typing import Annotated, Any, Optional
import operator
from pydantic import BaseModel


class AgentState(str, Enum):
    GREETING = "greeting"
    AUTHENTICATION = "authentication"
    CONTEXT_GATHERING = "context_gathering"
    DIAGNOSIS = "diagnosis"
    TOOL_EXECUTION = "tool_execution"
    KNOWLEDGE_RETRIEVAL = "knowledge_retrieval"
    NEGOTIATION = "negotiation"
    ESCALATION = "escalation"
    RESOLUTION = "resolution"
    FOLLOW_UP = "follow_up"
    END = "end"


class Intent(str, Enum):
    DELAY = "delay"             # Borrower wants to delay payment
    REFUSAL = "refusal"         # Borrower refuses to pay
    DISPUTE = "dispute"         # Borrower disputes the amount
    SETTLEMENT = "settlement"   # Borrower wants settlement
    ANGRY = "angry"             # Emotional/angry borrower
    COOPERATIVE = "cooperative" # Willing to pay
    UNKNOWN = "unknown"


class ResolutionOutcome(str, Enum):
    PTP_LOGGED = "ptp_logged"               # Promise to Pay scheduled
    SETTLEMENT_AGREED = "settlement_agreed"
    ESCALATED = "escalated"
    DISPUTE_RAISED = "dispute_raised"
    CALLBACK_SCHEDULED = "callback_scheduled"
    PAYMENT_CONFIRMED = "payment_confirmed"


# ── LangGraph state dict ──────────────────────────────────────────────────────
# operator.add merges lists across parallel nodes
from typing import TypedDict, List


class ConversationState(TypedDict, total=False):
    # Conversation history — messages are appended, never replaced
    messages: Annotated[List[dict], operator.add]

    # Session identity
    session_id: str
    customer_id: Optional[str]
    language: str                   # "hi", "en", "hinglish"

    # Current position in the state machine
    current_state: str              # AgentState value

    # Customer profile (loaded from CRM)
    customer_data: Optional[dict]
    authenticated: bool

    # Diagnosis
    intent: Optional[str]           # Intent value
    intent_confidence: float

    # RAG results
    retrieved_policies: Optional[List[str]]
    retrieved_sources: Optional[List[str]]

    # Tool results
    tool_results: Optional[dict]

    # Negotiation tracking
    negotiation_rounds: int
    proposed_amount: Optional[float]
    proposed_date: Optional[str]

    # Outcomes
    ptp_logged: bool
    ptp_date: Optional[str]
    ptp_amount: Optional[float]
    escalation_required: bool
    resolution_outcome: Optional[str]   # ResolutionOutcome value

    # Monitoring
    token_usage: int
    latency_ms: float
    tool_call_count: int
    hallucination_flags: int

    # Voice
    voice_enabled: bool
    audio_output_url: Optional[str]


# ── State transition rules (textual — documented for Part 1) ─────────────────
STATE_TRANSITIONS = {
    AgentState.GREETING: {
        "entry": "Session initialized; customer connected via voice/text/API",
        "exit": "Greeting delivered; customer acknowledged",
        "next": [AgentState.AUTHENTICATION],
        "failure": [AgentState.END],
    },
    AgentState.AUTHENTICATION: {
        "entry": "Agent requests DOB / loan-ID / OTP verification",
        "exit": "Identity verified OR max retries exceeded",
        "next": [AgentState.CONTEXT_GATHERING],
        "failure": [AgentState.ESCALATION],
    },
    AgentState.CONTEXT_GATHERING: {
        "entry": "CRM tool called; conversation history loaded from memory",
        "exit": "Customer record and payment history loaded",
        "next": [AgentState.DIAGNOSIS],
        "failure": [AgentState.ESCALATION],
    },
    AgentState.DIAGNOSIS: {
        "entry": "LLM classifies borrower intent from utterance",
        "exit": "Intent classified with confidence > 0.7",
        "next": [
            AgentState.TOOL_EXECUTION,      # delay / settlement
            AgentState.KNOWLEDGE_RETRIEVAL,  # dispute
            AgentState.ESCALATION,           # angry / emotional
        ],
        "failure": [AgentState.NEGOTIATION],
    },
    AgentState.TOOL_EXECUTION: {
        "entry": "Required tools selected based on intent",
        "exit": "Tool results returned and validated",
        "next": [AgentState.KNOWLEDGE_RETRIEVAL],
        "failure": [AgentState.ESCALATION],
    },
    AgentState.KNOWLEDGE_RETRIEVAL: {
        "entry": "RAG query built from intent + customer context",
        "exit": "Relevant policy / FAQ chunks retrieved",
        "next": [AgentState.NEGOTIATION],
        "failure": [AgentState.NEGOTIATION],   # proceed without RAG on failure
    },
    AgentState.NEGOTIATION: {
        "entry": "Agent presents options grounded in policy; tracks rounds",
        "exit": "Agreement reached OR max rounds exceeded",
        "next": [AgentState.RESOLUTION, AgentState.ESCALATION],
        "failure": [AgentState.ESCALATION],
    },
    AgentState.ESCALATION: {
        "entry": "High-risk flag set; supervisor notified via ticket",
        "exit": "Escalation ticket created; callback scheduled",
        "next": [AgentState.FOLLOW_UP],
        "failure": [AgentState.FOLLOW_UP],
    },
    AgentState.RESOLUTION: {
        "entry": "PTP / settlement agreement confirmed",
        "exit": "Outcome logged to CRM; confirmation sent to customer",
        "next": [AgentState.FOLLOW_UP],
        "failure": [AgentState.ESCALATION],
    },
    AgentState.FOLLOW_UP: {
        "entry": "Summary prepared; SMS/email confirmation dispatched",
        "exit": "Confirmation sent; session closed",
        "next": [AgentState.END],
        "failure": [AgentState.END],
    },
}
