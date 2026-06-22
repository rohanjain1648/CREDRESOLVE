"""
Part 1 + Part 2: LangGraph State Machine — the core agent workflow
Defines all nodes, conditional routing, and checkpointing.
"""
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver

from backend.agent.state_machine import ConversationState, AgentState, Intent
from backend.agent.nodes import (
    greeting_node,
    authentication_node,
    context_gathering_node,
    diagnosis_node,
    tool_execution_node,
    knowledge_retrieval_node,
    negotiation_node,
    escalation_node,
    resolution_node,
    follow_up_node,
)


# ── Routing Functions ─────────────────────────────────────────────────────────

def route_after_auth(state: ConversationState) -> str:
    """After authentication: proceed to context OR escalate if auth failed."""
    if state.get("authenticated"):
        return "context_gathering"
    return "escalation"


def route_after_diagnosis(state: ConversationState) -> str:
    """Route based on detected borrower intent."""
    intent = state.get("intent", Intent.UNKNOWN)
    if intent == Intent.ANGRY:
        return "escalation"
    elif intent in [Intent.DISPUTE]:
        # Dispute: verify payment first via tools, then knowledge
        return "tool_execution"
    elif intent in [Intent.DELAY, Intent.SETTLEMENT, Intent.COOPERATIVE]:
        return "tool_execution"
    else:
        # Refusal or unknown → go to negotiation directly
        return "knowledge_retrieval"


def route_after_tool(state: ConversationState) -> str:
    """After tool execution, always retrieve relevant policies."""
    return "knowledge_retrieval"


# Borrower-acceptance signals (Hindi / Hinglish / English)
_AGREE_KEYWORDS = (
    "haan", "हाँ", "हां", "ठीक", "theek", " theek", "ok", "okay", "yes", "yep",
    "राज़ी", "razi", "agree", "करूँगा", "करूंगा", "करुंगा", "दूँगा", "दूंगा",
    "दे दूँगा", "पक्का", "pakka", "manjoor", "मंजूर", "सहमत", "deal", "done",
    "chalega", "चलेगा", "kar dunga", "bhar dunga", "भर दूंगा",
)


def route_after_negotiation(state: ConversationState) -> str:
    """After the borrower replies to an offer:
       - explicit acceptance  → resolution (log PTP / settlement)
       - 3 rounds exhausted    → escalation
       - otherwise             → another negotiation round
    """
    if state.get("escalation_required"):
        return "escalation"

    last_user = next(
        (m["content"] for m in reversed(state.get("messages", [])) if m.get("role") == "user"),
        "",
    ).lower()
    if any(k in last_user for k in _AGREE_KEYWORDS):
        return "resolution"

    if state.get("negotiation_rounds", 0) >= 3:
        return "escalation"

    return "negotiation"


def route_after_resolution(state: ConversationState) -> str:
    return "follow_up"


def route_after_escalation(state: ConversationState) -> str:
    return "follow_up"


# ── Graph Builder ─────────────────────────────────────────────────────────────

def build_graph(checkpointer=None):
    """
    Build and compile the CredResolve DCO LangGraph.

    Args:
        checkpointer: LangGraph checkpointer (SqliteSaver for persistence).
                      If None, no checkpointing is used.
    Returns:
        Compiled StateGraph
    """
    workflow = StateGraph(ConversationState)

    # Register all state nodes
    workflow.add_node(AgentState.GREETING, greeting_node)
    workflow.add_node(AgentState.AUTHENTICATION, authentication_node)
    workflow.add_node(AgentState.CONTEXT_GATHERING, context_gathering_node)
    workflow.add_node(AgentState.DIAGNOSIS, diagnosis_node)
    workflow.add_node(AgentState.TOOL_EXECUTION, tool_execution_node)
    workflow.add_node(AgentState.KNOWLEDGE_RETRIEVAL, knowledge_retrieval_node)
    workflow.add_node(AgentState.NEGOTIATION, negotiation_node)
    workflow.add_node(AgentState.ESCALATION, escalation_node)
    workflow.add_node(AgentState.RESOLUTION, resolution_node)
    workflow.add_node(AgentState.FOLLOW_UP, follow_up_node)

    # Entry point
    workflow.set_entry_point(AgentState.GREETING)

    # Fixed transitions
    workflow.add_edge(AgentState.GREETING, AgentState.AUTHENTICATION)
    workflow.add_edge(AgentState.CONTEXT_GATHERING, AgentState.DIAGNOSIS)
    workflow.add_edge(AgentState.KNOWLEDGE_RETRIEVAL, AgentState.NEGOTIATION)
    workflow.add_edge(AgentState.FOLLOW_UP, END)

    # Conditional transitions
    workflow.add_conditional_edges(
        AgentState.AUTHENTICATION,
        route_after_auth,
        {
            "context_gathering": AgentState.CONTEXT_GATHERING,
            "escalation": AgentState.ESCALATION,
        },
    )
    workflow.add_conditional_edges(
        AgentState.DIAGNOSIS,
        route_after_diagnosis,
        {
            "tool_execution": AgentState.TOOL_EXECUTION,
            "knowledge_retrieval": AgentState.KNOWLEDGE_RETRIEVAL,
            "escalation": AgentState.ESCALATION,
        },
    )
    workflow.add_conditional_edges(
        AgentState.TOOL_EXECUTION,
        route_after_tool,
        {"knowledge_retrieval": AgentState.KNOWLEDGE_RETRIEVAL},
    )
    workflow.add_conditional_edges(
        AgentState.NEGOTIATION,
        route_after_negotiation,
        {
            "resolution": AgentState.RESOLUTION,
            "escalation": AgentState.ESCALATION,
            "negotiation": AgentState.NEGOTIATION,
        },
    )
    workflow.add_conditional_edges(
        AgentState.RESOLUTION,
        route_after_resolution,
        {"follow_up": AgentState.FOLLOW_UP},
    )
    workflow.add_conditional_edges(
        AgentState.ESCALATION,
        route_after_escalation,
        {"follow_up": AgentState.FOLLOW_UP},
    )

    # Pause AFTER negotiation so the agent delivers its offer and waits for the
    # borrower's reply (turn-based conversation) instead of running the whole
    # pipeline to follow_up on every call.
    if checkpointer:
        return workflow.compile(
            checkpointer=checkpointer,
            interrupt_after=[AgentState.NEGOTIATION],
        )
    return workflow.compile(interrupt_after=[AgentState.NEGOTIATION])


# ── Singleton with SQLite checkpointing ──────────────────────────────────────
_graph = None
_checkpointer_cm = None


def get_graph():
    global _graph, _checkpointer_cm
    if _graph is None:
        import sqlite3
        conn = sqlite3.connect("./langgraph_checkpoints.db", check_same_thread=False)
        memory = SqliteSaver(conn)
        _graph = build_graph(checkpointer=memory)
    return _graph
