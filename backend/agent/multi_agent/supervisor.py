"""Supervisor — builds and runs the multi-agent LangGraph.

Architecture:
  SupervisorRouter (reads next_agent from state) → dispatches to one of:
    ContextAgent    → RetrievalAgent → DecisionAgent → ExecutionAgent → QAAgent → END
                                   ↑_______________|
                      (DecisionAgent loops back to RetrievalAgent on first visit)

The Supervisor itself is not an LLM — it is a pure Python routing function
that reads state.next_agent and returns the name of the next graph node.
This keeps orchestration deterministic and fast.

Graph structure:
  START → context_agent
  context_agent ──[next_agent]──► retrieval / decision / context / __end__
  decision_agent ─[next_agent]──► retrieval / execution
  retrieval_agent─[next_agent]──► decision
  execution_agent─[next_agent]──► qa
  qa_agent ──────────────────────► END

Agent-to-agent messages accumulate in state.agent_messages throughout
the graph run, giving a full reasoning trace per turn.
"""


from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver

from .state import MultiAgentState
from .context_agent import context_agent
from .retrieval_agent import retrieval_agent
from .decision_agent import decision_agent
from .execution_agent import execution_agent
from .qa_agent import qa_agent


# ── Routing functions (Supervisor logic) ──────────────────────────────────────

def route_from_context(state: MultiAgentState) -> str:
    """After ContextAgent: route based on auth result."""
    next_a = state.get("next_agent", "context")
    if next_a == "__end__":
        return END
    return next_a  # "context" | "decision"


def route_from_decision(state: MultiAgentState) -> str:
    """After DecisionAgent: route to retrieval (phase A) or execution (phase B)."""
    next_a = state.get("next_agent", "retrieval")
    return next_a  # "retrieval" | "execution"


def route_from_retrieval(state: MultiAgentState) -> str:
    """After RetrievalAgent: always back to decision for negotiation."""
    return "decision"


def route_from_execution(state: MultiAgentState) -> str:
    """After ExecutionAgent: always to QA."""
    return "qa"


def route_from_qa(state: MultiAgentState) -> str:
    """After QAAgent: always END."""
    return END


# ── Graph builder ─────────────────────────────────────────────────────────────

def build_multi_agent_graph(checkpointer=None) -> StateGraph:
    """Construct and compile the 5-agent LangGraph."""
    graph = StateGraph(MultiAgentState)

    # Register agent nodes
    graph.add_node("context", context_agent)
    graph.add_node("retrieval", retrieval_agent)
    graph.add_node("decision", decision_agent)
    graph.add_node("execution", execution_agent)
    graph.add_node("qa", qa_agent)

    # Entry point
    graph.set_entry_point("context")

    # Conditional edges from each agent
    graph.add_conditional_edges(
        "context",
        route_from_context,
        {
            "context": "context",   # still in auth phase
            "decision": "decision", # authenticated — go to diagnosis
            END: END,
        },
    )

    graph.add_conditional_edges(
        "decision",
        route_from_decision,
        {
            "retrieval": "retrieval",   # phase A — fetch policies first
            "execution": "execution",   # phase B — execute negotiation result
        },
    )

    graph.add_conditional_edges(
        "retrieval",
        route_from_retrieval,
        {"decision": "decision"},
    )

    graph.add_conditional_edges(
        "execution",
        route_from_execution,
        {"qa": "qa"},
    )

    graph.add_conditional_edges(
        "qa",
        route_from_qa,
        {END: END},
    )

    return graph.compile(checkpointer=checkpointer)


_ma_graph = None


def get_multi_agent_graph():
    """Singleton with SQLite checkpointing (one per process)."""
    global _ma_graph
    if _ma_graph is None:
        import sqlite3
        conn = sqlite3.connect("./langgraph_ma_checkpoints.db", check_same_thread=False)
        checkpointer = SqliteSaver(conn)
        _ma_graph = build_multi_agent_graph(checkpointer=checkpointer)
    return _ma_graph


# ── Convenience run helper ────────────────────────────────────────────────────

def run_multi_agent(
    customer_message: str,
    session_id: str,
    loan_id: str = "",
    language: str = "hi",
) -> dict:
    """
    Single-call entrypoint. Returns:
      {
        final_response:    str,
        agent_trace:       list[AgentMessage],   # full agent-to-agent log
        evaluation_passed: bool,
        intent:            str,
        ptp_id:            str,
        ticket_id:         str,
      }
    """
    graph = get_multi_agent_graph()

    initial_state: MultiAgentState = {
        "session_id": session_id,
        "loan_id": loan_id,
        "customer_message": customer_message,
        "language": language,
        "agent_messages": [],
        "messages": [{"role": "user", "content": customer_message}],
        "next_agent": "context",
        # All other fields left as None/default — agents populate them
        "authenticated": False,
        "auth_attempts": 0,
        "customer_data": {},
        "memory_summary": "",
        "retrieved_policies": [],
        "retrieval_query": "",
        "intent": "",
        "intent_confidence": 0.0,
        "negotiation_round": 0,
        "proposed_amount": 0.0,
        "proposed_date": "",
        "escalation_required": False,
        "negotiation_response": "",
        "tool_results": {},
        "ptp_logged": False,
        "ptp_id": "",
        "ticket_id": "",
        "sms_sent": False,
        "final_response": "",
        "evaluation_passed": False,
        "compliance_violation": False,
        "hallucination_detected": False,
        "accuracy_score": 1.0,
        "tone_score": 1.0,
    }

    config = {"configurable": {"thread_id": session_id}}
    final_state = graph.invoke(initial_state, config)

    return {
        "final_response": final_state.get("final_response", ""),
        "agent_trace": final_state.get("agent_messages", []),
        "evaluation_passed": final_state.get("evaluation_passed", False),
        "intent": final_state.get("intent", "unknown"),
        "ptp_id": final_state.get("ptp_id", ""),
        "ticket_id": final_state.get("ticket_id", ""),
        "hallucination_detected": final_state.get("hallucination_detected", False),
        "compliance_violation": final_state.get("compliance_violation", False),
        "accuracy_score": final_state.get("accuracy_score", 1.0),
        "tone_score": final_state.get("tone_score", 1.0),
    }
