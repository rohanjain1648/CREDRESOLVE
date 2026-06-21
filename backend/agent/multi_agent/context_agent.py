"""ContextAgent — handles GREETING, AUTHENTICATION, CONTEXT_GATHERING.

Responsibilities:
  1. Greet the customer in Hindi (Arya persona).
  2. Verify the borrower's Loan ID via the CRM tool.
  3. Load cross-session SQLite memory summary.
  4. Pass authenticated customer data downstream to DecisionAgent.

Agent-to-agent output:
  Appends an AgentMessage with role="handoff" summarising auth result
  and routes next_agent → "decision" (or "__end__" after 3 failures).
"""

import json
import time

from backend.tools.crm_tool import fetch_customer_data
from backend.memory.user_memory import upsert_user_profile, build_memory_summary
from backend.monitoring.metrics import AGENT_HANDOFFS
from .state import MultiAgentState, AgentMessage

GREETING_HINDI = (
    "नमस्ते! मैं आर्या हूँ, CredResolve की डिजिटल DCO। "
    "आपकी सहायता करने के लिए तैयार हूँ। "
    "कृपया अपना Loan ID बताएं ताकि मैं आपका खाता देख सकूँ।"
)

MAX_AUTH_ATTEMPTS = 3


def context_agent(state: MultiAgentState) -> dict:
    """LangGraph node function — returns a partial state update."""
    t0 = time.time()
    loan_id = state.get("loan_id", "").strip()
    attempts = state.get("auth_attempts", 0)
    messages_out = []
    agent_msgs = []

    # ── Step 1: Greeting (first turn only) ───────────────────────────────────
    if not state.get("customer_data"):
        messages_out.append({"role": "assistant", "content": GREETING_HINDI})
        agent_msgs.append(AgentMessage(
            agent="context",
            role="output",
            content="Greeted customer in Hindi. Awaiting Loan ID.",
            metadata={"step": "greeting"},
        ))

    # ── Step 2: Authentication via CRM tool ───────────────────────────────────
    if not state.get("authenticated", False):
        if not loan_id:
            agent_msgs.append(AgentMessage(
                agent="context",
                role="handoff",
                content="No loan_id provided — staying in context phase.",
                metadata={"next": "context", "reason": "missing_loan_id"},
            ))
            return {
                "agent_messages": agent_msgs,
                "messages": messages_out,
                "next_agent": "context",
                "auth_attempts": attempts,
            }

        attempts += 1
        crm_raw = fetch_customer_data.invoke({"loan_id": loan_id})
        crm_result = json.loads(crm_raw) if isinstance(crm_raw, str) else crm_raw

        if crm_result.get("error"):
            if attempts >= MAX_AUTH_ATTEMPTS:
                msg = (
                    f"Loan ID {loan_id} को {attempts} बार verify करने में विफल। "
                    "आपका case senior DCO को transfer किया जा रहा है।"
                )
                messages_out.append({"role": "assistant", "content": msg})
                agent_msgs.append(AgentMessage(
                    agent="context",
                    role="handoff",
                    content=f"Auth failed after {attempts} attempts — routing to escalation via decision.",
                    metadata={"next": "decision", "reason": "auth_failed", "attempts": attempts},
                ))
                return {
                    "authenticated": False,
                    "auth_attempts": attempts,
                    "agent_messages": agent_msgs,
                    "messages": messages_out,
                    "next_agent": "decision",
                    "intent": "angry",
                }
            else:
                remaining = MAX_AUTH_ATTEMPTS - attempts
                msg = (
                    f"Loan ID नहीं मिला। कृपया फिर से दर्ज करें "
                    f"({remaining} प्रयास शेष)।"
                )
                messages_out.append({"role": "assistant", "content": msg})
                agent_msgs.append(AgentMessage(
                    agent="context",
                    role="output",
                    content=f"CRM lookup failed for {loan_id}. Attempt {attempts}/{MAX_AUTH_ATTEMPTS}.",
                    metadata={"loan_id": loan_id, "attempts": attempts},
                ))
                return {
                    "authenticated": False,
                    "auth_attempts": attempts,
                    "agent_messages": agent_msgs,
                    "messages": messages_out,
                    "next_agent": "context",
                }

        # ── Step 3: Load SQLite cross-session memory ──────────────────────────
        customer_id = crm_result.get("customer_id", loan_id)
        upsert_user_profile(
            customer_id=customer_id,
            loan_id=loan_id,
            name=crm_result.get("name", ""),
        )
        memory_summary = build_memory_summary(customer_id)

        latency_ms = (time.time() - t0) * 1000
        AGENT_HANDOFFS.labels(from_agent="context", to_agent="decision").inc()

        agent_msgs.append(AgentMessage(
            agent="context",
            role="handoff",
            content=(
                f"Authenticated: {crm_result.get('name')} | "
                f"Loan: {loan_id} | DPD: {crm_result.get('days_past_due')} | "
                f"Outstanding: ₹{crm_result.get('outstanding_amount')} | "
                f"Memory: {memory_summary[:80]}..."
            ),
            metadata={
                "next": "decision",
                "customer_id": customer_id,
                "dpd": crm_result.get("days_past_due"),
                "outstanding": crm_result.get("outstanding_amount"),
                "latency_ms": round(latency_ms, 1),
            },
        ))

        return {
            "authenticated": True,
            "auth_attempts": attempts,
            "customer_data": crm_result,
            "memory_summary": memory_summary,
            "agent_messages": agent_msgs,
            "messages": messages_out,
            "next_agent": "decision",
        }

    # Already authenticated — context already gathered, pass through
    AGENT_HANDOFFS.labels(from_agent="context", to_agent="decision").inc()
    agent_msgs.append(AgentMessage(
        agent="context",
        role="handoff",
        content="Already authenticated. Forwarding new turn to decision.",
        metadata={"next": "decision"},
    ))
    return {
        "agent_messages": agent_msgs,
        "messages": messages_out,
        "next_agent": "decision",
    }
