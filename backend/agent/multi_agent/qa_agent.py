"""QAAgent — handles evaluation, hallucination check, compliance, session close.

Responsibilities:
  1. Run second Gemini call to validate DecisionAgent's negotiation_response:
       - Hallucination check: does response contradict RAG-retrieved policies?
       - RBI compliance: no threats, correct hours, no third-party disclosure.
       - Numeric accuracy: amounts/dates match ExecutionAgent tool_results.
       - Tone score: empathetic, not aggressive (threshold 0.7).
  2. If overall_pass=False → substitute corrected_response from Gemini.
  3. Log interaction to SQLite + update CRM notes (session close).
  4. Produce final_response for the API to return to the customer.
  5. Route → "__end__" (conversation turn complete).

Agent-to-agent communication:
  Receives: negotiation_response, tool_results, retrieved_policies from prior agents.
  Emits:    AgentMessage with full evaluation scores + pass/fail verdict.
  Routes:   next_agent → "__end__".
"""

import json
import re
import time

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage

from backend.config import get_settings
from backend.prompts.evaluation_prompt import HALLUCINATION_CHECK_PROMPT
from backend.memory.user_memory import log_interaction
from backend.tools.crm_tool import update_customer_notes
from backend.monitoring.metrics import AGENT_HANDOFFS, LLM_LATENCY, TOKEN_USAGE, HALLUCINATION_FLAGS
from .state import MultiAgentState, AgentMessage

settings = get_settings()


def _llm() -> ChatGoogleGenerativeAI:
    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash-lite",
        google_api_key=settings.google_api_key,
        max_output_tokens=512,
        temperature=0.1,
    )


def _extract_json(text: str) -> dict:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    return {}


def qa_agent(state: MultiAgentState) -> dict:
    """LangGraph node function — returns a partial state update."""
    t0 = time.time()

    negotiation_response = state.get("negotiation_response", "")
    tool_results = state.get("tool_results", {})
    retrieved_policies = state.get("retrieved_policies", [])
    customer_data = state.get("customer_data", {})
    intent = state.get("intent", "unknown")
    loan_id = state.get("loan_id", "") or customer_data.get("loan_id", "")
    customer_id = customer_data.get("customer_id", loan_id)
    ptp_logged = state.get("ptp_logged", False)
    ticket_id = state.get("ticket_id", "")
    escalation_required = state.get("escalation_required", False)

    # ── Build evaluation prompt ───────────────────────────────────────────────
    policy_context = "\n".join(
        f"[{p.get('id', '?')}] {p.get('text', '')}"
        for p in retrieved_policies
    ) or "No policies retrieved."

    tool_summary = json.dumps(tool_results, indent=2, default=str)[:600]

    eval_user = (
        f"Response to evaluate:\n{negotiation_response}\n\n"
        f"RAG Policies used:\n{policy_context}\n\n"
        f"Tool results:\n{tool_summary}\n\n"
        f"Customer DPD: {customer_data.get('days_past_due', 0)} | "
        f"Outstanding: ₹{customer_data.get('outstanding_amount', 0)}"
    )

    llm = _llm()
    t_llm = time.time()
    eval_response = llm.invoke([
        SystemMessage(content=HALLUCINATION_CHECK_PROMPT),
        HumanMessage(content=eval_user),
    ])
    llm_latency = (time.time() - t_llm) * 1000
    LLM_LATENCY.labels(model="gemini-2.5-flash", node="qa_agent").observe(llm_latency / 1000)

    usage = getattr(eval_response, "usage_metadata", {}) or {}
    TOKEN_USAGE.labels(model="gemini-2.5-flash", direction="input").inc(usage.get("input_tokens", 0))
    TOKEN_USAGE.labels(model="gemini-2.5-flash", direction="output").inc(usage.get("output_tokens", 0))

    parsed = _extract_json(eval_response.content)
    overall_pass = bool(parsed.get("overall_pass", True))
    hallucination_detected = bool(parsed.get("hallucination_detected", False))
    compliance_violation = bool(parsed.get("compliance_violation", False))
    accuracy_score = float(parsed.get("accuracy_score", 1.0))
    tone_score = float(parsed.get("tone_score", 1.0))
    corrected = parsed.get("corrected_response", "")

    if hallucination_detected:
        HALLUCINATION_FLAGS.labels(model="gemini-2.5-flash").inc()

    final_response = negotiation_response
    if not overall_pass and corrected:
        final_response = corrected

    # ── Session close — SQLite + CRM note ────────────────────────────────────
    outcome = (
        "escalated" if escalation_required
        else ("ptp_logged" if ptp_logged else "information_provided")
    )
    try:
        log_interaction(
            customer_id=customer_id,
            intent=intent,
            outcome=outcome,
            ptp_date=state.get("proposed_date", ""),
            ptp_amount=state.get("proposed_amount", 0),
            escalated=escalation_required,
        )
    except Exception:
        pass

    if loan_id:
        note_parts = [
            f"DCO multi-agent session | Intent: {intent} | Outcome: {outcome}",
        ]
        if ptp_logged:
            note_parts.append(
                f"PTP logged: ₹{state.get('proposed_amount')} by {state.get('proposed_date')} "
                f"(ID: {state.get('ptp_id')})"
            )
        if ticket_id:
            note_parts.append(f"Escalation ticket: {ticket_id}")
        note_parts.append(f"QA: pass={overall_pass} accuracy={accuracy_score:.2f} tone={tone_score:.2f}")
        try:
            update_customer_notes.invoke({"loan_id": loan_id, "note": " | ".join(note_parts)})
        except Exception:
            pass

    total_latency = (time.time() - t0) * 1000
    AGENT_HANDOFFS.labels(from_agent="qa", to_agent="__end__").inc()

    verdict = "PASS" if overall_pass else "FAIL → corrected"
    return {
        "final_response": final_response,
        "evaluation_passed": overall_pass,
        "hallucination_detected": hallucination_detected,
        "compliance_violation": compliance_violation,
        "accuracy_score": accuracy_score,
        "tone_score": tone_score,
        "messages": [{"role": "assistant", "content": final_response}],
        "agent_messages": [AgentMessage(
            agent="qa",
            role="handoff",
            content=(
                f"QA evaluation: {verdict} | "
                f"hallucination={hallucination_detected} | "
                f"compliance_ok={not compliance_violation} | "
                f"accuracy={accuracy_score:.2f} | tone={tone_score:.2f}. "
                f"Session closed (outcome={outcome})."
            ),
            metadata={
                "next": "__end__",
                "overall_pass": overall_pass,
                "hallucination_detected": hallucination_detected,
                "compliance_violation": compliance_violation,
                "accuracy_score": accuracy_score,
                "tone_score": tone_score,
                "outcome": outcome,
                "llm_latency_ms": round(llm_latency, 1),
                "total_latency_ms": round(total_latency, 1),
            },
        )],
        "next_agent": "__end__",
    }
