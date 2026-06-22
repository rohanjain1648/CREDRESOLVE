"""DecisionAgent — handles DIAGNOSIS and NEGOTIATION.

Responsibilities (two sub-phases in one agent):

  Phase A — DIAGNOSIS (first visit, no intent yet):
    1. Call Gemini with structured JSON prompt to classify borrower intent.
    2. Set intent, confidence, and route to RetrievalAgent for policy lookup.

  Phase B — NEGOTIATION (after RetrievalAgent has returned policies):
    1. Build a full Hindi prompt from customer data + policies + memory.
    2. Call Gemini for negotiation response (JSON with hindi_pitch, amounts, dates).
    3. Decide whether to escalate or continue.
    4. Route to ExecutionAgent (agreement) or QAAgent (escalation).

Agent-to-agent communication:
  Phase A emits intent classification result → routes to "retrieval".
  Phase B emits negotiation pitch + routing decision → "execution" or "qa".
"""

import json
import re
import time

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage

from backend.config import get_settings
from backend.prompts.reasoning_prompt import INTENT_CLASSIFICATION_PROMPT, NEGOTIATION_REASONING_PROMPT
from backend.monitoring.metrics import AGENT_HANDOFFS, LLM_LATENCY, TOKEN_USAGE
from .state import MultiAgentState, AgentMessage

settings = get_settings()

MAX_NEGOTIATION_ROUNDS = 3


def _llm() -> ChatGoogleGenerativeAI:
    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash-lite",
        google_api_key=settings.google_api_key,
        max_output_tokens=1024,
        temperature=0.3,
    )


def _call_gemini(system: str, user: str) -> tuple[str, dict]:
    llm = _llm()
    t0 = time.time()
    response = llm.invoke([SystemMessage(content=system), HumanMessage(content=user)])
    latency = (time.time() - t0) * 1000
    usage = getattr(response, "usage_metadata", {}) or {}
    LLM_LATENCY.labels(model="gemini-2.5-flash", node="decision_agent").observe(latency / 1000)
    TOKEN_USAGE.labels(model="gemini-2.5-flash", direction="input").inc(usage.get("input_tokens", 0))
    TOKEN_USAGE.labels(model="gemini-2.5-flash", direction="output").inc(usage.get("output_tokens", 0))
    return response.content, {"latency_ms": round(latency, 1), **usage}


def _extract_json(text: str) -> dict:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    return {}


def decision_agent(state: MultiAgentState) -> dict:
    """LangGraph node function — returns a partial state update."""
    intent = state.get("intent", "")
    retrieved_policies = state.get("retrieved_policies")
    customer_data = state.get("customer_data", {})
    memory_summary = state.get("memory_summary", "No prior history.")
    customer_message = state.get("customer_message", "")
    negotiation_round = state.get("negotiation_round", 0)

    # ══════════════════════════════════════════════════════════════════════════
    # PHASE A — DIAGNOSIS  (run when intent is not yet classified)
    # ══════════════════════════════════════════════════════════════════════════
    if not intent:
        user_prompt = (
            f"Customer message: {customer_message}\n"
            f"Customer DPD: {customer_data.get('days_past_due', 0)}\n"
            f"Outstanding: ₹{customer_data.get('outstanding_amount', 0)}\n"
            f"Memory: {memory_summary}"
        )
        raw, usage = _call_gemini(INTENT_CLASSIFICATION_PROMPT, user_prompt)
        parsed = _extract_json(raw)

        intent = parsed.get("intent", "unknown")
        confidence = float(parsed.get("confidence", 0.5))

        AGENT_HANDOFFS.labels(from_agent="decision", to_agent="retrieval").inc()

        return {
            "intent": intent,
            "intent_confidence": confidence,
            "agent_messages": [AgentMessage(
                agent="decision",
                role="handoff",
                content=(
                    f"DIAGNOSIS complete → intent='{intent}' "
                    f"confidence={confidence:.2f}. "
                    f"Handing off to RetrievalAgent for policy lookup."
                ),
                metadata={
                    "next": "retrieval",
                    "intent": intent,
                    "confidence": confidence,
                    "raw_llm_output": raw[:200],
                    **usage,
                },
            )],
            "next_agent": "retrieval",
        }

    # ══════════════════════════════════════════════════════════════════════════
    # PHASE B — NEGOTIATION  (run after RetrievalAgent has returned policies)
    # ══════════════════════════════════════════════════════════════════════════

    # Direct escalation for angry/auth-failed intents
    if intent == "angry" or state.get("auth_attempts", 0) >= 3:
        AGENT_HANDOFFS.labels(from_agent="decision", to_agent="qa").inc()
        escalation_msg = (
            "मैं समझ सकती हूँ कि आप परेशान हैं। "
            "मैं आपका case तुरंत senior DCO को transfer कर रही हूँ।"
        )
        return {
            "negotiation_round": negotiation_round + 1,
            "escalation_required": True,
            "negotiation_response": escalation_msg,
            "agent_messages": [AgentMessage(
                agent="decision",
                role="handoff",
                content="Intent=angry — immediate escalation. Routing to ExecutionAgent for ticket creation.",
                metadata={"next": "execution", "escalate": True},
            )],
            "next_agent": "execution",
        }

    # Build policy context from RetrievalAgent output
    policy_context = "\n".join(
        f"[{p.get('id', '?')}] {p.get('text', '')}"
        for p in (retrieved_policies or [])
    ) or "No specific policies retrieved."

    user_prompt = (
        f"Intent: {intent} | Round: {negotiation_round + 1}/{MAX_NEGOTIATION_ROUNDS}\n"
        f"Customer: {customer_data.get('name', 'Customer')} | "
        f"DPD: {customer_data.get('days_past_due', 0)} | "
        f"Outstanding: ₹{customer_data.get('outstanding_amount', 0)}\n"
        f"Memory: {memory_summary}\n"
        f"Policies:\n{policy_context}\n"
        f"Customer message: {customer_message}"
    )

    raw, usage = _call_gemini(NEGOTIATION_REASONING_PROMPT, user_prompt)
    parsed = _extract_json(raw)

    hindi_pitch = parsed.get("hindi_pitch", raw[:400])
    proposed_amount = float(parsed.get("proposed_amount", 0) or 0)
    proposed_date = parsed.get("proposed_date", "")
    should_escalate = bool(parsed.get("escalate", False)) or (negotiation_round + 1 >= MAX_NEGOTIATION_ROUNDS)

    next_agent = "execution" if not should_escalate else "execution"

    AGENT_HANDOFFS.labels(from_agent="decision", to_agent="execution").inc()

    return {
        "negotiation_round": negotiation_round + 1,
        "proposed_amount": proposed_amount,
        "proposed_date": proposed_date,
        "escalation_required": should_escalate,
        "negotiation_response": hindi_pitch,
        "agent_messages": [AgentMessage(
            agent="decision",
            role="handoff",
            content=(
                f"NEGOTIATION round {negotiation_round + 1}: "
                f"proposed ₹{proposed_amount} by {proposed_date or 'TBD'}. "
                f"Escalate={should_escalate}. Routing to ExecutionAgent."
            ),
            metadata={
                "next": "execution",
                "round": negotiation_round + 1,
                "proposed_amount": proposed_amount,
                "proposed_date": proposed_date,
                "escalate": should_escalate,
                **usage,
            },
        )],
        "next_agent": "execution",
    }
