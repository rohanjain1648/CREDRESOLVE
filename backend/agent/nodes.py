"""
Part 1 + Part 9: LangGraph Node implementations
Each function handles one state in the state machine.
"""
import json
import time
import uuid
from datetime import date, timedelta
from typing import Any

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

from backend.agent.state_machine import ConversationState, AgentState, Intent
from backend.prompts.system_prompt import get_system_prompt
from backend.prompts.context_prompt import build_context_prompt
from backend.prompts.reasoning_prompt import get_intent_prompt, get_negotiation_prompt
from backend.prompts.evaluation_prompt import get_hallucination_check_prompt
from backend.tools.crm_tool import fetch_customer_data, log_ptp, update_customer_notes
from backend.tools.ticketing_tool import create_ticket
from backend.tools.payment_tool import (
    verify_payment, calculate_outstanding, calculate_settlement_offer
)
from backend.tools.communication_tool import send_sms, SMS_TEMPLATES
from backend.rag.retriever import retrieve_for_intent
from backend.memory.user_memory import (
    upsert_user_profile, log_interaction, build_memory_summary
)
from backend.memory.conversation_memory import log_transition
from backend.monitoring.metrics import (
    INTENT_DETECTED, TOOL_CALLS, STATE_TRANSITIONS,
    RESOLUTION_OUTCOME, ESCALATION_RATE, PTP_AMOUNT, record_llm_call
)
from backend.config import get_settings

settings = get_settings()


def _get_llm() -> ChatGoogleGenerativeAI:
    return ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        google_api_key=settings.google_api_key,
        max_output_tokens=1024,
        temperature=0.3,
    )


def _call_llm(messages: list, system: str = None) -> tuple[str, dict]:
    """Call LLM and return (text_response, usage_dict)."""
    llm = _get_llm()
    start = time.time()
    all_messages = []
    if system:
        all_messages.append(SystemMessage(content=system))
    all_messages.extend(messages)
    response = llm.invoke(all_messages)
    latency = (time.time() - start) * 1000

    usage = getattr(response, "usage_metadata", {}) or {}
    input_tokens = usage.get("input_tokens", 0)
    output_tokens = usage.get("output_tokens", 0)
    record_llm_call("gemini-2.0-flash", input_tokens, output_tokens, latency)

    return response.content, {"input_tokens": input_tokens, "output_tokens": output_tokens, "latency_ms": latency}


# ── 1. GREETING ───────────────────────────────────────────────────────────────
def greeting_node(state: ConversationState) -> ConversationState:
    STATE_TRANSITIONS.labels(from_state="start", to_state="greeting").inc()
    log_transition(state.get("session_id", ""), "start", "greeting")

    greeting_hi = (
        "नमस्ते! मैं आर्या हूं, CredResolve से। "
        "क्या मैं आपसे आपके ऋण खाते के बारे में बात कर सकती हूं? "
        "कृपया अपना Loan ID बताएं।"
    )
    greeting_en = (
        "Hello! I'm Arya from CredResolve. "
        "May I speak with you about your loan account? "
        "Please provide your Loan ID."
    )
    lang = state.get("language", "hi")
    text = greeting_hi if lang == "hi" else greeting_en

    return {
        **state,
        "current_state": AgentState.GREETING,
        "messages": [{"role": "assistant", "content": text}],
    }


# ── 2. AUTHENTICATION ─────────────────────────────────────────────────────────
def authentication_node(state: ConversationState) -> ConversationState:
    STATE_TRANSITIONS.labels(from_state="greeting", to_state="authentication").inc()

    last_msg = state.get("messages", [])[-1] if state.get("messages") else {}
    user_text = last_msg.get("content", "")

    # Extract loan ID from user message (simple heuristic for demo)
    loan_id = None
    for word in user_text.upper().split():
        if word.startswith("LOAN"):
            loan_id = word
            break

    if not loan_id:
        # Ask again
        text = "कृपया अपना Loan ID बताएं (जैसे LOAN001)।"
        return {
            **state,
            "current_state": AgentState.AUTHENTICATION,
            "authenticated": False,
            "messages": [{"role": "assistant", "content": text}],
        }

    # Verify via CRM tool
    try:
        result = json.loads(fetch_customer_data.invoke({"loan_id": loan_id}))
        TOOL_CALLS.labels(tool_name="fetch_customer_data", status="success").inc()
        if "error" in result:
            return {
                **state,
                "current_state": AgentState.AUTHENTICATION,
                "authenticated": False,
                "messages": [{"role": "assistant", "content": f"क्षमा करें, {loan_id} के लिए कोई रिकॉर्ड नहीं मिला।"}],
            }
        return {
            **state,
            "current_state": AgentState.AUTHENTICATION,
            "customer_id": result["customer_id"],
            "customer_data": result,
            "authenticated": True,
            "messages": [{"role": "assistant", "content": f"धन्यवाद! {result['name']} जी, आपकी पहचान सत्यापित हो गई।"}],
        }
    except Exception as e:
        TOOL_CALLS.labels(tool_name="fetch_customer_data", status="error").inc()
        return {**state, "authenticated": False, "escalation_required": True}


# ── 3. CONTEXT GATHERING ──────────────────────────────────────────────────────
def context_gathering_node(state: ConversationState) -> ConversationState:
    STATE_TRANSITIONS.labels(from_state="authentication", to_state="context_gathering").inc()

    customer = state.get("customer_data", {})
    memory_summary = build_memory_summary(customer.get("customer_id", ""))

    # Upsert user profile
    upsert_user_profile(
        customer_id=customer.get("customer_id", ""),
        loan_id=customer.get("loan_id", ""),
        name=customer.get("name", ""),
        preferred_language=state.get("language", "hi"),
    )

    outstanding = customer.get("outstanding_amount", 0)
    dpd = customer.get("days_past_due", 0)
    name = customer.get("name", "")

    text_hi = (
        f"{name} जी, आपके ऋण खाते की जानकारी: "
        f"बकाया राशि ₹{outstanding:,.0f} है और {dpd} दिन से देरी चल रही है। "
        f"क्या आप इस बारे में बात करना चाहेंगे?"
    )

    return {
        **state,
        "current_state": AgentState.CONTEXT_GATHERING,
        "messages": [{"role": "assistant", "content": text_hi}],
        "token_usage": state.get("token_usage", 0),
    }


# ── 4. DIAGNOSIS ──────────────────────────────────────────────────────────────
def diagnosis_node(state: ConversationState) -> ConversationState:
    STATE_TRANSITIONS.labels(from_state="context_gathering", to_state="diagnosis").inc()

    messages = state.get("messages", [])
    last_user_msg = next(
        (m["content"] for m in reversed(messages) if m["role"] == "user"), ""
    )

    intent_prompt = get_intent_prompt(
        last_user_msg,
        context_summary=f"DPD={state.get('customer_data', {}).get('days_past_due', 0)}",
    )

    system = get_system_prompt(state.get("language", "hi"))
    response, usage = _call_llm(
        [HumanMessage(content=intent_prompt)],
        system=system,
    )

    # Parse JSON response
    try:
        parsed = json.loads(response.strip().strip("```json").strip("```"))
        intent = parsed.get("intent", Intent.UNKNOWN)
        confidence = parsed.get("confidence", 0.5)
    except Exception:
        intent = Intent.UNKNOWN
        confidence = 0.5

    INTENT_DETECTED.labels(intent=intent).inc()
    STATE_TRANSITIONS.labels(from_state="diagnosis", to_state=intent).inc()

    return {
        **state,
        "current_state": AgentState.DIAGNOSIS,
        "intent": intent,
        "intent_confidence": confidence,
        "token_usage": state.get("token_usage", 0) + usage.get("input_tokens", 0) + usage.get("output_tokens", 0),
    }


# ── 5. TOOL EXECUTION ─────────────────────────────────────────────────────────
def tool_execution_node(state: ConversationState) -> ConversationState:
    STATE_TRANSITIONS.labels(from_state="diagnosis", to_state="tool_execution").inc()

    customer = state.get("customer_data", {})
    intent = state.get("intent", "")
    tool_results = {}

    if intent == Intent.DISPUTE:
        # Verify payment claim
        messages = state.get("messages", [])
        last_user = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")
        # Extract UTR from message (heuristic)
        utr = next((w for w in last_user.split() if len(w) > 6 and w.isalnum()), "UNKNOWN")
        result = verify_payment.invoke({"loan_id": customer.get("loan_id", ""), "txn_id": utr})
        TOOL_CALLS.labels(tool_name="verify_payment", status="success").inc()
        tool_results["payment_verification"] = json.loads(result)

    if intent in [Intent.DELAY, Intent.SETTLEMENT, Intent.COOPERATIVE]:
        # Calculate outstanding with interest
        result = calculate_outstanding.invoke({
            "loan_id": customer.get("loan_id", ""),
            "days_past_due": customer.get("days_past_due", 0),
            "principal": customer.get("outstanding_amount", 0),
        })
        TOOL_CALLS.labels(tool_name="calculate_outstanding", status="success").inc()
        tool_results["outstanding"] = json.loads(result)

    if intent == Intent.SETTLEMENT:
        # Get settlement options
        result = calculate_settlement_offer.invoke({
            "loan_id": customer.get("loan_id", ""),
            "outstanding": customer.get("outstanding_amount", 0),
            "days_past_due": customer.get("days_past_due", 0),
        })
        TOOL_CALLS.labels(tool_name="calculate_settlement_offer", status="success").inc()
        tool_results["settlement_options"] = json.loads(result)

    return {
        **state,
        "current_state": AgentState.TOOL_EXECUTION,
        "tool_results": tool_results,
        "tool_call_count": state.get("tool_call_count", 0) + len(tool_results),
    }


# ── 6. KNOWLEDGE RETRIEVAL ────────────────────────────────────────────────────
def knowledge_retrieval_node(state: ConversationState) -> ConversationState:
    STATE_TRANSITIONS.labels(from_state="tool_execution", to_state="knowledge_retrieval").inc()

    intent = state.get("intent", "unknown")
    customer = state.get("customer_data", {})
    context_snippet = f"DPD={customer.get('days_past_due', 0)}, outstanding={customer.get('outstanding_amount', 0)}"

    rag_result = retrieve_for_intent(intent, context_snippet)

    return {
        **state,
        "current_state": AgentState.KNOWLEDGE_RETRIEVAL,
        "retrieved_policies": rag_result["documents"],
        "retrieved_sources": rag_result["sources"],
    }


# ── 7. NEGOTIATION ────────────────────────────────────────────────────────────
def negotiation_node(state: ConversationState) -> ConversationState:
    STATE_TRANSITIONS.labels(from_state="knowledge_retrieval", to_state="negotiation").inc()

    customer = state.get("customer_data", {})
    intent = state.get("intent", "unknown")
    tool_results = state.get("tool_results", {})
    policies = state.get("retrieved_policies", [])
    negotiation_round = state.get("negotiation_rounds", 0) + 1

    # Build negotiation prompt
    neg_prompt = get_negotiation_prompt(
        intent=intent,
        outstanding_amount=customer.get("outstanding_amount", 0),
        days_past_due=customer.get("days_past_due", 0),
        policy_options="\n".join(policies) if policies else "No policies retrieved.",
        negotiation_round=negotiation_round,
    )

    context_prompt = build_context_prompt(
        customer_data=customer,
        retrieved_policies=policies,
        language=state.get("language", "hi"),
    )

    system = get_system_prompt(state.get("language", "hi"))
    full_prompt = f"{context_prompt}\n\n{neg_prompt}"

    # Add tool result context
    if tool_results.get("settlement_options"):
        opts = tool_results["settlement_options"].get("settlement_options", [])
        opts_text = "\n".join(
            f"- {o['label']}: ₹{o.get('settlement_amount', o.get('emi', 'N/A'))} "
            f"({o['waiver_pct']}% छूट, {o['validity_days']} दिन मान्य)"
            for o in opts
        )
        full_prompt += f"\n\nउपलब्ध समझौता विकल्प:\n{opts_text}"

    response, usage = _call_llm(
        [HumanMessage(content=full_prompt)],
        system=system,
    )

    # Parse JSON negotiation response
    escalate = False
    proposed_amount = None
    proposed_date = None
    try:
        parsed = json.loads(response.strip().strip("```json").strip("```"))
        response_text = parsed.get("hindi_pitch", response)
        proposed_amount = parsed.get("proposed_amount")
        proposed_date = parsed.get("proposed_date")
        escalate = bool(parsed.get("escalate_reason"))
    except Exception:
        response_text = response

    return {
        **state,
        "current_state": AgentState.NEGOTIATION,
        "negotiation_rounds": negotiation_round,
        "proposed_amount": proposed_amount,
        "proposed_date": proposed_date,
        "escalation_required": escalate or (negotiation_round >= 3),
        "messages": [{"role": "assistant", "content": response_text}],
        "token_usage": state.get("token_usage", 0) + usage.get("input_tokens", 0) + usage.get("output_tokens", 0),
    }


# ── 8. ESCALATION ─────────────────────────────────────────────────────────────
def escalation_node(state: ConversationState) -> ConversationState:
    STATE_TRANSITIONS.labels(from_state="negotiation", to_state="escalation").inc()

    customer = state.get("customer_data", {})
    intent = state.get("intent", "unknown")

    # Create escalation ticket
    reason = f"Intent={intent}, DPD={customer.get('days_past_due', 0)}, Rounds={state.get('negotiation_rounds', 0)}"
    ticket_result = json.loads(create_ticket.invoke({
        "loan_id": customer.get("loan_id", ""),
        "category": "escalation",
        "priority": "high" if customer.get("days_past_due", 0) > 90 else "medium",
        "description": reason,
        "assigned_to": "senior_dco",
    }))
    TOOL_CALLS.labels(tool_name="create_ticket", status="success").inc()
    ESCALATION_RATE.labels(reason=intent).inc()

    text_hi = (
        "आपकी बात समझ आई। मैं आपका मामला अपने वरिष्ठ अधिकारी को भेज रही हूं। "
        "वे आपसे 24 घंटों के भीतर संपर्क करेंगे। "
        f"आपका टिकट नंबर है: {ticket_result.get('ticket_id', 'N/A')}। "
        "आपकी सहायता के लिए धन्यवाद।"
    )

    return {
        **state,
        "current_state": AgentState.ESCALATION,
        "resolution_outcome": "escalated",
        "messages": [{"role": "assistant", "content": text_hi}],
        "tool_results": {**state.get("tool_results", {}), "escalation_ticket": ticket_result},
    }


# ── 9. RESOLUTION ─────────────────────────────────────────────────────────────
def resolution_node(state: ConversationState) -> ConversationState:
    STATE_TRANSITIONS.labels(from_state="negotiation", to_state="resolution").inc()

    customer = state.get("customer_data", {})
    proposed_date = state.get("proposed_date") or (date.today() + timedelta(days=7)).isoformat()
    proposed_amount = state.get("proposed_amount") or customer.get("outstanding_amount", 0)
    intent = state.get("intent", "cooperative")

    # Log PTP in CRM
    ptp_result = json.loads(log_ptp.invoke({
        "loan_id": customer.get("loan_id", ""),
        "ptp_date": proposed_date,
        "ptp_amount": proposed_amount,
    }))
    TOOL_CALLS.labels(tool_name="log_ptp", status="success").inc()
    PTP_AMOUNT.observe(proposed_amount)

    ptp_id = ptp_result.get("ptp_id", "N/A")
    outcome = "settlement_agreed" if intent == "settlement" else "ptp_logged"
    RESOLUTION_OUTCOME.labels(outcome=outcome).inc()

    # Send SMS confirmation
    sms_text = SMS_TEMPLATES["ptp_confirmation"].format(
        name=customer.get("name", ""),
        loan_id=customer.get("loan_id", ""),
        amount=f"{proposed_amount:,.0f}",
        date=proposed_date,
        ptp_id=ptp_id,
    )
    send_sms.invoke({"phone": customer.get("phone", ""), "message": sms_text})

    text_hi = (
        f"बहुत अच्छा! आपका भुगतान वचन (PTP) दर्ज कर दिया गया है। "
        f"₹{proposed_amount:,.0f} का भुगतान {proposed_date} तक। "
        f"PTP ID: {ptp_id}। "
        "आपके पंजीकृत मोबाइल पर SMS भेज दिया गया है।"
    )

    return {
        **state,
        "current_state": AgentState.RESOLUTION,
        "ptp_logged": True,
        "ptp_date": proposed_date,
        "ptp_amount": proposed_amount,
        "resolution_outcome": outcome,
        "messages": [{"role": "assistant", "content": text_hi}],
    }


# ── 10. FOLLOW-UP ─────────────────────────────────────────────────────────────
def follow_up_node(state: ConversationState) -> ConversationState:
    STATE_TRANSITIONS.labels(from_state="resolution", to_state="follow_up").inc()

    customer = state.get("customer_data", {})

    # Log to interaction history
    log_interaction(
        session_id=state.get("session_id", ""),
        customer_id=customer.get("customer_id", ""),
        loan_id=customer.get("loan_id", ""),
        intent=state.get("intent", "unknown"),
        outcome=state.get("resolution_outcome", "unknown"),
        agent_response_summary=f"PTP={state.get('ptp_logged', False)}, Amount={state.get('ptp_amount')}",
        ptp_date=state.get("ptp_date"),
        ptp_amount=state.get("ptp_amount"),
        escalated=state.get("escalation_required", False),
    )

    # Update CRM notes
    note = (
        f"Session {state.get('session_id', '')} | "
        f"Intent={state.get('intent')} | "
        f"Outcome={state.get('resolution_outcome')} | "
        f"PTP={state.get('ptp_date')}"
    )
    update_customer_notes.invoke({"loan_id": customer.get("loan_id", ""), "note": note})

    text_hi = (
        "आपसे बात करके अच्छा लगा। यदि आपको कोई सहायता चाहिए, "
        "तो हमारे helpline 1800-XXX-XXXX पर कॉल करें। "
        "धन्यवाद और आपका दिन शुभ हो!"
    )

    return {
        **state,
        "current_state": AgentState.FOLLOW_UP,
        "messages": [{"role": "assistant", "content": text_hi}],
    }
