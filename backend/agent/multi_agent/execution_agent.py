"""ExecutionAgent — handles TOOL_EXECUTION, RESOLUTION, ESCALATION.

Responsibilities:
  1. Select and call the right LangChain tools based on intent + escalation flag.
  2. Resolution path:  calculate_outstanding → log_ptp → send_sms
  3. Escalation path:  create_ticket → send_sms (callback_scheduled template)
  4. Settlement path:  calculate_settlement_offer → log_ptp → send_sms
  5. Dispute path:     verify_payment → update_customer_notes
  6. Aggregate all tool results into state.tool_results.

Agent-to-agent communication:
  Receives: intent, proposed_amount, proposed_date, escalation_required from DecisionAgent.
  Emits:    AgentMessage listing every tool call + result summary.
  Routes:   next_agent → "qa" (always — QAAgent validates before final response).
"""

import json
import time
from datetime import datetime


def _parse(raw) -> dict:
    """Parse a JSON string returned by a LangChain @tool into a dict."""
    if isinstance(raw, str):
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return {"raw": raw}
    return raw if isinstance(raw, dict) else {}

from backend.tools.crm_tool import log_ptp, update_customer_notes, fetch_customer_data
from backend.tools.payment_tool import (
    calculate_outstanding,
    calculate_settlement_offer,
    verify_payment,
)
from backend.tools.ticketing_tool import create_ticket
from backend.tools.communication_tool import send_sms
from backend.monitoring.metrics import AGENT_HANDOFFS, TOOL_CALLS
from .state import MultiAgentState, AgentMessage


def execution_agent(state: MultiAgentState) -> dict:
    """LangGraph node function — returns a partial state update."""
    t0 = time.time()

    intent = state.get("intent", "unknown")
    escalation_required = state.get("escalation_required", False)
    customer_data = state.get("customer_data", {})
    loan_id = state.get("loan_id", "")
    proposed_amount = state.get("proposed_amount", 0.0)
    proposed_date = state.get("proposed_date", "")

    loan_id = loan_id or customer_data.get("loan_id", "")
    phone = customer_data.get("phone", "")
    name = customer_data.get("name", "Customer")
    dpd = customer_data.get("days_past_due", 0)
    outstanding = customer_data.get("outstanding_amount", 0)

    tool_results = {}
    tool_call_log = []   # human-readable log for AgentMessage
    ptp_logged = False
    ptp_id = ""
    ticket_id = ""
    sms_sent = False

    # ══════════════════════════════════════════════════════════════════════════
    # ESCALATION PATH
    # ══════════════════════════════════════════════════════════════════════════
    if escalation_required:
        priority = "high" if intent == "angry" or dpd > 90 else "medium"
        description = (
            f"Customer {name} (Loan: {loan_id}, DPD: {dpd}) "
            f"escalated from AI DCO. Intent: {intent}. "
            f"Negotiation rounds exhausted or emotional distress detected."
        )
        ticket_result = _parse(create_ticket.invoke({
            "loan_id": loan_id,
            "category": "escalation",
            "priority": priority,
            "description": description,
            "assigned_to": "senior_dco",
        }))
        TOOL_CALLS.labels(tool_name="create_ticket", status="success").inc()
        ticket_id = ticket_result.get("ticket_id", "")
        tool_results["ticket"] = ticket_result
        tool_call_log.append(f"create_ticket → {ticket_id} (priority={priority})")

        if phone:
            sms_result = _parse(send_sms.invoke({
                "phone": phone,
                "message": (
                    f"प्रिय {name} ji, आपका case (Ticket: {ticket_id}) "
                    "senior DCO को transfer किया गया है। "
                    "24 घंटे में call back होगी।"
                ),
                "template_id": "callback_scheduled",
            }))
            TOOL_CALLS.labels(tool_name="send_sms", status="success").inc()
            tool_results["sms"] = sms_result
            sms_sent = True
            tool_call_log.append(f"send_sms (callback) → {phone}")

    # ══════════════════════════════════════════════════════════════════════════
    # RESOLUTION PATH — agreement on PTP or settlement
    # ══════════════════════════════════════════════════════════════════════════
    elif intent in ("delay", "cooperative") and proposed_amount > 0 and proposed_date:
        # Calculate final outstanding for accuracy
        calc_result = _parse(calculate_outstanding.invoke({
            "loan_id": loan_id,
            "days_past_due": dpd,
            "principal": outstanding,
        }))
        TOOL_CALLS.labels(tool_name="calculate_outstanding", status="success").inc()
        tool_results["outstanding"] = calc_result
        tool_call_log.append(
            f"calculate_outstanding → total=₹{calc_result.get('total_due', outstanding)}"
        )

        # Log PTP
        ptp_result = _parse(log_ptp.invoke({
            "loan_id": loan_id,
            "ptp_date": proposed_date,
            "ptp_amount": proposed_amount,
        }))
        TOOL_CALLS.labels(tool_name="log_ptp", status="success").inc()
        ptp_id = ptp_result.get("ptp_id", "")
        ptp_logged = True
        tool_results["ptp"] = ptp_result
        tool_call_log.append(f"log_ptp → {ptp_id} (₹{proposed_amount} by {proposed_date})")

        # Confirmation SMS
        if phone:
            sms_result = _parse(send_sms.invoke({
                "phone": phone,
                "message": (
                    f"प्रिय {name} ji, आपका PTP confirm हुआ। "
                    f"राशि ₹{proposed_amount} तारीख {proposed_date}। "
                    f"Reference: {ptp_id}। CredResolve"
                ),
                "template_id": "ptp_confirmation",
            }))
            TOOL_CALLS.labels(tool_name="send_sms", status="success").inc()
            tool_results["sms"] = sms_result
            sms_sent = True
            tool_call_log.append(f"send_sms (ptp_confirmation) → {phone}")

    # ══════════════════════════════════════════════════════════════════════════
    # SETTLEMENT PATH
    # ══════════════════════════════════════════════════════════════════════════
    elif intent == "settlement":
        settlement_result = _parse(calculate_settlement_offer.invoke({
            "loan_id": loan_id,
            "outstanding": outstanding,
            "days_past_due": dpd,
        }))
        TOOL_CALLS.labels(tool_name="calculate_settlement_offer", status="success").inc()
        tool_results["settlement"] = settlement_result
        tool_call_log.append(
            f"calculate_settlement_offer → "
            f"waiver={settlement_result.get('waiver_percent', 0)}% "
            f"amount=₹{settlement_result.get('settlement_amount', 0)}"
        )

        if proposed_amount > 0 and proposed_date:
            ptp_result = _parse(log_ptp.invoke({
                "loan_id": loan_id,
                "ptp_date": proposed_date,
                "ptp_amount": proposed_amount,
            }))
            TOOL_CALLS.labels(tool_name="log_ptp", status="success").inc()
            ptp_id = ptp_result.get("ptp_id", "")
            ptp_logged = True
            tool_results["ptp"] = ptp_result
            tool_call_log.append(f"log_ptp (settlement) → {ptp_id}")

            if phone:
                sms_result = _parse(send_sms.invoke({
                    "phone": phone,
                    "message": (
                        f"प्रिय {name} ji, Settlement offer: "
                        f"₹{proposed_amount} तारीख {proposed_date}। "
                        f"Ref: {ptp_id}। CredResolve"
                    ),
                    "template_id": "settlement_offer",
                }))
                TOOL_CALLS.labels(tool_name="send_sms", status="success").inc()
                tool_results["sms"] = sms_result
                sms_sent = True
                tool_call_log.append(f"send_sms (settlement_offer) → {phone}")

    # ══════════════════════════════════════════════════════════════════════════
    # DISPUTE PATH
    # ══════════════════════════════════════════════════════════════════════════
    elif intent == "dispute":
        txn_id = customer_data.get("last_txn_id", "")
        if txn_id:
            verify_result = _parse(verify_payment.invoke({
                "loan_id": loan_id,
                "txn_id": txn_id,
            }))
            TOOL_CALLS.labels(tool_name="verify_payment", status="success").inc()
            tool_results["payment_verify"] = verify_result
            tool_call_log.append(
                f"verify_payment → status={verify_result.get('status', 'unknown')}"
            )

    latency_ms = (time.time() - t0) * 1000
    AGENT_HANDOFFS.labels(from_agent="execution", to_agent="qa").inc()

    return {
        "tool_results": tool_results,
        "ptp_logged": ptp_logged,
        "ptp_id": ptp_id,
        "ticket_id": ticket_id,
        "sms_sent": sms_sent,
        "agent_messages": [AgentMessage(
            agent="execution",
            role="handoff",
            content=(
                f"Executed {len(tool_call_log)} tool(s): "
                + " | ".join(tool_call_log)
                if tool_call_log else "No tool calls required for this intent."
            ),
            metadata={
                "next": "qa",
                "intent": intent,
                "escalation": escalation_required,
                "ptp_logged": ptp_logged,
                "ptp_id": ptp_id,
                "ticket_id": ticket_id,
                "sms_sent": sms_sent,
                "tool_count": len(tool_call_log),
                "latency_ms": round(latency_ms, 1),
            },
        )],
        "next_agent": "qa",
    }
