"""
Part 6: Tool Calling — Payment / Calculator Tool
Verifies payment history, computes dues, calculates settlement amounts.
"""
import json
from datetime import date, timedelta
from langchain.tools import tool


PAYMENT_LEDGER: dict[str, list] = {
    "LOAN001": [
        {"txn_id": "TXN001", "date": "2024-01-15", "amount": 4500, "mode": "UPI", "status": "cleared"},
        {"txn_id": "TXN002", "date": "2024-02-15", "amount": 4500, "mode": "NEFT", "status": "cleared"},
    ],
    "LOAN002": [
        {"txn_id": "TXN003", "date": "2024-03-15", "amount": 22000, "mode": "Auto-debit", "status": "cleared"},
        {"txn_id": "TXN004", "date": "2024-04-15", "amount": 22000, "mode": "Auto-debit", "status": "cleared"},
    ],
    "LOAN003": [
        {"txn_id": "UPI123456", "date": "2024-04-01", "amount": 12000, "mode": "UPI", "status": "unverified"},
    ],
}

INTEREST_RATE_ANNUAL = 0.18  # 18% p.a.


@tool
def verify_payment(loan_id: str, txn_id: str) -> str:
    """
    Verify if a specific payment transaction exists and is cleared.
    Args:
        loan_id: Loan identifier
        txn_id: Transaction/UTR reference number
    Returns:
        Payment verification status JSON
    """
    ledger = PAYMENT_LEDGER.get(loan_id.upper(), [])
    for txn in ledger:
        if txn["txn_id"].upper() == txn_id.upper():
            return json.dumps({"status": "found", "transaction": txn}, ensure_ascii=False)
    return json.dumps({"status": "not_found", "message": f"No transaction {txn_id} found for {loan_id}."})


@tool
def calculate_outstanding(loan_id: str, days_past_due: int, principal: float) -> str:
    """
    Calculate current outstanding amount including penalty interest.
    Args:
        loan_id: Loan identifier
        days_past_due: Number of days overdue
        principal: Original outstanding principal amount
    Returns:
        JSON with breakdown: principal, interest, penalty, total
    """
    daily_rate = INTEREST_RATE_ANNUAL / 365
    accrued_interest = round(principal * daily_rate * days_past_due, 2)
    penalty = round(principal * 0.02, 2) if days_past_due > 30 else 0.0
    total = round(principal + accrued_interest + penalty, 2)
    return json.dumps({
        "loan_id": loan_id,
        "principal": principal,
        "accrued_interest": accrued_interest,
        "penalty_charges": penalty,
        "total_outstanding": total,
        "calculated_on": date.today().isoformat(),
    }, ensure_ascii=False)


@tool
def calculate_settlement_offer(loan_id: str, outstanding: float, days_past_due: int) -> str:
    """
    Calculate eligible settlement/one-time payment discount.
    Args:
        loan_id: Loan identifier
        outstanding: Total outstanding amount
        days_past_due: Days overdue
    Returns:
        JSON with settlement tiers and waiver percentages
    """
    tiers = []
    if days_past_due < 30:
        tiers.append({"label": "Standard", "waiver_pct": 0, "settlement_amount": outstanding, "validity_days": 7})
    elif days_past_due < 90:
        waiver = 0.05
        tiers.append({"label": "Early Settlement", "waiver_pct": waiver * 100, "settlement_amount": round(outstanding * (1 - waiver), 2), "validity_days": 7})
    elif days_past_due < 180:
        waiver = 0.10
        tiers.append({"label": "Hardship Settlement", "waiver_pct": waiver * 100, "settlement_amount": round(outstanding * (1 - waiver), 2), "validity_days": 5})
        tiers.append({"label": "Extended EMI Plan", "waiver_pct": 5, "emi": round(outstanding / 24, 2), "tenure_months": 24, "validity_days": 3})
    else:
        waiver = 0.20
        tiers.append({"label": "Max Hardship Settlement", "waiver_pct": waiver * 100, "settlement_amount": round(outstanding * (1 - waiver), 2), "validity_days": 3})
        tiers.append({"label": "Long-term EMI", "waiver_pct": 10, "emi": round(outstanding * 0.90 / 36, 2), "tenure_months": 36, "validity_days": 3})

    return json.dumps({
        "loan_id": loan_id,
        "outstanding": outstanding,
        "settlement_options": tiers,
        "policy_ref": "POL-SETTLE-2024-001",
    }, ensure_ascii=False)


@tool
def get_payment_history(loan_id: str) -> str:
    """
    Retrieve full payment transaction history for a loan.
    Args:
        loan_id: Loan identifier
    Returns:
        JSON list of transactions
    """
    ledger = PAYMENT_LEDGER.get(loan_id.upper(), [])
    return json.dumps({"loan_id": loan_id, "transactions": ledger}, ensure_ascii=False)


PAYMENT_TOOLS = [verify_payment, calculate_outstanding, calculate_settlement_offer, get_payment_history]
