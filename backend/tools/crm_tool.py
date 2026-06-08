"""
Part 6: Tool Calling — CRM Tool
Fetches customer loan and payment data.
"""
import json
from datetime import date, timedelta
from langchain.tools import tool


MOCK_CRM_DB = {
    "LOAN001": {
        "customer_id": "CUST001",
        "name": "Rajesh Kumar",
        "phone": "+91-9876543210",
        "loan_id": "LOAN001",
        "loan_type": "Personal Loan",
        "principal": 150000,
        "outstanding_amount": 87500,
        "emi_amount": 4500,
        "due_date": (date.today() + timedelta(days=5)).isoformat(),
        "days_past_due": 45,
        "last_contact": (date.today() - timedelta(days=15)).isoformat(),
        "last_outcome": "No response",
        "risk_category": "Medium",
        "payment_history": [
            {"date": "2024-01-15", "amount": 4500, "status": "paid"},
            {"date": "2024-02-15", "amount": 4500, "status": "paid"},
            {"date": "2024-03-15", "amount": 0, "status": "missed"},
            {"date": "2024-04-15", "amount": 0, "status": "missed"},
        ],
        "ptp_history": [],
        "notes": "Customer cited job loss in March 2024.",
    },
    "LOAN002": {
        "customer_id": "CUST002",
        "name": "Priya Sharma",
        "phone": "+91-9123456789",
        "loan_id": "LOAN002",
        "loan_type": "Home Loan",
        "principal": 2500000,
        "outstanding_amount": 1950000,
        "emi_amount": 22000,
        "due_date": (date.today() - timedelta(days=3)).isoformat(),
        "days_past_due": 3,
        "last_contact": (date.today() - timedelta(days=3)).isoformat(),
        "last_outcome": "Payment promised by end of week",
        "risk_category": "Low",
        "payment_history": [
            {"date": "2024-03-15", "amount": 22000, "status": "paid"},
            {"date": "2024-04-15", "amount": 22000, "status": "paid"},
        ],
        "ptp_history": [],
        "notes": "",
    },
    "LOAN003": {
        "customer_id": "CUST003",
        "name": "Mohammed Ali",
        "phone": "+91-9988776655",
        "loan_id": "LOAN003",
        "loan_type": "Business Loan",
        "principal": 500000,
        "outstanding_amount": 320000,
        "emi_amount": 12000,
        "due_date": (date.today() - timedelta(days=95)).isoformat(),
        "days_past_due": 95,
        "last_contact": (date.today() - timedelta(days=30)).isoformat(),
        "last_outcome": "Dispute raised — claims payment made",
        "risk_category": "High",
        "payment_history": [],
        "ptp_history": [],
        "notes": "Customer claims payment via UPI on April 1. Reference: UPI123456.",
    },
}


@tool
def fetch_customer_data(loan_id: str) -> str:
    """
    Fetch complete customer loan and payment history from CRM.
    Args:
        loan_id: The unique loan identifier (e.g., LOAN001)
    Returns:
        JSON string with customer data or error message
    """
    data = MOCK_CRM_DB.get(loan_id.upper())
    if not data:
        return json.dumps({"error": f"No record found for loan_id: {loan_id}"})
    return json.dumps(data, ensure_ascii=False)


@tool
def update_customer_notes(loan_id: str, note: str) -> str:
    """
    Append a note to the customer's CRM record.
    Args:
        loan_id: Loan identifier
        note: Note text to append (include date and outcome)
    Returns:
        Confirmation message
    """
    if loan_id.upper() in MOCK_CRM_DB:
        MOCK_CRM_DB[loan_id.upper()]["notes"] += f"\n[{date.today()}] {note}"
        return json.dumps({"status": "success", "message": "Note appended to CRM."})
    return json.dumps({"status": "error", "message": "Loan not found."})


@tool
def log_ptp(loan_id: str, ptp_date: str, ptp_amount: float) -> str:
    """
    Log a Promise-to-Pay (PTP) commitment in CRM.
    Args:
        loan_id: Loan identifier
        ptp_date: Date borrower promised to pay (YYYY-MM-DD)
        ptp_amount: Amount promised
    Returns:
        Confirmation with PTP ID
    """
    if loan_id.upper() not in MOCK_CRM_DB:
        return json.dumps({"status": "error", "message": "Loan not found."})
    entry = {
        "ptp_id": f"PTP{date.today().strftime('%Y%m%d')}{loan_id[-3:]}",
        "date_logged": date.today().isoformat(),
        "ptp_date": ptp_date,
        "ptp_amount": ptp_amount,
        "status": "active",
    }
    MOCK_CRM_DB[loan_id.upper()]["ptp_history"].append(entry)
    return json.dumps(
        {"status": "success", "ptp_id": entry["ptp_id"], "details": entry},
        ensure_ascii=False,
    )


CRM_TOOLS = [fetch_customer_data, update_customer_notes, log_ptp]
