"""
Part 6: Tool Calling — Ticketing System
Creates and updates support/escalation tickets.
"""
import json
import uuid
from datetime import datetime
from langchain.tools import tool


TICKET_DB: dict[str, dict] = {}


@tool
def create_ticket(
    loan_id: str,
    category: str,
    priority: str,
    description: str,
    assigned_to: str = "supervisor",
) -> str:
    """
    Create a new support or escalation ticket.
    Args:
        loan_id: Related loan identifier
        category: 'escalation' | 'dispute' | 'callback' | 'complaint'
        priority: 'low' | 'medium' | 'high' | 'critical'
        description: Detailed description of the issue
        assigned_to: Team or person to assign (default: supervisor)
    Returns:
        JSON with ticket_id and details
    """
    ticket_id = f"TKT-{uuid.uuid4().hex[:8].upper()}"
    ticket = {
        "ticket_id": ticket_id,
        "loan_id": loan_id,
        "category": category,
        "priority": priority,
        "description": description,
        "assigned_to": assigned_to,
        "status": "open",
        "created_at": datetime.now().isoformat(),
        "updated_at": datetime.now().isoformat(),
        "comments": [],
    }
    TICKET_DB[ticket_id] = ticket
    return json.dumps(
        {"status": "success", "ticket_id": ticket_id, "ticket": ticket},
        ensure_ascii=False,
    )


@tool
def update_ticket(ticket_id: str, status: str, comment: str = "") -> str:
    """
    Update an existing ticket's status and add a comment.
    Args:
        ticket_id: Ticket identifier (e.g., TKT-ABC12345)
        status: 'open' | 'in_progress' | 'resolved' | 'closed'
        comment: Optional comment to add
    Returns:
        Updated ticket JSON
    """
    if ticket_id not in TICKET_DB:
        return json.dumps({"status": "error", "message": "Ticket not found."})
    TICKET_DB[ticket_id]["status"] = status
    TICKET_DB[ticket_id]["updated_at"] = datetime.now().isoformat()
    if comment:
        TICKET_DB[ticket_id]["comments"].append(
            {"timestamp": datetime.now().isoformat(), "text": comment}
        )
    return json.dumps(
        {"status": "success", "ticket": TICKET_DB[ticket_id]},
        ensure_ascii=False,
    )


@tool
def get_ticket(ticket_id: str) -> str:
    """
    Retrieve an existing ticket by ID.
    Args:
        ticket_id: Ticket identifier
    Returns:
        Ticket details JSON
    """
    ticket = TICKET_DB.get(ticket_id)
    if not ticket:
        return json.dumps({"status": "error", "message": "Ticket not found."})
    return json.dumps({"status": "success", "ticket": ticket}, ensure_ascii=False)


TICKETING_TOOLS = [create_ticket, update_ticket, get_ticket]
