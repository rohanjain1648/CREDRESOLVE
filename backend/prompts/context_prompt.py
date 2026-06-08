"""
Part 3: Context Prompt — injects customer data + retrieved policies
"""
from typing import Optional


CONTEXT_TEMPLATE_HI = """
## ग्राहक जानकारी
- नाम: {customer_name}
- ऋण ID: {loan_id}
- बकाया राशि: ₹{outstanding_amount}
- देरी के दिन: {days_past_due} दिन
- पिछली EMI: ₹{emi_amount} (देय तिथि: {due_date})
- पिछला संपर्क: {last_contact}
- पिछला परिणाम: {last_outcome}

## पुनः प्राप्त नीतियां (RAG)
{retrieved_policies}

## बातचीत इतिहास सारांश
{conversation_summary}
"""

CONTEXT_TEMPLATE_EN = """
## Customer Profile
- Name: {customer_name}
- Loan ID: {loan_id}
- Outstanding: ₹{outstanding_amount}
- Days Past Due: {days_past_due}
- EMI Amount: ₹{emi_amount} (Due: {due_date})
- Last Contact: {last_contact}
- Last Outcome: {last_outcome}

## Retrieved Policies (RAG)
{retrieved_policies}

## Conversation Summary
{conversation_summary}
"""


def build_context_prompt(
    customer_data: dict,
    retrieved_policies: Optional[list] = None,
    conversation_summary: str = "First interaction",
    language: str = "hi",
) -> str:
    policies_text = "\n".join(
        f"- {p}" for p in (retrieved_policies or ["कोई नीति नहीं मिली।"])
    )
    template = CONTEXT_TEMPLATE_HI if language == "hi" else CONTEXT_TEMPLATE_EN
    return template.format(
        customer_name=customer_data.get("name", "अज्ञात"),
        loan_id=customer_data.get("loan_id", "N/A"),
        outstanding_amount=customer_data.get("outstanding_amount", 0),
        days_past_due=customer_data.get("days_past_due", 0),
        emi_amount=customer_data.get("emi_amount", 0),
        due_date=customer_data.get("due_date", "N/A"),
        last_contact=customer_data.get("last_contact", "पहली बार"),
        last_outcome=customer_data.get("last_outcome", "N/A"),
        retrieved_policies=policies_text,
        conversation_summary=conversation_summary,
    )
