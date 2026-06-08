"""
Part 6: Tool Calling — Email / SMS / WhatsApp Communication Tool
Sends confirmation, reminders, and follow-up messages.
"""
import json
from datetime import datetime
from langchain.tools import tool


MESSAGE_LOG: list[dict] = []


@tool
def send_sms(phone: str, message: str, template_id: str = "CUSTOM") -> str:
    """
    Send an SMS to the customer (mock — logs the message).
    Args:
        phone: Customer phone number
        message: SMS text content
        template_id: DLT-registered template ID
    Returns:
        Delivery confirmation JSON
    """
    entry = {
        "channel": "SMS",
        "to": phone,
        "message": message,
        "template_id": template_id,
        "sent_at": datetime.now().isoformat(),
        "status": "delivered",
        "msg_id": f"SMS-{datetime.now().strftime('%Y%m%d%H%M%S')}",
    }
    MESSAGE_LOG.append(entry)
    return json.dumps({"status": "success", "delivery": entry}, ensure_ascii=False)


@tool
def send_email(to_email: str, subject: str, body: str) -> str:
    """
    Send a confirmation or reminder email to the customer.
    Args:
        to_email: Customer email address
        subject: Email subject
        body: Email body (plain text)
    Returns:
        Delivery confirmation JSON
    """
    entry = {
        "channel": "Email",
        "to": to_email,
        "subject": subject,
        "body": body,
        "sent_at": datetime.now().isoformat(),
        "status": "sent",
        "msg_id": f"EMAIL-{datetime.now().strftime('%Y%m%d%H%M%S')}",
    }
    MESSAGE_LOG.append(entry)
    return json.dumps({"status": "success", "delivery": entry}, ensure_ascii=False)


@tool
def send_whatsapp(phone: str, message: str) -> str:
    """
    Send a WhatsApp message to the customer via Business API.
    Args:
        phone: Customer phone number with country code
        message: Message content
    Returns:
        Delivery confirmation JSON
    """
    entry = {
        "channel": "WhatsApp",
        "to": phone,
        "message": message,
        "sent_at": datetime.now().isoformat(),
        "status": "delivered",
        "msg_id": f"WA-{datetime.now().strftime('%Y%m%d%H%M%S')}",
    }
    MESSAGE_LOG.append(entry)
    return json.dumps({"status": "success", "delivery": entry}, ensure_ascii=False)


def get_message_log() -> list:
    return MESSAGE_LOG


COMMUNICATION_TOOLS = [send_sms, send_email, send_whatsapp]


# ── Hindi SMS Templates ──────────────────────────────────────────────────────
SMS_TEMPLATES = {
    "ptp_confirmation": (
        "प्रिय {name}, आपके {loan_id} ऋण के लिए "
        "₹{amount} का भुगतान {date} तक देने का वादा दर्ज किया गया है। "
        "PTP ID: {ptp_id}। धन्यवाद। - CredResolve"
    ),
    "settlement_offer": (
        "प्रिय {name}, आपके ऋण {loan_id} पर एक विशेष "
        "समझौता प्रस्ताव उपलब्ध है। ₹{settlement_amount} में खाता बंद करें। "
        "प्रस्ताव {validity_days} दिनों तक मान्य। - CredResolve"
    ),
    "callback_scheduled": (
        "प्रिय {name}, हमारे वरिष्ठ अधिकारी {callback_time} पर "
        "आपसे संपर्क करेंगे। ऋण ID: {loan_id}। - CredResolve"
    ),
}
