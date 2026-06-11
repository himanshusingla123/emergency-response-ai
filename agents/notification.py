"""
Notification Agent — deterministic rule engine, not LLM.
Calls Twilio when criticality = Extreme.
"""
from typing import List
from models.input_models import EmergencyContact
from tools.twilio_notifier import notify_emergency_contacts


async def run_notifications(
    contacts: List[EmergencyContact],
    emergency_type: str,
    severity: str,
    hospital_name: str,
) -> dict:
    if not contacts:
        return {"notify": False, "contacts_notified": []}

    notified = notify_emergency_contacts(
        contacts=contacts,
        emergency_type=emergency_type,
        severity=severity,
        hospital_name=hospital_name,
    )
    return {"notify": True, "contacts_notified": notified}
