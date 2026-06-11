"""
Twilio SMS + voice call tool.
Triggered by the Notification Agent when criticality = Extreme.
"""
from twilio.rest import Client
from config import settings
from models.input_models import EmergencyContact
from typing import List


def notify_emergency_contacts(
    contacts: List[EmergencyContact],
    emergency_type: str,
    severity: str,
    hospital_name: str,
) -> List[dict]:
    """
    Sends SMS + initiates voice call to all emergency contacts.
    Returns a list of notification result dicts.
    """
    client = Client(settings.twilio_account_sid, settings.twilio_auth_token)
    results = []

    sms_body = (
        f"🚨 EMERGENCY ALERT\n"
        f"Type: {emergency_type}\n"
        f"Severity: {severity}\n"
        f"Nearest Hospital: {hospital_name}\n"
        f"Please respond immediately."
    )

    for contact in contacts:
        sms_result = client.messages.create(
            body=sms_body,
            from_=settings.twilio_from_number,
            to=contact.phone,
        )

        call_result = client.calls.create(
            twiml=(
                f"<Response><Say voice='Polly.Joanna'>"
                f"Emergency Alert. {contact.name}, your contact has a {emergency_type} "
                f"with {severity} severity. Please call them or proceed to {hospital_name} immediately."
                f"</Say></Response>"
            ),
            from_=settings.twilio_from_number,
            to=contact.phone,
        )

        results.append({
            "name": contact.name,
            "phone": contact.phone,
            "sms_sid": sms_result.sid,
            "call_sid": call_result.sid,
            "action": "Call + SMS",
        })

    return results
