"""
agents/final_response.py

Final Response Agent — aggregates outputs from all upstream agents into a
single, structured EmergencyResponse object that the FastAPI layer returns
to the client (REST) or streams over WebSocket.

Reads from ADK session state keys:
    detection_data      → agents/emergency_detection.py
    severity_data       → agents/severity.py
    criticality_data    → agents/criticality_engine.py
    location_data       → agents/location.py
    resource_data       → agents/resource.py
    recommendation_data → agents/recommendation.py
    notification_data   → agents/notification.py
"""

import json
import logging
from datetime import datetime, timezone
from typing import Any
from google.adk.agents import LlmAgent
from config import settings
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

FINAL_RESPONSE_PROMPT = """
You are the Final Response Aggregator in a medical emergency response system.

You receive the complete outputs from every specialist agent and your job is to
compile them into ONE clean, structured JSON response for the user.

## Inputs available to you (from session state)
- detection_data      : emergency type, classification, confidence
- severity_data       : severity score (0-100), urgency label
- criticality_data    : hybrid criticality score, rule triggers
- location_data       : resolved address, coordinates
- resource_data       : nearest hospitals, ambulance contacts
- recommendation_data : immediate actions, do's & don'ts
- notification_data   : SMS/call status, contacts notified

## Output Format  (return ONLY this JSON, nothing else)
{
  "response_id": "<uuid-v4>",
  "timestamp": "<ISO-8601 UTC>",
  "status": "success" | "partial" | "failed",

  "emergency": {
    "type": "<e.g. Cardiac Emergency>",
    "description": "<one-line summary>",
    "confidence": <0.0–1.0>,
    "is_emergency": true | false
  },

  "severity": {
    "score": <0–100>,
    "level": "Low" | "Medium" | "High" | "Critical",
    "reasoning": "<brief explanation>"
  },

  "criticality": {
    "score": <0–100>,
    "level": "Low" | "Medium" | "High" | "Critical",
    "rule_triggers": ["<rule1>", "<rule2>"],
    "llm_assessment": "<brief LLM reasoning>"
  },

  "location": {
    "resolved": true | false,
    "formatted_address": "<full address>",
    "city": "<city>",
    "state": "<state>",
    "coordinates": { "lat": <float>, "lng": <float> }
  },

  "resources": {
    "nearest_hospital": {
      "name": "<hospital name>",
      "address": "<address>",
      "phone": "<phone>",
      "distance_km": <float>,
      "google_maps_url": "<url>"
    },
    "other_hospitals": [
      {
        "name": "<name>",
        "address": "<address>",
        "distance_km": <float>
      }
    ],
    "emergency_number": "112"
  },

  "recommendations": {
    "immediate_actions": ["<step 1>", "<step 2>"],
    "dos": ["<do 1>", "<do 2>"],
    "donts": ["<don't 1>", "<don't 2>"],
    "call_ambulance": true | false
  },

  "notifications": {
    "sent": true | false,
    "channels": ["sms", "call"],
    "contacts_notified": ["<name or number>"],
    "message_preview": "<first 100 chars of SMS sent>"
  },

  "agent_pipeline": {
    "agents_executed": ["detection", "severity", "criticality",
                        "location", "resource", "recommendation", "notification"],
    "execution_time_ms": <int>,
    "any_agent_failed": false
  },

  "display": {
    "alert_color": "green" | "yellow" | "orange" | "red",
    "banner_message": "<short message for UI banner>",
    "show_ambulance_button": true | false,
    "show_hospital_map": true | false
  }
}

## Rules
1. If severity level is Critical or High → alert_color = "red", show_ambulance_button = true.
2. If severity level is Medium → alert_color = "orange".
3. If severity level is Low → alert_color = "yellow".
4. If is_emergency is false → alert_color = "green", banner_message = "No emergency detected."
5. show_hospital_map = true whenever location.resolved is true AND resources.nearest_hospital exists.
6. If any agent failed (its data is null or contains an error key), set status = "partial" and
   any_agent_failed = true; only set status = "failed" if BOTH detection and severity failed.
7. Never expose raw API keys, internal tracebacks, or personal data beyond what is listed above.
""".strip()


# ---------------------------------------------------------------------------
# Agent factory
# ---------------------------------------------------------------------------

def create_final_response_agent() -> LlmAgent:
    """Return the configured Final Response LlmAgent."""
    return LlmAgent(
        name="final_response_agent",
        model="gemini-2.5-flash",
        description=(
            "Aggregates outputs from all emergency-response agents into a single "
            "structured JSON response for the API layer."
        ),
        instruction=FINAL_RESPONSE_PROMPT,
        # No tools needed — this agent only reads session state and reasons.
        tools=[],
        output_key="final_response",
    )


# ---------------------------------------------------------------------------
# Standalone aggregator (used by coordinator without full agent runner)
# ---------------------------------------------------------------------------

def aggregate_final_response(
    session_state: dict[str, Any],
    execution_time_ms: int = 0,
    response_id: str | None = None,
) -> dict:
    """
    Pure-Python fallback aggregator.  The coordinator calls this when it wants
    a guaranteed-valid response dict even if the LLM agent is unavailable or
    returns malformed JSON.

    Parameters
    ----------
    session_state    : ADK session state dict (all agent output_key values).
    execution_time_ms: Total pipeline wall-clock time.
    response_id      : Optional pre-generated UUID string.

    Returns
    -------
    dict matching the schema in FINAL_RESPONSE_PROMPT.
    """
    import uuid

    detection      = session_state.get("detection_data")      or {}
    severity       = session_state.get("severity_data")       or {}
    criticality    = session_state.get("criticality_data")    or {}
    location       = session_state.get("location_data")       or {}
    resource       = session_state.get("resource_data")       or {}
    recommendation = session_state.get("recommendation_data") or {}
    notification   = session_state.get("notification_data")   or {}

    # ---- severity helpers ------------------------------------------------
    severity_score = severity.get("score", 0)
    severity_level = severity.get("level", "Low")
    is_emergency   = detection.get("is_emergency", False)

    alert_color = _alert_color(severity_level, is_emergency)
    show_ambulance = severity_level in ("High", "Critical")

    # ---- hospitals -------------------------------------------------------
    hospitals      = resource.get("hospitals", [])
    nearest        = hospitals[0] if hospitals else {}
    other_hospitals = hospitals[1:4] if len(hospitals) > 1 else []

    # ---- notifications ---------------------------------------------------
    notif_sent     = notification.get("sent", False)
    notif_channels = notification.get("channels", [])
    notif_contacts = notification.get("contacts_notified", [])
    notif_preview  = (notification.get("message", "") or "")[:100]

    # ---- agent health ----------------------------------------------------
    agents_run = [
        name for name, data in {
            "detection":      detection,
            "severity":       severity,
            "criticality":    criticality,
            "location":       location,
            "resource":       resource,
            "recommendation": recommendation,
            "notification":   notification,
        }.items()
        if data  # truthy → agent produced output
    ]

    any_failed   = len(agents_run) < 7
    core_missing = not detection and not severity
    status = "failed" if core_missing else ("partial" if any_failed else "success")

    banner = _banner(severity_level, is_emergency, detection.get("type", ""))

    return {
        "response_id": response_id or str(uuid.uuid4()),
        "timestamp":   datetime.now(timezone.utc).isoformat(),
        "status":      status,

        "emergency": {
            "type":        detection.get("type",        "Unknown"),
            "description": detection.get("description", ""),
            "confidence":  detection.get("confidence",  0.0),
            "is_emergency": is_emergency,
        },

        "severity": {
            "score":     severity_score,
            "level":     severity_level,
            "reasoning": severity.get("reasoning", ""),
        },

        "criticality": {
            "score":          criticality.get("score",          severity_score),
            "level":          criticality.get("level",          severity_level),
            "rule_triggers":  criticality.get("rule_triggers",  []),
            "llm_assessment": criticality.get("llm_assessment", ""),
        },

        "location": {
            "resolved":          location.get("location_resolved", False),
            "formatted_address": location.get("formatted_address"),
            "city":              location.get("city"),
            "state":             location.get("state"),
            "coordinates": {
                "lat": location.get("latitude"),
                "lng": location.get("longitude"),
            },
        },

        "resources": {
            "nearest_hospital": {
                "name":           nearest.get("name"),
                "address":        nearest.get("address"),
                "phone":          nearest.get("phone"),
                "distance_km":    nearest.get("distance_km"),
                "google_maps_url": nearest.get("google_maps_url"),
            } if nearest else None,
            "other_hospitals": [
                {
                    "name":        h.get("name"),
                    "address":     h.get("address"),
                    "distance_km": h.get("distance_km"),
                }
                for h in other_hospitals
            ],
            "emergency_number": "112",
        },

        "recommendations": {
            "immediate_actions": recommendation.get("immediate_actions", []),
            "dos":               recommendation.get("dos",   []),
            "donts":             recommendation.get("donts", []),
            "call_ambulance":    recommendation.get("call_ambulance", show_ambulance),
        },

        "notifications": {
            "sent":              notif_sent,
            "channels":          notif_channels,
            "contacts_notified": notif_contacts,
            "message_preview":   notif_preview,
        },

        "agent_pipeline": {
            "agents_executed":  agents_run,
            "execution_time_ms": execution_time_ms,
            "any_agent_failed": any_failed,
        },

        "display": {
            "alert_color":           alert_color,
            "banner_message":        banner,
            "show_ambulance_button": show_ambulance,
            "show_hospital_map":     bool(
                location.get("location_resolved") and nearest
            ),
        },
    }


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _alert_color(level: str, is_emergency: bool) -> str:
    if not is_emergency:
        return "green"
    return {
        "Critical": "red",
        "High":     "red",
        "Medium":   "orange",
        "Low":      "yellow",
    }.get(level, "yellow")


def _banner(level: str, is_emergency: bool, etype: str) -> str:
    if not is_emergency:
        return "No emergency detected. Monitor symptoms and consult a doctor if needed."
    messages = {
        "Critical": f"🚨 CRITICAL EMERGENCY — {etype}. Call 112 immediately!",
        "High":     f"⚠️  HIGH SEVERITY — {etype}. Seek emergency care now.",
        "Medium":   f"⚠️  MEDIUM SEVERITY — {etype}. Visit a hospital soon.",
        "Low":      f"ℹ️  LOW SEVERITY — {etype}. Monitor and consult a doctor.",
    }
    return messages.get(level, f"Emergency detected: {etype}.")