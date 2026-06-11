"""
Deterministic Criticality Engine — NOT an LLM call.
Uses a weighted formula for reproducible, safe results.

Formula:
  criticality_score = 0.5 * severity_score
                    + 0.3 * emergency_confidence
                    + 0.2 * symptom_risk_score
"""
from models.output_models import CriticalityLevel, CriticalityResult
from config import settings


def run_criticality_engine(
    severity_score: int,
    emergency_confidence: int,
    symptom_risk_score: int,
) -> CriticalityResult:
    score = (
        0.5 * severity_score
        + 0.3 * emergency_confidence
        + 0.2 * symptom_risk_score
    )
    score = round(score, 2)

    if score >= settings.criticality_extreme_threshold:
        level = CriticalityLevel.EXTREME
    elif score >= settings.criticality_high_threshold:
        level = CriticalityLevel.HIGH
    elif score >= settings.criticality_medium_threshold:
        level = CriticalityLevel.MEDIUM
    else:
        level = CriticalityLevel.LOW

    return CriticalityResult(
        criticality_score=score,
        criticality_level=level,
        requires_ambulance=level in (CriticalityLevel.EXTREME, CriticalityLevel.HIGH),
        requires_hospital=level != CriticalityLevel.LOW,
        requires_contact_notification=level == CriticalityLevel.EXTREME,
    )
