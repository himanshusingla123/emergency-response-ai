from pydantic import BaseModel, Field
from typing import List, Optional
from enum import Enum


class SeverityLabel(str, Enum):
    MINIMAL = "Minimal"
    MILD = "Mild"
    MODERATE = "Moderate"
    SEVERE = "Severe"
    CRITICAL = "Critical"


class CriticalityLevel(str, Enum):
    LOW = "Low"
    MEDIUM = "Medium"
    HIGH = "High"
    EXTREME = "Extreme"


class EmergencyDetectionResult(BaseModel):
    emergency_type: str
    confidence_score: int = Field(..., ge=0, le=100)
    supporting_symptoms: List[str]
    reasoning: str


class SeverityResult(BaseModel):
    severity_score: int = Field(..., ge=0, le=100)
    severity_label: SeverityLabel
    confidence_score: int = Field(..., ge=0, le=100)
    reasoning: str


class CriticalityResult(BaseModel):
    criticality_score: float
    criticality_level: CriticalityLevel
    requires_ambulance: bool
    requires_hospital: bool
    requires_contact_notification: bool


class HospitalInfo(BaseModel):
    name: str
    address: str
    distance_km: float
    eta_minutes: int = 0        # ← add default
    phone: Optional[str] = None
    google_maps_url: Optional[str] = None  # ← add this, LLM includes it



class RecommendationResult(BaseModel):
    immediate_actions: List[str]
    dos: List[str]
    donts: List[str]


class NotificationResult(BaseModel):
    notify: bool
    contacts_notified: List[dict]


class FinalEmergencyResponse(BaseModel):
    detected_emergency: str
    severity: str
    criticality: str
    nearest_hospital: Optional[HospitalInfo] = None
    ambulance_required: bool
    immediate_actions: List[str]
    dos: List[str]
    donts: List[str]
    contacts_notified: List[str]
    confidence: float           # ← int → float
    reasoning_summary: str