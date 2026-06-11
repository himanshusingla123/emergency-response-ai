from pydantic import BaseModel, Field
from typing import Optional, List


class Location(BaseModel):
    latitude: float
    longitude: float


class EmergencyContact(BaseModel):
    name: str
    phone: str
    relationship: str


class EmergencyRequest(BaseModel):
    user_id: str
    symptoms: str = Field(..., min_length=5)
    age: int = Field(..., ge=1, le=120)
    gender: Optional[str] = None
    medical_history: Optional[str] = ""
    location: Location
    emergency_contacts: List[EmergencyContact] = []