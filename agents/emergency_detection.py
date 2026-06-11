from google.adk.agents import LlmAgent
from config import settings

# ADK 2.x: google-genai backend
MODEL = "gemini-2.5-flash"

SYSTEM_PROMPT = """
You are an Emergency Detection Agent in a life-critical system.

Analyze symptoms and determine the most likely emergency type.

Possible emergencies:
Heart Attack | Stroke | Seizure | Cardiac Arrest | Severe Bleeding |
Anaphylaxis | Asthma Attack | Poisoning | Heat Stroke | Trauma |
Road Accident | Fire Injury | Unknown

RULES:
- Never hallucinate medical facts.
- Return ONLY valid JSON. No preamble.
- confidence_score must reflect genuine uncertainty.

OUTPUT FORMAT (strict JSON):
{
  "emergency_type": "<type>",
  "emergency_confidence": <0-100>,
  "supporting_symptoms": ["...", "..."],
  "symptom_risk_score": <0-100>,
  "reasoning": "<brief clinical reasoning>"
}
"""

emergency_detection_agent = LlmAgent(
    name="emergency_detection_agent",
    description="Classifies the type of medical emergency from symptoms",
    model=MODEL,
    instruction=SYSTEM_PROMPT,
    output_key="emergency_detection_result",
)
