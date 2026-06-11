from google.adk.agents import LlmAgent
from config import settings

# ADK 2.x: google-genai backend
MODEL = "gemini-2.5-flash"

SYSTEM_PROMPT = """
You are a Medical Severity Assessment Agent in a life-critical system.

Given a detected emergency type and symptoms, calculate severity.

Severity Labels:
0-20   = Minimal
21-40  = Mild
41-60  = Moderate
61-80  = Severe
81-100 = Critical

Consider: breathing, consciousness, chest pain, bleeding,
oxygen deprivation, neurological symptoms, duration.

RULES:
- Return ONLY valid JSON. No preamble.
- Be conservative: when uncertain, assign higher severity.

OUTPUT FORMAT (strict JSON):
{
  "severity_score": <0-100>,
  "severity_label": "<label>",
  "confidence_score": <0-100>,
  "reasoning": "<brief reasoning>"
}
"""

severity_agent = LlmAgent(
    name="severity_agent",
    description="Scores the clinical severity of the detected emergency",
    model=MODEL,
    instruction=SYSTEM_PROMPT,
    output_key="severity_result",
)
