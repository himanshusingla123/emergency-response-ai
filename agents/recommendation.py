from google.adk.agents import LlmAgent
from config import settings

# ADK 2.x: google-genai backend
MODEL = "gemini-2.5-flash"

SYSTEM_PROMPT = """
You are an Emergency Recommendation Agent in a life-critical system.

Based on the emergency type, severity and criticality:
1. Generate 3-5 immediate actions (ordered by urgency)
2. List 3-5 do's
3. List 3-5 don'ts

RULES:
- Be specific and actionable.
- Never invent drug names or doses.
- Return ONLY valid JSON. No preamble.

OUTPUT FORMAT (strict JSON):
{
  "immediate_actions": ["...", "...", "Call ambulance immediately"],
  "dos": ["..."],
  "donts": ["..."]
}
"""

recommendation_agent = LlmAgent(
    name="recommendation_agent",
    description="Generates emergency first-aid actions and guidance",
    model=MODEL,
    instruction=SYSTEM_PROMPT,
    output_key="recommendation_result",
)
