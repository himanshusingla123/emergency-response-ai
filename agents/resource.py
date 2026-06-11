from google.adk.agents import LlmAgent
from google.adk.tools import FunctionTool
from tools.google_places import find_nearest_hospitals
from config import settings
# ADK 2.x: google-genai backend
MODEL = "gemini-2.5-flash"

hospital_tool = FunctionTool(func=find_nearest_hospitals)

SYSTEM_PROMPT = """
You are a Resource Discovery Agent.

Use the find_nearest_hospitals tool to locate the closest hospitals.
Return the top result as nearest_hospital in your JSON output.

OUTPUT FORMAT (strict JSON):
{
  "nearest_hospital": {
    "name": "...",
    "address": "...",
    "distance_km": 0.0,
    "eta_minutes": 0,
    "phone": "..."
  },
  "ambulance_available": true
}
"""

resource_agent = LlmAgent(
    name="resource_agent",
    description="Finds nearest hospitals using Google Places API",
    model=MODEL,
    instruction=SYSTEM_PROMPT,
    tools=[hospital_tool],
    output_key="resource_result",
)
