"""
Location Agent — reverse geocodes the user's coordinates using Google Maps API.
Built as an ADK LlmAgent with a FunctionTool wrapping reverse_geocode().

Reads latitude/longitude from session state (set by the Coordinator Agent)
and writes structured location info to output_key="location_result".
"""
from google.adk.agents import LlmAgent
from google.adk.tools import FunctionTool
from tools.reverse_geocode import reverse_geocode_tool
from config import settings

# ADK 2.x: google-genai backend
MODEL = "gemini-2.5-flash"

# Wrap the async reverse geocoding function as an ADK tool
geocode_tool = FunctionTool(func=reverse_geocode_tool)

SYSTEM_PROMPT = """
You are a Location Agent in a life-critical Emergency Response system.

Your only job is to determine the user's location from their coordinates.

STEPS:
1. Extract latitude and longitude from the input JSON.
2. Call the reverse_geocode tool with those coordinates.
3. Return the structured location result as JSON.

RULES:
- Always call the reverse_geocode tool — never guess or invent location data.
- If the tool returns empty strings for city/state, return them as empty strings.
  Do NOT fill in assumed values.
- Return ONLY valid JSON. No preamble, no explanation.

INPUT (from session state):
{
  "latitude": <float>,
  "longitude": <float>
}

OUTPUT FORMAT (strict JSON):
{
  "city": "<city name or empty string>",
  "state": "<state/province or empty string>",
  "country": "<country name or empty string>",
  "formatted_address": "<full formatted address>",
  "latitude": <float>,
  "longitude": <float>
}
"""

location_agent = LlmAgent(
    name="location_agent",
    description=(
        "Reverse geocodes the user's GPS coordinates into city, state, "
        "and country using the Google Maps Geocoding API"
    ),
    model=MODEL,
    instruction=SYSTEM_PROMPT,
    tools=[geocode_tool],
    output_key="location_result",
)