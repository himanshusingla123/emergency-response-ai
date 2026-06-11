"""
agents/self_improvement.py

Self-Improvement Loop — uses the OFFICIAL Phoenix MCP server to query
its own traces at runtime, then patches weak agent prompts in the
persistent PromptStore.

Fixes:
  #1 — Prompt patches actually applied at request time (agents read from store)
  #2 — PromptStore persisted to disk (survives restarts)
  #4 — Real Phoenix MCP protocol via @arizeai/phoenix-mcp npm package

Flow every 5 pipeline runs:
  1. Spawn Phoenix MCP subprocess (npx @arizeai/phoenix-mcp)
  2. Self-improvement agent queries its own traces/evals/experiments via MCP
  3. LLM identifies weakest agent and generates improved prompt
  4. prompt_store.apply_patch() → written to disk
  5. Next request uses improved prompt automatically
"""
import json
import logging
from contextlib import asynccontextmanager
from typing import AsyncIterator

from google.adk.agents import LlmAgent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai.types import Content, Part

from agents.prompt_store import prompt_store
from tools.phoenix_mcp import create_phoenix_mcp_toolset, get_recent_spans
from config import settings

logger = logging.getLogger(__name__)

MODEL = "gemini-2.5-flash"


SELF_IMPROVEMENT_PROMPT = """
You are a Self-Improvement Agent for an AI emergency response system.

You have access to the Phoenix MCP server tools. Use them to:
1. List recent traces/spans from the "google-rapid-hackathon" project
2. Look at span annotations (evaluation scores) if available
3. Identify which agent has the most errors or lowest quality scores

Then generate a specific improved system prompt for the weakest agent.

Valid agent names to improve:
  emergency_detection | severity | location | resource | recommendation | final_response

Return ONLY this JSON (no preamble, no markdown fences):
{
  "analysis_summary": "<2-3 sentences on what you found in the traces>",
  "weakest_agent": "<agent name, or 'none' if all healthy>",
  "failure_reason": "<specific reason from trace data>",
  "current_issues": ["<issue 1>", "<issue 2>"],
  "improved_prompt_patch": "<complete improved system prompt for that agent, or empty string>",
  "expected_improvement": "<what specific improvement is expected>",
  "confidence": <0.0-1.0>
}

Rules:
- Base your analysis ENTIRELY on what the Phoenix tools return
- If all agents look healthy (no errors, good scores) → weakest_agent = "none"
- The improved_prompt_patch must be a COMPLETE drop-in system prompt
- Focus on the single highest-impact change only
""".strip()


async def run_self_improvement(force: bool = False) -> dict:
    """
    Run the self-improvement loop.
    Called via asyncio.create_task() — non-blocking.

    Parameters
    ----------
    force : bypass the every-5-runs throttle
    """
    run = prompt_store.increment_run()

    if not force and run % 5 != 0:
        return {
            "improved": False,
            "skipped": True,
            "reason": f"Run {run} — improvement scheduled at run {(run // 5 + 1) * 5}",
        }

    logger.info("Self-improvement loop triggered at run #%d", run)

    # Try real Phoenix MCP toolset first
    mcp_toolset = create_phoenix_mcp_toolset()

    if mcp_toolset is None:
        # npx unavailable — fall back to REST-based analysis
        logger.warning("Phoenix MCP unavailable — running REST-based self-improvement")
        return await _run_rest_fallback()

    # Use real MCP tools
    try:
        async with mcp_toolset as toolset:
            tools = await toolset.get_tools()

            agent = LlmAgent(
                name="self_improvement_agent",
                model=MODEL,
                description="Queries Phoenix via MCP and improves agent prompts",
                instruction=SELF_IMPROVEMENT_PROMPT,
                tools=tools,
                output_key="improvement_result",
            )

            session_service = InMemorySessionService()
            runner = Runner(
                agent=agent,
                app_name="self_improvement",
                session_service=session_service,
            )
            session = await session_service.create_session(
                app_name="self_improvement",
                user_id="system",
            )

            raw_result = ""
            final_seen = False  # ← fix 1: flag instead of break
            async for event in runner.run_async(
                user_id="system",
                session_id=session.id,
                new_message=Content(
                    role="user",
                    parts=[Part(text=(
                        "Query Phoenix for recent traces and evaluations in project "
                        "'google-rapid-hackathon', then identify and improve the "
                        "weakest agent's prompt."
                    ))]
                ),
            ):
                if event.is_final_response() and not final_seen:
                    raw_result = event.content.parts[0].text
                    final_seen = True
                # no break — let the generator exhaust naturally

    except Exception as e:
        logger.error("Self-improvement agent (MCP) failed: %s", e)
        return {"improved": False, "skipped": False, "reason": str(e)}

    return _parse_and_apply(raw_result)


async def _run_rest_fallback() -> dict:
    """
    Fallback when npx is unavailable.
    Uses REST to get spans, then runs a simpler analysis.
    """
    from google.adk.tools import FunctionTool

    agent = LlmAgent(
        name="self_improvement_agent_rest",
        model=MODEL,
        description="Queries Phoenix via REST and improves agent prompts",
        instruction=SELF_IMPROVEMENT_PROMPT,
        tools=[FunctionTool(func=get_recent_spans)],
        output_key="improvement_result",
    )

    session_service = InMemorySessionService()
    runner = Runner(
        agent=agent,
        app_name="self_improvement_rest",
        session_service=session_service,
    )
    session = await session_service.create_session(
        app_name="self_improvement_rest",
        user_id="system",
    )

    raw_result = ""
    final_seen = False  # ← fix 2: flag instead of break
    try:
        async for event in runner.run_async(
            user_id="system",
            session_id=session.id,
            new_message=Content(
                role="user",
                parts=[Part(text=(
                    "Use get_recent_spans to fetch recent spans from project "
                    "'google-rapid-hackathon', then identify and improve the weakest agent."
                ))]
            ),
        ):
            if event.is_final_response() and not final_seen:
                raw_result = event.content.parts[0].text
                final_seen = True
            # no break — let the generator exhaust naturally
    except Exception as e:
        logger.error("Self-improvement REST fallback failed: %s", e)
        return {"improved": False, "skipped": False, "reason": str(e)}

    return _parse_and_apply(raw_result)


def _parse_and_apply(raw_result: str) -> dict:
    """Parse LLM output and apply prompt patch if valid."""
    try:
        cleaned = raw_result.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("```")[1]
            if cleaned.startswith("json"):
                cleaned = cleaned[4:]
            cleaned = cleaned.strip()
        result = json.loads(cleaned)
    except Exception as e:
        logger.error("Failed to parse self-improvement result: %s | raw: %.200s", e, raw_result)
        return {"improved": False, "skipped": False, "reason": "Parse error"}

    agent_name = result.get("weakest_agent", "none")
    patch = result.get("improved_prompt_patch", "").strip()

    if agent_name == "none" or not patch:
        logger.info("Self-improvement: all agents healthy, no changes made")
        return {
            "improved": False,
            "skipped": False,
            "reason": result.get("analysis_summary", "All agents performing well"),
        }

    prompt_store.apply_patch(agent_name, patch, result.get("failure_reason", ""))

    return {
        "improved": True,
        "skipped": False,
        "agent": agent_name,
        "reason": result.get("failure_reason"),
        "expected_improvement": result.get("expected_improvement"),
        "confidence": result.get("confidence", 0.0),
        "total_improvements": len(prompt_store.get_history()),
    }


def get_improvement_history() -> list[dict]:
    return prompt_store.get_history()


def get_current_overrides() -> dict[str, int]:
    return prompt_store.get_overrides_summary()