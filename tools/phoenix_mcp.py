"""
tools/phoenix_mcp.py

Phoenix MCP integration — two layers:

Layer 1 (ADK MCPToolset):
    The self-improvement agent uses the OFFICIAL @arizeai/phoenix-mcp npm
    package as a subprocess MCP server. This is the real MCP integration
    the Arize track requires. The agent gets Phoenix tools natively via
    ADK's MCPToolset — no manual REST calls needed.

    Requires Node.js on the host machine (npx must be available).

Layer 2 (REST fallback):
    For the evaluator posting annotations back to Phoenix, we use the
    confirmed REST API endpoints directly (POST /v1/span_annotations).
    This is reliable and doesn't require the MCP subprocess.

Usage:
    # In self_improvement agent — gets real Phoenix MCP tools:
    from tools.phoenix_mcp import create_phoenix_mcp_toolset

    # In evaluator — posts scores back to Phoenix:
    from tools.phoenix_mcp import post_evaluation_to_phoenix, get_recent_spans
"""
import logging
import shutil
from typing import Optional
import httpx
from config import settings

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Layer 1 — Official Phoenix MCP server (real MCP protocol)
# ---------------------------------------------------------------------------

def create_phoenix_mcp_toolset():
    """
    Launch the official @arizeai/phoenix-mcp npm package as a stdio MCP
    subprocess and return an ADK MCPToolset connected to it.

    The self-improvement agent uses this to query its own traces, prompts,
    datasets and experiments via the real MCP protocol.

    Requires: Node.js / npx installed on the host.

    Returns MCPToolset or None if npx is unavailable.
    """
    from google.adk.tools.mcp_tool.mcp_toolset import MCPToolset, StdioServerParameters

    if not shutil.which("npx"):
        logger.warning(
            "npx not found — Phoenix MCP server unavailable. "
            "Install Node.js to enable real MCP integration."
        )
        return None

    # Derive base URL from collector endpoint
    # Collector: https://app.phoenix.arize.com/s/him250870/v1/traces
    # Base URL:  https://app.phoenix.arize.com
    base_url = settings.phoenix_collector_endpoint.rstrip("/")
    for suffix in ["/v1/traces", "/s/" + base_url.split("/s/")[-1].split("/")[0]]:
        if base_url.endswith(suffix.rstrip("/")):
            base_url = base_url[: -len(suffix.rstrip("/"))]
            break
    # Fallback to Phoenix Cloud root
    if "phoenix.arize.com" in base_url and "/s/" in settings.phoenix_collector_endpoint:
        base_url = "https://app.phoenix.arize.com"

    logger.info("Phoenix MCP server connecting to: %s", base_url)

    return MCPToolset(
        connection_params=StdioServerParameters(
            command="npx",
            args=[
                "-y",
                "@arizeai/phoenix-mcp@latest",
                "--baseUrl", base_url,
                "--apiKey", settings.phoenix_api_key,
            ],
        )
    )


# ---------------------------------------------------------------------------
# Layer 2 — REST helpers (used by evaluator for span annotations)
# ---------------------------------------------------------------------------

def _rest_base() -> str:
    """
    Return the Phoenix REST API base URL derived from collector endpoint.
    Strips /v1/traces to get the base.
    """
    endpoint = settings.phoenix_collector_endpoint.rstrip("/")
    if endpoint.endswith("/v1/traces"):
        return endpoint[: -len("/v1/traces")]
    return endpoint


def _headers() -> dict:
    return {
        "Authorization": f"Bearer {settings.phoenix_api_key}",
        "Content-Type": "application/json",
    }


async def get_recent_spans(
    project_name: str = "google-rapid-hackathon",
    limit: int = 10,
) -> dict:
    """
    Fetch recent spans via REST API.
    Used by evaluator and as fallback when MCP subprocess isn't available.

    GET /v1/projects/{project_identifier}/spans
    """
    try:
        url = f"{_rest_base()}/v1/projects/{project_name}/spans"
        params = {"limit": min(limit, 50)}

        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.get(url, headers=_headers(), params=params)
            if resp.status_code != 200:
                return {"spans": [], "count": 0,
                        "error": f"HTTP {resp.status_code}: {resp.text[:300]}"}
            data = resp.json()

        spans = data.get("data", [])
        return {
            "spans": [
                {
                    "span_id": s.get("context", {}).get("span_id"),
                    "trace_id": s.get("context", {}).get("trace_id"),
                    "name": s.get("name"),
                    "status_code": s.get("status_code"),
                    "start_time": s.get("start_time"),
                    "end_time": s.get("end_time"),
                    "attributes": s.get("attributes", {}),
                }
                for s in spans
            ],
            "count": len(spans),
            "error": None,
        }
    except Exception as e:
        logger.error("get_recent_spans error: %s", e)
        return {"spans": [], "count": 0, "error": str(e)}


async def post_evaluation_to_phoenix(
    span_id: str,
    eval_name: str,
    score: float,
    label: str,
    explanation: str,
    trace_id: Optional[str] = None,
) -> dict:
    """
    Post LLM-as-a-Judge score to Phoenix as a span annotation.

    POST /v1/span_annotations
    Confirmed endpoint from official Phoenix REST API docs.
    """
    try:
        url = f"{_rest_base()}/v1/span_annotations"
        payload = {
            "data": [
                {
                    "name": eval_name,
                    "annotator_kind": "LLM",
                    "span_id": span_id,
                    "result": {
                        "label": label,
                        "score": score,
                        "explanation": explanation[:500],
                    },
                    "metadata": {
                        "trace_id": trace_id or "",
                        "source": "emergency_evaluator",
                    },
                }
            ]
        }

        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.post(url, headers=_headers(), json=payload)
            if resp.status_code not in (200, 201):
                return {"success": False,
                        "error": f"HTTP {resp.status_code}: {resp.text[:300]}"}

        logger.info("Annotation posted to Phoenix: span=%s score=%.3f label=%s",
                    span_id, score, label)
        return {"success": True, "error": None}

    except Exception as e:
        logger.error("post_evaluation_to_phoenix error: %s", e)
        return {"success": False, "error": str(e)}