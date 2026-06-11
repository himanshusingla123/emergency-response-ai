"""
FastAPI entry point.
Arize Phoenix tracing initialised before any ADK imports.
"""
# ⚠️ MUST be first — triggers GoogleGenAIInstrumentor before any ADK imports
from observability.phoenix_setup import tracer  # noqa: F401

import os
from config import settings as _settings
os.environ.setdefault("GOOGLE_API_KEY", _settings.gemini_api_key)
os.environ.setdefault("GOOGLE_GENAI_USE_VERTEXAI", "false")

import uuid
import asyncio
from datetime import datetime, timezone
from typing import Optional

import uvicorn
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager

from models.input_models import EmergencyRequest, Location
from models.output_models import FinalEmergencyResponse
from agents.coordinator import run_emergency_pipeline
from agents.evaluator import get_cached_evaluation
from agents.self_improvement import run_self_improvement, get_improvement_history, get_current_overrides
from agents.criticality_engine import run_criticality_engine
import structlog
from fastapi.middleware.cors import CORSMiddleware

from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse


logger = structlog.get_logger()

# In-memory request history (last 50 requests this session)
_request_history: list[dict] = []
_MAX_HISTORY = 50


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Phoenix tracing initialised")
    yield
    from observability.phoenix_setup import _force_flush
    _force_flush()


app = FastAPI(
    title="Emergency Detection & Response System",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def serve_frontend():
    return FileResponse("index.html")
# ---------------------------------------------------------------------------
# 1. Core — analyze emergency
# ---------------------------------------------------------------------------

@app.post("/emergency/analyze", response_model=FinalEmergencyResponse)
async def analyze_emergency(request: EmergencyRequest) -> FinalEmergencyResponse:
    """
    Main pipeline: detect emergency → severity → location → resources →
    recommendations → notifications.
    """
    request_id = str(uuid.uuid4())
    started_at = datetime.now(timezone.utc)

    try:
        result = await run_emergency_pipeline(request, request_id=request_id)
        logger.info("Pipeline complete", emergency=result.detected_emergency, criticality=result.criticality)

        # Store in history
        _push_history({
            "request_id": request_id,
            "timestamp": started_at.isoformat(),
            "user_id": request.user_id,
            "symptoms": request.symptoms,
            "detected_emergency": result.detected_emergency,
            "severity": result.severity,
            "criticality": str(result.criticality),
            "ambulance_required": result.ambulance_required,
        })

        return result

    except Exception as e:
        import traceback
        traceback.print_exc()
        logger.error("Pipeline failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------------
# 2. Quick triage — severity + criticality only, no location/hospital lookup
# ---------------------------------------------------------------------------

@app.get("/emergency/triage")
async def quick_triage(
    symptoms: str = Query(..., description="Describe the symptoms"),
    age: int = Query(..., ge=1, le=120),
    severity_score: int = Query(..., ge=0, le=100, description="Estimated severity 0-100"),
    emergency_confidence: int = Query(..., ge=0, le=100),
):
    """
    Lightweight endpoint — runs only the criticality engine (no LLM, no maps).
    Useful for a fast frontend badge before the full analysis completes.
    """
    result = run_criticality_engine(
        severity_score=severity_score,
        emergency_confidence=emergency_confidence,
        symptom_risk_score=severity_score,
    )
    return {
        "criticality_level": result.criticality_level,
        "criticality_score": result.criticality_score,
        "requires_ambulance": result.requires_ambulance,
        "requires_hospital": result.requires_hospital,
        "requires_contact_notification": result.requires_contact_notification,
        "symptoms_preview": symptoms[:100],
        "age": age,
    }


# ---------------------------------------------------------------------------
# 3. Evaluation result for a past request
# ---------------------------------------------------------------------------

@app.get("/emergency/evaluation/{request_id}")
async def get_evaluation(request_id: str):
    """
    Poll for the LLM-as-a-Judge evaluation score for a completed request.
    The evaluation runs async after /emergency/analyze responds, so call
    this a few seconds later.
    """
    result = get_cached_evaluation(request_id)
    if not result:
        raise HTTPException(
            status_code=404,
            detail=f"No evaluation found for request_id '{request_id}'. "
                   "It may still be processing — try again in a few seconds."
        )
    return result


# ---------------------------------------------------------------------------
# 4. Request history (this session)
# ---------------------------------------------------------------------------

@app.get("/emergency/history")
async def request_history(limit: int = Query(10, ge=1, le=50)):
    """
    Returns the last N emergency analyses run this session.
    Useful for reviewing patterns or debugging.
    """
    return {
        "total": len(_request_history),
        "limit": limit,
        "requests": _request_history[-limit:][::-1],  # newest first
    }


# ---------------------------------------------------------------------------
# 5. Re-analyze from history (replay a past request)
# ---------------------------------------------------------------------------

@app.post("/emergency/replay/{request_id}")
async def replay_request(request_id: str):
    """
    Re-run the pipeline for a previously analyzed request (from session history).
    Useful for testing self-improvement — run /admin/improve then replay to see
    whether prompt patches improved the result.
    """
    past = next((r for r in _request_history if r["request_id"] == request_id), None)
    if not past:
        raise HTTPException(status_code=404, detail=f"Request '{request_id}' not in session history.")

    return JSONResponse(content={
        "message": "Use the original request body with /emergency/analyze to replay.",
        "original_summary": past,
        "hint": "Session history stores summaries only — resend the full request body."
    })


# ---------------------------------------------------------------------------
# 6. Health — detailed
# ---------------------------------------------------------------------------

@app.get("/health")
async def health():
    """Basic liveness check."""
    return {"status": "ok"}


@app.get("/health/detailed")
async def health_detailed():
    """
    Detailed health check — reports on all subsystems.
    Use this to verify Phoenix, config, and session state on startup.
    """
    import shutil
    from tools.phoenix_mcp import _rest_base

    checks = {}

    # Phoenix reachability
    try:
        import httpx
        async with httpx.AsyncClient(timeout=5.0) as client:
            r = await client.get(
                f"{_rest_base()}/v1/projects",
                headers={"Authorization": f"Bearer {_settings.phoenix_api_key}"},
            )
        checks["phoenix"] = "ok" if r.status_code == 200 else f"http_{r.status_code}"
    except Exception as e:
        checks["phoenix"] = f"error: {str(e)[:60]}"

    # MCP (npx) availability
    checks["mcp_npx"] = "available" if shutil.which("npx") else "unavailable"

    # Gemini API key set
    checks["gemini_api_key"] = "set" if os.environ.get("GOOGLE_API_KEY") else "missing"

    # Session stats
    checks["session_requests"] = len(_request_history)
    checks["active_prompt_overrides"] = len(get_current_overrides())
    checks["total_improvements"] = len(get_improvement_history())

    overall = "ok" if all(
        v in ("ok", "available", "set") or isinstance(v, int)
        for v in checks.values()
    ) else "degraded"

    return {"status": overall, "checks": checks, "timestamp": datetime.now(timezone.utc).isoformat()}


# ---------------------------------------------------------------------------
# 7. Admin — self-improvement
# ---------------------------------------------------------------------------

@app.post("/admin/improve")
async def trigger_improvement():
    """
    Manually trigger the self-improvement loop.
    Queries Phoenix traces, identifies weak agents, patches prompts.
    """
    try:
        result = await run_self_improvement(force=True)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/admin/improvement-history")
async def improvement_history():
    """Full prompt improvement history for this session."""
    return {
        "history": get_improvement_history(),
        "active_overrides": list(get_current_overrides().keys()),
        "total_improvements": len(get_improvement_history()),
    }


@app.get("/admin/prompt-overrides")
async def prompt_overrides():
    """Which agents currently have improved prompts active."""
    overrides = get_current_overrides()
    return {
        "count": len(overrides),
        "agents_with_overrides": list(overrides.keys()),
    }


@app.delete("/admin/prompt-overrides")
async def reset_prompt_overrides():
    """
    Reset all prompt overrides — agents revert to their hardcoded defaults.
    Useful for A/B testing: compare before and after /admin/improve.
    """
    from agents.prompt_store import prompt_store
    overrides_before = list(get_current_overrides().keys())
    prompt_store._overrides.clear()
    prompt_store._save()
    return {
        "message": "All prompt overrides cleared.",
        "agents_reset": overrides_before,
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _push_history(entry: dict):
    if len(_request_history) >= _MAX_HISTORY:
        _request_history.pop(0)
    _request_history.append(entry)


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
