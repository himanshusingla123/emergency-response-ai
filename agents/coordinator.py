"""
Root Coordinator Agent — built with Google ADK.
Orchestrates all sub-agents sequentially and aggregates the final response.
All LLM calls use Gemini 2.5 Flash via native google-genai SDK.
All calls are traced via Arize Phoenix.

v2 additions:
  - LLM-as-a-Judge evaluation (fire-and-forget via asyncio.create_task)
  - Self-improvement loop every 5 runs
  - Span ID captured and passed to evaluator for Phoenix annotation
"""
import json
import uuid
import asyncio
import logging
from opentelemetry import trace as otel_trace

from google.adk.agents import SequentialAgent
from google.adk.sessions import InMemorySessionService
from google.adk.runners import Runner

from agents.emergency_detection import emergency_detection_agent
from agents.criticality_engine import run_criticality_engine
from agents.serverity import severity_agent
from agents.location import location_agent
from agents.resource import resource_agent
from agents.recommendation import recommendation_agent
from agents.final_response import create_final_response_agent
from agents.evaluator import run_evaluation
from agents.self_improvement import run_self_improvement

from models.input_models import EmergencyRequest
from models.output_models import FinalEmergencyResponse
from observability.phoenix_setup import tracer, trace
from config import settings

logger = logging.getLogger(__name__)

MODEL = "gemini-2.5-flash"

# Module-level set — holds references to background tasks so they are never
# garbage-collected (and therefore never silently cancelled) before finishing.
_background_tasks: set = set()

coordinator = SequentialAgent(
    name="coordinator_agent",
    description="Orchestrates all emergency response sub-agents in sequence",
    sub_agents=[
        emergency_detection_agent,
        severity_agent,
        location_agent,
        resource_agent,
        recommendation_agent,
        create_final_response_agent(),
    ],
)


def _clean_json(raw: str) -> dict:
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("```")[1]
        if cleaned.startswith("json"):
            cleaned = cleaned[4:]
        cleaned = cleaned.strip()
    return json.loads(cleaned)


async def run_emergency_pipeline(request: EmergencyRequest, request_id: str = None) -> FinalEmergencyResponse:
    """
    Full pipeline:
      1. Sequential ADK agents
      2. Deterministic criticality engine
      3. Twilio notifications (EXTREME only)
      4. Build FinalEmergencyResponse
      5. Fire-and-forget: LLM-as-a-Judge evaluation + self-improvement
    """
    span = tracer.start_span("emergency_pipeline")

    # Capture span_id and trace_id for Phoenix annotation
    span_ctx = span.get_span_context()
    span_id = format(span_ctx.span_id, "016x") if span_ctx.span_id else None
    trace_id = format(span_ctx.trace_id, "032x") if span_ctx.trace_id else None

    try:
        span.set_attribute("user_id", request.user_id)
        span.set_attribute("symptoms", request.symptoms)

        # ----------------------------------------------------------------
        # 1. Sequential agent pipeline
        # ----------------------------------------------------------------
        session_service = InMemorySessionService()
        runner = Runner(
            agent=coordinator,
            app_name="emergency_response",
            session_service=session_service,
        )
        session = await session_service.create_session(
            app_name="emergency_response",
            user_id=request.user_id,
        )

        from google.genai.types import Content, Part
        user_message = Content(
            role="user",
            parts=[Part(text=json.dumps(request.model_dump()))]
        )

        result_state = {}
        async for event in runner.run_async(
            user_id=request.user_id,
            session_id=session.id,
            new_message=user_message,
        ):
            if event.is_final_response():
                try:
                    raw = event.content.parts[0].text
                    if not raw or not raw.strip():
                        raise ValueError("Agent returned empty response")
                    result_state = _clean_json(raw)
                except (AttributeError, IndexError) as e:
                    raise ValueError(f"Agent response has no text content: {event.content}") from e
                except json.JSONDecodeError as e:
                    raise ValueError(f"Agent returned invalid JSON: {raw!r}") from e

        # ----------------------------------------------------------------
        # 2. Deterministic criticality engine
        # ----------------------------------------------------------------
        criticality = run_criticality_engine(
            severity_score=result_state.get("severity", {}).get("score", 0),
            emergency_confidence=int(
                result_state.get("emergency", {}).get("confidence", 0) * 100
            ),
            symptom_risk_score=result_state.get("severity", {}).get("score", 0),
        )

        span.set_attribute("criticality_level", criticality.criticality_level.value)
        span.set_attribute("emergency_type", result_state.get("emergency", {}).get("type", "Unknown"))
        span.set_attribute("severity_score", result_state.get("severity", {}).get("score", 0))

        # ----------------------------------------------------------------
        # 3. Notifications (EXTREME criticality only)
        # ----------------------------------------------------------------
        notification_result = {"notify": False, "contacts_notified": []}
        if criticality.requires_contact_notification and request.emergency_contacts:
            from agents.notification import run_notifications
            notification_result = await run_notifications(
                contacts=request.emergency_contacts,
                emergency_type=result_state.get("emergency", {}).get("type", "Unknown"),
                severity=result_state.get("severity", {}).get("level", "Unknown"),
                hospital_name=result_state.get("resources", {}).get(
                    "nearest_hospital", {}).get("name", "Unknown"),
            )

        # ----------------------------------------------------------------
        # 4. Build response
        # ----------------------------------------------------------------
        nearest = result_state.get("resources", {}).get("nearest_hospital")
        if nearest and "eta_minutes" not in nearest:
            dist = nearest.get("distance_km", 0)
            nearest["eta_minutes"] = max(1, int(dist / 0.5))

        response = FinalEmergencyResponse(
            detected_emergency=result_state.get("emergency", {}).get("type", "Unknown"),
            severity=result_state.get("severity", {}).get("level", "Unknown"),
            criticality=criticality.criticality_level,
            nearest_hospital=nearest,
            ambulance_required=criticality.requires_ambulance,
            immediate_actions=result_state.get("recommendations", {}).get("immediate_actions", []),
            dos=result_state.get("recommendations", {}).get("dos", []),
            donts=result_state.get("recommendations", {}).get("donts", []),
            contacts_notified=[c["name"] for c in notification_result.get("contacts_notified", [])],
            confidence=float(result_state.get("emergency", {}).get("confidence", 0.0)),
            reasoning_summary=result_state.get("severity", {}).get("reasoning", ""),
        )

        # ----------------------------------------------------------------
        # 5. Fire-and-forget: evaluation + self-improvement
        #    _background_tasks holds references so tasks are never cancelled
        # ----------------------------------------------------------------
        for coro in [
            run_evaluation(
                request_data=request.model_dump(),
                pipeline_response=response.model_dump(),
                request_id=request_id or str(uuid.uuid4()),
                span_id=span_id,
                trace_id=trace_id,
            ),
            run_self_improvement(),
        ]:
            task = asyncio.create_task(coro)
            _background_tasks.add(task)
            task.add_done_callback(_background_tasks.discard)

        return response

    except Exception as e:
        span.record_exception(e)
        span.set_status(trace.StatusCode.ERROR, str(e))
        raise

    finally:
        span.end()
