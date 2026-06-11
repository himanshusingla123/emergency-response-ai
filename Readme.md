# 🚨 Emergency Response AI

> A multi-agent AI system that detects medical emergencies, finds the nearest hospital, and alerts emergency contacts — in under 30 seconds.

Built for the **Google × Arize AI Hackathon** · Powered by Google ADK, Gemini 2.5 Flash, and Arize Phoenix.

---

## What it does

You describe symptoms. Six AI agents fire sequentially:

```
Detect → Severity → Location → Hospital → Recommend → Notify
```

The system identifies the emergency type, scores criticality, finds the nearest hospital via Google Places, generates immediate action steps, and sends a Twilio SMS to emergency contacts — all in one pipeline call.

After every response, a separate **LLM-as-a-Judge** agent evaluates quality across four dimensions and posts scores to Arize Phoenix as span annotations. Every 5 runs, a **self-improvement agent** queries its own Phoenix traces via MCP, identifies the weakest agent, and patches its prompt automatically.

---

## Demo

![Pipeline running](https://img.shields.io/badge/status-live-green)

```
POST /emergency/analyze
{
  "symptoms": "severe chest pain radiating to left arm, sweating",
  "age": 55,
  "location": { "latitude": 19.076, "longitude": 72.877 }
}
```

Response: emergency type, criticality badge, nearest hospital, immediate actions, do's & don'ts.

---

## Architecture

```
                        ┌─────────────────────────────────────┐
                        │         Google ADK SequentialAgent   │
                        │                                      │
  EmergencyRequest ───► │  Detection → Severity → Location    │
                        │  → Resources → Recommendation       │
                        │  → Notification                     │
                        │                                      │
                        └──────────────┬──────────────────────┘
                                       │
                          ┌────────────▼────────────┐
                          │   Criticality Engine     │
                          │   (deterministic rules)  │
                          └────────────┬────────────┘
                                       │
               ┌───────────────────────┼───────────────────────┐
               │                       │                       │
    ┌──────────▼──────────┐  ┌────────▼────────┐  ┌──────────▼──────────┐
    │   LLM-as-a-Judge    │  │  Arize Phoenix  │  │  Self-Improvement   │
    │   Evaluator Agent   │  │  (OTel Tracing) │  │  Agent + MCP        │
    └─────────────────────┘  └─────────────────┘  └─────────────────────┘
```

---

## Stack

| Layer | Technology |
|-------|-----------|
| Agent runtime | Google ADK 2.2 (`SequentialAgent`, `LlmAgent`) |
| LLM | Gemini 2.5 Flash |
| Observability | Arize Phoenix Cloud + OpenInference |
| MCP integration | `@arizeai/phoenix-mcp` (stdio) |
| Notifications | Twilio SMS |
| Hospital lookup | Google Places API |
| API | FastAPI on Google Cloud Run |
| Instrumentation | `openinference-instrumentation-google-genai` |

---

## Hackathon requirements

| Requirement | Implementation |
|-------------|---------------|
| Move beyond chat | 6-agent pipeline with tools, maps, SMS |
| Multi-step mission | Sequential ADK agents with deterministic criticality engine |
| Partner MCP integration | Arize Phoenix MCP server — agent queries own traces at runtime |
| Code-owned agent runtime | Google ADK (not visual builder) |
| OpenInference instrumentation | `GoogleGenAIInstrumentor` auto-instruments all LLM calls |
| Traces sent to Phoenix | `phoenix.otel.register` + OTLP exporter to Phoenix Cloud |
| LLM-as-a-Judge evals | 4-dimension evaluator posts scores as span annotations |
| Self-improvement loop | Agent reads traces via MCP → patches weak agent prompts |

---

## Agents

| Agent | Role |
|-------|------|
| `EmergencyDetectionAgent` | Identifies emergency type and confidence score |
| `SeverityAgent` | Scores severity 0–100, determines ambulance need |
| `LocationAgent` | Reverse geocodes coordinates |
| `ResourceAgent` | Finds nearest hospital via Google Places |
| `RecommendationAgent` | Generates immediate actions, do's & don'ts |
| `NotificationAgent` | Sends Twilio SMS to emergency contacts |
| `EvaluatorAgent` | LLM-as-a-Judge — scores clinical accuracy, completeness, safety |
| `SelfImprovementAgent` | Queries Phoenix MCP → rewrites weak prompts |
| `CriticalityEngine` | Deterministic rule engine — EXTREME / HIGH / MEDIUM / LOW |

---

## API endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/emergency/analyze` | Full 6-agent pipeline |
| `GET` | `/emergency/triage` | Fast criticality check (no LLM) |
| `GET` | `/emergency/evaluation/{id}` | Poll async eval results |
| `GET` | `/emergency/history` | Session request history |
| `POST` | `/emergency/replay/{id}` | Replay a past request |
| `POST` | `/admin/improve` | Trigger self-improvement loop |
| `GET` | `/admin/improvement-history` | View prompt patch history |
| `DELETE` | `/admin/prompt-overrides` | Reset to default prompts |
| `GET` | `/health/detailed` | System health + subsystem checks |

---

## Self-improvement loop

```
Every 5 runs (or via POST /admin/improve):

1. SelfImprovementAgent queries Phoenix MCP
   → fetches recent traces and span scores

2. Identifies the agent with lowest eval score

3. Generates an improved system prompt via Gemini

4. Patches the prompt in PromptStore (persisted to disk)

5. All subsequent runs use the improved prompt
```

This means the system gets measurably better over time using its own observability data.

---

## Quickstart

```bash
# 1. Clone and install
git clone <repo>
cd emergency-workflow-system
pip install -r requirements.txt

# 2. Configure
cp .env.example .env
# Fill in: GEMINI_API_KEY, PHOENIX_API_KEY, TWILIO_*, GOOGLE_MAPS_API_KEY, etc.

# 3. Run
uvicorn main:app --port 8000 --reload

# 4. Open frontend
open index.html
```

---

## Environment variables

```env
GEMINI_API_KEY=
GOOGLE_CLOUD_PROJECT=
GOOGLE_MAPS_API_KEY=
GOOGLE_PLACES_API_KEY=
TWILIO_ACCOUNT_SID=
TWILIO_AUTH_TOKEN=
TWILIO_FROM_NUMBER=
PHOENIX_API_KEY=
PHOENIX_COLLECTOR_ENDPOINT=
```

---

## Deploy to Cloud Run

```bash
gcloud run deploy emergency-response \
  --source . \
  --region us-central1 \
  --allow-unauthenticated \
  --port 8080 \
  --set-env-vars "GEMINI_API_KEY=...,PHOENIX_API_KEY=...,..."
```

---

## Project structure

```
emergency-workflow-system/
├── main.py                    # FastAPI app + all endpoints
├── config.py                  # Pydantic settings
├── agents/
│   ├── coordinator.py         # SequentialAgent pipeline
│   ├── emergency_detection.py
│   ├── serverity.py
│   ├── location.py
│   ├── resource.py
│   ├── recommendation.py
│   ├── notification.py
│   ├── criticality_engine.py  # Deterministic scoring
│   ├── evaluator.py           # LLM-as-a-Judge
│   ├── self_improvement.py    # MCP-powered prompt patching
│   └── prompt_store.py        # Persisted prompt overrides
├── tools/
│   ├── google_places.py
│   ├── reverse_geocode.py
│   ├── twilio_notifier.py
│   └── phoenix_mcp.py         # MCP + REST client for Phoenix
├── observability/
│   └── phoenix_setup.py       # OTel + GoogleGenAIInstrumentor
├── models/
│   ├── input_models.py
│   └── output_models.py
├── index.html                 # Frontend
└── requirements.txt
```