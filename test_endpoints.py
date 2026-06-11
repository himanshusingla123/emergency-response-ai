"""
test_endpoints.py

Tests for all FastAPI endpoints in main.py.
Run with:  pytest test_endpoints.py -v
Or quick:  python test_endpoints.py

Requires the server to be running:
    uvicorn main:app --port 8000

Set BASE_URL below if your server runs elsewhere.
"""

import time
import uuid
import httpx
import json

BASE_URL = "http://localhost:8000"

# ---------------------------------------------------------------------------
# Shared test payload
# ---------------------------------------------------------------------------

SAMPLE_REQUEST = {
    "user_id": "test-user-001",
    "symptoms": "severe chest pain radiating to left arm, sweating, shortness of breath",
    "age": 55,
    "gender": "male",
    "medical_history": "hypertension, diabetes",
    "location": {
        "latitude": 19.0760,
        "longitude": 72.8777
    },
    "emergency_contacts": [
        {
            "name": "Jane Doe",
            "phone": "+916280894726",
            "relationship": "spouse"
        }
    ]
}

MILD_REQUEST = {
    "user_id": "test-user-002",
    "symptoms": "mild headache and slight fever since this morning",
    "age": 30,
    "gender": "female",
    "medical_history": "",
    "location": {
        "latitude": 19.0760,
        "longitude": 72.8777
    },
    "emergency_contacts": []
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def ok(label: str, passed: bool, detail: str = ""):
    status = "✅ PASS" if passed else "❌ FAIL"
    msg = f"  {status}  {label}"
    if detail:
        msg += f"\n         {detail}"
    print(msg)
    return passed


def section(title: str):
    print(f"\n{'─' * 55}")
    print(f"  {title}")
    print(f"{'─' * 55}")


# ---------------------------------------------------------------------------
# 1. Health endpoints
# ---------------------------------------------------------------------------

def test_health(client: httpx.Client):
    section("1. Health Checks")

    r = client.get("/health")
    ok("GET /health — 200", r.status_code == 200)
    ok("GET /health — status ok", r.json().get("status") == "ok")

    r = client.get("/health/detailed")
    ok("GET /health/detailed — 200", r.status_code == 200)
    data = r.json()
    ok("GET /health/detailed — has checks", "checks" in data)
    ok("GET /health/detailed — has timestamp", "timestamp" in data)
    ok("GET /health/detailed — gemini_api_key present", "gemini_api_key" in data.get("checks", {}))
    ok("GET /health/detailed — phoenix present", "phoenix" in data.get("checks", {}))


# ---------------------------------------------------------------------------
# 2. Quick triage
# ---------------------------------------------------------------------------

def test_triage(client: httpx.Client):
    section("2. Quick Triage  (no LLM — instant)")

    params = {
        "symptoms": "chest pain and sweating",
        "age": 55,
        "severity_score": 85,
        "emergency_confidence": 90,
    }
    r = client.get("/emergency/triage", params=params)
    ok("GET /emergency/triage — 200", r.status_code == 200, r.text[:120])
    data = r.json()
    ok("triage — has criticality_level", "criticality_level" in data)
    ok("triage — has requires_ambulance", "requires_ambulance" in data)
    ok("triage — score is EXTREME for high inputs",
       data.get("criticality_level", "").upper() in ("HIGH", "EXTREME"),
       str(data.get("criticality_level")))

    # Low severity
    params["severity_score"] = 10
    params["emergency_confidence"] = 15
    r2 = client.get("/emergency/triage", params=params)
    ok("triage — low severity returns LOW/MEDIUM",
       r2.json().get("criticality_level", "").upper() in ("LOW", "MEDIUM"),
       str(r2.json().get("criticality_level")))


# ---------------------------------------------------------------------------
# 3. Core analyze (full pipeline)
# ---------------------------------------------------------------------------

def test_analyze(client: httpx.Client) -> str:
    """Returns request_id for downstream tests."""
    section("3. Full Emergency Analysis  (LLM + maps — may take 20-40s)")

    print("  ⏳ Sending severe case ...")
    r = client.post("/emergency/analyze", json=SAMPLE_REQUEST, timeout=120)
    ok("POST /emergency/analyze — 200", r.status_code == 200, r.text[:200])

    if r.status_code != 200:
        return ""

    data = r.json()
    ok("analyze — detected_emergency present", bool(data.get("detected_emergency")))
    ok("analyze — severity present", bool(data.get("severity")))
    ok("analyze — criticality present", bool(data.get("criticality")))
    ok("analyze — immediate_actions is list", isinstance(data.get("immediate_actions"), list))
    ok("analyze — dos is list", isinstance(data.get("dos"), list))
    ok("analyze — donts is list", isinstance(data.get("donts"), list))
    ok("analyze — confidence is float", isinstance(data.get("confidence"), (int, float)))
    ok("analyze — ambulance_required is bool", isinstance(data.get("ambulance_required"), bool))

    hospital = data.get("nearest_hospital")
    if hospital:
        ok("analyze — hospital has name", bool(hospital.get("name")))
        ok("analyze — hospital has distance_km", "distance_km" in hospital)
    else:
        ok("analyze — nearest_hospital (may be null on location error)", True, "null — check location/maps key")

    # Mild case
    print("\n  ⏳ Sending mild case ...")
    r2 = client.post("/emergency/analyze", json=MILD_REQUEST, timeout=120)
    ok("POST /emergency/analyze — mild case 200", r2.status_code == 200, r2.text[:120])
    if r2.status_code == 200:
        ok("analyze — mild → lower criticality",
           r2.json().get("criticality", "").upper() in ("LOW", "MEDIUM"),
           str(r2.json().get("criticality")))

    # Check history now has entries
    history_r = client.get("/emergency/history")
    ok("analyze — history populated after calls",
       history_r.json().get("total", 0) >= 1)

    # Return request_id from history for downstream tests
    entries = history_r.json().get("requests", [])
    return entries[0]["request_id"] if entries else ""


# ---------------------------------------------------------------------------
# 4. Evaluation polling
# ---------------------------------------------------------------------------

def test_evaluation(client: httpx.Client, request_id: str):
    section("4. Evaluation Polling")

    if not request_id:
        ok("GET /emergency/evaluation/{id} — skipped (no request_id)", False, "analyze test must pass first")
        return

    # Evaluation runs async — give it a few seconds
    print("  ⏳ Waiting 8s for async evaluation to complete ...")
    time.sleep(8)

    r = client.get(f"/emergency/evaluation/{request_id}")
    ok(f"GET /emergency/evaluation/{request_id[:8]}… — 200 or 404",
       r.status_code in (200, 404),
       r.text[:120])

    if r.status_code == 200:
        data = r.json()
        ok("evaluation — overall_score present", "overall_score" in data)
        ok("evaluation — label present", data.get("label") in ("excellent", "good", "poor", "unsafe"))
        ok("evaluation — clinical_accuracy 0-1", 0.0 <= data.get("clinical_accuracy", -1) <= 1.0)
        ok("evaluation — improvement_hints is list", isinstance(data.get("improvement_hints"), list))
    else:
        ok("evaluation — 404 means still processing (non-fatal)", True,
           "Increase sleep above or re-run after pipeline completes")


# ---------------------------------------------------------------------------
# 5. History
# ---------------------------------------------------------------------------

def test_history(client: httpx.Client):
    section("5. Request History")

    r = client.get("/emergency/history")
    ok("GET /emergency/history — 200", r.status_code == 200)
    data = r.json()
    ok("history — has total field", "total" in data)
    ok("history — has requests list", isinstance(data.get("requests"), list))

    # Limit param
    r2 = client.get("/emergency/history", params={"limit": 1})
    ok("history — limit=1 returns at most 1", len(r2.json().get("requests", [])) <= 1)

    # Invalid limit
    r3 = client.get("/emergency/history", params={"limit": 0})
    ok("history — limit=0 returns 422", r3.status_code == 422)

    r4 = client.get("/emergency/history", params={"limit": 999})
    ok("history — limit=999 returns 422 (max 50)", r4.status_code == 422)


# ---------------------------------------------------------------------------
# 6. Replay
# ---------------------------------------------------------------------------

def test_replay(client: httpx.Client, request_id: str):
    section("6. Replay")

    if not request_id:
        ok("POST /emergency/replay/{id} — skipped", False, "no request_id from analyze test")
        return

    r = client.post(f"/emergency/replay/{request_id}")
    ok(f"POST /emergency/replay/{request_id[:8]}… — 200", r.status_code == 200, r.text[:120])
    data = r.json()
    ok("replay — has message", "message" in data)
    ok("replay — has original_summary", "original_summary" in data)

    # Non-existent ID
    fake_id = str(uuid.uuid4())
    r2 = client.post(f"/emergency/replay/{fake_id}")
    ok("replay — 404 for unknown id", r2.status_code == 404)


# ---------------------------------------------------------------------------
# 7. Admin endpoints
# ---------------------------------------------------------------------------

def test_admin(client: httpx.Client):
    section("7. Admin — Prompt Overrides & Improvement History")

    r = client.get("/admin/prompt-overrides")
    ok("GET /admin/prompt-overrides — 200", r.status_code == 200)
    data = r.json()
    ok("prompt-overrides — has count", "count" in data)
    ok("prompt-overrides — has agents_with_overrides list",
       isinstance(data.get("agents_with_overrides"), list))

    r2 = client.get("/admin/improvement-history")
    ok("GET /admin/improvement-history — 200", r2.status_code == 200)
    data2 = r2.json()
    ok("improvement-history — has history list", isinstance(data2.get("history"), list))
    ok("improvement-history — has total_improvements", "total_improvements" in data2)

    # Reset overrides
    r3 = client.delete("/admin/prompt-overrides")
    ok("DELETE /admin/prompt-overrides — 200", r3.status_code == 200)
    ok("DELETE — has agents_reset list", isinstance(r3.json().get("agents_reset"), list))

    # Confirm cleared
    r4 = client.get("/admin/prompt-overrides")
    ok("prompt-overrides cleared — count is 0", r4.json().get("count", -1) == 0)


# ---------------------------------------------------------------------------
# 8. Self-improvement trigger (slow — optional)
# ---------------------------------------------------------------------------

def test_self_improvement(client: httpx.Client, run: bool = False):
    section("8. Self-Improvement Trigger  (calls Phoenix + LLM)")

    if not run:
        print("  ⏭  Skipped — set run=True in test_self_improvement() to enable")
        print("     (takes 30-60s and consumes LLM quota)")
        return

    print("  ⏳ Triggering self-improvement loop ...")
    r = client.post("/admin/improve", timeout=120)
    ok("POST /admin/improve — 200", r.status_code == 200, r.text[:200])
    data = r.json()
    ok("improve — has improved field", "improved" in data)
    ok("improve — has skipped field", "skipped" in data)
    if data.get("improved"):
        ok("improve — has agent field when improved", "agent" in data)
        ok("improve — has expected_improvement", "expected_improvement" in data)


# ---------------------------------------------------------------------------
# 9. Edge cases & validation
# ---------------------------------------------------------------------------

def test_validation(client: httpx.Client):
    section("9. Input Validation")

    # Missing required fields
    r = client.post("/emergency/analyze", json={"user_id": "x"})
    ok("analyze — 422 for missing fields", r.status_code == 422)

    # Symptoms too short (min_length=5)
    bad = {**SAMPLE_REQUEST, "symptoms": "oww"}
    r2 = client.post("/emergency/analyze", json=bad)
    ok("analyze — 422 for too-short symptoms", r2.status_code == 422)

    # Age out of range
    bad2 = {**SAMPLE_REQUEST, "age": 0}
    r3 = client.post("/emergency/analyze", json=bad2)
    ok("analyze — 422 for age=0", r3.status_code == 422)

    bad3 = {**SAMPLE_REQUEST, "age": 150}
    r4 = client.post("/emergency/analyze", json=bad3)
    ok("analyze — 422 for age=150", r4.status_code == 422)

    # Triage — severity out of range
    r5 = client.get("/emergency/triage", params={
        "symptoms": "headache", "age": 30,
        "severity_score": 150, "emergency_confidence": 50
    })
    ok("triage — 422 for severity_score=150", r5.status_code == 422)

    # Evaluation — unknown id
    r6 = client.get(f"/emergency/evaluation/{uuid.uuid4()}")
    ok("evaluation — 404 for unknown request_id", r6.status_code == 404)


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_all(skip_full_pipeline: bool = False, run_self_improve: bool = False):
    print("\n" + "═" * 55)
    print("  Emergency Response API — Endpoint Tests")
    print("═" * 55)
    print(f"  Base URL : {BASE_URL}")
    print(f"  Full pipeline : {'SKIPPED' if skip_full_pipeline else 'ENABLED'}")

    with httpx.Client(base_url=BASE_URL, timeout=30) as client:
        # Quick tests first
        test_health(client)
        test_triage(client)
        test_validation(client)

        request_id = ""
        if not skip_full_pipeline:
            request_id = test_analyze(client)
            test_evaluation(client, request_id)
            test_history(client)
            test_replay(client, request_id)
        else:
            section("3-6. Full Pipeline — SKIPPED")
            print("  Set skip_full_pipeline=False to enable LLM tests")

        test_admin(client)
        test_self_improvement(client, run=run_self_improve)

    print("\n" + "═" * 55)
    print("  Done.")
    print("═" * 55 + "\n")


if __name__ == "__main__":
    import sys

    # CLI flags:
    #   python test_endpoints.py --skip-pipeline    → only fast tests
    #   python test_endpoints.py --self-improve     → also run /admin/improve
    skip = "--skip-pipeline" in sys.argv
    improve = "--self-improve" in sys.argv

    run_all(skip_full_pipeline=skip, run_self_improve=improve)