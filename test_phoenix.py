"""
test_phoenix_extract.py

Extracts all data for project 'google-rapid-hackathon' from Phoenix Cloud.
Run: python test_phoenix_extract.py

Exports:
  - projects list
  - traces
  - spans
  - span annotations (eval scores)
  - prompts
  - sessions
  Saves everything to phoenix_export/ folder as JSON files.
"""
import asyncio
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import httpx

from config import settings

# ── Config ────────────────────────────────────────────────────────────────
PROJECT_NAME   = "google-rapid-hackathon"
BASE_URL       = settings.phoenix_collector_endpoint.strip().strip("'\"").rstrip("/")
if "/v1/traces" in BASE_URL:
    BASE_URL   = BASE_URL.replace("/v1/traces", "")
if "/v1/" in BASE_URL:
    BASE_URL   = BASE_URL.split("/v1/")[0]

OUTPUT_DIR     = Path("phoenix_export")
TIMEOUT        = 30.0

HEADERS = {
    "Authorization": f"Bearer {settings.phoenix_api_key.strip().strip('\"')}",
    "Content-Type":  "application/json",
}


def save(filename: str, data) -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    path = OUTPUT_DIR / filename
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)
    print(f"  ✅ Saved {path}  ({len(data) if isinstance(data, list) else 1} records)")


async def get(client: httpx.AsyncClient, path: str, params: dict = None) -> dict | list:
    url = f"{BASE_URL}{path}"
    try:
        r = await client.get(url, headers=HEADERS, params=params or {}, timeout=TIMEOUT)
        if r.status_code == 200:
            return r.json()
        else:
            print(f"  ⚠️  {path} → HTTP {r.status_code}: {r.text[:200]}")
            return {}
    except Exception as e:
        print(f"  ❌ {path} → {e}")
        return {}


async def extract():
    print(f"\n{'='*60}")
    print(f"Phoenix Cloud Extractor")
    print(f"Base URL   : {BASE_URL}")
    print(f"Project    : {PROJECT_NAME}")
    print(f"Output     : {OUTPUT_DIR}/")
    print(f"Timestamp  : {datetime.now(timezone.utc).isoformat()}")
    print(f"{'='*60}\n")

    async with httpx.AsyncClient(timeout=TIMEOUT) as client:

        # ── 1. Projects ───────────────────────────────────────────────
        print("📁 Fetching projects...")
        data = await get(client, "/v1/projects")
        projects = data.get("data", [])
        save("projects.json", projects)

        # Find our project ID
        project = next((p for p in projects if p.get("name") == PROJECT_NAME), None)
        if not project:
            print(f"\n❌ Project '{PROJECT_NAME}' not found.")
            print(f"   Available: {[p.get('name') for p in projects]}")
            return
        project_id = project["id"]
        print(f"   Project ID: {project_id}")

        # ── 2. Traces ─────────────────────────────────────────────────
        print("\n🔍 Fetching traces...")
        data = await get(client, f"/v1/projects/{PROJECT_NAME}/traces", {"limit": 100})
        traces = data.get("data", [])
        save("traces.json", traces)
        print(f"   Found {len(traces)} traces")

        # ── 3. Spans ──────────────────────────────────────────────────
        print("\n📊 Fetching spans...")
        data = await get(client, f"/v1/projects/{PROJECT_NAME}/spans", {"limit": 1000})
        spans = data.get("data", [])
        save("spans.json", spans)
        print(f"   Found {len(spans)} spans")

        # ── 4. Span annotations (eval scores) ─────────────────────────
        print("\n🏷️  Fetching span annotations...")
        # collect span_ids from spans
        span_ids = [
            s.get("context", {}).get("span_id")
            for s in spans
            if s.get("context", {}).get("span_id")
        ]
        print(f"   Using {len(span_ids)} span IDs...")
        data = await get(
            client,
            f"/v1/projects/{PROJECT_NAME}/span_annotations",
            {"limit": 1000},
        )
        annotations = data.get("data", [])
        save("span_annotations.json", annotations)
        print(f"   Found {len(annotations)} annotations")

        # ── 5. Sessions ───────────────────────────────────────────────
        print("\n💬 Fetching sessions...")
        data = await get(client, f"/v1/projects/{PROJECT_NAME}/sessions", {"limit": 100})
        sessions = data.get("data", [])
        save("sessions.json", sessions)
        print(f"   Found {len(sessions)} sessions")

        # ── 6. Prompts ────────────────────────────────────────────────
        print("\n📝 Fetching prompts...")
        data = await get(client, "/v1/prompts", {"limit": 100})
        prompts = data.get("data", [])
        save("prompts.json", prompts)
        print(f"   Found {len(prompts)} prompts")

        # ── 7. Datasets ───────────────────────────────────────────────
        print("\n🗄️  Fetching datasets...")
        data = await get(client, "/v1/datasets", {"limit": 100})
        datasets = data.get("data", [])
        save("datasets.json", datasets)
        print(f"   Found {len(datasets)} datasets")

        # ── 8. Per-trace span detail ──────────────────────────────────
        if traces:
            print("\n🔬 Fetching per-trace details...")
            trace_details = []
            for t in traces[:20]:  # limit to last 20 to avoid rate limits
                trace_id = t.get("traceId") or t.get("id") or t.get("trace_id")
                if not trace_id:
                    continue
                detail = await get(
                    client,
                    f"/v1/projects/{PROJECT_NAME}/spans",
                    {"trace_id": trace_id, "limit": 50},
                )
                trace_details.append({
                    "trace_id": trace_id,
                    "spans": detail.get("data", []),
                })
            save("trace_details.json", trace_details)

        # ── 9. Summary report ─────────────────────────────────────────
        print("\n📋 Building summary report...")

        # compute eval score stats from annotations
        scores = [
            a.get("result", {}).get("score")
            for a in annotations
            if a.get("result", {}).get("score") is not None
        ]
        avg_score = round(sum(scores) / len(scores), 3) if scores else None

        labels = [
            a.get("result", {}).get("label")
            for a in annotations
            if a.get("result", {}).get("label")
        ]
        label_counts = {l: labels.count(l) for l in set(labels)}

        # emergency types from span attributes
        emergency_types = []
        for s in spans:
            attrs = s.get("attributes", {})
            et = attrs.get("emergency_type") or attrs.get("openinference.span.kind")
            if et and et not in ("LLM", "CHAIN", "AGENT", "TOOL"):
                emergency_types.append(et)

        summary = {
            "extracted_at":     datetime.now(timezone.utc).isoformat(),
            "project":          PROJECT_NAME,
            "base_url":         BASE_URL,
            "counts": {
                "projects":     len(projects),
                "traces":       len(traces),
                "spans":        len(spans),
                "annotations":  len(annotations),
                "sessions":     len(sessions),
                "prompts":      len(prompts),
                "datasets":     len(datasets),
            },
            "eval_scores": {
                "count":        len(scores),
                "average":      avg_score,
                "min":          round(min(scores), 3) if scores else None,
                "max":          round(max(scores), 3) if scores else None,
                "label_counts": label_counts,
            },
            "emergency_types_seen": list(set(emergency_types)),
            "span_kinds": list({
                s.get("attributes", {}).get("openinference.span.kind", "unknown")
                for s in spans
            }),
        }
        save("summary.json", summary)

        # ── Done ──────────────────────────────────────────────────────
        print(f"\n{'='*60}")
        print(f"✅ Extraction complete → {OUTPUT_DIR}/")
        print(f"\n📊 Summary:")
        print(f"   Traces      : {len(traces)}")
        print(f"   Spans       : {len(spans)}")
        print(f"   Annotations : {len(annotations)}")
        print(f"   Avg score   : {avg_score}")
        print(f"   Labels      : {label_counts}")
        print(f"{'='*60}\n")


if __name__ == "__main__":
    asyncio.run(extract())