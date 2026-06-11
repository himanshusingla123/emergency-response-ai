"""
agents/prompt_store.py

Persistent prompt store — fixes issue #2 (improvements lost on restart).

Stores improved prompts to disk as JSON so they survive server restarts.
Falls back to hardcoded defaults if the file doesn't exist yet.

Usage in every agent:
    from agents.prompt_store import prompt_store
    instruction = prompt_store.get("emergency_detection", SYSTEM_PROMPT)
"""
import json
import logging
import os
from pathlib import Path
from datetime import datetime, timezone
from threading import Lock

logger = logging.getLogger(__name__)

STORE_PATH = Path(os.environ.get("PROMPT_STORE_PATH", "data/prompt_store.json"))


class PromptStore:
    """
    Thread-safe, file-backed store for agent prompt overrides.
    Writes to STORE_PATH on every update so state survives restarts.
    """

    def __init__(self):
        self._lock = Lock()
        self._overrides: dict[str, str] = {}
        self._history: list[dict] = []
        self._run_count: int = 0
        self._load()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get(self, agent_name: str, default: str) -> str:
        """Return the improved prompt for agent_name, or default if none."""
        with self._lock:
            return self._overrides.get(agent_name, default)

    def apply_patch(self, agent_name: str, new_prompt: str, reason: str) -> None:
        """Persist an improved prompt for agent_name."""
        with self._lock:
            self._overrides[agent_name] = new_prompt
            self._history.append({
                "agent": agent_name,
                "reason": reason,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "run_count": self._run_count,
                "prompt_chars": len(new_prompt),
            })
            self._save()
        logger.info("Prompt improved and persisted for '%s': %s", agent_name, reason)

    def increment_run(self) -> int:
        with self._lock:
            self._run_count += 1
            run = self._run_count
        return run

    @property
    def run_count(self) -> int:
        with self._lock:
            return self._run_count

    def get_history(self) -> list[dict]:
        with self._lock:
            return list(self._history)

    def get_overrides_summary(self) -> dict[str, int]:
        """Returns {agent_name: prompt_char_count} — safe to expose via API."""
        with self._lock:
            return {k: len(v) for k, v in self._overrides.items()}

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _load(self) -> None:
        if not STORE_PATH.exists():
            logger.info("No prompt store found at %s — starting fresh", STORE_PATH)
            return
        try:
            data = json.loads(STORE_PATH.read_text())
            self._overrides = data.get("overrides", {})
            self._history = data.get("history", [])
            self._run_count = data.get("run_count", 0)
            logger.info(
                "Prompt store loaded: %d overrides, %d history entries",
                len(self._overrides), len(self._history)
            )
        except Exception as e:
            logger.error("Failed to load prompt store: %s — starting fresh", e)

    def _save(self) -> None:
        try:
            STORE_PATH.parent.mkdir(parents=True, exist_ok=True)
            STORE_PATH.write_text(json.dumps({
                "overrides": self._overrides,
                "history": self._history,
                "run_count": self._run_count,
            }, indent=2))
        except Exception as e:
            logger.error("Failed to save prompt store: %s", e)


# Global singleton — imported by all agents and self_improvement
prompt_store = PromptStore()