from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Optional, Any, Dict


def _base_dir() -> str:
    """Return absolute directory for conflict logs, ensure it exists."""
    here = os.path.dirname(__file__)
    base = os.path.abspath(os.path.join(here, "..", "conversation_histories", "conflicts"))
    os.makedirs(base, exist_ok=True)
    return base


def get_conflict_dir() -> str:
    """Public accessor for the conflict logs directory."""
    return _base_dir()


def get_conflict_path(session_id: str) -> str:
    """Get absolute path to the JSON file for a given session_id."""
    return os.path.join(_base_dir(), f"{session_id}.json")


def start_conflict_session(session_id: str) -> None:
    """Create a new conflict log file if it doesn't exist."""
    path = get_conflict_path(session_id)
    if os.path.exists(path):
        return
    payload = {
        "session_id": session_id,
        "created_at": datetime.utcnow().isoformat(),
        "updated_at": datetime.utcnow().isoformat(),
        "events": [],
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def append_event(session_id: str, event: Dict[str, Any]) -> None:
    """Append one conflict-check event to the session log."""
    path = get_conflict_path(session_id)
    if not os.path.exists(path):
        start_conflict_session(session_id)
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    data.setdefault("events", []).append(event)
    data["updated_at"] = datetime.utcnow().isoformat()
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def get_log(session_id: str) -> dict:
    """Return the full JSON log for a given session_id."""
    path = get_conflict_path(session_id)
    if not os.path.exists(path):
        return {"session_id": session_id, "created_at": None, "updated_at": None, "events": []}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


