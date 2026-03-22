from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Optional


def _base_dir() -> str:
    """Return absolute directory for trace logs, ensure it exists."""
    here = os.path.dirname(__file__)
    base = os.path.abspath(os.path.join(here, "..", "conversation_histories", "traces"))
    os.makedirs(base, exist_ok=True)
    return base

def get_trace_dir() -> str:
    """Public accessor for the trace logs directory."""
    return _base_dir()

_CURRENT_FILE = "current.json"

def _current_path() -> str:
    return os.path.join(_base_dir(), _CURRENT_FILE)

def set_current_trace_id(trace_id: str) -> None:
    """Persist the current active trace_id to disk for non-UI contexts."""
    payload = {"trace_id": trace_id, "updated_at": datetime.utcnow().isoformat()}
    with open(_current_path(), "w", encoding="utf-8") as f:
        json.dump(payload, f)

def get_current_trace_id() -> Optional[str]:
    """Load the current active trace_id from disk, if any."""
    path = _current_path()
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data.get("trace_id")
    except Exception:
        return None


def get_trace_path(trace_id: str) -> str:
    """Get absolute path to the JSON file for a given trace_id."""
    return os.path.join(_base_dir(), f"{trace_id}.json")


def start_trace(trace_id: str) -> None:
    """Create a new trace file if it doesn't exist."""
    path = get_trace_path(trace_id)
    if not os.path.exists(path):
        payload = {
            "trace_id": trace_id,
            "created_at": datetime.utcnow().isoformat(),
            "points": [],
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f)


def append_point(
    trace_id: str,
    *,
    tool: str,
    summary: str,
    detail: str,
    time: Optional[str] = None,
) -> None:
    """Append a single summary point to the trace log."""
    path = get_trace_path(trace_id)
    if not os.path.exists(path):
        start_trace(trace_id)
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    data.setdefault("points", []).append(
        {
            "tool": tool,
            "summary": (summary or "").strip(),
            "detail": (detail or "").strip(),
            "time": time or datetime.utcnow().isoformat(),
            "trace_id": trace_id,
        }
    )
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def get_log(trace_id: str) -> dict:
    """Return the full JSON log for a given trace_id."""
    path = get_trace_path(trace_id)
    if not os.path.exists(path):
        return {"trace_id": trace_id, "created_at": None, "points": []}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


