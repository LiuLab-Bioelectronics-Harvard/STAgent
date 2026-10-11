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




def _event_key(e: Dict[str, Any]) -> str:
    return str(e.get("event_id") or f"{e.get('time')}|{e.get('message_index')}|{e.get('trigger_tool')}")


def merge_logs(session_id: str, *logs: Optional[dict]) -> dict:
    """Union of events (deduplicated by event_id) across versions of one session's log, ordered by time."""
    events: Dict[str, Dict[str, Any]] = {}
    created = [l.get("created_at") for l in logs if isinstance(l, dict) and l.get("created_at")]
    for log in logs:
        for e in (log or {}).get("events", []) if isinstance(log, dict) else []:
            events.setdefault(_event_key(e), e)
    return {
        "session_id": session_id,
        "created_at": min(created) if created else datetime.utcnow().isoformat(),
        "updated_at": datetime.utcnow().isoformat(),
        "events": sorted(events.values(), key=lambda e: str(e.get("time") or "")),
    }


def _read(path: str) -> Optional[dict]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def snapshot_log(session_id: str, dest_dir: str) -> Optional[str]:
    """Save this session's conflict log into dest_dir/<session_id>.json (merged with any earlier snapshot,
    so a cleaned-up or re-created live log can never shrink the saved copy). Returns the path, or None
    when there are no events to save."""
    os.makedirs(dest_dir, exist_ok=True)
    dest = os.path.join(dest_dir, f"{session_id}.json")
    merged = merge_logs(session_id, _read(dest), get_log(session_id))
    if not merged["events"]:
        return None
    with open(dest, "w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)
    return dest


def restore_log(session_id: str, snapshot_path: str) -> int:
    """Restore events from a saved snapshot into the live log (merge; nothing is dropped).
    Returns the number of events added to the live log."""
    snap = _read(snapshot_path)
    if not snap:
        return 0
    live = get_log(session_id)
    merged = merge_logs(session_id, live, snap)
    added = len(merged["events"]) - len(live.get("events", []))
    if added > 0:
        with open(get_conflict_path(session_id), "w", encoding="utf-8") as f:
            json.dump(merged, f, ensure_ascii=False, indent=2)
    return max(0, added)
