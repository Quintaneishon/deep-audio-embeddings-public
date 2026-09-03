"""Validation shared by the full and evaluation-only Flask applications."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from uuid import UUID


MAX_JSON_BYTES = 4_096
MAX_RESPONSE_TIME_MS = 30 * 60 * 1_000
MAX_SHOWN_IDS = 100
ALLOWED_RESPONDENT_TYPES = {"public"}


@dataclass(frozen=True)
class EvalResponse:
    session_id: str
    triplet_id: int
    choice: str
    response_time_ms: int
    respondent_type: str


def _plain_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field} must be an integer")
    return value


def normalize_session_id(value: Any) -> str:
    """Return the canonical UUID string used as an anonymous session key."""
    if not isinstance(value, str) or len(value) > 36:
        raise ValueError("session_id must be a UUID")
    try:
        return str(UUID(value))
    except (ValueError, AttributeError) as exc:
        raise ValueError("session_id must be a UUID") from exc


def parse_shown_ids(value: str) -> list[int]:
    """Validate the bounded comma-separated exclusion list from the UI."""
    if len(value) > 1_024:
        raise ValueError("shown list is too long")
    if not value:
        return []
    parts = value.split(",")
    if len(parts) > MAX_SHOWN_IDS:
        raise ValueError("shown list is too long")
    try:
        ids = [int(part) for part in parts]
    except ValueError as exc:
        raise ValueError("shown must contain positive integer IDs") from exc
    if any(identifier <= 0 for identifier in ids):
        raise ValueError("shown must contain positive integer IDs")
    return list(dict.fromkeys(ids))


def validate_eval_response(data: Any) -> EvalResponse:
    """Return a normalized response or raise ``ValueError``."""
    if not isinstance(data, dict):
        raise ValueError("JSON object required")

    session_id = normalize_session_id(data.get("session_id"))

    triplet_id = _plain_int(data.get("triplet_id"), "triplet_id")
    if triplet_id <= 0:
        raise ValueError("triplet_id must be positive")

    choice = data.get("choice")
    if choice not in {"a", "b"}:
        raise ValueError("choice must be 'a' or 'b'")

    response_time_ms = _plain_int(data.get("response_time_ms"), "response_time_ms")
    if not 0 <= response_time_ms <= MAX_RESPONSE_TIME_MS:
        raise ValueError("response_time_ms is outside the allowed range")

    respondent_type = data.get("respondent_type", "public")
    if respondent_type not in ALLOWED_RESPONDENT_TYPES:
        raise ValueError("respondent_type must be 'public'")

    return EvalResponse(
        session_id=session_id,
        triplet_id=triplet_id,
        choice=choice,
        response_time_ms=response_time_ms,
        respondent_type=respondent_type,
    )
