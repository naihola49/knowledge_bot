"""Deterministic contract helpers for Anthropic topic hydration"""

from __future__ import annotations


def build_topic_skeleton(candidate_ids: list[str], *, max_topics: int) -> list[dict]:
    """Create deterministic output rows keyed by id."""
    rows: list[dict] = []
    for idx, cid in enumerate(candidate_ids[:max_topics], start=1):
        rows.append(
            {
                "id": str(idx),
                "candidate_id": cid,
                "topic": cid.replace("_", " ")[:64] or "topic",
                "error_explanation": f"Need research for: {cid}",
                "confidence": 0.5,
            }
        )
    return rows


def _clamp_score(value: object, default: float = 0.5) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return min(1.0, max(0.0, parsed))


def merge_hydrations(skeleton: list[dict], hydrations: list[dict]) -> list[dict]:
    """Hydrate deterministic rows without allowing schema drift."""
    by_id = {str(row.get("id")): row for row in hydrations if isinstance(row, dict)}
    merged: list[dict] = []
    for row in skeleton:
        incoming = by_id.get(str(row["id"]), {})
        topic = str(incoming.get("topic") or row["topic"]).strip()[:64]
        explanation = str(incoming.get("error_explanation") or row["error_explanation"]).strip()[:180]
        merged.append(
            {
                "topic": topic or row["topic"],
                "error_explanation": explanation or row["error_explanation"],
                "confidence": _clamp_score(incoming.get("confidence"), default=row["confidence"]),
            }
        )
    return merged
