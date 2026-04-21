"""Candidate topic ids from Node 1 signals (before research briefs are built)."""

from __future__ import annotations

from book_companion.config import (
    HIGH_CONTRADICTION_THRESHOLD,
    LOW_COVERAGE_THRESHOLD,
    MAX_TOPICS_DEFAULT,
)
from book_companion.state import Output1


def extract_candidate_topics(output_1: Output1, *, max_topics: int = MAX_TOPICS_DEFAULT) -> list[str]:
    """Return ordered candidate topic identifiers from Node 1 signals"""
    candidates: list[str] = []

    weak_topics = output_1.get("weak_topics")
    if weak_topics:
        candidates.extend(weak_topics)

    for row in output_1.get("nli_results", []):
        if row.get("contradiction", 0.0) >= HIGH_CONTRADICTION_THRESHOLD:
            candidates.append(row["chunk_id"])

    if output_1.get("coverage_score", 0.0) < LOW_COVERAGE_THRESHOLD:
        candidates.append("low_evidence_coverage")

    if not candidates:
        candidates.append("overall_comprehension_gap")

    seen: set[str] = set()
    deduped: list[str] = []
    for item in candidates:
        if item in seen:
            continue
        seen.add(item)
        deduped.append(item)
        if len(deduped) >= max_topics:
            break
    return deduped


"""
How this works:
1. Start from Node 1 weak_topics (if any): string ids already chosen by comprehension (e.g. high-contradiction chunk ids).
2. Add any NLI row whose scaled contradiction meets HIGH_CONTRADICTION_THRESHOLD (per chunk_id, including merged premise id when present).
3. If mean retrieval coverage (coverage_score) is below LOW_COVERAGE_THRESHOLD, add the synthetic id low_evidence_coverage.
4. If nothing was flagged, use overall_comprehension_gap so Node 2 still has a handle for research.
5. Dedupe while preserving order, then cap the list at MAX_TOPICS_DEFAULT.

Thresholds live in config.py
"""
