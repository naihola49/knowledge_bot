"""Turn candidate topic ids + Output1 into research-oriented text briefs for Node 3."""

from __future__ import annotations

from book_companion.config import (
    HIGH_CONTRADICTION_THRESHOLD,
    LOW_COVERAGE_THRESHOLD,
    RESEARCH_TOPIC_SNIPPET_MAX_CHARS,
)
from book_companion.nodes.comprehension import MERGED_PREMISE_CHUNK_ID
from book_companion.state import ClarificationTopic, NLIResult, Output1, RetrievedChunk


def _truncate(text: str, max_chars: int) -> str:
    text = text.strip()
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3].rstrip() + "..."


def _retrieved_by_chunk_id(chunks: list[RetrievedChunk]) -> dict[str, RetrievedChunk]:
    return {c["chunk_id"]: c for c in chunks}


def _merged_evidence_excerpt(output_1: Output1) -> str:
    """Same join as Node 1 merged premise, for quoting to the research LLM."""
    parts = [c["text"] for c in output_1.get("retrieved_chunks", [])]
    merged = "\n\n".join(parts)
    return _truncate(merged, RESEARCH_TOPIC_SNIPPET_MAX_CHARS)


def _metrics_lines(output_1: Output1) -> str:
    return (
        f"comprehension_score={output_1.get('comprehension_score', 0.0):.4f} (NLI entailment-derived); "
        f"coverage_score={output_1.get('coverage_score', 0.0):.4f} (mean cosine chunk↔user embedding); "
        f"contradiction_score={output_1.get('contradiction_score', 0.0):.4f} (NLI contradiction-derived)."
    )


def _nli_row_for_topic(nli_results: list[NLIResult], topic: str) -> NLIResult | None:
    for row in nli_results:
        if row.get("chunk_id") == topic:
            return row
    return None


def build_topic_explanations(output_1: Output1, topics: list[str]) -> list[ClarificationTopic]:
    """
    Build `ClarificationTopic` entries whose `error_explanation` is a **research brief**:
    signals from Node 1, short excerpts, and explicit questions for the downstream research node.
    """
    contradiction_score = output_1.get("contradiction_score", 0.0)
    coverage_score = output_1.get("coverage_score", 0.0)
    comprehension_score = output_1.get("comprehension_score", 0.0)
    user_note = output_1.get("user_input", "")
    nli_results = output_1.get("nli_results", [])
    retrieved = output_1.get("retrieved_chunks", [])
    by_id = _retrieved_by_chunk_id(retrieved)

    user_snip = _truncate(user_note, RESEARCH_TOPIC_SNIPPET_MAX_CHARS)
    metrics = _metrics_lines(output_1)

    entries: list[ClarificationTopic] = []

    for topic in topics:
        nli_row = _nli_row_for_topic(nli_results, topic)
        contradiction = float(nli_row["contradiction"]) if nli_row else contradiction_score
        entailment = float(nli_row["entailment"]) if nli_row else comprehension_score
        neutral = float(nli_row["neutral"]) if nli_row else 0.0

        lines: list[str] = [
            "[Node 1 context for research]",
            metrics,
            f"Learner note (excerpt): {user_snip!r}",
        ]

        if topic == "low_evidence_coverage":
            confidence = min(1.0, max(0.0, 1.0 - coverage_score))
            lines.extend(
                [
                    "",
                    "[Issue]",
                    f"Coverage is below the configured threshold ({LOW_COVERAGE_THRESHOLD}): retrieved chunks "
                    "do not align strongly with the learner embedding (cosine mean = coverage_score).",
                    "",
                    "[Research directions]",
                    "- Identify definitions or claims in the learner note that are not anchored in the source excerpts.",
                    "- Suggest significance and historical context from reference material that would strengthen grounding.",
                ]
            )
        elif topic == "overall_comprehension_gap":
            confidence = min(1.0, max(0.0, 0.45 + contradiction_score * 0.5))
            lines.extend(
                [
                    "",
                    "[Issue]",
                    "No specific weak-topic id was produced; overall comprehension/grounding is still insufficient.",
                    "",
                    "[Research directions]",
                    "- Summarize the core thesis of the learner note in one sentence, then list what must be verified.",
                    "- Propose a minimal reading list or search queries to close the gap.",
                ]
            )
        elif topic == "high contradiction overall":
            confidence = min(1.0, max(0.0, contradiction_score))
            lines.extend(
                [
                    "",
                    "[Issue]",
                    f"Aggregate contradiction_score ({contradiction_score:.4f}) is high relative to typical pass.",
                    "",
                    "[Research directions]",
                    "- List the strongest tensions between the note and standard accounts of the subject.",
                    "- For each tension, state what evidence would resolve it (sources, mechanisms, dates).",
                ]
            )
        elif topic == MERGED_PREMISE_CHUNK_ID:
            confidence = min(1.0, max(0.0, contradiction))
            ev = _merged_evidence_excerpt(output_1)
            lines.extend(
                [
                    "",
                    "[NLI target]",
                    "Single MNLI pass over merged top-k source text vs learner note.",
                    f"NLI (scaled): entailment={entailment:.4f}, neutral={neutral:.4f}, contradiction={contradiction:.4f}.",
                    "",
                    "[Merged source evidence (excerpt)]",
                    ev,
                    "",
                    "[Research directions]",
                    "- Compare the learner claims to this evidence: what is unsupported or overstated?",
                    "- What background concepts would help the learner reconcile note vs source?",
                ]
            )
        elif topic in by_id:
            confidence = min(1.0, max(0.0, contradiction))
            ch = by_id[topic]
            excerpt = _truncate(ch["text"], RESEARCH_TOPIC_SNIPPET_MAX_CHARS)
            lines.extend(
                [
                    "",
                    "[Retrieved chunk signal]",
                    f"chunk_id={topic!r}; cosine_similarity_to_user_embedding={ch['similarity']:.4f}.",
                    f"NLI for this row (if present): contradiction={contradiction:.4f}.",
                    "",
                    "[Source excerpt]",
                    excerpt,
                    "",
                    "[Research directions]",
                    "- Explain how this passage relates to the learner's claims.",
                    "- If contradiction is high, outline what would need to change in the note or what nuance is missing.",
                ]
            )
        elif contradiction >= HIGH_CONTRADICTION_THRESHOLD:
            confidence = min(1.0, max(0.0, contradiction))
            lines.extend(
                [
                    "",
                    "[Issue]",
                    f"High per-row contradiction (≥ {HIGH_CONTRADICTION_THRESHOLD}).",
                    f"NLI (scaled): entailment={entailment:.4f}, neutral={neutral:.4f}, contradiction={contradiction:.4f}.",
                    "",
                    "[Research directions]",
                    "- Pin down which claim in the learner note conflicts with usual interpretations of the topic.",
                    "- Provide neutral reference facts and common misconceptions.",
                ]
            )
        else:
            confidence = min(1.0, max(0.0, 0.3 + contradiction * 0.5))
            lines.extend(
                [
                    "",
                    "[Issue]",
                    "Topic flagged as underdeveloped or imprecise relative to retrieved evidence.",
                    f"NLI (scaled): entailment={entailment:.4f}, neutral={neutral:.4f}, contradiction={contradiction:.4f}.",
                    "",
                    "[Research directions]",
                    "- Clarify terminology the learner uses ambiguously.",
                    "- Give 2–3 crisp criteria a good answer on this topic should satisfy.",
                ]
            )

        explanation = "\n".join(lines)
        entries.append(
            {
                "topic": topic,
                "error_explanation": explanation,
                "confidence": round(confidence, 4),
            }
        )

    return entries


"""
How this works:
1. Input is Output1 from Node 1 (scores, weak_topics, retrieved_chunks with cosine similarity, nli_results with
   scaled entailment/neutral/contradiction per logical chunk, plus the learner note text).
2. Caller supplies a list of topic ids (usually from topic_candidates.extract_candidate_topics): synthetic strings
   like low_evidence_coverage or chunk ids like chunk_0 / premise_top_k_merged.
3. For each id, we assemble a multi-section plain-text brief: Node 1 metrics line, learner note excerpt (truncated),
   then branch-specific [Issue] / [NLI target] / [Retrieved chunk signal] and source excerpts where applicable.
4. Branches use the same thresholds as Node 1/2 config (HIGH_CONTRADICTION_THRESHOLD, LOW_COVERAGE_THRESHOLD);
   MERGED_PREMISE_CHUNK_ID matches the single MNLI call over merged top-k text in comprehension.py.
5. confidence is a heuristic 0..1 for prioritization, not a calibrated probability.
6. The resulting error_explanation string is meant to be pasted or passed into an LLM research step (Node 3) as
   structured context—no web calls here, only string construction.
"""
