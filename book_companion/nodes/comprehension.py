"""Node 1 orchestration node: chunk, embed, retrieve, NLI aggregate"""

from __future__ import annotations

import re
from pathlib import Path
from typing import TypedDict

from book_companion.config import (
    CLARIFICATION_TRIGGER_SCORE,
    COMPREHENSION_PASS_SCORE,
    HIGH_CONTRADICTION_THRESHOLD,
    MAX_LOOPS,
    MIN_WORD_COUNT,
    TOP_K_RETRIEVAL,
)
from book_companion.nodes.node1_methods import (
    chunk_text,
    clean_text,
    retrieve_top_k,
    vectorize_text,
    vectorize_texts,
)
from book_companion.services.bart_nli_inference import get_bart_nli_client
from book_companion.state import GraphState, NLIResult, Output1, RetrievedChunk


def _count_words(text: str) -> int:
    return len(re.findall(r"\b\w+\b", text))


def _derive_weak_topics(nli_results: list[dict], contradiction_score: float) -> list[str] | None:
    weak_topics: list[str] = []
    contradiction_chunks = [
        item["chunk_id"] for item in nli_results if item["contradiction"] >= HIGH_CONTRADICTION_THRESHOLD
    ]
    if contradiction_chunks:
        weak_topics.extend(contradiction_chunks)
    if contradiction_score >= HIGH_CONTRADICTION_THRESHOLD:
        weak_topics.append("high contradiction overall")
    return weak_topics or None


class Node1NLIStageResult(TypedDict):
    retrieved_chunks: list[RetrievedChunk]
    nli_results: list[NLIResult]
    comprehension_score: float
    coverage_score: float
    contradiction_score: float


MERGED_PREMISE_CHUNK_ID = "premise_top_k_merged"


def aggregate_comprehension(
    nli_results: list[NLIResult],
    retrieved_chunks: list[RetrievedChunk],
) -> tuple[float, float, float]:
    """Return (comprehension_score, coverage_score, contradiction_score)."""
    if not nli_results:
        return 0.0, 0.0, 1.0

    entailment_avg = sum(item["entailment"] for item in nli_results) / len(nli_results)
    contradiction_avg = sum(item["contradiction"] for item in nli_results) / len(nli_results)
    coverage_avg = (
        sum(item["similarity"] for item in retrieved_chunks) / len(retrieved_chunks)
        if retrieved_chunks
        else 0.0
    )
    return round(entailment_avg, 6), round(coverage_avg, 6), round(contradiction_avg, 6)


def run_premise_hypothesis_pipeline(
    raw_content: str,
    user_input: str,
    *,
    k: int | None = None,
    chunk_size: int = 120,
    overlap: int = 20,
) -> Node1NLIStageResult:
    """
    1. Chunk premise (article).
    2. Embed all premise chunks and the hypothesis (user input).
    3. Retrieve top-k premise chunks by cosine similarity to hypothesis embedding.
    4. Merge those chunk texts into one premise string; run a single BART-MNLI call
       (premise=merged chunks, hypothesis=user input).
    """
    k = k if k is not None else TOP_K_RETRIEVAL
    chunks = chunk_text(raw_content, chunk_size=chunk_size, overlap=overlap)
    if not chunks:
        return {
            "retrieved_chunks": [],
            "nli_results": [],
            "comprehension_score": 0.0,
            "coverage_score": 0.0,
            "contradiction_score": 1.0,
        }

    chunk_vectors = vectorize_texts(chunks)
    hypothesis_vec = vectorize_text(user_input)

    top = retrieve_top_k(
        user_vec=hypothesis_vec,
        chunk_vectors=chunk_vectors,
        chunks=chunks,
        k=min(k, len(chunks)),
    )

    retrieved_chunks: list[RetrievedChunk] = [
        {"chunk_id": cid, "text": text, "similarity": sim} for cid, text, sim in top
    ]

    merged_premise = "\n\n".join(rc["text"] for rc in retrieved_chunks)

    nli_client = get_bart_nli_client()
    ent, neu, con = nli_client.predict(merged_premise, user_input)
    nli_results: list[NLIResult] = [
        {
            "chunk_id": MERGED_PREMISE_CHUNK_ID,
            "entailment": ent,
            "neutral": neu,
            "contradiction": con,
        }
    ]

    comp, cov, contra = aggregate_comprehension(nli_results, retrieved_chunks)
    return {
        "retrieved_chunks": retrieved_chunks,
        "nli_results": nli_results,
        "comprehension_score": comp,
        "coverage_score": cov,
        "contradiction_score": contra,
    }


def run_comprehension_node(state: GraphState) -> GraphState:
    """Read source + user input, run chunk → embed → retrieve → NLI, write output_1"""
    raw_content_path = Path(state["raw_content_path"])
    user_input_path = Path(state["user_input_path"])
    raw_content = raw_content_path.read_text(encoding="utf-8")
    user_input = clean_text(user_input_path.read_text(encoding="utf-8"))

    word_count = _count_words(user_input)
    day = state.get("day", "unknown_day")
    loop_count = state.get("loop_count", 0)

    stage = run_premise_hypothesis_pipeline(raw_content, user_input)
    weak_topics = _derive_weak_topics(stage["nli_results"], stage["contradiction_score"])

    needs_clarification = (
        word_count < MIN_WORD_COUNT
        or stage["comprehension_score"] < CLARIFICATION_TRIGGER_SCORE
        or stage["contradiction_score"] >= HIGH_CONTRADICTION_THRESHOLD
    )

    output_1: Output1 = {
        "day": day,
        "comprehension_score": stage["comprehension_score"],
        "needs_clarification": needs_clarification,
        "weak_topics": weak_topics,
        "user_input": user_input,
        "retrieved_chunks": stage["retrieved_chunks"],
        "nli_results": stage["nli_results"],
        "coverage_score": stage["coverage_score"],
        "contradiction_score": stage["contradiction_score"],
    }

    score = output_1["comprehension_score"]
    hit_max_loops = loop_count >= state.get("max_loops", MAX_LOOPS)

    if hit_max_loops:
        exit_reason = "max_loops"
        store_ready = False
    elif score >= COMPREHENSION_PASS_SCORE and not output_1["needs_clarification"]:
        exit_reason = "done"
        store_ready = True
    else:
        exit_reason = "continue"
        store_ready = False

    return {
        **state,
        "output_1": output_1,
        "weak_topics": output_1["weak_topics"] or [],
        "store_ready": store_ready,
        "exit_reason": exit_reason,
        "loop_count": loop_count,
        "max_loops": state.get("max_loops", MAX_LOOPS),
    }
