"""Manual debug script for Node 2 topic extraction and research-brief concatenation.
"""

from __future__ import annotations

from book_companion.nodes.comprehension import MERGED_PREMISE_CHUNK_ID
from book_companion.nodes.node2_methods import build_topic_explanations, extract_candidate_topics
from book_companion.state import NLIResult, Output1, RetrievedChunk


def _make_retrieved_chunks() -> list[RetrievedChunk]:
    return [
        {
            "chunk_id": "chunk_0",
            "text": "Gradient descent updates parameters by following the negative gradient of a loss function.",
            "similarity": 0.74,
        },
        {
            "chunk_id": "chunk_1",
            "text": "Overfitting means a model memorizes training noise and fails to generalize to unseen samples.",
            "similarity": 0.59,
        },
        {
            "chunk_id": "chunk_2",
            "text": "Regularization methods like L2 and dropout can reduce variance and improve generalization.",
            "similarity": 0.48,
        },
    ]


def _make_nli_results() -> list[NLIResult]:
    return [
        {
            "chunk_id": MERGED_PREMISE_CHUNK_ID,
            "entailment": 0.62,
            "neutral": 0.19,
            "contradiction": 0.84,
        },
        {
            "chunk_id": "chunk_0",
            "entailment": 0.45,
            "neutral": 0.25,
            "contradiction": 0.73,
        },
        {
            "chunk_id": "chunk_1",
            "entailment": 0.39,
            "neutral": 0.41,
            "contradiction": 0.51,
        },
        {
            "chunk_id": "chunk_2",
            "entailment": 0.22,
            "neutral": 0.18,
            "contradiction": 0.81,
        },
        {
            # Not in retrieved chunks, but high contradiction to exercise that branch.
            "chunk_id": "unknown_claim_cluster",
            "entailment": 0.11,
            "neutral": 0.20,
            "contradiction": 0.91,
        },
    ]


def _make_output_1() -> Output1:
    return {
        "day": "day_debug",
        "comprehension_score": 0.44,
        "needs_clarification": True,
        # 8+ ids so we can test max topic cap behavior.
        "weak_topics": [
            "high contradiction overall",
            MERGED_PREMISE_CHUNK_ID,
            "chunk_0",
            "chunk_1",
            "chunk_2",
            "unknown_claim_cluster",
            "terminology_confusion",
            "causal_chain_missing",
            "quantitative_claims_unsupported",
            "timeline_inconsistency",
        ],
        "user_input": (
            "I think overfitting means the model always improves with more epochs and that "
            "dropout memorizes important examples. I also believe gradient descent increases "
            "loss first so the model can escape local minima."
        ),
        "retrieved_chunks": _make_retrieved_chunks(),
        "nli_results": _make_nli_results(),
        "coverage_score": 0.22,  # low on purpose: triggers low_evidence_coverage id
        "contradiction_score": 0.85,
    }


def main() -> None:
    output_1 = _make_output_1()

    print("\n=== Candidate Topic Extraction ===")
    extracted = extract_candidate_topics(output_1, max_topics=10)
    print(f"candidate_count={len(extracted)}")
    for i, topic in enumerate(extracted, start=1):
        print(f"{i:>2}. {topic}")

    # Force all branch ids so each explanation path is exercised.
    all_branch_topics = [
        "low_evidence_coverage",
        "overall_comprehension_gap",
        "high contradiction overall",
        MERGED_PREMISE_CHUNK_ID,
        "chunk_0",
        "unknown_claim_cluster",
        "terminology_confusion",
        "causal_chain_missing",
    ]

    print("\n=== Research Brief String Concatenation (all branches) ===")
    briefs = build_topic_explanations(output_1, all_branch_topics)
    print(f"brief_count={len(briefs)}")

    for i, row in enumerate(briefs, start=1):
        print("\n" + "=" * 100)
        print(f"[{i}] topic={row['topic']}  confidence={row['confidence']}")
        print("-" * 100)
        print(row["error_explanation"])
        print(f"\n[debug] explanation_len_chars={len(row['error_explanation'])}")


if __name__ == "__main__":
    main()

