"""NLI pipeline tests: HF Inference embeddings (requires HF_TOKEN) + local BART-MNLI (transformers)"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from book_companion.config import TOP_K_RETRIEVAL
from book_companion.nodes.comprehension import (
    MERGED_PREMISE_CHUNK_ID,
    run_premise_hypothesis_pipeline,
)

_TEST_INFER = Path(__file__).resolve().parent / "test_infer"


requires_hf = pytest.mark.skipif(
    not os.environ.get("HF_TOKEN"),
    reason="HF_TOKEN not set (e.g. load .env.dev or export HF_TOKEN)",
)


@requires_hf
def test_run_premise_hypothesis_pipeline_structure() -> None:
    article = (_TEST_INFER / "test_article.txt").read_text(encoding="utf-8")
    hypothesis = (_TEST_INFER / "test_hypothesis.txt").read_text(encoding="utf-8")
    k = min(2, TOP_K_RETRIEVAL)

    out = run_premise_hypothesis_pipeline(article, hypothesis, k=k)

    assert len(out["retrieved_chunks"]) <= k
    assert len(out["retrieved_chunks"]) >= 1
    assert len(out["nli_results"]) == 1
    assert out["nli_results"][0]["chunk_id"] == MERGED_PREMISE_CHUNK_ID

    ent = out["nli_results"][0]["entailment"]
    neu = out["nli_results"][0]["neutral"]
    con = out["nli_results"][0]["contradiction"]
    assert 0.0 <= ent <= 2.0
    assert 0.0 <= neu <= 1.0
    assert 0.0 <= con <= 1.0

    assert "comprehension_score" in out
    assert "coverage_score" in out
    assert "contradiction_score" in out
