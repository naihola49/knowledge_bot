"""Minimum spacing between Hugging Face Inference API calls (embeddings + NLI)."""

from __future__ import annotations

import time

from book_companion.config import HF_INFERENCE_MIN_INTERVAL_SECONDS

_last_request_end: float = 0.0


def before_hf_request() -> None:
    """Sleep if needed so consecutive requests are at least HF_INFERENCE_MIN_INTERVAL_SECONDS apart"""
    global _last_request_end
    min_interval = HF_INFERENCE_MIN_INTERVAL_SECONDS
    if min_interval <= 0:
        return
    now = time.monotonic()
    if _last_request_end > 0:
        elapsed = now - _last_request_end
        wait = min_interval - elapsed
        if wait > 0:
            time.sleep(wait)


def after_hf_request() -> None:
    """Call after each HF Inference request completes (success or HTTP error)"""
    global _last_request_end
    _last_request_end = time.monotonic()
