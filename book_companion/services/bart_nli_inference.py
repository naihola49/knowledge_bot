"""BART-MNLI via local transformers"""

from __future__ import annotations

import os
from typing import Protocol

import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from book_companion.config import HF_TOKEN_ENV_VAR, NLI_MAX_SEQ_LENGTH, NLI_MODEL_NAME


class BartNLIClient(Protocol):
    def predict(self, premise: str, hypothesis: str) -> tuple[float, float, float]:
        """Return (entailment, neutral, contradiction) on scales [0, 2], [0, 1], [0, 1]"""
        ...


def _probabilities_to_triple(
    contradiction_p: float,
    neutral_p: float,
    entailment_p: float,
) -> tuple[float, float, float]:
    """Match state.py NLIResult scaling: entailment up to 2, neutral/contradiction up to 1."""
    return (
        round(min(2.0, entailment_p * 2.0), 6),
        round(min(1.0, neutral_p * 1.0), 6),
        round(min(1.0, contradiction_p * 1.0), 6),
    )


def _logits_to_triple(logits: torch.Tensor) -> tuple[float, float, float]:
    """Map 3-class logits to (entailment, neutral, contradiction) scaled scores.

    facebook/bart-large-mnli: index 0 = contradiction, 1 = neutral, 2 = entailment.
    """
    probs = F.softmax(logits, dim=-1).squeeze(0)
    c, n, e = float(probs[0].item()), float(probs[1].item()), float(probs[2].item())
    return _probabilities_to_triple(c, n, e)


class LocalBartMNLIClient:
    """BART-MNLI: tokenizer(premise, hypothesis), softmax over three-class logits."""

    def __init__(
        self,
        model_name: str = NLI_MODEL_NAME,
        max_length: int = NLI_MAX_SEQ_LENGTH,
    ) -> None:
        self.model_name = model_name
        self.max_length = max_length
        self._tokenizer: AutoTokenizer | None = None
        self._model: AutoModelForSequenceClassification | None = None
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _ensure_loaded(self) -> None:
        if self._tokenizer is not None and self._model is not None:
            return
        token = os.getenv(HF_TOKEN_ENV_VAR)
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name, token=token)
        self._model = AutoModelForSequenceClassification.from_pretrained(self.model_name, token=token)
        self._model.eval()
        self._model.to(self._device)

    def predict(self, premise: str, hypothesis: str) -> tuple[float, float, float]:
        self._ensure_loaded()
        assert self._tokenizer is not None and self._model is not None

        inputs = self._tokenizer(
            premise,
            hypothesis,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        )
        inputs = {k: v.to(self._device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self._model(**inputs)

        return _logits_to_triple(outputs.logits)


_NLI_CLIENT: LocalBartMNLIClient | None = None


def get_bart_nli_client() -> LocalBartMNLIClient:
    global _NLI_CLIENT
    if _NLI_CLIENT is None:
        _NLI_CLIENT = LocalBartMNLIClient()
    return _NLI_CLIENT
