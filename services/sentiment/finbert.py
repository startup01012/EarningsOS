from __future__ import annotations

from pydantic import BaseModel, Field


class SentimentResult(BaseModel):
    label: str
    score: float = Field(ge=0, le=1)
    positive: float = Field(ge=0, le=1)
    negative: float = Field(ge=0, le=1)
    neutral: float = Field(ge=0, le=1)


class FinBERTSentiment:
    """Pretrained financial sentiment adapter; no project-specific training."""

    model_id = "ProsusAI/finbert"

    def __init__(self, model_id: str | None = None):
        self.model_id = model_id or self.model_id
        self._pipeline = None

    def _load(self):
        if self._pipeline is None:
            from transformers import pipeline
            self._pipeline = pipeline(
                "text-classification",
                model=self.model_id,
                tokenizer=self.model_id,
                top_k=None,
            )
        return self._pipeline

    @staticmethod
    def _normalize_outputs(outputs) -> list[dict]:
        """Normalize Transformers single-input and batched output shapes."""
        if not outputs:
            return []
        if isinstance(outputs[0], list):
            return outputs[0]
        return outputs

    def predict(self, text: str) -> SentimentResult:
        if not text or not text.strip():
            raise ValueError("text must not be empty")
        outputs = self._normalize_outputs(self._load()(text[:4000]))
        scores = {item["label"].lower(): float(item["score"]) for item in outputs}
        if not scores:
            raise RuntimeError("FinBERT returned no sentiment scores")
        label = max(scores, key=scores.get)
        return SentimentResult(
            label=label,
            score=scores[label],
            positive=scores.get("positive", 0.0),
            negative=scores.get("negative", 0.0),
            neutral=scores.get("neutral", 0.0),
        )
