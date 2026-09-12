from services.sentiment.finbert import FinBERTSentiment, SentimentResult
from services.sentiment.service import SentimentService


def test_sentiment_text_uses_title_and_summary():
    class Article:
        title = "Quarterly revenue rises"
        summary = "Operating profit also improved."

    assert SentimentService._text(Article()) == (
        "Quarterly revenue rises\nOperating profit also improved."
    )


def test_sentiment_adapter_can_be_injected_without_loading_model():
    class FakeAdapter(FinBERTSentiment):
        model_id = "test-finbert"

        def predict(self, text: str) -> SentimentResult:
            assert text == "Positive news"
            return SentimentResult(
                label="positive",
                score=0.9,
                positive=0.9,
                negative=0.05,
                neutral=0.05,
            )

    service = SentimentService(adapter=FakeAdapter())
    assert service.adapter.model_id == "test-finbert"
