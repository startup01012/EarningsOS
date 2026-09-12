from services.sentiment.finbert import FinBERTSentiment


def test_finbert_adapter_maps_pretrained_output_without_loading_model():
    adapter = FinBERTSentiment()

    class FakePipeline:
        def __call__(self, text):
            assert text == "Strong quarterly growth"
            return [
                {"label": "positive", "score": 0.90},
                {"label": "negative", "score": 0.05},
                {"label": "neutral", "score": 0.05},
            ]

    adapter._pipeline = FakePipeline()
    result = adapter.predict("Strong quarterly growth")

    assert result.label == "positive"
    assert result.score == 0.90
    assert result.positive == 0.90
    assert result.negative == 0.05
    assert result.neutral == 0.05
