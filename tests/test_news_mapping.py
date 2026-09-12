from apps.api.db.models import Stock
from services.news.mapping import match_stock


def stock(symbol: str, company_name: str) -> Stock:
    return Stock(symbol=symbol, company_name=company_name)


def test_symbol_match_has_priority():
    stocks = [stock("TCS", "Tata Consultancy Services Limited"), stock("INFY", "Infosys Limited")]
    result = match_stock("TCS reports strong quarterly growth", stocks)
    assert result is not None
    assert result.symbol == "TCS"
    assert result.reason == "symbol"


def test_unique_company_name_match():
    stocks = [stock("TCS", "Tata Consultancy Services Limited"), stock("INFY", "Infosys Limited")]
    result = match_stock("Infosys Limited wins a large contract", stocks)
    assert result is not None
    assert result.symbol == "INFY"
    assert result.reason == "company_name"


def test_ambiguous_or_unmatched_news_is_not_guessed():
    stocks = [stock("ABC", "ABC Holdings Limited"), stock("XYZ", "XYZ Holdings Limited")]
    assert match_stock("The banking sector outlook improves", stocks) is None
