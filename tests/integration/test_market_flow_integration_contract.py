import pytest


pytestmark = [pytest.mark.integration, pytest.mark.contract]


def test_market_flow_uses_mocked_provider_and_cache(require_attr, fake_market_quote) -> None:
    MarketAnalysisAgent = require_attr("agents.market", "MarketAnalysisAgent")
    TTLCache = require_attr("utils.cache", "TTLCache")
    calls = {"count": 0}

    def provider(ticker):
        calls["count"] += 1
        return fake_market_quote

    agent = MarketAnalysisAgent(provider=provider, cache=TTLCache(ttl_seconds=1800))

    first = agent.lookup("VTI")
    second = agent.lookup("VTI")

    assert first.price == second.price
    assert calls["count"] == 1


def test_market_flow_reports_data_freshness(require_attr, fake_market_quote) -> None:
    MarketAnalysisAgent = require_attr("agents.market", "MarketAnalysisAgent")
    format_market_response = require_attr("utils.formatting", "format_market_response")
    agent = MarketAnalysisAgent(provider=lambda ticker: fake_market_quote)

    quote = agent.lookup("VTI")
    message = format_market_response(quote)

    assert "as of" in message.lower()
    assert "test-provider" in message
