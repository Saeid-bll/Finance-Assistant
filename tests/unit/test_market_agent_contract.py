from datetime import timedelta

import pytest


pytestmark = [pytest.mark.unit, pytest.mark.contract]


def test_market_agent_normalizes_ticker_symbols(require_attr) -> None:
    MarketAnalysisAgent = require_attr("agents.market", "MarketAnalysisAgent")
    agent = MarketAnalysisAgent(provider=None)

    assert agent.normalize_ticker(" vti ") == "VTI"


def test_market_agent_returns_quote_with_freshness(require_attr, fake_market_quote) -> None:
    MarketAnalysisAgent = require_attr("agents.market", "MarketAnalysisAgent")
    agent = MarketAnalysisAgent(provider=lambda ticker: fake_market_quote)

    quote = agent.lookup("VTI")

    assert quote.ticker == "VTI"
    assert quote.price == pytest.approx(250.00)
    assert quote.provider == "test-provider"
    assert quote.as_of is not None


def test_market_agent_uses_cache_for_repeated_lookup(require_attr, fake_market_quote) -> None:
    MarketAnalysisAgent = require_attr("agents.market", "MarketAnalysisAgent")
    calls = {"count": 0}

    def provider(ticker):
        calls["count"] += 1
        return fake_market_quote

    agent = MarketAnalysisAgent(provider=provider, cache_ttl=timedelta(minutes=30))

    agent.lookup("VTI")
    agent.lookup("VTI")

    assert calls["count"] == 1


def test_market_agent_handles_unknown_ticker(require_attr) -> None:
    MarketAnalysisAgent = require_attr("agents.market", "MarketAnalysisAgent")
    agent = MarketAnalysisAgent(provider=lambda ticker: None)

    result = agent.lookup("NOT_A_TICKER")

    assert result.error_code == "UNKNOWN_TICKER"
    assert "could not find" in result.message.lower()


def test_market_agent_handles_provider_failure(require_attr) -> None:
    MarketAnalysisAgent = require_attr("agents.market", "MarketAnalysisAgent")

    def failing_provider(ticker):
        raise TimeoutError("market provider timed out")

    agent = MarketAnalysisAgent(provider=failing_provider)

    result = agent.lookup("VTI")

    assert result.error_code == "MARKET_DATA_UNAVAILABLE"
    assert result.fallback_used is True
