"""Market analysis agent."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Dict, Optional, Union

from agents.base import BaseAgent
from core.models import AgentResponse, MarketDataResult, MarketQuote
from core.tracing import traceable_span
from utils.cache import TTLCache


MarketProvider = Callable[[str], Optional[Union[Dict[str, Any], MarketQuote]]]


class MarketAnalysisAgent(BaseAgent):
    """Look up market quotes with injectable providers and TTL caching."""

    agent_name = "market"

    def __init__(
        self,
        *,
        provider: Optional[MarketProvider] = None,
        cache: Optional[TTLCache] = None,
        cache_ttl: timedelta = timedelta(minutes=30),
    ) -> None:
        self.provider = provider or self._yfinance_provider
        self.cache = cache or TTLCache(ttl_seconds=cache_ttl.total_seconds())

    @traceable_span(name="market.normalize_ticker", run_type="tool", tags=["agent", "market"])
    def normalize_ticker(self, ticker: str) -> str:
        normalized = ticker.strip().upper()
        if not normalized:
            raise ValueError("ticker cannot be empty")
        return normalized

    @traceable_span(name="market.lookup", run_type="tool", tags=["agent", "market"])
    def lookup(self, ticker: str) -> MarketQuote | MarketDataResult:
        normalized = self.normalize_ticker(ticker)
        cache_key = f"quote:{normalized}"
        cached = self.cache.get(cache_key)
        if cached is not None:
            return cached

        try:
            raw_quote = self.provider(normalized)
        except Exception:
            return MarketDataResult(
                ticker=normalized,
                error_code="MARKET_DATA_UNAVAILABLE",
                message="Market data is temporarily unavailable. Please try again later.",
                fallback_used=True,
            )

        if raw_quote is None:
            return MarketDataResult(
                ticker=normalized,
                error_code="UNKNOWN_TICKER",
                message=f"Could not find market data for {normalized}.",
            )

        quote = self._normalize_quote(normalized, raw_quote)
        self.cache.set(cache_key, quote)
        return quote

    @traceable_span(name="market.run", run_type="chain", tags=["agent", "market"])
    def run(self, payload: Any) -> AgentResponse:
        ticker = payload.get("ticker") if isinstance(payload, dict) else payload
        result = self.lookup(str(ticker or ""))
        if isinstance(result, MarketDataResult):
            return self.response(
                result.message,
                confidence=0.2,
                error_code=result.error_code,
                metadata=result.model_dump(),
            )
        return self.response(
            f"{result.ticker} is trading at {result.price:.2f} {result.currency} as of {result.as_of.isoformat()}.",
            confidence=0.8,
            metadata={"quote": result.model_dump()},
        )

    @traceable_span(name="market.normalize_quote", run_type="tool", tags=["agent", "market"])
    def _normalize_quote(self, ticker: str, quote: dict[str, Any] | MarketQuote) -> MarketQuote:
        if isinstance(quote, MarketQuote):
            return quote
        return MarketQuote(
            ticker=str(quote.get("ticker") or ticker),
            price=float(quote["price"]),
            currency=str(quote.get("currency") or "USD"),
            as_of=quote.get("as_of") or datetime.now(timezone.utc),
            provider=str(quote.get("provider") or "unknown"),
            metadata=dict(quote.get("metadata") or {}),
        )

    @traceable_span(name="market.yfinance_provider", run_type="tool", tags=["agent", "market"])
    def _yfinance_provider(self, ticker: str) -> dict[str, Any] | None:
        import yfinance as yf

        info = yf.Ticker(ticker).fast_info
        price = getattr(info, "last_price", None)
        if price is None:
            price = info.get("last_price") if hasattr(info, "get") else None
        if price is None:
            return None
        currency = getattr(info, "currency", None)
        if currency is None:
            currency = info.get("currency", "USD") if hasattr(info, "get") else "USD"
        return {
            "ticker": ticker,
            "price": float(price),
            "currency": currency or "USD",
            "as_of": datetime.now(timezone.utc),
            "provider": "yfinance",
        }
