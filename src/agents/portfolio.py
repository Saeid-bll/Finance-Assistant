"""Portfolio analysis agent."""

from __future__ import annotations

from typing import Any, Iterable

from pydantic import ValidationError

from agents.base import BaseAgent
from core.disclaimers import disclaimer_text
from core.models import AgentResponse, PortfolioAnalysis, PortfolioHolding
from core.tracing import traceable_span


class PortfolioAnalysisAgent(BaseAgent):
    """Calculate beginner-friendly portfolio metrics from supplied holdings."""

    agent_name = "portfolio"

    @traceable_span(name="portfolio.analyze", run_type="chain", tags=["agent", "portfolio"])
    def analyze(self, holdings: Iterable[dict[str, Any] | PortfolioHolding]) -> PortfolioAnalysis:
        parsed_holdings = [self._parse_holding(holding) for holding in holdings]
        if not parsed_holdings:
            raise ValueError("portfolio must include at least one holding")

        values = {holding.ticker: holding.market_value for holding in parsed_holdings}
        total_value = sum(values.values())
        if total_value <= 0:
            raise ValueError("portfolio total value must be positive")

        allocations = {ticker: value / total_value for ticker, value in values.items()}
        warnings = self._concentration_warnings(allocations)
        diversification_score = self._diversification_score(allocations)
        summary = self._summary(total_value, allocations, warnings, diversification_score)

        return PortfolioAnalysis(
            total_value=total_value,
            allocations=allocations,
            concentration_warnings=warnings,
            diversification_score=diversification_score,
            educational_summary=summary,
            disclaimer=disclaimer_text(),
        )

    @traceable_span(name="portfolio.run", run_type="chain", tags=["agent", "portfolio"])
    def run(self, payload: Any) -> AgentResponse:
        try:
            holdings = payload.get("holdings", []) if isinstance(payload, dict) else payload
            analysis = self.analyze(holdings)
        except (TypeError, ValueError, ValidationError) as exc:
            return self.response(
                f"I could not analyze this portfolio because the input looks invalid: {exc}",
                confidence=0.2,
                error_code="INVALID_PORTFOLIO",
            )

        return self.response(
            analysis.educational_summary,
            confidence=0.9,
            metadata={"analysis": analysis.model_dump()},
        )

    def _parse_holding(self, holding: dict[str, Any] | PortfolioHolding) -> PortfolioHolding:
        if isinstance(holding, PortfolioHolding):
            return holding
        return PortfolioHolding.model_validate(holding)

    @traceable_span(name="portfolio.concentration_warnings", run_type="tool", tags=["agent", "portfolio"])
    def _concentration_warnings(self, allocations: dict[str, float]) -> list[str]:
        warnings = []
        for ticker, allocation in allocations.items():
            if allocation >= 0.35:
                warnings.append(
                    f"{ticker} represents {allocation:.1%} of the portfolio, which may create concentration risk."
                )
        return warnings

    @traceable_span(name="portfolio.diversification_score", run_type="tool", tags=["agent", "portfolio"])
    def _diversification_score(self, allocations: dict[str, float]) -> float:
        largest_allocation = max(allocations.values())
        holding_count = len(allocations)
        concentration_penalty = max(0.0, largest_allocation - 0.25) * 80
        breadth_bonus = min(holding_count, 10) * 4
        score = 75 + breadth_bonus - concentration_penalty
        return max(0.0, min(100.0, score))

    def _summary(
        self,
        total_value: float,
        allocations: dict[str, float],
        warnings: list[str],
        diversification_score: float,
    ) -> str:
        largest_ticker = max(allocations, key=allocations.get)
        summary = (
            f"The portfolio value is ${total_value:,.2f}. "
            f"The largest holding is {largest_ticker} at {allocations[largest_ticker]:.1%}. "
            f"The diversification score is {diversification_score:.0f} out of 100."
        )
        if warnings:
            summary += " Review concentration risk before relying heavily on one holding."
        else:
            summary += " The holdings are spread across multiple positions."
        return summary
