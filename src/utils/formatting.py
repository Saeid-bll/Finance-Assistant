"""Formatting helpers for UI-ready outputs."""

from __future__ import annotations

from core.models import GoalProjection, MarketDataResult, MarketQuote, PortfolioAnalysis


def format_portfolio_chart_data(analysis: PortfolioAnalysis) -> dict[str, list[float] | list[str]]:
    """Convert portfolio allocations into chart-ready labels and values."""

    return {
        "labels": list(analysis.allocations.keys()),
        "values": list(analysis.allocations.values()),
    }


def format_market_response(result: MarketQuote | MarketDataResult) -> str:
    """Format a market quote or graceful market-data error."""

    if isinstance(result, MarketDataResult):
        return result.message
    return (
        f"{result.ticker}: {result.price:.2f} {result.currency} "
        f"as of {result.as_of.isoformat()} from {result.provider}."
    )


def format_goal_summary(projection: GoalProjection) -> str:
    """Format a goal projection for display."""

    assumptions = " ".join(projection.assumptions)
    return (
        f"{projection.name}: projected balance ${projection.projected_balance:,.2f}; "
        f"target ${projection.target_amount:,.2f}; "
        f"shortfall or surplus ${projection.shortfall_or_surplus:,.2f}. "
        f"{projection.educational_summary} Assumptions: {assumptions}"
    )
