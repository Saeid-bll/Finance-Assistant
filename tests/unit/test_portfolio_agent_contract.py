import pytest


pytestmark = [pytest.mark.unit, pytest.mark.contract]


def test_portfolio_agent_calculates_total_value(require_attr, sample_holdings) -> None:
    PortfolioAnalysisAgent = require_attr("agents.portfolio", "PortfolioAnalysisAgent")
    agent = PortfolioAnalysisAgent()

    result = agent.analyze(sample_holdings)

    assert result.total_value == pytest.approx(4780.00)


def test_portfolio_agent_calculates_allocation_percentages(require_attr, sample_holdings) -> None:
    PortfolioAnalysisAgent = require_attr("agents.portfolio", "PortfolioAnalysisAgent")
    agent = PortfolioAnalysisAgent()

    result = agent.analyze(sample_holdings)

    assert result.allocations["VTI"] == pytest.approx(2500 / 4780, rel=1e-3)
    assert result.allocations["VXUS"] == pytest.approx(1200 / 4780, rel=1e-3)
    assert result.allocations["BND"] == pytest.approx(1080 / 4780, rel=1e-3)


def test_portfolio_agent_flags_concentration_risk(require_attr, concentrated_holdings) -> None:
    PortfolioAnalysisAgent = require_attr("agents.portfolio", "PortfolioAnalysisAgent")
    agent = PortfolioAnalysisAgent()

    result = agent.analyze(concentrated_holdings)

    assert result.concentration_warnings
    assert any("AAPL" in warning for warning in result.concentration_warnings)


def test_portfolio_agent_rejects_empty_portfolio(require_attr) -> None:
    PortfolioAnalysisAgent = require_attr("agents.portfolio", "PortfolioAnalysisAgent")
    agent = PortfolioAnalysisAgent()

    with pytest.raises(ValueError):
        agent.analyze([])


def test_portfolio_agent_includes_educational_disclaimer(require_attr, sample_holdings) -> None:
    PortfolioAnalysisAgent = require_attr("agents.portfolio", "PortfolioAnalysisAgent")
    agent = PortfolioAnalysisAgent()

    result = agent.analyze(sample_holdings)

    assert "not financial advice" in result.disclaimer.lower()
