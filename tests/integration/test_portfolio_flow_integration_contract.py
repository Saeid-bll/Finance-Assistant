import pytest


pytestmark = [pytest.mark.integration, pytest.mark.contract]
pending = pytest.mark.xfail(strict=True, reason="Pending portfolio flow integration")


@pending
def test_portfolio_flow_validates_analyzes_and_returns_chart_ready_data(
    require_attr, sample_holdings
) -> None:
    PortfolioAnalysisAgent = require_attr("agents.portfolio", "PortfolioAnalysisAgent")
    format_portfolio_chart_data = require_attr("utils.formatting", "format_portfolio_chart_data")

    agent = PortfolioAnalysisAgent()
    analysis = agent.analyze(sample_holdings)
    chart_data = format_portfolio_chart_data(analysis)

    assert analysis.total_value == pytest.approx(4780.00)
    assert chart_data["labels"] == ["VTI", "VXUS", "BND"]
    assert sum(chart_data["values"]) == pytest.approx(1.0)


@pending
def test_portfolio_flow_returns_beginner_friendly_error_for_bad_input(require_attr) -> None:
    PortfolioAnalysisAgent = require_attr("agents.portfolio", "PortfolioAnalysisAgent")
    agent = PortfolioAnalysisAgent()

    response = agent.run({"holdings": [{"ticker": "VTI", "quantity": "many"}]})

    assert response.error_code == "INVALID_PORTFOLIO"
    assert "quantity" in response.content.lower()

