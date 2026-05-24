import os

import pytest


pytestmark = pytest.mark.unit


def test_user_query_rejects_empty_text(require_attr) -> None:
    UserQuery = require_attr("core.models", "UserQuery")

    with pytest.raises(ValueError):
        UserQuery(text="")


def test_agent_response_requires_agent_name_and_content(require_attr) -> None:
    AgentResponse = require_attr("core.models", "AgentResponse")

    response = AgentResponse(agent_name="finance_qa", content="Diversification spreads risk.")

    assert response.agent_name == "finance_qa"
    assert response.content
    assert response.citations == []


def test_citation_requires_traceable_source(require_attr) -> None:
    Citation = require_attr("core.models", "Citation")

    citation = Citation(
        source_id="investor-gov-diversification",
        title="Diversification",
        url="https://www.investor.gov/example",
        excerpt="Diversification spreads investments across assets.",
    )

    assert citation.source_id
    assert citation.title
    assert citation.url.startswith("https://")


def test_citation_rejects_untraceable_url(require_attr) -> None:
    Citation = require_attr("core.models", "Citation")

    with pytest.raises(ValueError):
        Citation(
            source_id="bad-source",
            title="Bad Source",
            url="not-a-url",
            excerpt="This URL is not traceable.",
        )


def test_portfolio_holding_rejects_negative_quantity(require_attr) -> None:
    PortfolioHolding = require_attr("core.models", "PortfolioHolding")

    with pytest.raises(ValueError):
        PortfolioHolding(ticker="VTI", quantity=-1, price=250.00, asset_type="ETF")


def test_portfolio_holding_normalizes_ticker(require_attr) -> None:
    PortfolioHolding = require_attr("core.models", "PortfolioHolding")

    holding = PortfolioHolding(ticker=" vti ", quantity=2, price=250.00, asset_type="ETF")

    assert holding.ticker == "VTI"
    assert holding.market_value == pytest.approx(500.00)


def test_financial_goal_rejects_invalid_horizon(require_attr) -> None:
    FinancialGoal = require_attr("core.models", "FinancialGoal")

    with pytest.raises(ValueError):
        FinancialGoal(
            name="Retirement",
            target_amount=100000,
            current_amount=1000,
            monthly_contribution=500,
            time_horizon_years=0,
            risk_appetite="moderate",
        )


def test_financial_goal_rejects_unknown_risk_appetite(require_attr) -> None:
    FinancialGoal = require_attr("core.models", "FinancialGoal")

    with pytest.raises(ValueError):
        FinancialGoal(
            name="House",
            target_amount=50000,
            current_amount=1000,
            monthly_contribution=500,
            time_horizon_years=5,
            risk_appetite="reckless",
        )


def test_market_quote_normalizes_ticker_and_currency(require_attr, fake_market_quote) -> None:
    MarketQuote = require_attr("core.models", "MarketQuote")

    quote = MarketQuote(**{**fake_market_quote, "ticker": " vti ", "currency": " usd "})

    assert quote.ticker == "VTI"
    assert quote.currency == "USD"


def test_market_data_result_requires_quote_or_error(require_attr) -> None:
    MarketDataResult = require_attr("core.models", "MarketDataResult")

    with pytest.raises(ValueError):
        MarketDataResult(ticker="VTI")


def test_config_loader_reads_yaml(require_attr, project_root) -> None:
    load_config = require_attr("core.config", "load_config")

    config = load_config(project_root / "config.yaml", load_env_file=False)

    assert config.app.name == "AI Finance Assistant"
    assert config.rag.embedding_provider == "gemini"
    assert config.rag.embedding_model == "models/gemini-embedding-001"
    assert config.rag.embedding_dimensions == 768
    assert config.langsmith.tracing_enabled is False
    assert config.langsmith.project == "ai-finance-assistant"
    assert config.testing.coverage_target_percent == 80


def test_config_loader_applies_env_overrides(require_attr, project_root, monkeypatch) -> None:
    load_config = require_attr("core.config", "load_config")
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setenv("RAG_EMBEDDING_MODEL", "gemini-embedding-001")
    monkeypatch.setenv("RAG_EMBEDDING_DIMENSIONS", "1536")
    monkeypatch.setenv("ALPHA_VANTAGE_API_KEY", "alpha-key")
    monkeypatch.setenv("LANGSMITH_TRACING", "true")
    monkeypatch.setenv("LANGSMITH_API_KEY", "smith-key")
    monkeypatch.setenv("LANGSMITH_PROJECT", "finance-test")

    config = load_config(project_root / "config.yaml", load_env_file=False)

    assert config.llm.provider == "openai"
    assert config.llm.api_key == "test-key"
    assert config.rag.embedding_model == "gemini-embedding-001"
    assert config.rag.embedding_dimensions == 1536
    assert config.market_data.alpha_vantage_api_key == "alpha-key"
    assert config.langsmith.tracing_enabled is True
    assert config.langsmith.api_key == "smith-key"
    assert config.langsmith.project == "finance-test"


def test_langsmith_tracing_sets_langchain_environment(require_attr, monkeypatch) -> None:
    ProjectConfig = require_attr("core.config", "ProjectConfig")
    LangSmithSettings = require_attr("core.config", "LangSmithSettings")
    configure_langsmith_tracing = require_attr("core.tracing", "configure_langsmith_tracing")
    for key in (
        "LANGSMITH_TRACING",
        "LANGCHAIN_TRACING_V2",
        "LANGSMITH_API_KEY",
        "LANGCHAIN_API_KEY",
        "LANGSMITH_PROJECT",
        "LANGCHAIN_PROJECT",
    ):
        monkeypatch.delenv(key, raising=False)

    config = ProjectConfig(
        langsmith=LangSmithSettings(
            tracing_enabled=True,
            api_key="smith-key",
            project="finance-test",
        )
    )

    try:
        enabled = configure_langsmith_tracing(config)

        assert enabled is True
        assert os.environ["LANGSMITH_TRACING"] == "true"
        assert os.environ["LANGCHAIN_TRACING_V2"] == "true"
        assert os.environ["LANGSMITH_API_KEY"] == "smith-key"
        assert os.environ["LANGSMITH_PROJECT"] == "finance-test"
    finally:
        os.environ["LANGSMITH_TRACING"] = "false"
        os.environ["LANGCHAIN_TRACING_V2"] = "false"
        os.environ.pop("LANGSMITH_API_KEY", None)
        os.environ.pop("LANGCHAIN_API_KEY", None)


def test_config_loader_rejects_missing_file(require_attr, tmp_path) -> None:
    load_config = require_attr("core.config", "load_config")
    ConfigError = require_attr("core.config", "ConfigError")

    with pytest.raises(ConfigError):
        load_config(tmp_path / "missing.yaml", load_env_file=False)


def test_disclaimer_is_attached_to_educational_response(require_attr) -> None:
    append_disclaimer = require_attr("core.disclaimers", "append_disclaimer")

    response = append_disclaimer("Diversification can reduce concentration risk.")

    assert "education" in response.lower()
    assert "not financial advice" in response.lower()


def test_disclaimer_is_not_duplicated(require_attr) -> None:
    append_disclaimer = require_attr("core.disclaimers", "append_disclaimer")

    once = append_disclaimer("This is not financial advice.")
    twice = append_disclaimer(once)

    assert twice == once


def test_logging_rejects_unknown_level(require_attr) -> None:
    configure_logging = require_attr("core.logging", "configure_logging")

    with pytest.raises(ValueError):
        configure_logging("LOUD")


def test_logging_returns_named_logger(require_attr) -> None:
    configure_logging = require_attr("core.logging", "configure_logging")
    get_logger = require_attr("core.logging", "get_logger")

    logger = configure_logging("INFO", "ai_finance_assistant.test")

    assert get_logger("ai_finance_assistant.test") is logger
