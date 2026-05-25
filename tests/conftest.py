"""Shared pytest fixtures for the AI Finance Assistant."""

from __future__ import annotations

import importlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import pytest


@pytest.fixture
def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def disable_external_services(monkeypatch) -> None:
    """Keep tests from calling external LLM and tracing services from a local .env file."""

    monkeypatch.setenv("GEMINI_API_KEY", "")
    monkeypatch.setenv("GOOGLE_API_KEY", "")
    monkeypatch.setenv("LANGSMITH_TRACING", "false")
    monkeypatch.setenv("LANGCHAIN_TRACING_V2", "false")
    monkeypatch.setenv("LANGSMITH_API_KEY", "")
    monkeypatch.setenv("LANGCHAIN_API_KEY", "")


@pytest.fixture
def require_attr() -> Callable[[str, str], Any]:
    def _require_attr(module_name: str, attr_name: str) -> Any:
        module = importlib.import_module(module_name)
        if not hasattr(module, attr_name):
            raise AssertionError(f"{module_name}.{attr_name} is not implemented yet")
        return getattr(module, attr_name)

    return _require_attr


@pytest.fixture
def sample_holdings() -> list[dict[str, Any]]:
    return [
        {
            "ticker": "VTI",
            "quantity": 10,
            "price": 250.00,
            "asset_type": "ETF",
            "expense_ratio": 0.0003,
        },
        {
            "ticker": "VXUS",
            "quantity": 20,
            "price": 60.00,
            "asset_type": "ETF",
            "expense_ratio": 0.0007,
        },
        {
            "ticker": "BND",
            "quantity": 15,
            "price": 72.00,
            "asset_type": "Bond ETF",
            "expense_ratio": 0.0003,
        },
    ]


@pytest.fixture
def concentrated_holdings() -> list[dict[str, Any]]:
    return [
        {
            "ticker": "AAPL",
            "quantity": 20,
            "price": 200.00,
            "asset_type": "Stock",
            "expense_ratio": None,
        },
        {
            "ticker": "VTI",
            "quantity": 1,
            "price": 250.00,
            "asset_type": "ETF",
            "expense_ratio": 0.0003,
        },
    ]


@pytest.fixture
def sample_goal() -> dict[str, Any]:
    return {
        "name": "Emergency fund and first investment goal",
        "target_amount": 10000.00,
        "current_amount": 1500.00,
        "monthly_contribution": 250.00,
        "time_horizon_years": 3,
        "risk_appetite": "moderate",
    }


@pytest.fixture
def sample_documents() -> list[dict[str, str]]:
    return [
        {
            "source_id": "investor-gov-diversification",
            "title": "Diversification",
            "url": "https://www.investor.gov/introduction-investing/investing-basics/glossary/diversification",
            "content": "Diversification means spreading investments across different assets to manage risk.",
        },
        {
            "source_id": "investor-gov-compound-interest",
            "title": "Compound Interest",
            "url": "https://www.investor.gov/financial-tools-calculators/calculators/compound-interest-calculator",
            "content": "Compound interest is interest earned on both principal and prior interest.",
        },
    ]


@pytest.fixture
def fake_market_quote() -> dict[str, Any]:
    return {
        "ticker": "VTI",
        "price": 250.00,
        "currency": "USD",
        "as_of": datetime(2026, 5, 20, 19, 0, tzinfo=timezone.utc),
        "provider": "test-provider",
    }
