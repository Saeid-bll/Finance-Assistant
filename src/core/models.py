"""Shared data models for the AI Finance Assistant."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


def _clean_non_empty(value: str, field_name: str) -> str:
    cleaned = value.strip()
    if not cleaned:
        raise ValueError(f"{field_name} cannot be empty")
    return cleaned


class Citation(BaseModel):
    """Traceable source reference used in educational responses."""

    source_id: str = Field(..., min_length=1)
    title: str = Field(..., min_length=1)
    url: Optional[str] = None
    excerpt: str = Field(..., min_length=1)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("source_id", "title", "excerpt")
    @classmethod
    def validate_required_text(cls, value: str, info) -> str:
        return _clean_non_empty(value, info.field_name)

    @field_validator("url")
    @classmethod
    def validate_url(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return value
        cleaned = value.strip()
        if not cleaned.startswith(("http://", "https://")):
            raise ValueError("url must start with http:// or https://")
        return cleaned


class UserQuery(BaseModel):
    """A normalized user query entering the assistant workflow."""

    text: str = Field(..., min_length=1)
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("text")
    @classmethod
    def validate_text(cls, value: str) -> str:
        return _clean_non_empty(value, "text")


class AgentResponse(BaseModel):
    """Structured response returned by an assistant agent."""

    agent_name: str = Field(..., min_length=1)
    content: str = Field(..., min_length=1)
    citations: List[Citation] = Field(default_factory=list)
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    disclaimer: Optional[str] = None
    error_code: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("agent_name", "content")
    @classmethod
    def validate_required_text(cls, value: str, info) -> str:
        return _clean_non_empty(value, info.field_name)


class PortfolioHolding(BaseModel):
    """A single holding supplied by a user for portfolio analysis."""

    ticker: str = Field(..., min_length=1)
    quantity: float = Field(..., gt=0)
    price: float = Field(..., ge=0)
    asset_type: str = Field(..., min_length=1)
    expense_ratio: Optional[float] = Field(default=None, ge=0)
    name: Optional[str] = None

    @field_validator("ticker")
    @classmethod
    def normalize_ticker(cls, value: str) -> str:
        return _clean_non_empty(value, "ticker").upper()

    @field_validator("asset_type")
    @classmethod
    def validate_asset_type(cls, value: str) -> str:
        return _clean_non_empty(value, "asset_type")

    @property
    def market_value(self) -> float:
        return self.quantity * self.price


class PortfolioAnalysis(BaseModel):
    """Calculated portfolio analysis output."""

    total_value: float = Field(..., ge=0)
    allocations: Dict[str, float] = Field(default_factory=dict)
    concentration_warnings: List[str] = Field(default_factory=list)
    diversification_score: float = Field(..., ge=0, le=100)
    educational_summary: str = Field(..., min_length=1)
    disclaimer: str = Field(..., min_length=1)


class MarketQuote(BaseModel):
    """Normalized market quote from a provider."""

    ticker: str = Field(..., min_length=1)
    price: float = Field(..., ge=0)
    currency: str = Field(default="USD", min_length=1)
    as_of: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    provider: str = Field(..., min_length=1)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("ticker", "currency")
    @classmethod
    def normalize_upper_text(cls, value: str, info) -> str:
        return _clean_non_empty(value, info.field_name).upper()

    @field_validator("provider")
    @classmethod
    def validate_provider(cls, value: str) -> str:
        return _clean_non_empty(value, "provider")


class MarketDataResult(BaseModel):
    """Market lookup result, including graceful error states."""

    ticker: str = Field(..., min_length=1)
    quote: Optional[MarketQuote] = None
    error_code: Optional[str] = None
    message: str = ""
    fallback_used: bool = False

    @field_validator("ticker")
    @classmethod
    def normalize_ticker(cls, value: str) -> str:
        return _clean_non_empty(value, "ticker").upper()

    @model_validator(mode="after")
    def validate_success_or_error(self) -> "MarketDataResult":
        if self.quote is None and not self.error_code:
            raise ValueError("market result must include either quote or error_code")
        return self


RiskAppetite = Literal["conservative", "moderate", "aggressive"]


class FinancialGoal(BaseModel):
    """User financial goal used for projection education."""

    name: str = Field(..., min_length=1)
    target_amount: float = Field(..., gt=0)
    current_amount: float = Field(default=0, ge=0)
    monthly_contribution: float = Field(default=0, ge=0)
    time_horizon_years: float = Field(..., gt=0)
    risk_appetite: RiskAppetite = "moderate"

    @field_validator("name")
    @classmethod
    def validate_name(cls, value: str) -> str:
        return _clean_non_empty(value, "name")


class GoalProjection(BaseModel):
    """Projection output for a user's financial goal."""

    name: str = Field(..., min_length=1)
    target_amount: float = Field(..., gt=0)
    projected_balance: float = Field(..., ge=0)
    shortfall_or_surplus: float
    expected_return: float = Field(..., ge=0)
    assumptions: List[str] = Field(default_factory=list)
    educational_summary: str = Field(..., min_length=1)
    disclaimer: str = Field(..., min_length=1)


class ChatMessage(BaseModel):
    """A conversation message stored in workflow state."""

    role: Literal["user", "assistant", "system"]
    content: str = Field(..., min_length=1)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("content")
    @classmethod
    def validate_content(cls, value: str) -> str:
        return _clean_non_empty(value, "content")


class ErrorResponse(BaseModel):
    """Consistent non-crashing error response."""

    error_code: str = Field(..., min_length=1)
    message: str = Field(..., min_length=1)
    recoverable: bool = True
    details: Dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(frozen=True)
