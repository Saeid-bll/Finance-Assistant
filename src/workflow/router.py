"""LLM-backed intent routing for the assistant workflow."""

from __future__ import annotations

import os
import re
from functools import lru_cache
from typing import Any, Optional

from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from typing_extensions import Literal, TypedDict

from core.config import load_config


RouteAgent = Literal["finance_qa", "portfolio", "market", "goals", "tax", "news", "workflow"]


class WorkflowRoute(TypedDict):
    """Selected agent route for one user message."""

    agent_name: RouteAgent
    needs_clarification: bool
    confidence: float
    reason: str


SUPPORTED_ROUTER_AGENTS = {"finance_qa", "portfolio", "market", "goals", "tax", "news"}
KNOWN_ROUTE_AGENTS = {*SUPPORTED_ROUTER_AGENTS, "workflow"}

ROUTER_SYSTEM_PROMPT = """
You are the routing controller for an AI finance assistant. Your only job is to inspect
one user message and choose the single best assistant agent to handle it. You do not
answer the user, provide finance education, retrieve documents, calculate results, or
make recommendations. You classify intent.

Available agents:

1. finance_qa
Use this for beginner finance education and conceptual investing questions. Route here
when the user asks what something means, how a general concept works, or asks for a safe
educational explanation. Examples include ETFs, mutual funds, index funds, bonds, stocks,
diversification, asset classes, compounding, interest, risk, volatility, dollar-cost
averaging, expense ratios, emergency funds, inflation, brokerage accounts, and general
"how should I think about this?" questions. Also route unsafe personalized prediction or
recommendation requests here, because this agent has safety handling for requests like
"Should I buy/sell/hold X?", "Which stock will double?", "Will X crash?", and other
personalized trading advice.

2. portfolio
Use this when the user wants analysis of holdings, allocations, concentration, asset mix,
portfolio diversification, portfolio risk, rebalancing education, or feedback on a list of
positions. Route here even if the holdings are incomplete; the portfolio agent can ask for
or work with portfolio context. Strong signals include words such as portfolio, holdings,
allocation, exposure, concentration, diversification of my account, rebalance, asset mix,
and lists of tickers/quantities that represent a user's investments.

3. market
Use this for market-data lookup or ticker quote requests. Route here when the user asks
for a current price, quote, market price, ticker lookup, latest trading value, or recent
market data for a security. Strong signals include price, quote, ticker, current price,
trading at, market data, and questions like "What is VTI trading at?" If the user asks for
a prediction or a buy/sell decision, do not route to market; route to finance_qa for safe
educational handling.

4. goals
Use this for financial goal planning, savings projections, contribution planning, time
horizon planning, retirement goal calculations, house/down payment planning, emergency
fund targets, and "how much should I save per month" style questions. The goals agent is
for educational projections and planning assumptions, not personalized guarantees.

5. tax
Use this for tax-education topics and tax-advantaged account concepts. Strong signals
include Roth IRA, traditional IRA, 401(k), HSA, tax deduction, deductible, capital gains,
tax loss harvesting, taxable brokerage, contribution limits, and tax treatment. If the
question combines taxes with a goal or portfolio, choose tax when the main user intent is
understanding tax rules or tax concepts.

6. news
Use this for requests to summarize market news, headlines, recent events, macro news,
earnings news, or "what happened in markets today" style questions. Use news only for
news/context requests. If the user asks for a live quote, route to market. If the user
asks what to buy because of news, route to finance_qa for safety.

Clarification policy:
- Set needs_clarification to true when the message is too vague to choose a helpful agent,
  such as "Can you help me?", "What should I do?", or "I need advice" without enough
  financial topic detail.
- Set needs_clarification to false when one agent is clearly best, even if the agent may
  later ask for missing details.

Safety policy:
- Do not route personalized investment advice, predictions, or trade instructions to an
  execution-like agent. Requests asking what to buy, sell, hold, predict, double, moon,
  crash, or guarantee should go to finance_qa so the assistant can respond educationally
  and safely.
- The assistant is educational only. Routing must never imply financial, tax, legal, or
  investment advice.

Output contract:
- Return exactly one JSON object and no markdown.
- JSON keys must be: agent_name, needs_clarification, confidence, reason.
- agent_name must be one of: finance_qa, portfolio, market, goals, tax, news.
- needs_clarification must be a boolean.
- confidence must be a number from 0.0 to 1.0.
- reason must be one concise sentence explaining the routing decision.
""".strip()

ROUTER_HUMAN_PROMPT = """
User message:
{query}

Choose the best route using the routing rules. Return only the JSON object.
""".strip()


def create_workflow_route(
    agent_name: str,
    *,
    needs_clarification: bool = False,
    confidence: float = 1.0,
    reason: str = "",
) -> WorkflowRoute:
    """Create a normalized route typed dict."""

    normalized_agent = agent_name.strip().lower()
    if normalized_agent not in KNOWN_ROUTE_AGENTS:
        normalized_agent = "finance_qa"
        needs_clarification = True
        confidence = min(confidence, 0.25)
        reason = reason or f"Unsupported route '{agent_name}' returned by router."

    return {
        "agent_name": normalized_agent,  # type: ignore[typeddict-item]
        "needs_clarification": bool(needs_clarification),
        "confidence": _clamp_confidence(confidence),
        "reason": str(reason or "").strip(),
    }


def route_query(query: str, *, llm: Optional[Any] = None) -> WorkflowRoute:
    """Classify a user query into the agent that should handle it."""

    cleaned = (query or "").strip()
    if not cleaned:
        return create_workflow_route(
            agent_name="finance_qa",
            needs_clarification=True,
            confidence=0.0,
            reason="empty_query",
        )

    router_llm = llm if llm is not None else _default_router_llm()
    if router_llm is not None:
        try:
            raw_route = _router_chain(router_llm).invoke({"query": cleaned})
            return _normalize_route(raw_route)
        except Exception:
            pass

    return _fallback_route(cleaned)


def _router_chain(llm: Any) -> Any:
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", ROUTER_SYSTEM_PROMPT),
            ("human", ROUTER_HUMAN_PROMPT),
        ]
    )
    return prompt | llm | JsonOutputParser()


@lru_cache(maxsize=1)
def _default_router_llm() -> Optional[Any]:
    config = load_config(require_api_keys=False)
    provider = config.llm.provider.strip().lower()
    if provider != "gemini":
        return None

    api_key = config.llm.api_key or os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not api_key:
        return None

    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
    except ImportError as exc:
        raise RuntimeError(
            "Gemini routing requires the langchain-google-genai package. "
            "Install project dependencies with `python -m pip install -r requirements.txt`."
        ) from exc

    return ChatGoogleGenerativeAI(
        model=config.llm.model,
        temperature=0,
        max_tokens=config.llm.max_output_tokens,
        api_key=api_key,
        response_mime_type="application/json",
    )


def _normalize_route(raw_route: Any) -> WorkflowRoute:
    if not isinstance(raw_route, dict):
        return create_workflow_route(
            agent_name="finance_qa",
            needs_clarification=True,
            confidence=0.2,
            reason="Router returned a non-JSON response.",
        )

    return create_workflow_route(
        agent_name=str(raw_route.get("agent_name") or ""),
        needs_clarification=bool(raw_route.get("needs_clarification", False)),
        confidence=_coerce_confidence(raw_route.get("confidence", 0.5)),
        reason=str(raw_route.get("reason") or ""),
    )


def _fallback_route(query: str) -> WorkflowRoute:
    lowered = query.lower()
    if _matches(lowered, r"\b(news|headline|headlines|summarize.+market news)\b"):
        return create_workflow_route(agent_name="news", reason="fallback_news_terms")
    if _matches(lowered, r"\b(roth|ira|401k|tax|taxes|hsa|deduction|deductible)\b"):
        return create_workflow_route(agent_name="tax", reason="fallback_tax_terms")
    if _matches(lowered, r"\b(portfolio|holding|holdings|allocation|allocations|concentration)\b"):
        return create_workflow_route(agent_name="portfolio", reason="fallback_portfolio_terms")
    if _matches(lowered, r"\b(price|quote|ticker|market price|current price|trading at)\b"):
        return create_workflow_route(agent_name="market", reason="fallback_market_terms")
    if _matches(lowered, r"\b(goal|save|saving|savings|house|retirement|contribution|each month)\b"):
        return create_workflow_route(agent_name="goals", reason="fallback_goal_terms")
    if _matches(lowered, r"\b(should i|which stock|double next week|predict|buy|sell|hold)\b"):
        return create_workflow_route(agent_name="finance_qa", reason="fallback_financial_safety_terms")
    if _matches(lowered, r"\b(etf|fund|bond|stock|invest|investing|diversification|compound|interest)\b"):
        return create_workflow_route(agent_name="finance_qa", reason="fallback_education_terms")

    return create_workflow_route(
        agent_name="finance_qa",
        needs_clarification=True,
        confidence=0.35,
        reason="fallback_needs_clarification",
    )


def _coerce_confidence(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.5


def _clamp_confidence(value: float) -> float:
    return max(0.0, min(1.0, value))


def _matches(text: str, pattern: str) -> bool:
    return re.search(pattern, text) is not None
