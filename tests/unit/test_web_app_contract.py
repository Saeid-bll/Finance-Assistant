import json

import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel


pytestmark = [pytest.mark.unit, pytest.mark.contract]


def test_chat_runtime_runs_with_langgraph_memory(require_attr) -> None:
    create_chat_runtime = require_attr("web_app.chat", "create_chat_runtime")
    run_chat_turn = require_attr("web_app.chat", "run_chat_turn")

    runtime = create_chat_runtime(
        thread_id="test-chat",
        router_llm=FakeListChatModel(
            responses=[
                _route_response("finance_qa"),
                _route_response("portfolio"),
            ]
        ),
    )

    first = run_chat_turn(runtime, "What is diversification?")
    second = run_chat_turn(runtime, "How does that affect my portfolio?")

    assert first["messages"][0].type == "human"
    assert second["route"]["agent_name"] == "portfolio"
    assert [message.type for message in second["messages"]] == ["human", "ai", "human", "ai"]


def test_display_messages_maps_langchain_roles(require_attr) -> None:
    from langchain_core.messages import AIMessage, HumanMessage

    display_messages = require_attr("web_app.chat", "display_messages")

    messages = display_messages(
        {"messages": [HumanMessage(content="hello"), AIMessage(content="hi")]}
    )

    assert messages == [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi"},
    ]


def test_portfolio_rows_to_records_filters_blank_rows(require_attr) -> None:
    holding_rows_to_records = require_attr("web_app.portfolio_view", "holding_rows_to_records")

    records = holding_rows_to_records(
        [
            {"ticker": " vti ", "quantity": 2, "price": 250, "asset_type": "ETF"},
            {"ticker": "", "quantity": 1, "price": 10, "asset_type": "ETF"},
        ]
    )

    assert records == [
        {"ticker": "VTI", "quantity": 2.0, "price": 250.0, "asset_type": "ETF"}
    ]


def test_goal_payload_preserves_agent_fields(require_attr) -> None:
    goal_payload = require_attr("web_app.goals_view", "goal_payload")

    payload = goal_payload(
        name=" Retirement ",
        target_amount=100000,
        current_amount=5000,
        monthly_contribution=500,
        time_horizon_years=10,
        risk_appetite="moderate",
    )

    assert payload["name"] == "Retirement"
    assert payload["risk_appetite"] == "moderate"


def _route_response(agent_name: str) -> str:
    return json.dumps(
        {
            "agent_name": agent_name,
            "needs_clarification": False,
            "confidence": 0.9,
            "reason": "test route",
        }
    )
