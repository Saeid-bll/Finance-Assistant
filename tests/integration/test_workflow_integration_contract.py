import json

import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel


pytestmark = [pytest.mark.integration, pytest.mark.contract]


def test_workflow_routes_multi_turn_conversation(require_attr) -> None:
    build_graph = require_attr("workflow.graph", "build_graph")
    initial_state = require_attr("workflow.state", "initial_state")

    graph = build_graph(
        router_llm=FakeListChatModel(
            responses=[
                _route_response("finance_qa"),
                _route_response("portfolio"),
            ]
        )
    )
    state = initial_state(user_id="demo")

    first = graph.invoke({**state, "message": "What is diversification?"})
    second = graph.invoke({**first, "message": "How does that affect my portfolio?"})

    assert first["route"]["agent_name"] == "finance_qa"
    assert second["route"]["agent_name"] == "portfolio"
    assert len(second["messages"]) >= 4


def test_workflow_checkpoint_memory_preserves_thread_history(require_attr) -> None:
    build_graph = require_attr("workflow.graph", "build_graph")
    conversation_config = require_attr("workflow.graph", "conversation_config")
    create_memory_checkpointer = require_attr("workflow.graph", "create_memory_checkpointer")

    graph = build_graph(
        checkpointer=create_memory_checkpointer(),
        router_llm=FakeListChatModel(
            responses=[
                _route_response("finance_qa"),
                _route_response("portfolio"),
            ]
        ),
    )
    config = conversation_config("demo-thread")

    first = graph.invoke({"message": "What is diversification?"}, config=config)
    second = graph.invoke(
        {"message": "How does that affect my portfolio?"},
        config=config,
    )

    assert first["messages"][0].content == "What is diversification?"
    assert second["route"]["agent_name"] == "portfolio"
    assert [message.type for message in second["messages"]] == ["human", "ai", "human", "ai"]


def test_workflow_falls_back_when_agent_raises(require_attr) -> None:
    build_graph = require_attr("workflow.graph", "build_graph")

    class FailingAgent:
        def run(self, state):
            raise RuntimeError("agent failed")

    graph = build_graph(agents={"market": FailingAgent()})

    result = graph.invoke({"message": "What is the price of VTI?"})

    assert result["response"].error_code == "AGENT_UNAVAILABLE"
    assert "try again" in result["response"].content.lower()


def test_workflow_preserves_financial_safety_disclaimer(require_attr) -> None:
    build_graph = require_attr("workflow.graph", "build_graph")
    graph = build_graph()

    result = graph.invoke({"message": "Should I buy VTI?"})

    assert "not financial advice" in result["response"].content.lower()


def _route_response(agent_name: str) -> str:
    return json.dumps(
        {
            "agent_name": agent_name,
            "needs_clarification": False,
            "confidence": 0.9,
            "reason": "test route",
        }
    )
