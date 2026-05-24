import json

import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel


pytestmark = [pytest.mark.unit, pytest.mark.contract]


@pytest.mark.parametrize(
    ("query", "expected_route"),
    [
        ("What is an ETF?", "finance_qa"),
        ("Analyze my portfolio: 10 VTI and 5 BND", "portfolio"),
        ("What is the current price of VTI?", "market"),
        ("How much should I save each month for a house?", "goals"),
        ("What is a Roth IRA?", "tax"),
        ("Summarize today's market news", "news"),
    ],
)
def test_router_classifies_supported_intents(require_attr, query, expected_route) -> None:
    route_query = require_attr("workflow.router", "route_query")

    route = route_query(query, llm=_fake_router_llm(expected_route))

    assert route["agent_name"] == expected_route


def test_router_falls_back_for_ambiguous_query(require_attr) -> None:
    route_query = require_attr("workflow.router", "route_query")

    route = route_query("Can you help me?", llm=_fake_router_llm("finance_qa", clarify=True))

    assert route["agent_name"] == "finance_qa"
    assert route["needs_clarification"] is True


def test_workflow_state_preserves_conversation_history(require_attr) -> None:
    import operator
    from typing import Annotated, get_args, get_origin, get_type_hints

    WorkflowState = require_attr("workflow.state", "WorkflowState")
    initial_state = require_attr("workflow.state", "initial_state")

    hints = get_type_hints(WorkflowState, include_extras=True)
    messages_annotation = hints["messages"]

    assert get_origin(messages_annotation) is Annotated
    assert get_args(messages_annotation)[1] is operator.add
    assert initial_state()["messages"] == []


def test_graph_builds_compilable_langgraph(require_attr) -> None:
    build_graph = require_attr("workflow.graph", "build_graph")

    graph = build_graph(agents={})

    assert hasattr(graph, "invoke")


def _fake_router_llm(agent_name: str, *, clarify: bool = False) -> FakeListChatModel:
    return FakeListChatModel(
        responses=[
            json.dumps(
                {
                    "agent_name": agent_name,
                    "needs_clarification": clarify,
                    "confidence": 0.9,
                    "reason": "test route",
                }
            )
        ]
    )
