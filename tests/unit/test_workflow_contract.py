import pytest


pytestmark = [pytest.mark.unit, pytest.mark.contract]
pending = pytest.mark.xfail(strict=True, reason="Pending workflow implementation")


@pending
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

    route = route_query(query)

    assert route.agent_name == expected_route


@pending
def test_router_falls_back_for_ambiguous_query(require_attr) -> None:
    route_query = require_attr("workflow.router", "route_query")

    route = route_query("Can you help me?")

    assert route.agent_name == "finance_qa"
    assert route.needs_clarification is True


@pending
def test_workflow_state_preserves_conversation_history(require_attr) -> None:
    WorkflowState = require_attr("workflow.state", "WorkflowState")

    state = WorkflowState()
    state = state.add_user_message("What is an ETF?")
    state = state.add_agent_message("An ETF is an exchange-traded fund.")

    assert len(state.messages) == 2
    assert state.messages[-1].role == "assistant"


@pending
def test_graph_builds_compilable_langgraph(require_attr) -> None:
    build_graph = require_attr("workflow.graph", "build_graph")

    graph = build_graph(agents={})

    assert hasattr(graph, "invoke")

