import pytest


pytestmark = [pytest.mark.integration, pytest.mark.contract]
pending = pytest.mark.xfail(strict=True, reason="Pending LangGraph workflow integration")


@pending
def test_workflow_routes_multi_turn_conversation(require_attr) -> None:
    build_graph = require_attr("workflow.graph", "build_graph")
    WorkflowState = require_attr("workflow.state", "WorkflowState")

    graph = build_graph()
    state = WorkflowState(user_id="demo")

    first = graph.invoke({"state": state, "message": "What is diversification?"})
    second = graph.invoke({"state": first["state"], "message": "How does that affect my portfolio?"})

    assert first["route"].agent_name == "finance_qa"
    assert second["route"].agent_name == "portfolio"
    assert len(second["state"].messages) >= 4


@pending
def test_workflow_falls_back_when_agent_raises(require_attr) -> None:
    build_graph = require_attr("workflow.graph", "build_graph")

    class FailingAgent:
        def run(self, state):
            raise RuntimeError("agent failed")

    graph = build_graph(agents={"market": FailingAgent()})

    result = graph.invoke({"message": "What is the price of VTI?"})

    assert result["response"].error_code == "AGENT_UNAVAILABLE"
    assert "try again" in result["response"].content.lower()


@pending
def test_workflow_preserves_financial_safety_disclaimer(require_attr) -> None:
    build_graph = require_attr("workflow.graph", "build_graph")
    graph = build_graph()

    result = graph.invoke({"message": "Should I buy VTI?"})

    assert "not financial advice" in result["response"].content.lower()

