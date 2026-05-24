"""LangGraph workflow orchestration."""

from __future__ import annotations

import re
from typing import Any, Callable, Mapping, Optional

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableLambda
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph

from agents.finance_qa import FinanceQAAgent
from agents.goals import GoalPlanningAgent
from agents.market import MarketAnalysisAgent
from agents.portfolio import PortfolioAnalysisAgent
from core.disclaimers import append_disclaimer, disclaimer_text
from core.models import AgentResponse
from workflow.router import create_workflow_route, route_query
from workflow.state import WorkflowState


StateUpdate = dict[str, Any]

AGENT_NODES = ("finance_qa", "portfolio", "market", "goals")
STRETCH_NODES = ("tax", "news")


def build_graph(
    *,
    agents: Optional[Mapping[str, Any]] = None,
    router_llm: Optional[Any] = None,
    checkpointer: Optional[Any] = None,
) -> Any:
    """Build the assistant workflow as a compiled LangGraph graph."""

    agent_map = {**_default_agents(), **dict(agents or {})}

    graph = StateGraph(WorkflowState)
    graph.add_node("validate_input", RunnableLambda(_validate_input))
    graph.add_node("route_message", RunnableLambda(_route_message(router_llm)))
    graph.add_node("clarify", RunnableLambda(_clarify))
    graph.add_node("unavailable", RunnableLambda(_unavailable))
    graph.add_node("finalize", RunnableLambda(_finalize))

    for agent_name in AGENT_NODES:
        graph.add_node(agent_name, RunnableLambda(_agent_node(agent_name, agent_map)))
    for agent_name in STRETCH_NODES:
        graph.add_node(agent_name, RunnableLambda(_stretch_node(agent_name)))

    graph.set_entry_point("validate_input")
    graph.add_conditional_edges(
        "validate_input",
        _input_status,
        {
            "valid": "route_message",
            "invalid": "finalize",
        },
    )
    graph.add_conditional_edges(
        "route_message",
        _route_target,
        {
            "clarify": "clarify",
            "finance_qa": "finance_qa",
            "portfolio": "portfolio",
            "market": "market",
            "goals": "goals",
            "tax": "tax",
            "news": "news",
            "unavailable": "unavailable",
        },
    )

    for node_name in (*AGENT_NODES, *STRETCH_NODES, "clarify", "unavailable"):
        graph.add_edge(node_name, "finalize")
    graph.add_edge("finalize", END)
    return graph.compile(checkpointer=checkpointer)


def create_memory_checkpointer() -> MemorySaver:
    """Create an in-process LangGraph checkpointer for conversation memory."""

    return MemorySaver()


def conversation_config(thread_id: str) -> dict[str, Any]:
    """Create LangGraph config that selects a conversation memory thread."""

    cleaned = thread_id.strip()
    if not cleaned:
        raise ValueError("thread_id cannot be empty")
    return {"configurable": {"thread_id": cleaned}}


def _validate_input(state: WorkflowState) -> StateUpdate:
    raw_message = state.get("message")
    if not isinstance(raw_message, str) or not raw_message.strip():
        return {
            "route": create_workflow_route(
                agent_name="finance_qa",
                needs_clarification=True,
                confidence=0.0,
                reason="empty_query",
            ),
            "response": _error_response(
                agent_name="workflow",
                content="Please enter a question so I can route it to the right assistant.",
                error_code="INVALID_INPUT",
            ),
            "input_status": "invalid",
        }

    message = raw_message.strip()
    return {
        "message": message,
        "messages": [HumanMessage(content=message)],
        "input_status": "valid",
    }


def _route_message(router_llm: Optional[Any]) -> Callable[[WorkflowState], StateUpdate]:
    def _run_router(state: WorkflowState) -> StateUpdate:
        return {
            "route": route_query(
                state["message"],
                llm=router_llm,
                history=_prior_messages(state),
            )
        }

    return _run_router


def _clarify(state: WorkflowState) -> StateUpdate:
    route = state["route"]
    return {
        "response": _error_response(
            agent_name=route["agent_name"],
            content=(
                "I can help with beginner finance education, portfolio review, market lookup, "
                "or goal planning. Please add a little more detail."
            ),
            error_code="NEEDS_CLARIFICATION",
            confidence=0.35,
        )
    }


def _unavailable(state: WorkflowState) -> StateUpdate:
    route = state.get("route") or create_workflow_route(agent_name="workflow")
    return {
        "response": _error_response(
            agent_name=route["agent_name"],
            content="That assistant is not available yet. Please try again later.",
            error_code="AGENT_UNAVAILABLE",
            confidence=0.1,
        )
    }


def _agent_node(
    agent_name: str, agents: Mapping[str, Any]
) -> Callable[[WorkflowState], StateUpdate]:
    def _run_agent(state: WorkflowState) -> StateUpdate:
        agent = agents.get(agent_name)
        if agent is None or not hasattr(agent, "run"):
            return _unavailable(state)

        try:
            response = agent.run(_agent_payload(agent_name, state["message"], state))
        except Exception:
            response = _error_response(
                agent_name=agent_name,
                content="That assistant is temporarily unavailable. Please try again in a moment.",
                error_code="AGENT_UNAVAILABLE",
                confidence=0.1,
            )

        if not isinstance(response, AgentResponse):
            response = _error_response(
                agent_name=agent_name,
                content="That assistant returned an invalid response. Please try again.",
                error_code="AGENT_UNAVAILABLE",
                confidence=0.1,
            )
        return {"response": response}

    return _run_agent


def _stretch_node(agent_name: str) -> Callable[[WorkflowState], StateUpdate]:
    def _run_stretch_agent(state: WorkflowState) -> StateUpdate:
        return {"response": _stretch_agent_response(agent_name, state["message"])}

    return _run_stretch_agent


def _finalize(state: WorkflowState) -> StateUpdate:
    response = state["response"]
    return {"messages": [AIMessage(content=response.content)]}


def _input_status(state: WorkflowState) -> str:
    return state["input_status"]


def _route_target(state: WorkflowState) -> str:
    route = state["route"]
    if route["needs_clarification"]:
        return "clarify"
    agent_name = route["agent_name"]
    if agent_name in {*AGENT_NODES, *STRETCH_NODES}:
        return agent_name
    return "unavailable"


def _prior_messages(state: WorkflowState) -> list[Any]:
    messages = list(state.get("messages", []))
    if messages and getattr(messages[-1], "type", None) == "human":
        return messages[:-1]
    return messages


def _default_agents() -> dict[str, Any]:
    return {
        "finance_qa": FinanceQAAgent(),
        "portfolio": PortfolioAnalysisAgent(),
        "market": MarketAnalysisAgent(),
        "goals": GoalPlanningAgent(),
    }


def _agent_payload(agent_name: str, message: str, state: WorkflowState) -> Any:
    if agent_name == "finance_qa":
        return {"message": message, "state": state}
    if agent_name == "market":
        return {"ticker": _extract_ticker(message), "message": message, "state": state}
    if agent_name == "portfolio":
        return {"message": message, "holdings": [], "state": state}
    if agent_name == "goals":
        return {"message": message, "state": state}
    return {"message": message, "state": state}


def _extract_ticker(message: str) -> str:
    candidates = re.findall(r"\b[A-Z]{1,5}\b", message.upper())
    ignored = {"WHAT", "PRICE", "CURRENT"}
    for candidate in candidates:
        if candidate not in ignored:
            return candidate
    return message.strip()


def _error_response(
    *,
    agent_name: str,
    content: str,
    error_code: str,
    confidence: float = 0.2,
) -> AgentResponse:
    return AgentResponse(
        agent_name=agent_name,
        content=append_disclaimer(content),
        confidence=confidence,
        disclaimer=disclaimer_text(),
        error_code=error_code,
    )


def _stretch_agent_response(agent_name: str, message: str) -> AgentResponse:
    if agent_name == "tax":
        content = (
            "I can explain tax-related account concepts at a high level, but this project "
            "has not implemented personalized tax education yet."
        )
    else:
        content = (
            "I can discuss market-news context at a high level, but live news synthesis is "
            "not implemented yet."
        )
    return AgentResponse(
        agent_name=agent_name,
        content=append_disclaimer(content),
        confidence=0.4,
        disclaimer=disclaimer_text(),
        metadata={"message": message},
    )
