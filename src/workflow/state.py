"""Workflow state for LangGraph orchestration."""

from __future__ import annotations

import operator
from typing import Annotated, Any, Optional

from langchain_core.messages import AnyMessage
from typing_extensions import NotRequired, TypedDict


class WorkflowState(TypedDict):
    """LangGraph state passed between workflow nodes and invocations."""

    messages: Annotated[list[AnyMessage], operator.add]
    user_id: NotRequired[Optional[str]]
    session_id: NotRequired[Optional[str]]
    message: NotRequired[str]
    input_status: NotRequired[str]
    route: NotRequired[Any]
    response: NotRequired[Any]


def initial_state(
    *, user_id: Optional[str] = None, session_id: Optional[str] = None
) -> WorkflowState:
    """Create the initial workflow state for a new conversation."""

    state: WorkflowState = {"messages": []}
    if user_id is not None:
        state["user_id"] = user_id
    if session_id is not None:
        state["session_id"] = session_id
    return state
