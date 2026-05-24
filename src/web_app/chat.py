"""Chat view and app-facing workflow helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional
from uuid import uuid4

from langchain_core.messages import BaseMessage

from core.models import AgentResponse
from core.tracing import langsmith_trace_context
from workflow.graph import build_graph, conversation_config, create_memory_checkpointer
from workflow.state import WorkflowState, initial_state


CHAT_RUNTIME_KEY = "chat_runtime"
CHAT_STATE_KEY = "chat_state"


@dataclass
class ChatRuntime:
    """Compiled workflow plus its LangGraph memory thread id."""

    graph: Any
    thread_id: str


def new_thread_id() -> str:
    """Create a unique local conversation id."""

    return f"chat-{uuid4().hex}"


def create_chat_runtime(
    *,
    thread_id: Optional[str] = None,
    agents: Optional[dict[str, Any]] = None,
    router_llm: Optional[Any] = None,
    checkpointer: Optional[Any] = None,
) -> ChatRuntime:
    """Create a chat runtime backed by LangGraph checkpoint memory."""

    resolved_thread_id = thread_id or new_thread_id()
    graph = build_graph(
        agents=agents,
        router_llm=router_llm,
        checkpointer=checkpointer or create_memory_checkpointer(),
    )
    return ChatRuntime(graph=graph, thread_id=resolved_thread_id)


def run_chat_turn(
    runtime: ChatRuntime,
    message: str,
    *,
    user_id: Optional[str] = None,
) -> WorkflowState:
    """Send one user message through the workflow memory thread."""

    cleaned = message.strip()
    if not cleaned:
        raise ValueError("message cannot be empty")

    payload: dict[str, Any] = {
        "message": cleaned,
        "session_id": runtime.thread_id,
    }
    if user_id:
        payload["user_id"] = user_id

    with langsmith_trace_context(
        tags=["streamlit", "chat"],
        metadata={"thread_id": runtime.thread_id, "surface": "web_app"},
    ):
        return runtime.graph.invoke(payload, config=conversation_config(runtime.thread_id))


def reset_chat_runtime(session_state: dict[str, Any]) -> None:
    """Reset Streamlit chat state to a fresh LangGraph memory thread."""

    runtime = create_chat_runtime()
    session_state[CHAT_RUNTIME_KEY] = runtime
    session_state[CHAT_STATE_KEY] = initial_state(session_id=runtime.thread_id)


def ensure_chat_runtime(session_state: dict[str, Any]) -> tuple[ChatRuntime, WorkflowState]:
    """Return the current chat runtime and state, creating them if needed."""

    if CHAT_RUNTIME_KEY not in session_state:
        reset_chat_runtime(session_state)
    return session_state[CHAT_RUNTIME_KEY], session_state[CHAT_STATE_KEY]


def display_messages(state: WorkflowState) -> list[dict[str, str]]:
    """Convert LangChain messages into Streamlit chat-message payloads."""

    return [
        {"role": _display_role(message), "content": _message_content(message)}
        for message in state.get("messages", [])
        if _message_content(message)
    ]


def latest_response(state: WorkflowState) -> Optional[AgentResponse]:
    """Return the latest structured agent response from workflow state."""

    response = state.get("response")
    return response if isinstance(response, AgentResponse) else None


def latest_route_label(state: WorkflowState) -> str:
    """Return a compact route label for UI metadata."""

    route = state.get("route")
    if not isinstance(route, dict):
        return "unknown"
    agent = route.get("agent_name") or "unknown"
    confidence = route.get("confidence")
    if isinstance(confidence, (float, int)):
        return f"{agent} ({confidence:.0%})"
    return str(agent)


def render_chat_view() -> None:
    """Render the Streamlit chat tab."""

    import streamlit as st

    runtime, state = ensure_chat_runtime(st.session_state)

    with st.sidebar:
        st.text_input("Conversation ID", value=runtime.thread_id, disabled=True)
        if st.button("New chat", use_container_width=True):
            reset_chat_runtime(st.session_state)
            st.rerun()

    for message in display_messages(state):
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    response = latest_response(state)
    if response is not None:
        with st.expander("Sources and route", expanded=False):
            st.caption(f"Route: {latest_route_label(state)}")
            if response.citations:
                for citation in response.citations:
                    st.markdown(f"- [{citation.title}]({citation.url})" if citation.url else f"- {citation.title}")
            if response.error_code:
                st.caption(f"Status: {response.error_code}")

    prompt = st.chat_input("Ask a finance question")
    if prompt:
        with st.spinner("Thinking"):
            st.session_state[CHAT_STATE_KEY] = run_chat_turn(runtime, prompt)
        st.rerun()


def _display_role(message: Any) -> str:
    role = getattr(message, "type", None)
    if role == "human":
        return "user"
    if role == "ai":
        return "assistant"
    if isinstance(message, dict):
        mapped = message.get("role") or message.get("type")
        if mapped == "human":
            return "user"
        if mapped == "ai":
            return "assistant"
        return str(mapped or "assistant")
    return "assistant"


def _message_content(message: Any) -> str:
    if isinstance(message, BaseMessage):
        return str(message.content)
    if isinstance(message, dict):
        return str(message.get("content") or "")
    return str(getattr(message, "content", "") or "")
