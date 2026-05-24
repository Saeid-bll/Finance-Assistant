"""LangSmith tracing setup."""

from __future__ import annotations

import os
from contextlib import nullcontext
from typing import Any, ContextManager, Optional

from core.config import ProjectConfig, load_config


def configure_langsmith_tracing(config: Optional[ProjectConfig] = None) -> bool:
    """Configure LangSmith environment variables for LangChain and LangGraph tracing."""

    project_config = config or load_config(require_api_keys=False)
    settings = project_config.langsmith
    enabled = bool(settings.tracing_enabled and settings.api_key)

    os.environ["LANGSMITH_TRACING"] = _bool_env(enabled)
    os.environ["LANGCHAIN_TRACING_V2"] = _bool_env(enabled)
    os.environ["LANGSMITH_PROJECT"] = settings.project
    os.environ["LANGCHAIN_PROJECT"] = settings.project
    os.environ["LANGSMITH_ENDPOINT"] = settings.endpoint
    os.environ["LANGCHAIN_ENDPOINT"] = settings.endpoint

    if settings.api_key:
        os.environ["LANGSMITH_API_KEY"] = settings.api_key
        os.environ["LANGCHAIN_API_KEY"] = settings.api_key

    return enabled


def langsmith_trace_context(
    *,
    config: Optional[ProjectConfig] = None,
    tags: Optional[list[str]] = None,
    metadata: Optional[dict[str, Any]] = None,
) -> ContextManager[None]:
    """Create a LangSmith tracing context when tracing is configured."""

    project_config = config or load_config(require_api_keys=False)
    enabled = configure_langsmith_tracing(project_config)
    if not enabled:
        return nullcontext()

    from langsmith import tracing_context

    return tracing_context(
        project_name=project_config.langsmith.project,
        tags=tags,
        metadata=metadata,
        enabled=True,
    )


def _bool_env(value: bool) -> str:
    return "true" if value else "false"
