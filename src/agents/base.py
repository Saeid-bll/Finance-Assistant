"""Base interfaces for finance assistant agents."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Iterable, Optional

from core.disclaimers import append_disclaimer, disclaimer_text
from core.models import AgentResponse, Citation
from core.tracing import traceable_span


class BaseAgent(ABC):
    """Common response helpers for assistant agents."""

    agent_name: str = "base"

    @traceable_span(name="agent.build_response", run_type="chain", tags=["agent"])
    def response(
        self,
        content: str,
        *,
        citations: Optional[Iterable[Citation]] = None,
        confidence: float = 1.0,
        error_code: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
        include_disclaimer: bool = True,
    ) -> AgentResponse:
        final_content = append_disclaimer(content) if include_disclaimer else content.strip()
        return AgentResponse(
            agent_name=self.agent_name,
            content=final_content,
            citations=list(citations or []),
            confidence=confidence,
            disclaimer=disclaimer_text() if include_disclaimer else None,
            error_code=error_code,
            metadata=metadata or {},
        )

    @abstractmethod
    def run(self, payload: Any) -> AgentResponse:
        """Run the agent with workflow-style input."""
