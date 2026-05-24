"""Finance Q&A agent backed by LangChain retrieval."""

from __future__ import annotations

import re
from typing import Any, Optional

from langchain_core.documents import Document

from agents.base import BaseAgent
from core.models import AgentResponse, Citation
from rag.types import SOURCE_ID_KEY, TITLE_KEY, URL_KEY


PERSONALIZED_RECOMMENDATION_PATTERNS = (
    r"\bshould\s+i\s+(buy|sell|hold|invest)\b",
    r"\bwhat\s+stock\s+should\s+i\b",
    r"\bwhich\s+stock\s+will\s+(double|moon|crash)\b",
    r"\bwill\s+(double|moon|crash)\b",
    r"\bwill\s+.+\s+(double|moon|crash)\b",
)


class FinanceQAAgent(BaseAgent):
    """Answer beginner finance questions with retrieved educational sources."""

    agent_name = "finance_qa"

    def __init__(self, *, retriever: Any = None, llm: Any = None) -> None:
        self.retriever = retriever
        self.llm = llm

    def answer(self, question: str) -> AgentResponse:
        cleaned = question.strip()
        if not cleaned:
            raise ValueError("question cannot be empty")

        if self._asks_for_personalized_recommendation(cleaned):
            return self.response(
                "I cannot predict short-term stock moves or tell you whether to buy, sell, "
                "or hold a specific investment. I can explain the concepts, risks, "
                "diversification, time horizon, fees, and tradeoffs that investors often "
                "review before making decisions.",
                confidence=0.4,
                error_code="PERSONALIZED_ADVICE_REQUEST",
            )

        documents = self._retrieve(cleaned)
        if not documents:
            return self.response(
                "I could not find enough reliable source material in the knowledge base "
                "to answer with citations. Try asking about a beginner investing concept "
                "such as diversification, ETFs, compound interest, bonds, or emergency funds.",
                citations=[],
                confidence=0.3,
                error_code="LOW_CONFIDENCE_RETRIEVAL",
            )

        citations = [self._citation_from_document(document) for document in documents]
        content = self._generate_answer(cleaned, documents)
        return self.response(
            content,
            citations=citations,
            confidence=0.85,
            metadata={"source_count": len(citations)},
        )

    def run(self, payload: Any) -> AgentResponse:
        if isinstance(payload, str):
            return self.answer(payload)
        if isinstance(payload, dict):
            return self.answer(str(payload.get("question") or payload.get("message") or ""))
        raise ValueError("payload must be a question string or mapping")

    def _retrieve(self, question: str) -> list[Document | dict[str, Any] | Any]:
        if self.retriever is None:
            return []
        if hasattr(self.retriever, "invoke"):
            return list(self.retriever.invoke(question))
        if hasattr(self.retriever, "retrieve"):
            return list(self.retriever.retrieve(question))
        return []

    def _generate_answer(self, question: str, documents: list[Document | dict[str, Any] | Any]) -> str:
        context = "\n\n".join(
            f"Source: {self._document_title(document)}\n{self._document_content(document)}"
            for document in documents
        )
        prompt = (
            "Answer the beginner finance question using only the educational context. "
            "Avoid personalized buy/sell recommendations.\n\n"
            f"Question: {question}\n\nContext:\n{context}"
        )
        if self.llm is not None and hasattr(self.llm, "generate"):
            generated = str(self.llm.generate(prompt)).strip()
            if generated:
                return generated

        primary = documents[0]
        return (
            f"{self._document_title(primary)}: {self._document_content(primary)} "
            "This explanation is educational and should be considered alongside your goals, "
            "risk tolerance, costs, and time horizon."
        )

    def _citation_from_document(self, document: Document | dict[str, Any] | Any) -> Citation:
        metadata = self._document_metadata(document)
        source_id = str(metadata.get(SOURCE_ID_KEY) or metadata.get("source") or "knowledge-base")
        title = str(metadata.get(TITLE_KEY) or source_id.replace("-", " ").title())
        content = self._document_content(document)
        return Citation(
            source_id=source_id,
            title=title,
            url=metadata.get(URL_KEY),
            excerpt=content[:240],
            metadata=metadata,
        )

    def _document_metadata(self, document: Document | dict[str, Any] | Any) -> dict[str, Any]:
        if isinstance(document, Document):
            return dict(document.metadata or {})
        if isinstance(document, dict):
            metadata = dict(document.get("metadata") or {})
            for key in (SOURCE_ID_KEY, TITLE_KEY, URL_KEY):
                if document.get(key):
                    metadata[key] = document[key]
            return metadata
        return dict(getattr(document, "metadata", {}) or {})

    def _document_content(self, document: Document | dict[str, Any] | Any) -> str:
        if isinstance(document, Document):
            return document.page_content.strip()
        if isinstance(document, dict):
            return str(document.get("page_content") or document.get("content") or "").strip()
        return str(getattr(document, "page_content", None) or getattr(document, "content", "")).strip()

    def _document_title(self, document: Document | dict[str, Any] | Any) -> str:
        metadata = self._document_metadata(document)
        source_id = str(metadata.get(SOURCE_ID_KEY) or "knowledge-base")
        return str(metadata.get(TITLE_KEY) or source_id.replace("-", " ").title())

    def _asks_for_personalized_recommendation(self, question: str) -> bool:
        lowered = question.lower()
        return any(re.search(pattern, lowered) for pattern in PERSONALIZED_RECOMMENDATION_PATTERNS)
