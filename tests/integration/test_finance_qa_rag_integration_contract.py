import pytest


pytestmark = [pytest.mark.integration, pytest.mark.contract]
pending = pytest.mark.xfail(strict=True, reason="Pending Finance Q&A plus RAG integration")


@pending
def test_finance_qa_retrieves_sources_and_generates_cited_answer(require_attr, sample_documents) -> None:
    Retriever = require_attr("rag.retriever", "Retriever")
    FinanceQAAgent = require_attr("agents.finance_qa", "FinanceQAAgent")

    retriever = Retriever.from_documents(sample_documents)
    agent = FinanceQAAgent(retriever=retriever)

    response = agent.answer("What is diversification?")

    assert response.content
    assert response.citations
    assert response.citations[0].source_id == "investor-gov-diversification"
    assert "not financial advice" in response.content.lower()


@pending
def test_finance_qa_handles_empty_knowledge_base(require_attr) -> None:
    Retriever = require_attr("rag.retriever", "Retriever")
    FinanceQAAgent = require_attr("agents.finance_qa", "FinanceQAAgent")

    retriever = Retriever.from_documents([])
    agent = FinanceQAAgent(retriever=retriever)

    response = agent.answer("What is an ETF?")

    assert response.confidence < 0.5
    assert response.citations == []
    assert "source" in response.content.lower()

