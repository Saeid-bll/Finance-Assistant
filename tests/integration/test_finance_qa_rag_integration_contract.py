import pytest


pytestmark = [pytest.mark.integration, pytest.mark.contract]


def test_finance_qa_retrieves_sources_and_generates_cited_answer(require_attr, sample_documents) -> None:
    create_test_embedding_model = require_attr("rag.embeddings", "create_test_embedding_model")
    create_retriever_from_documents = require_attr(
        "rag.retriever", "create_retriever_from_documents"
    )
    FinanceQAAgent = require_attr("agents.finance_qa", "FinanceQAAgent")

    retriever = create_retriever_from_documents(
        sample_documents,
        embedding_model=create_test_embedding_model(),
    )
    agent = FinanceQAAgent(retriever=retriever)

    response = agent.answer(sample_documents[0]["content"])

    assert response.content
    assert response.citations
    assert response.citations[0].source_id == "investor-gov-diversification"
    assert "not financial advice" in response.content.lower()


def test_finance_qa_handles_empty_knowledge_base(require_attr) -> None:
    create_test_embedding_model = require_attr("rag.embeddings", "create_test_embedding_model")
    create_retriever_from_documents = require_attr(
        "rag.retriever", "create_retriever_from_documents"
    )
    FinanceQAAgent = require_attr("agents.finance_qa", "FinanceQAAgent")

    retriever = create_retriever_from_documents([], embedding_model=create_test_embedding_model())
    agent = FinanceQAAgent(retriever=retriever)

    response = agent.answer("What is an ETF?")

    assert response.confidence < 0.5
    assert response.citations == []
    assert "source" in response.content.lower()
