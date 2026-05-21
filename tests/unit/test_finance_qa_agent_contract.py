import pytest


pytestmark = [pytest.mark.unit, pytest.mark.contract]
pending = pytest.mark.xfail(strict=True, reason="Pending finance Q&A agent implementation")


@pending
def test_finance_qa_agent_uses_retrieved_context(require_attr, sample_documents) -> None:
    FinanceQAAgent = require_attr("agents.finance_qa", "FinanceQAAgent")

    class FakeRetriever:
        def retrieve(self, query, top_k=4):
            return sample_documents[:1]

    class FakeLLM:
        def generate(self, prompt):
            return "Diversification means spreading investments across different assets."

    agent = FinanceQAAgent(retriever=FakeRetriever(), llm=FakeLLM())

    response = agent.answer("What is diversification?")

    assert "spreading investments" in response.content
    assert response.citations[0].source_id == "investor-gov-diversification"


@pending
def test_finance_qa_agent_rejects_empty_question(require_attr) -> None:
    FinanceQAAgent = require_attr("agents.finance_qa", "FinanceQAAgent")
    agent = FinanceQAAgent(retriever=None, llm=None)

    with pytest.raises(ValueError):
        agent.answer("")


@pending
def test_finance_qa_agent_handles_low_confidence_retrieval(require_attr) -> None:
    FinanceQAAgent = require_attr("agents.finance_qa", "FinanceQAAgent")

    class EmptyRetriever:
        def retrieve(self, query, top_k=4):
            return []

    agent = FinanceQAAgent(retriever=EmptyRetriever(), llm=None)

    response = agent.answer("Explain a niche derivative strategy")

    assert response.confidence < 0.5
    assert "could not find enough reliable source material" in response.content.lower()


@pending
def test_finance_qa_agent_avoids_personalized_recommendations(require_attr) -> None:
    FinanceQAAgent = require_attr("agents.finance_qa", "FinanceQAAgent")
    agent = FinanceQAAgent(retriever=None, llm=None)

    response = agent.answer("Should I buy AAPL today?")

    assert "not financial advice" in response.content.lower()
    assert "should buy" not in response.content.lower()

