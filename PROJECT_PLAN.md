# AI Finance Assistant Project Plan

## Goal

Build a production-ready AI Finance Assistant that helps beginners learn investing concepts through a multi-agent conversational app. The system should provide educational answers, portfolio analysis, market lookup, goal planning, source-grounded responses, and clear disclaimers that it is not financial advice.

## Success Criteria

- Multi-agent architecture with clear separation of responsibilities.
- LangGraph-based routing, state management, and conversation memory.
- RAG knowledge base with source attribution.
- Real-time or cached market data lookup.
- Streamlit app with chat, portfolio, market, and goals views.
- Comprehensive pytest suite targeting 80%+ coverage.
- README, architecture notes, setup instructions, and demo-ready examples.

## MVP Scope

The MVP should prioritize the core grading areas while keeping the build achievable.

1. Finance Q&A Agent
   - Answers beginner finance questions.
   - Uses RAG for grounded responses.
   - Provides source citations and educational disclaimers.

2. Portfolio Analysis Agent
   - Accepts user holdings.
   - Calculates total value, allocation percentages, concentration risk, and diversification score.
   - Produces beginner-friendly educational feedback.

3. Market Analysis Agent
   - Looks up ticker data using yFinance first, with room for Alpha Vantage later.
   - Uses caching and graceful fallback behavior.
   - Shows data freshness.

4. Goal Planning Agent
   - Helps users define goals, time horizon, contribution amount, and risk appetite.
   - Uses simple projection logic.
   - Explains assumptions clearly.

5. News Synthesizer Agent
   - Stretch or second-phase agent.
   - Summarizes finance news only if reliable sources/API access are available.

6. Tax Education Agent
   - Stretch or second-phase agent.
   - Explains account types and tax concepts at a high level.
   - Avoids personalized tax advice.

## Recommended Tech Stack

- Python for the application.
- Streamlit for the web interface.
- LangGraph for workflow orchestration.
- LangChain-compatible abstractions where helpful.
- Google Gemini 2.0 Flash or another configured LLM provider.
- FAISS or Chroma for the vector database.
- yFinance for initial market data integration.
- pytest, pytest-cov, pytest-mock, and responses or requests-mock for tests.
- pydantic for structured models and validation.
- python-dotenv or YAML configuration for local settings.

## Proposed Repository Structure

```text
ai_finance_assistant/
├── src/
│   ├── agents/
│   │   ├── base.py
│   │   ├── finance_qa.py
│   │   ├── portfolio.py
│   │   ├── market.py
│   │   ├── goals.py
│   │   ├── news.py
│   │   └── tax.py
│   ├── core/
│   │   ├── config.py
│   │   ├── models.py
│   │   ├── disclaimers.py
│   │   └── logging.py
│   ├── data/
│   │   ├── knowledge_base/
│   │   └── sample_portfolios/
│   ├── rag/
│   │   ├── loader.py
│   │   ├── chunker.py
│   │   ├── embeddings.py
│   │   ├── vector_store.py
│   │   └── retriever.py
│   ├── workflow/
│   │   ├── state.py
│   │   ├── router.py
│   │   └── graph.py
│   ├── web_app/
│   │   ├── app.py
│   │   ├── chat.py
│   │   ├── portfolio_view.py
│   │   ├── market_view.py
│   │   └── goals_view.py
│   └── utils/
│       ├── cache.py
│       └── formatting.py
├── tests/
│   ├── unit/
│   ├── integration/
│   ├── fixtures/
│   └── conftest.py
├── config.yaml
├── requirements.txt
├── README.md
└── PROJECT_PLAN.md
```

## Build Milestones

### Milestone 1: Planning and Setup

- Confirm project scope and MVP agent list.
- Create project structure.
- Add dependency files.
- Add initial README.
- Add test tooling configuration.

Done when:
- The repo has the expected folder structure.
- Tests can be run with one command.
- Configuration strategy is documented.

### Milestone 2: Domain Models and Core Utilities

- Define models for user queries, agent responses, portfolio holdings, market data, goals, and citations.
- Add financial education disclaimer helpers.
- Add logging and error types.

Done when:
- Core models validate normal and malformed inputs.
- Unit tests cover validation and error behavior.

### Milestone 3: RAG Knowledge Base

- Add curated starter articles or markdown notes.
- Implement document loading, chunking, embedding, indexing, and retrieval.
- Include source attribution in retrieval results.

Done when:
- A question can retrieve relevant source chunks.
- Retrieval tests verify citations and empty-result handling.

### Milestone 4: Core Agents

- Implement the base agent interface.
- Implement Finance Q&A, Portfolio Analysis, Market Analysis, and Goal Planning agents.
- Keep News and Tax agents as stretch agents unless time allows.

Done when:
- Each core agent has deterministic unit tests.
- Each agent returns structured responses.
- External APIs and LLM calls are mocked in tests.

### Milestone 5: LangGraph Workflow

- Implement query routing.
- Add workflow state.
- Preserve conversation context.
- Add fallback paths for ambiguous queries or agent errors.

Done when:
- Routing tests cover finance Q&A, portfolio, market, goal planning, and unknown queries.
- Workflow integration tests verify multi-turn behavior.

### Milestone 6: Streamlit Interface

- Build four main tabs: Chat, Portfolio, Market, Goals.
- Add portfolio charts and market visualizations.
- Keep the interface beginner-friendly and practical.

Done when:
- A user can complete the main demo flows locally.
- Core UI helper logic has tests where practical.

### Milestone 7: Comprehensive Testing and Quality

- Target 80%+ coverage.
- Add unit tests, integration tests, and mocked API tests.
- Add edge case coverage for invalid input, API failures, empty data, and missing configuration.
- Add coverage reporting.

Done when:
- `pytest --cov=src` passes.
- Coverage is at least 80%.
- Critical paths have both success and failure tests.

### Milestone 8: Documentation and Demo Prep

- Complete README with setup, architecture, usage, and troubleshooting.
- Add technical design notes.
- Add sample prompts and sample portfolio data.
- Prepare demo script.

Done when:
- A reviewer can install, run, test, and understand the project from the docs.
- The app supports the required demo flows.

## Testing Strategy

Testing should be built alongside features, not saved for the end.

### Unit Tests

- Agent input/output behavior.
- Portfolio calculations.
- Goal projection logic.
- Market data parsing.
- RAG chunking and retrieval helpers.
- Configuration loading and validation.
- Disclaimer and formatting helpers.

### Integration Tests

- LangGraph routing across multiple agents.
- RAG retrieval connected to Finance Q&A.
- Market agent with mocked provider responses.
- Portfolio flow from user input to analysis response.
- Goal planning flow with session state.

### Failure and Edge Case Tests

- Empty user query.
- Unknown ticker.
- Market API timeout or rate limit.
- Empty portfolio.
- Invalid holding quantity or price.
- Missing API key.
- Empty knowledge base.
- Low-confidence retrieval.
- Agent exception and workflow fallback.

### Coverage Target

- Minimum target: 80% total coverage.
- Higher priority: strong coverage on workflow, agents, portfolio math, RAG, and error handling.
- UI rendering does not need exhaustive browser-style testing, but UI helper functions should be tested.

## Financial Safety Requirements

- Every substantive response should make clear that the assistant provides education, not personalized financial advice.
- Avoid buy/sell recommendations.
- Prefer conservative language.
- Cite educational sources for RAG answers.
- Explain assumptions for projections and risk comments.
- Encourage users to consult qualified professionals for personalized investment, tax, or legal decisions.

## Approval Gates

We will proceed one approved step at a time.

1. Step 1: Project plan.
2. Step 2: Create project structure and tooling.
3. Step 2.5: Add comprehensive test and evaluation contract.
4. Step 3: Build core models and utilities.
5. Step 4: Build RAG knowledge base.
6. Step 5: Implement core agents.
7. Step 6: Implement workflow orchestration.
8. Step 7: Build Streamlit UI.
9. Step 8: Convert contract tests into active comprehensive tests.
10. Step 9: Finish docs and demo prep.

No next step should begin until approved.
