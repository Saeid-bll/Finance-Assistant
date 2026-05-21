# Testing and Evaluation Plan

This project uses testing as an implementation contract. The current scaffold has active smoke tests, plus contract tests marked with `@pytest.mark.contract` and `xfail(strict=True)`. Contract tests describe expected behavior for future milestones. As each feature is implemented, remove the matching `xfail` marker and make the test active.

## Test Commands

```bash
.venv/bin/python -m pytest
.venv/bin/python -m pytest --cov=src
.venv/bin/python -m pytest -m contract
.venv/bin/python -m pytest --runxfail
```

Use `--runxfail` during implementation to see which future contracts still fail.

## Coverage Target

- Minimum total coverage: 80%.
- Highest priority coverage: workflow routing, agent behavior, portfolio math, RAG retrieval, error handling, and market-data fallback logic.
- UI rendering does not require exhaustive browser testing, but UI helper logic should be covered.

## Unit Test Matrix

| Area | Required Coverage |
| --- | --- |
| Core models | Query validation, response shape, citations, holdings, market quotes, goals |
| Config | YAML loading, environment overrides, missing key behavior |
| Disclaimers | Educational disclaimer presence, no personalized advice wording |
| Portfolio agent | Total value, allocation percentages, concentration risk, diversification score, invalid holdings |
| Market agent | Provider parsing, cache behavior, unknown tickers, rate limit/API failures, data freshness |
| Goal agent | Projection math, risk appetite assumptions, shortfall/surplus, invalid goals |
| Finance Q&A agent | RAG usage, citations, empty query handling, low-confidence retrieval |
| RAG | Loading, chunking, source metadata, retrieval scoring, empty knowledge base |
| Workflow router | Intent routing for Q&A, portfolio, market, goals, tax, news, and unknown queries |
| Utilities | Formatting, cache TTL, error-safe output |

## Integration Test Matrix

| Flow | Required Coverage |
| --- | --- |
| Workflow orchestration | Multi-turn routing, state preservation, fallback on agent failure |
| Q&A + RAG | Query retrieves source chunks and produces cited educational response |
| Portfolio flow | User holdings validate, analyze, and return chart-ready values |
| Market flow | Market lookup uses mocked provider, cache, freshness, and fallback behavior |
| Goal flow | User goal validates, projects, and returns assumptions |
| Error handling | Malformed input, missing config, empty data, API timeout, low-confidence retrieval |

## Evaluation Rubric

The project will be evaluated against the assignment criteria:

| Category | Weight | Evidence |
| --- | ---: | --- |
| Technical implementation | 40% | Multi-agent architecture, LangGraph workflow, RAG, market data integration |
| User experience | 25% | Streamlit tabs, conversational flow, data visualizations |
| Financial domain knowledge | 20% | Accurate education, portfolio metrics, market context, safety wording |
| Code quality and documentation | 15% | Modular code, README/design docs, tests, edge-case handling |

## Definition of Done for Testing

- `pytest` passes.
- `pytest --cov=src` passes with at least 80% coverage.
- Contract tests for implemented features are converted to active tests.
- External LLM and market API calls are mocked in automated tests.
- Every critical success path has at least one failure-path test.
- Demo flows have integration coverage.

