# AI Finance Assistant

An educational multi-agent finance assistant for beginner investors. The app will combine conversational agents, retrieval-augmented generation, market data lookup, portfolio analysis, and goal planning.

This assistant is for financial education only. It must not provide personalized investment, tax, or legal advice.

## Planned Features

- Finance Q&A agent with source citations.
- Portfolio analysis agent.
- Market analysis agent with cached data lookup.
- Goal planning agent.
- Streamlit interface with Chat, Portfolio, Market, and Goals views.
- Comprehensive pytest suite targeting 80%+ coverage.

## RAG Knowledge Base

The current RAG layer uses local markdown documents from `src/data/knowledge_base/`, word-aware chunking, deterministic term-frequency embeddings, JSON-persisted vector indexes, and source-preserving retrieval results. This keeps development and tests reproducible before external embedding providers are introduced.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
cp .env.example .env
```

## Run Tests

```bash
pytest
```

Without activating the virtual environment:

```bash
.venv/bin/python -m pytest
```

With coverage:

```bash
pytest --cov=src
```

Run only future behavior contracts:

```bash
.venv/bin/python -m pytest -m contract
```

Run contract tests as active failures while implementing a feature:

```bash
.venv/bin/python -m pytest --runxfail
```

See [TESTING_AND_EVALUATION.md](TESTING_AND_EVALUATION.md) for the full test matrix and grading rubric.

## Run App

```bash
streamlit run src/web_app/app.py
```

## Project Status

Step 1 planning is complete. Step 2 scaffold and tooling are complete. Step 2.5 test and evaluation contracts are complete. Step 3 core models and utilities are complete. Step 4 RAG knowledge base is complete. Core agent implementation begins after approval of the next step.
