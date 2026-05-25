"""LangChain embedding model factories for RAG."""

from __future__ import annotations

from typing import Optional

from langchain_core.embeddings import DeterministicFakeEmbedding, Embeddings

from core.tracing import traceable_span


DEFAULT_EMBEDDING_PROVIDER = "gemini"
DEFAULT_GEMINI_EMBEDDING_MODEL = "models/gemini-embedding-001"
DEFAULT_EMBEDDING_DIMENSIONS = 768


@traceable_span(name="rag.create_embedding_model", run_type="tool", tags=["rag", "embeddings"])
def create_embedding_model(
    *,
    provider: str = DEFAULT_EMBEDDING_PROVIDER,
    model: Optional[str] = None,
    dimensions: Optional[int] = DEFAULT_EMBEDDING_DIMENSIONS,
    api_key: Optional[str] = None,
) -> Embeddings:
    """Create the production LangChain embedding model."""

    provider_name = provider.strip().lower()
    if provider_name != "gemini":
        raise ValueError(f"Unsupported embedding provider: {provider}")

    try:
        from langchain_google_genai import GoogleGenerativeAIEmbeddings
    except ImportError as exc:
        raise RuntimeError(
            "Gemini embeddings require the langchain-google-genai package. "
            "Install project dependencies with `python -m pip install -r requirements.txt`."
        ) from exc

    kwargs = {"model": model or DEFAULT_GEMINI_EMBEDDING_MODEL}
    if dimensions is not None:
        kwargs["output_dimensionality"] = dimensions
    if api_key:
        kwargs["google_api_key"] = api_key

    return GoogleGenerativeAIEmbeddings(**kwargs)


@traceable_span(name="rag.create_test_embedding_model", run_type="tool", tags=["rag", "embeddings"])
def create_test_embedding_model(*, dimensions: int = DEFAULT_EMBEDDING_DIMENSIONS) -> Embeddings:
    """Create an off-the-shelf deterministic fake embedding model for tests."""

    if dimensions <= 0:
        raise ValueError("dimensions must be positive")
    return DeterministicFakeEmbedding(size=dimensions)
