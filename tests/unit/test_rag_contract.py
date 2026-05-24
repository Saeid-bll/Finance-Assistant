from numbers import Real

import pytest


pytestmark = pytest.mark.unit


def test_loader_reads_markdown_knowledge_base(require_attr, tmp_path) -> None:
    load_knowledge_base = require_attr("rag.loader", "load_knowledge_base")
    article = tmp_path / "diversification.md"
    article.write_text("# Diversification\n\nSpreading investments can manage risk.\n", encoding="utf-8")

    documents = load_knowledge_base(tmp_path)

    assert len(documents) == 1
    assert documents[0].metadata["title"] == "Diversification"
    assert documents[0].metadata["source_id"] == "diversification"
    assert documents[0].id == "diversification"


def test_loader_reads_project_starter_knowledge_base(require_attr, project_root) -> None:
    load_knowledge_base = require_attr("rag.loader", "load_knowledge_base")

    documents = load_knowledge_base(project_root / "src" / "data" / "knowledge_base")

    assert len(documents) >= 10
    assert {document.metadata["source_id"] for document in documents} >= {
        "compound-interest",
        "diversification",
        "exchange-traded-funds",
    }


def test_loader_rejects_missing_knowledge_base(require_attr, tmp_path) -> None:
    load_knowledge_base = require_attr("rag.loader", "load_knowledge_base")

    with pytest.raises(FileNotFoundError):
        load_knowledge_base(tmp_path / "missing")


def test_chunker_preserves_source_metadata(require_attr, sample_documents) -> None:
    chunk_documents = require_attr("rag.chunker", "chunk_documents")

    chunks = chunk_documents(sample_documents, chunk_size=80, chunk_overlap=10)

    assert chunks
    assert chunks[0].metadata["source_id"] == "investor-gov-diversification"
    assert chunks[0].metadata["title"] == "Diversification"
    assert chunks[0].metadata["chunk_index"] == 0
    assert chunks[0].id == "investor-gov-diversification:0"


def test_chunker_rejects_invalid_overlap(require_attr, sample_documents) -> None:
    chunk_documents = require_attr("rag.chunker", "chunk_documents")

    with pytest.raises(ValueError):
        chunk_documents(sample_documents, chunk_size=50, chunk_overlap=50)


def test_test_embedding_model_is_deterministic_and_offline(require_attr) -> None:
    create_test_embedding_model = require_attr("rag.embeddings", "create_test_embedding_model")

    embeddings = create_test_embedding_model(dimensions=16)
    first = embeddings.embed_query("compound interest")
    second = embeddings.embed_query("compound interest")

    assert first == second
    assert len(first) == 16


def test_embedding_factory_rejects_unknown_provider(require_attr) -> None:
    create_embedding_model = require_attr("rag.embeddings", "create_embedding_model")

    with pytest.raises(ValueError):
        create_embedding_model(provider="custom")


def test_retriever_returns_ranked_results(require_attr, sample_documents) -> None:
    create_test_embedding_model = require_attr("rag.embeddings", "create_test_embedding_model")
    create_retriever_from_documents = require_attr(
        "rag.retriever", "create_retriever_from_documents"
    )
    retriever = create_retriever_from_documents(
        sample_documents,
        embedding_model=create_test_embedding_model(),
        top_k=2,
    )

    results = retriever.invoke(sample_documents[1]["content"])

    assert results[0].metadata["source_id"] == "investor-gov-compound-interest"
    assert results[0].page_content


def test_retriever_handles_empty_knowledge_base(require_attr) -> None:
    create_test_embedding_model = require_attr("rag.embeddings", "create_test_embedding_model")
    create_retriever_from_documents = require_attr(
        "rag.retriever", "create_retriever_from_documents"
    )
    retriever = create_retriever_from_documents([], embedding_model=create_test_embedding_model())

    results = retriever.invoke("What is an ETF?")

    assert results == []


def test_retriever_respects_top_k(require_attr, sample_documents) -> None:
    create_test_embedding_model = require_attr("rag.embeddings", "create_test_embedding_model")
    create_retriever_from_documents = require_attr(
        "rag.retriever", "create_retriever_from_documents"
    )
    retriever = create_retriever_from_documents(
        sample_documents,
        embedding_model=create_test_embedding_model(),
        top_k=1,
    )

    results = retriever.invoke("interest diversification")

    assert len(results) == 1


def test_retriever_can_load_project_knowledge_base(require_attr, project_root) -> None:
    load_knowledge_base = require_attr("rag.loader", "load_knowledge_base")
    create_test_embedding_model = require_attr("rag.embeddings", "create_test_embedding_model")
    create_retriever_from_knowledge_base = require_attr(
        "rag.retriever", "create_retriever_from_knowledge_base"
    )
    documents = load_knowledge_base(project_root / "src" / "data" / "knowledge_base")
    expense_document = next(
        document for document in documents if document.metadata["source_id"] == "expense-ratios"
    )

    retriever = create_retriever_from_knowledge_base(
        project_root / "src" / "data" / "knowledge_base",
        embedding_model=create_test_embedding_model(),
        top_k=2,
    )
    results = retriever.invoke(expense_document.page_content)

    assert results
    assert results[0].metadata["source_id"] == "expense-ratios"


def test_faiss_vector_store_can_save_and_load_index(require_attr, sample_documents, tmp_path) -> None:
    create_test_embedding_model = require_attr("rag.embeddings", "create_test_embedding_model")
    create_vector_store = require_attr("rag.vector_store", "create_vector_store")
    save_vector_store = require_attr("rag.vector_store", "save_vector_store")
    load_vector_store = require_attr("rag.vector_store", "load_vector_store")
    embeddings = create_test_embedding_model()
    store = create_vector_store(sample_documents, embedding_model=embeddings)
    index_path = tmp_path / "index"

    assert store is not None
    save_vector_store(store, index_path)
    reloaded = load_vector_store(index_path, embedding_model=embeddings)

    results = reloaded.similarity_search_with_score(sample_documents[0]["content"], k=1)
    assert results[0][0].metadata["source_id"] == "investor-gov-diversification"
    assert isinstance(results[0][1], Real)


def test_retriever_rejects_invalid_top_k(require_attr, sample_documents) -> None:
    create_test_embedding_model = require_attr("rag.embeddings", "create_test_embedding_model")
    create_retriever_from_documents = require_attr(
        "rag.retriever", "create_retriever_from_documents"
    )

    with pytest.raises(ValueError):
        create_retriever_from_documents(
            sample_documents,
            embedding_model=create_test_embedding_model(),
            top_k=0,
        )
