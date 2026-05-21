import pytest


pytestmark = pytest.mark.unit


def test_loader_reads_markdown_knowledge_base(require_attr, tmp_path) -> None:
    load_knowledge_base = require_attr("rag.loader", "load_knowledge_base")
    article = tmp_path / "diversification.md"
    article.write_text("# Diversification\n\nSpreading investments can manage risk.\n", encoding="utf-8")

    documents = load_knowledge_base(tmp_path)

    assert len(documents) == 1
    assert documents[0].title == "Diversification"
    assert documents[0].source_id == "diversification"


def test_loader_reads_project_starter_knowledge_base(require_attr, project_root) -> None:
    load_knowledge_base = require_attr("rag.loader", "load_knowledge_base")

    documents = load_knowledge_base(project_root / "src" / "data" / "knowledge_base")

    assert len(documents) >= 10
    assert {document.source_id for document in documents} >= {
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
    assert chunks[0].source_id == "investor-gov-diversification"
    assert chunks[0].title == "Diversification"


def test_chunker_rejects_invalid_overlap(require_attr, sample_documents) -> None:
    chunk_documents = require_attr("rag.chunker", "chunk_documents")

    with pytest.raises(ValueError):
        chunk_documents(sample_documents, chunk_size=50, chunk_overlap=50)


def test_local_embedding_tokenizer_ignores_common_stopwords(require_attr) -> None:
    tokenize = require_attr("rag.embeddings", "tokenize")

    assert tokenize("What is the compound interest?") == ["compound", "interest"]


def test_cosine_similarity_returns_zero_for_no_overlap(require_attr) -> None:
    cosine_similarity = require_attr("rag.embeddings", "cosine_similarity")

    assert cosine_similarity({"stocks": 1.0}, {"bonds": 1.0}) == 0.0


def test_retriever_returns_ranked_results(require_attr, sample_documents) -> None:
    Retriever = require_attr("rag.retriever", "Retriever")
    retriever = Retriever.from_documents(sample_documents)

    results = retriever.retrieve("What is compound interest?", top_k=2)

    assert results[0].source_id == "investor-gov-compound-interest"
    assert results[0].score > 0


def test_retriever_handles_empty_knowledge_base(require_attr) -> None:
    Retriever = require_attr("rag.retriever", "Retriever")
    retriever = Retriever.from_documents([])

    results = retriever.retrieve("What is an ETF?", top_k=4)

    assert results == []


def test_retriever_respects_top_k(require_attr, sample_documents) -> None:
    Retriever = require_attr("rag.retriever", "Retriever")
    retriever = Retriever.from_documents(sample_documents)

    results = retriever.retrieve("interest diversification", top_k=1)

    assert len(results) == 1


def test_retriever_can_load_project_knowledge_base(require_attr, project_root) -> None:
    Retriever = require_attr("rag.retriever", "Retriever")

    retriever = Retriever.from_knowledge_base(project_root / "src" / "data" / "knowledge_base")
    results = retriever.retrieve("How do expense ratios affect fund costs?", top_k=2)

    assert results
    assert results[0].source_id == "expense-ratios"


def test_vector_store_can_save_and_load_index(require_attr, sample_documents, tmp_path) -> None:
    VectorStore = require_attr("rag.vector_store", "VectorStore")
    store = VectorStore.from_documents(sample_documents)
    index_path = tmp_path / "index"

    store.save(index_path)
    reloaded = VectorStore.load(index_path)

    assert reloaded.search("diversification", top_k=1)[0].source_id == "investor-gov-diversification"


def test_vector_store_rejects_invalid_top_k(require_attr, sample_documents) -> None:
    VectorStore = require_attr("rag.vector_store", "VectorStore")
    store = VectorStore.from_documents(sample_documents)

    with pytest.raises(ValueError):
        store.search("diversification", top_k=0)
