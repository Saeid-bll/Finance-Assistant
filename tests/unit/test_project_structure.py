import importlib
from pathlib import Path


def test_expected_project_directories_exist() -> None:
    root = Path(__file__).resolve().parents[2]

    expected_dirs = [
        root / "src" / "agents",
        root / "src" / "core",
        root / "src" / "data",
        root / "src" / "rag",
        root / "src" / "utils",
        root / "src" / "web_app",
        root / "src" / "workflow",
        root / "tests" / "unit",
        root / "tests" / "integration",
        root / "tests" / "fixtures",
    ]

    missing = [path for path in expected_dirs if not path.is_dir()]
    assert missing == []


def test_tooling_files_exist() -> None:
    root = Path(__file__).resolve().parents[2]

    expected_files = [
        root / "README.md",
        root / "PROJECT_PLAN.md",
        root / "requirements.txt",
        root / "config.yaml",
        root / "pyproject.toml",
        root / ".env.example",
    ]

    missing = [path for path in expected_files if not path.is_file()]
    assert missing == []


def test_placeholder_modules_import() -> None:
    modules = [
        "agents.base",
        "agents.finance_qa",
        "agents.goals",
        "agents.market",
        "agents.news",
        "agents.portfolio",
        "agents.tax",
        "core.config",
        "core.disclaimers",
        "core.logging",
        "core.models",
        "rag.chunker",
        "rag.embeddings",
        "rag.loader",
        "rag.retriever",
        "rag.vector_store",
        "utils.cache",
        "utils.formatting",
        "web_app.chat",
        "web_app.goals_view",
        "web_app.market_view",
        "web_app.portfolio_view",
        "workflow.graph",
        "workflow.router",
        "workflow.state",
    ]

    for module in modules:
        importlib.import_module(module)
