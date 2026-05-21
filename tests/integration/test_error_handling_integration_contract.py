import pytest


pytestmark = pytest.mark.integration
pending = pytest.mark.xfail(strict=True, reason="Pending cross-module error handling")


def test_missing_llm_api_key_has_actionable_error(require_attr, monkeypatch) -> None:
    load_config = require_attr("core.config", "load_config")
    ConfigError = require_attr("core.config", "ConfigError")
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    with pytest.raises(ConfigError, match="GEMINI_API_KEY"):
        load_config(require_api_keys=True, load_env_file=False)


@pending
@pytest.mark.contract
def test_low_confidence_retrieval_routes_to_safe_fallback(require_attr) -> None:
    build_graph = require_attr("workflow.graph", "build_graph")
    graph = build_graph()

    result = graph.invoke({"message": "Tell me exactly which stock will double next week"})

    assert result["response"].confidence < 0.5
    assert "cannot predict" in result["response"].content.lower()
    assert "not financial advice" in result["response"].content.lower()


@pending
@pytest.mark.contract
def test_malformed_user_input_does_not_crash_workflow(require_attr) -> None:
    build_graph = require_attr("workflow.graph", "build_graph")
    graph = build_graph()

    result = graph.invoke({"message": None})

    assert result["response"].error_code == "INVALID_INPUT"
    assert "enter a question" in result["response"].content.lower()
