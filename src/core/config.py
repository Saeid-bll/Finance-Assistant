"""Configuration loading for local development and tests."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional, Union

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field, ValidationError


class ConfigError(RuntimeError):
    """Raised when project configuration is invalid or incomplete."""


class AppSettings(BaseModel):
    name: str = "AI Finance Assistant"
    environment: str = "development"
    log_level: str = "INFO"


class LLMSettings(BaseModel):
    provider: str = "gemini"
    model: str = "gemini-2.0-flash"
    temperature: float = Field(default=0.2, ge=0, le=2)
    max_output_tokens: int = Field(default=1024, gt=0)
    api_key: Optional[str] = None


class RAGSettings(BaseModel):
    vector_store: str = "faiss"
    embedding_provider: str = "gemini"
    embedding_model: str = "models/gemini-embedding-001"
    embedding_dimensions: int = Field(default=768, gt=0)
    chunk_size: int = Field(default=800, gt=0)
    chunk_overlap: int = Field(default=120, ge=0)
    top_k: int = Field(default=4, gt=0)


class MarketDataSettings(BaseModel):
    primary_provider: str = "yfinance"
    fallback_provider: str = "alpha_vantage"
    cache_ttl_minutes: int = Field(default=30, gt=0)
    alpha_vantage_api_key: Optional[str] = None


class TestingSettings(BaseModel):
    coverage_target_percent: int = Field(default=80, ge=0, le=100)


class ProjectConfig(BaseModel):
    app: AppSettings = Field(default_factory=AppSettings)
    llm: LLMSettings = Field(default_factory=LLMSettings)
    rag: RAGSettings = Field(default_factory=RAGSettings)
    market_data: MarketDataSettings = Field(default_factory=MarketDataSettings)
    testing: TestingSettings = Field(default_factory=TestingSettings)


def _default_config_path() -> Path:
    return Path(__file__).resolve().parents[2] / "config.yaml"


def _read_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise ConfigError(f"Config file not found: {path}")

    try:
        with path.open("r", encoding="utf-8") as file:
            data = yaml.safe_load(file) or {}
    except yaml.YAMLError as exc:
        raise ConfigError(f"Could not parse config file: {path}") from exc

    if not isinstance(data, dict):
        raise ConfigError("Config file must contain a YAML mapping")
    return data


def _apply_env_overrides(data: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(data)
    llm = dict(merged.get("llm", {}))
    rag = dict(merged.get("rag", {}))
    market_data = dict(merged.get("market_data", {}))

    if os.getenv("LLM_PROVIDER"):
        llm["provider"] = os.environ["LLM_PROVIDER"]
    if os.getenv("GEMINI_API_KEY"):
        llm["api_key"] = os.environ["GEMINI_API_KEY"]
    if os.getenv("RAG_EMBEDDING_PROVIDER"):
        rag["embedding_provider"] = os.environ["RAG_EMBEDDING_PROVIDER"]
    if os.getenv("RAG_EMBEDDING_MODEL"):
        rag["embedding_model"] = os.environ["RAG_EMBEDDING_MODEL"]
    if os.getenv("RAG_EMBEDDING_DIMENSIONS"):
        rag["embedding_dimensions"] = os.environ["RAG_EMBEDDING_DIMENSIONS"]
    if os.getenv("ALPHA_VANTAGE_API_KEY"):
        market_data["alpha_vantage_api_key"] = os.environ["ALPHA_VANTAGE_API_KEY"]

    merged["llm"] = llm
    merged["rag"] = rag
    merged["market_data"] = market_data
    return merged


def _validate_required_api_keys(config: ProjectConfig) -> None:
    if config.llm.provider.lower() == "gemini" and not config.llm.api_key:
        raise ConfigError("GEMINI_API_KEY is required when using the Gemini provider")
    if config.rag.embedding_provider.lower() == "gemini" and not config.llm.api_key:
        raise ConfigError("GEMINI_API_KEY is required when using Gemini embeddings")


def load_config(
    path: Optional[Union[Path, str]] = None,
    *,
    require_api_keys: bool = False,
    load_env_file: bool = True,
) -> ProjectConfig:
    """Load project configuration from YAML plus supported environment overrides."""

    if load_env_file:
        load_dotenv()

    config_path = Path(path) if path is not None else _default_config_path()
    raw_config = _apply_env_overrides(_read_yaml(config_path))

    try:
        config = ProjectConfig.model_validate(raw_config)
    except ValidationError as exc:
        raise ConfigError("Project configuration is invalid") from exc

    if require_api_keys:
        _validate_required_api_keys(config)
    return config
