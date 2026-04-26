"""
Configuration management for the RAG Financial Filing Analyst.

Centralizes all settings — model parameters, chunking strategies, retrieval
tuning, guardrail thresholds, and infrastructure endpoints — into a single
validated Pydantic Settings object that reads from environment variables and
an optional `.env` file.
"""

from __future__ import annotations

import logging
from enum import Enum
from pathlib import Path
from typing import Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class LLMProvider(str, Enum):
    """Supported LLM providers."""

    OPENAI = "openai"
    ANTHROPIC = "anthropic"


class EmbeddingProvider(str, Enum):
    """Supported embedding providers."""

    OPENAI = "openai"
    SENTENCE_TRANSFORMERS = "sentence_transformers"
    HUGGINGFACE = "huggingface"


class ChunkingStrategy(str, Enum):
    """Available chunking strategies."""

    RECURSIVE = "recursive"
    SEMANTIC = "semantic"
    SECTION_AWARE = "section_aware"


class RetrievalStrategy(str, Enum):
    """Available retrieval strategies."""

    DENSE = "dense"
    SPARSE = "sparse"
    ENSEMBLE = "ensemble"
    MMR = "mmr"


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------


class Settings(BaseSettings):
    """Application-wide configuration loaded from environment variables.

    Reads from a `.env` file in the project root when present. Every field
    has a sensible default so the application can start with minimal
    configuration (only API keys are truly required for LLM calls).
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # -- LLM ----------------------------------------------------------------
    openai_api_key: Optional[str] = Field(default=None, description="OpenAI API key")
    anthropic_api_key: Optional[str] = Field(
        default=None, description="Anthropic API key"
    )
    llm_provider: LLMProvider = Field(
        default=LLMProvider.OPENAI, description="LLM provider to use"
    )
    llm_model_name: str = Field(
        default="gpt-4o", description="Model name for generation"
    )
    llm_temperature: float = Field(
        default=0.1, ge=0.0, le=2.0, description="Sampling temperature"
    )
    llm_max_tokens: int = Field(
        default=4096, ge=1, description="Max tokens for generation"
    )

    # -- Embeddings ---------------------------------------------------------
    embedding_provider: EmbeddingProvider = Field(
        default=EmbeddingProvider.OPENAI, description="Embedding provider"
    )
    embedding_model_name: str = Field(
        default="text-embedding-3-small", description="Embedding model name"
    )
    embedding_dimension: int = Field(
        default=1536, ge=1, description="Embedding vector dimension"
    )

    # -- ChromaDB -----------------------------------------------------------
    chroma_host: str = Field(default="localhost", description="ChromaDB host")
    chroma_port: int = Field(default=8000, ge=1, le=65535, description="ChromaDB port")
    chroma_collection_name: str = Field(
        default="sec_filings", description="Default ChromaDB collection"
    )
    chroma_persist_directory: Path = Field(
        default=Path("./data/chroma_db"),
        description="Local ChromaDB persistence directory",
    )

    # -- Chunking -----------------------------------------------------------
    chunk_size: int = Field(
        default=1000, ge=100, le=10000, description="Target chunk size in characters"
    )
    chunk_overlap: int = Field(
        default=200, ge=0, description="Overlap between consecutive chunks"
    )
    chunking_strategy: ChunkingStrategy = Field(
        default=ChunkingStrategy.SECTION_AWARE, description="Chunking strategy"
    )

    # -- Retrieval ----------------------------------------------------------
    retrieval_top_k: int = Field(
        default=5, ge=1, le=100, description="Number of chunks to retrieve"
    )
    retrieval_score_threshold: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Minimum similarity score for retrieval",
    )
    retrieval_strategy: RetrievalStrategy = Field(
        default=RetrievalStrategy.ENSEMBLE, description="Retrieval strategy"
    )
    mmr_diversity_score: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Lambda for MMR diversity (0=max diversity, 1=max relevance)",
    )
    dense_weight: float = Field(
        default=0.6,
        ge=0.0,
        le=1.0,
        description="Weight for dense retrieval in ensemble",
    )
    sparse_weight: float = Field(
        default=0.4,
        ge=0.0,
        le=1.0,
        description="Weight for sparse retrieval in ensemble",
    )

    # -- Guardrails ---------------------------------------------------------
    confidence_threshold: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Minimum confidence for accepting a response",
    )
    enable_hallucination_check: bool = Field(
        default=True, description="Enable hallucination detection"
    )
    enable_numerical_validation: bool = Field(
        default=True, description="Enable financial number cross-checking"
    )

    # -- SEC EDGAR ----------------------------------------------------------
    sec_edgar_user_agent: str = Field(
        default="RAGFinancialAnalyst admin@example.com",
        description="User-Agent header for EDGAR requests (SEC requires identification)",
    )
    sec_edgar_rate_limit: int = Field(
        default=10,
        ge=1,
        description="Max requests per second to EDGAR (SEC guideline: 10)",
    )

    # -- MLflow -------------------------------------------------------------
    mlflow_tracking_uri: str = Field(
        default="./mlruns", description="MLflow tracking URI"
    )
    mlflow_experiment_name: str = Field(
        default="rag-financial-analyst", description="MLflow experiment name"
    )

    # -- Logging ------------------------------------------------------------
    log_level: str = Field(default="INFO", description="Logging level")

    # -- Validators ---------------------------------------------------------

    @field_validator("dense_weight", "sparse_weight")
    @classmethod
    def _validate_ensemble_weights(cls, value: float) -> float:
        """Ensure ensemble weights are within [0, 1]."""
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"Ensemble weight must be in [0, 1], got {value}")
        return value

    @field_validator("log_level")
    @classmethod
    def _validate_log_level(cls, value: str) -> str:
        """Ensure the log level is a recognized Python logging level."""
        valid_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        upper = value.upper()
        if upper not in valid_levels:
            raise ValueError(f"Invalid log level '{value}'. Choose from {valid_levels}")
        return upper

    # -- Helpers ------------------------------------------------------------

    def configure_logging(self) -> None:
        """Apply the configured log level to the root logger."""
        logging.basicConfig(
            level=getattr(logging, self.log_level),
            format="%(asctime)s | %(name)-30s | %(levelname)-8s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        logger.info("Logging configured at %s level", self.log_level)

    def get_chroma_persist_path(self) -> Path:
        """Return the resolved, absolute persistence path for ChromaDB."""
        path = self.chroma_persist_directory.resolve()
        path.mkdir(parents=True, exist_ok=True)
        return path

    def validate_api_keys(self) -> list[str]:
        """Check that required API keys are set and return a list of warnings.

        Returns:
            A list of warning messages for any missing keys. An empty list
            means all required keys for the selected providers are present.
        """
        warnings: list[str] = []
        if self.llm_provider == LLMProvider.OPENAI and not self.openai_api_key:
            warnings.append(
                "OPENAI_API_KEY is not set but llm_provider is 'openai'."
            )
        if self.llm_provider == LLMProvider.ANTHROPIC and not self.anthropic_api_key:
            warnings.append(
                "ANTHROPIC_API_KEY is not set but llm_provider is 'anthropic'."
            )
        if (
            self.embedding_provider == EmbeddingProvider.OPENAI
            and not self.openai_api_key
        ):
            warnings.append(
                "OPENAI_API_KEY is not set but embedding_provider is 'openai'."
            )
        for warning in warnings:
            logger.warning(warning)
        return warnings
