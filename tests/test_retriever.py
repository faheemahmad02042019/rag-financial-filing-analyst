"""
Unit tests for the retrieval module.

Tests cover query expansion, contextual compression, MMR diversification,
and the ensemble retriever. Uses mock objects to avoid requiring a live
vector store or embedding model.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.retriever import (
    ContextualCompressor,
    EnsembleRetriever,
    QueryExpander,
    RetrievalResult,
    mmr_diversify,
)
from src.vector_store import SearchResult


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_results() -> list[SearchResult]:
    """Create a set of sample search results."""
    return [
        SearchResult(
            chunk_id=f"chunk_{i}",
            text=f"Sample text about financial topic {i}. "
            f"Revenue was ${i * 10} million in 2024.",
            score=1.0 - (i * 0.1),
            metadata={
                "section": "Financial Statements" if i % 2 == 0 else "Risk Factors",
                "company_name": "Test Corp.",
                "filing_date": "2024-10-31",
            },
        )
        for i in range(10)
    ]


@pytest.fixture
def sample_embeddings() -> np.ndarray:
    """Deterministic embeddings for 10 results."""
    np.random.seed(123)
    return np.random.rand(10, 64).astype(np.float32)


@pytest.fixture
def query_embedding() -> np.ndarray:
    """A single query embedding vector."""
    np.random.seed(456)
    return np.random.rand(64).astype(np.float32)


@pytest.fixture
def mock_embedding_pipeline() -> MagicMock:
    """Mock embedding pipeline."""
    pipeline = MagicMock()
    np.random.seed(789)
    pipeline.embed_query.return_value = np.random.rand(64).astype(np.float32)
    pipeline.embed_texts.return_value = np.random.rand(5, 64).astype(np.float32)
    return pipeline


# ---------------------------------------------------------------------------
# QueryExpander
# ---------------------------------------------------------------------------


class TestQueryExpander:
    """Tests for multi-query expansion."""

    def test_expansion_without_llm(self) -> None:
        """Without an LLM, should produce heuristic expansions."""
        expander = QueryExpander(llm_fn=None, num_variants=3)
        variants = expander.expand("What was total revenue?")
        assert len(variants) >= 2  # original + at least 1 heuristic
        assert variants[0] == "What was total revenue?"

    def test_expansion_with_llm(self) -> None:
        """With an LLM function, should produce LLM-generated variants."""
        mock_llm = MagicMock(
            return_value=(
                "How much revenue did the company earn?\n"
                "What were the total sales figures?\n"
                "Revenue totals for the fiscal year?"
            )
        )
        expander = QueryExpander(llm_fn=mock_llm, num_variants=3)
        variants = expander.expand("What was total revenue?")
        assert len(variants) == 4  # original + 3 LLM variants
        assert variants[0] == "What was total revenue?"
        mock_llm.assert_called_once()

    def test_expansion_llm_failure_fallback(self) -> None:
        """When LLM fails, should fall back to heuristic expansion."""
        mock_llm = MagicMock(side_effect=RuntimeError("API error"))
        expander = QueryExpander(llm_fn=mock_llm, num_variants=3)
        variants = expander.expand("What was total revenue?")
        assert len(variants) >= 2  # original + heuristic fallbacks

    def test_original_query_always_first(self) -> None:
        expander = QueryExpander(llm_fn=None)
        variants = expander.expand("test query")
        assert variants[0] == "test query"


# ---------------------------------------------------------------------------
# MMR diversification
# ---------------------------------------------------------------------------


class TestMMRDiversify:
    """Tests for Maximum Marginal Relevance diversification."""

    def test_basic_mmr(
        self,
        query_embedding: np.ndarray,
        sample_results: list[SearchResult],
        sample_embeddings: np.ndarray,
    ) -> None:
        diversified = mmr_diversify(
            query_embedding=query_embedding,
            results=sample_results,
            chunk_embeddings=sample_embeddings,
            top_k=5,
            lambda_param=0.7,
        )
        assert len(diversified) == 5
        # All results should be unique
        ids = [r.chunk_id for r in diversified]
        assert len(set(ids)) == len(ids)

    def test_mmr_returns_all_when_k_exceeds_results(
        self,
        query_embedding: np.ndarray,
    ) -> None:
        """When top_k > len(results), return all results."""
        results = [
            SearchResult(chunk_id="a", text="text a", score=0.9, metadata={}),
            SearchResult(chunk_id="b", text="text b", score=0.8, metadata={}),
        ]
        embeddings = np.random.rand(2, 64).astype(np.float32)

        diversified = mmr_diversify(
            query_embedding=query_embedding,
            results=results,
            chunk_embeddings=embeddings,
            top_k=10,
        )
        assert len(diversified) == 2

    def test_mmr_lambda_0_maximises_diversity(
        self,
        query_embedding: np.ndarray,
        sample_results: list[SearchResult],
        sample_embeddings: np.ndarray,
    ) -> None:
        """Lambda=0 should maximise diversity (minimise redundancy)."""
        diversified = mmr_diversify(
            query_embedding=query_embedding,
            results=sample_results,
            chunk_embeddings=sample_embeddings,
            top_k=5,
            lambda_param=0.0,
        )
        assert len(diversified) == 5

    def test_mmr_lambda_1_maximises_relevance(
        self,
        query_embedding: np.ndarray,
        sample_results: list[SearchResult],
        sample_embeddings: np.ndarray,
    ) -> None:
        """Lambda=1 should return the top-K most relevant (no diversity)."""
        diversified = mmr_diversify(
            query_embedding=query_embedding,
            results=sample_results,
            chunk_embeddings=sample_embeddings,
            top_k=5,
            lambda_param=1.0,
        )
        assert len(diversified) == 5


# ---------------------------------------------------------------------------
# ContextualCompressor
# ---------------------------------------------------------------------------


class TestContextualCompressor:
    """Tests for the contextual compression reranker."""

    def test_rerank(
        self,
        mock_embedding_pipeline: MagicMock,
        sample_results: list[SearchResult],
    ) -> None:
        compressor = ContextualCompressor(
            embedding_pipeline=mock_embedding_pipeline,
            top_n=3,
            compress=False,
        )
        reranked = compressor.rerank("test query", sample_results[:5])
        assert len(reranked) == 3
        # Scores should be set
        assert all(r.score > 0 or r.score == 0 for r in reranked)

    def test_rerank_empty_results(
        self, mock_embedding_pipeline: MagicMock
    ) -> None:
        compressor = ContextualCompressor(
            embedding_pipeline=mock_embedding_pipeline, top_n=3
        )
        reranked = compressor.rerank("query", [])
        assert reranked == []

    def test_rerank_fewer_than_top_n(
        self,
        mock_embedding_pipeline: MagicMock,
    ) -> None:
        """When results < top_n, all should be returned."""
        mock_embedding_pipeline.embed_texts.return_value = np.random.rand(
            2, 64
        ).astype(np.float32)
        results = [
            SearchResult(chunk_id="a", text="text a", score=0.9, metadata={}),
            SearchResult(chunk_id="b", text="text b", score=0.8, metadata={}),
        ]
        compressor = ContextualCompressor(
            embedding_pipeline=mock_embedding_pipeline, top_n=5
        )
        reranked = compressor.rerank("query", results)
        assert len(reranked) == 2


# ---------------------------------------------------------------------------
# RetrievalResult
# ---------------------------------------------------------------------------


class TestRetrievalResult:
    """Tests for the RetrievalResult data class."""

    def test_texts_property(self, sample_results: list[SearchResult]) -> None:
        result = RetrievalResult(results=sample_results[:3])
        assert len(result.texts) == 3
        assert all(isinstance(t, str) for t in result.texts)

    def test_scores_property(self, sample_results: list[SearchResult]) -> None:
        result = RetrievalResult(results=sample_results[:3])
        assert len(result.scores) == 3
        assert all(isinstance(s, float) for s in result.scores)

    def test_empty_result(self) -> None:
        result = RetrievalResult()
        assert result.texts == []
        assert result.scores == []
        assert result.strategy == ""


# ---------------------------------------------------------------------------
# EnsembleRetriever (integration-style with mocks)
# ---------------------------------------------------------------------------


class TestEnsembleRetriever:
    """Tests for the ensemble retriever with mocked dependencies."""

    def _make_retriever(
        self,
        mock_store: MagicMock,
        mock_embedder: MagicMock,
    ) -> EnsembleRetriever:
        from src.config import RetrievalStrategy, Settings

        settings = Settings(
            retrieval_strategy=RetrievalStrategy.ENSEMBLE,
            retrieval_top_k=5,
            openai_api_key="test-key",
        )
        return EnsembleRetriever(
            vector_store=mock_store,
            embedding_pipeline=mock_embedder,
            settings=settings,
        )

    def test_retrieve_returns_result(
        self,
        sample_results: list[SearchResult],
    ) -> None:
        mock_store = MagicMock()
        mock_store.hybrid_search.return_value = sample_results[:5]

        mock_embedder = MagicMock()
        mock_embedder.embed_texts.return_value = np.random.rand(1, 64).astype(
            np.float32
        )

        retriever = self._make_retriever(mock_store, mock_embedder)
        result = retriever.retrieve("test query")

        assert isinstance(result, RetrievalResult)
        assert len(result.results) == 5
        assert result.query == "test query"

    def test_retrieve_with_metadata_filter(
        self,
        sample_results: list[SearchResult],
    ) -> None:
        mock_store = MagicMock()
        mock_store.hybrid_search.return_value = sample_results[:3]

        mock_embedder = MagicMock()
        mock_embedder.embed_texts.return_value = np.random.rand(1, 64).astype(
            np.float32
        )

        retriever = self._make_retriever(mock_store, mock_embedder)
        result = retriever.retrieve(
            "test query",
            metadata_filter={"section": "Risk Factors"},
        )

        assert isinstance(result, RetrievalResult)
        # The filter should have been passed through
        mock_store.hybrid_search.assert_called_once()
        call_kwargs = mock_store.hybrid_search.call_args
        assert call_kwargs.kwargs.get("metadata_filter") == {
            "section": "Risk Factors"
        } or call_kwargs[1].get("metadata_filter") == {
            "section": "Risk Factors"
        }

    def test_retrieve_empty_store(self) -> None:
        mock_store = MagicMock()
        mock_store.hybrid_search.return_value = []

        mock_embedder = MagicMock()
        mock_embedder.embed_texts.return_value = np.random.rand(1, 64).astype(
            np.float32
        )

        retriever = self._make_retriever(mock_store, mock_embedder)
        result = retriever.retrieve("query about nothing")

        assert isinstance(result, RetrievalResult)
        assert len(result.results) == 0
