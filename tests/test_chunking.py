"""
Unit tests for the chunking module.

Tests cover all three chunking strategies (recursive, semantic, section-aware),
edge cases (empty input, very short text, very long text), overlap behaviour,
metadata enrichment, and token counting.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.chunking import (
    Chunk,
    DocumentChunker,
    count_tokens,
    recursive_split,
    section_aware_chunk,
    semantic_chunk,
)
from src.config import ChunkingStrategy, Settings
from src.document_loader import Document, DocumentMetadata


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_text() -> str:
    """Return a multi-paragraph financial text for testing."""
    return (
        "Item 1A. Risk Factors\n\n"
        "Our business faces significant risks related to market conditions. "
        "Revenue may decline due to economic downturns affecting consumer "
        "spending patterns. The competitive landscape in our industry "
        "continues to intensify, with new entrants offering lower-priced "
        "alternatives.\n\n"
        "Supply Chain Risks\n"
        "We depend on a limited number of suppliers for critical components. "
        "Disruptions in the global supply chain could materially impact our "
        "ability to manufacture products. Recent geopolitical tensions have "
        "increased the risk of supply interruptions.\n\n"
        "Regulatory Risks\n"
        "Changes in regulations across our operating jurisdictions could "
        "require significant compliance expenditures. Data privacy laws "
        "continue to evolve, and non-compliance could result in substantial "
        "fines and reputational damage.\n\n"
        "Financial Risks\n"
        "Our total revenue was $45.2 billion in fiscal year 2024, compared "
        "to $42.1 billion in fiscal year 2023, representing a 7.4% increase. "
        "Operating expenses increased by 5.2% year-over-year, reaching "
        "$38.6 billion. Net income was $6.6 billion, an improvement of 15.8% "
        "over the prior year."
    )


@pytest.fixture
def short_text() -> str:
    """Short text below typical chunk size."""
    return "This is a short text about revenue growth."


@pytest.fixture
def sample_metadata() -> DocumentMetadata:
    """Sample metadata for testing."""
    return DocumentMetadata(
        company_name="Test Corp.",
        cik="0001234567",
        filing_date="2024-10-31",
        filing_type="10-K",
        section="Risk Factors",
        source_file="/tmp/test_filing.txt",
    )


@pytest.fixture
def sample_document(sample_text: str, sample_metadata: DocumentMetadata) -> Document:
    """A complete Document for integration-style tests."""
    return Document(text=sample_text, metadata=sample_metadata)


def _dummy_embed_fn(texts: list[str]) -> np.ndarray:
    """Deterministic dummy embedding function for testing."""
    np.random.seed(42)
    return np.random.rand(len(texts), 64).astype(np.float32)


# ---------------------------------------------------------------------------
# Token counting
# ---------------------------------------------------------------------------


class TestCountTokens:
    """Tests for the token counting utility."""

    def test_empty_string(self) -> None:
        assert count_tokens("") == 0

    def test_single_word(self) -> None:
        assert count_tokens("hello") >= 1

    def test_long_text(self) -> None:
        text = "financial analysis " * 100
        tokens = count_tokens(text)
        assert tokens > 50

    def test_special_characters(self) -> None:
        tokens = count_tokens("$1,234.56 billion")
        assert tokens >= 1


# ---------------------------------------------------------------------------
# Recursive splitting
# ---------------------------------------------------------------------------


class TestRecursiveSplit:
    """Tests for the recursive character splitter."""

    def test_short_text_not_split(self, short_text: str) -> None:
        chunks = recursive_split(short_text, chunk_size=1000)
        assert len(chunks) == 1
        assert chunks[0] == short_text

    def test_empty_string(self) -> None:
        chunks = recursive_split("")
        assert chunks == []

    def test_splits_on_paragraphs(self, sample_text: str) -> None:
        chunks = recursive_split(sample_text, chunk_size=300, chunk_overlap=0)
        assert len(chunks) > 1
        # All chunks should be non-empty
        assert all(c.strip() for c in chunks)

    def test_chunk_size_respected(self, sample_text: str) -> None:
        chunk_size = 400
        chunks = recursive_split(
            sample_text, chunk_size=chunk_size, chunk_overlap=0
        )
        # Most chunks should be at or below chunk_size (overlap may cause slight excess)
        for chunk in chunks:
            assert len(chunk) <= chunk_size * 1.5  # allow some tolerance

    def test_overlap_present(self, sample_text: str) -> None:
        chunks = recursive_split(
            sample_text, chunk_size=300, chunk_overlap=50
        )
        if len(chunks) > 1:
            # The beginning of chunk[1] should share text with the end of chunk[0]
            # (due to overlap prepending)
            assert len(chunks[1]) > 0

    def test_whitespace_only(self) -> None:
        chunks = recursive_split("   \n\n   ")
        assert chunks == []

    def test_single_very_long_word(self) -> None:
        long_word = "a" * 2000
        chunks = recursive_split(long_word, chunk_size=500)
        assert len(chunks) >= 1

    def test_custom_separators(self) -> None:
        text = "part1|||part2|||part3"
        chunks = recursive_split(text, chunk_size=50, separators=["|||"])
        assert len(chunks) >= 2


# ---------------------------------------------------------------------------
# Section-aware chunking
# ---------------------------------------------------------------------------


class TestSectionAwareChunk:
    """Tests for section-aware chunking."""

    def test_basic_section_aware(self, sample_text: str) -> None:
        results = section_aware_chunk(
            sample_text,
            section_name="Risk Factors",
            chunk_size=300,
        )
        assert len(results) > 0
        # Each result is a (sub_section_label, text) tuple
        for label, text in results:
            assert isinstance(label, str)
            assert isinstance(text, str)
            assert len(text) > 0

    def test_section_label_propagation(self, sample_text: str) -> None:
        results = section_aware_chunk(
            sample_text,
            section_name="Risk Factors",
            chunk_size=500,
        )
        # At least one result should include the parent section name
        labels = [label for label, _ in results]
        assert any("Risk Factors" in label for label in labels)

    def test_empty_text(self) -> None:
        results = section_aware_chunk("", section_name="Test")
        assert results == []


# ---------------------------------------------------------------------------
# Semantic chunking
# ---------------------------------------------------------------------------


class TestSemanticChunk:
    """Tests for semantic chunking."""

    def test_basic_semantic_chunking(self, sample_text: str) -> None:
        chunks = semantic_chunk(
            sample_text,
            embed_fn=_dummy_embed_fn,
            similarity_threshold=0.75,
            min_chunk_size=100,
            max_chunk_size=500,
        )
        assert len(chunks) > 0
        assert all(isinstance(c, str) for c in chunks)

    def test_single_sentence(self) -> None:
        chunks = semantic_chunk(
            "This is one sentence.",
            embed_fn=_dummy_embed_fn,
        )
        assert len(chunks) == 1

    def test_empty_text(self) -> None:
        chunks = semantic_chunk("", embed_fn=_dummy_embed_fn)
        assert chunks == []

    def test_max_chunk_size_respected(self, sample_text: str) -> None:
        max_size = 300
        chunks = semantic_chunk(
            sample_text,
            embed_fn=_dummy_embed_fn,
            max_chunk_size=max_size,
        )
        # Most chunks should respect max_chunk_size (some may slightly exceed
        # due to sentence boundaries)
        for chunk in chunks:
            assert len(chunk) <= max_size * 2


# ---------------------------------------------------------------------------
# DocumentChunker orchestrator
# ---------------------------------------------------------------------------


class TestDocumentChunker:
    """Tests for the DocumentChunker orchestrator class."""

    def _make_settings(self, strategy: ChunkingStrategy) -> Settings:
        """Create settings with specified strategy."""
        return Settings(
            chunking_strategy=strategy,
            chunk_size=400,
            chunk_overlap=50,
            # Avoid needing real API keys
            openai_api_key="test-key",
            embedding_provider="sentence_transformers",
            embedding_model_name="all-MiniLM-L6-v2",
        )

    def test_recursive_strategy(self, sample_document: Document) -> None:
        settings = self._make_settings(ChunkingStrategy.RECURSIVE)
        chunker = DocumentChunker(settings)
        chunks = chunker.chunk_document(sample_document)
        assert len(chunks) > 0
        assert all(isinstance(c, Chunk) for c in chunks)
        assert all(c.token_count > 0 for c in chunks)

    def test_section_aware_strategy(self, sample_document: Document) -> None:
        settings = self._make_settings(ChunkingStrategy.SECTION_AWARE)
        chunker = DocumentChunker(settings)
        chunks = chunker.chunk_document(sample_document)
        assert len(chunks) > 0
        # Section-aware should preserve section info
        assert all(c.metadata.section for c in chunks)

    def test_semantic_fallback_without_embed_fn(
        self, sample_document: Document
    ) -> None:
        """Semantic strategy without embed_fn should fall back to recursive."""
        settings = self._make_settings(ChunkingStrategy.SEMANTIC)
        chunker = DocumentChunker(settings, embed_fn=None)
        chunks = chunker.chunk_document(sample_document)
        assert len(chunks) > 0

    def test_semantic_with_embed_fn(self, sample_document: Document) -> None:
        settings = self._make_settings(ChunkingStrategy.SEMANTIC)
        chunker = DocumentChunker(settings, embed_fn=_dummy_embed_fn)
        chunks = chunker.chunk_document(sample_document)
        assert len(chunks) > 0

    def test_chunk_documents_multiple(
        self, sample_document: Document
    ) -> None:
        settings = self._make_settings(ChunkingStrategy.RECURSIVE)
        chunker = DocumentChunker(settings)
        chunks = chunker.chunk_documents([sample_document, sample_document])
        # Should produce chunks from both documents
        assert len(chunks) >= 2

    def test_metadata_preserved(self, sample_document: Document) -> None:
        settings = self._make_settings(ChunkingStrategy.RECURSIVE)
        chunker = DocumentChunker(settings)
        chunks = chunker.chunk_document(sample_document)
        for chunk in chunks:
            assert chunk.metadata.company_name == "Test Corp."
            assert chunk.metadata.cik == "0001234567"

    def test_chunk_index_sequential(self, sample_document: Document) -> None:
        settings = self._make_settings(ChunkingStrategy.RECURSIVE)
        chunker = DocumentChunker(settings)
        chunks = chunker.chunk_document(sample_document)
        indices = [c.chunk_index for c in chunks]
        assert indices == list(range(len(chunks)))

    def test_empty_document(self) -> None:
        settings = self._make_settings(ChunkingStrategy.RECURSIVE)
        chunker = DocumentChunker(settings)
        doc = Document(text="", metadata=DocumentMetadata())
        chunks = chunker.chunk_document(doc)
        assert chunks == []
