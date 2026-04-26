"""
Advanced chunking strategies for SEC 10-K filings.

Provides three complementary chunking approaches:
1. **Recursive character splitting** — deterministic, overlap-based splitting
   with configurable separators.
2. **Semantic chunking** — groups sentences whose embeddings are similar,
   producing variable-size chunks that respect topical boundaries.
3. **Section-aware chunking** — first splits by detected section headers,
   then applies recursive splitting within each section, preserving
   document structure in chunk metadata.

All strategies enrich every chunk with metadata (source section, position
index, character offsets, estimated token count) so downstream retrieval
can apply metadata filters.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Optional, Protocol

import numpy as np
import tiktoken

from src.config import ChunkingStrategy, Settings
from src.document_loader import Document, DocumentMetadata

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Token counting
# ---------------------------------------------------------------------------

_TOKENIZER_CACHE: dict[str, tiktoken.Encoding] = {}


def count_tokens(text: str, model: str = "gpt-4o") -> int:
    """Return the number of tokens in *text* for the given model.

    Uses tiktoken with a per-model encoding cache so repeated calls avoid
    the overhead of re-loading the encoding.
    """
    if model not in _TOKENIZER_CACHE:
        try:
            _TOKENIZER_CACHE[model] = tiktoken.encoding_for_model(model)
        except KeyError:
            _TOKENIZER_CACHE[model] = tiktoken.get_encoding("cl100k_base")
    return len(_TOKENIZER_CACHE[model].encode(text))


# ---------------------------------------------------------------------------
# Chunk data class
# ---------------------------------------------------------------------------


@dataclass
class Chunk:
    """A text chunk with enriched metadata."""

    text: str
    metadata: DocumentMetadata = field(default_factory=DocumentMetadata)
    chunk_index: int = 0
    start_char: int = 0
    end_char: int = 0
    token_count: int = 0

    def __post_init__(self) -> None:
        if self.token_count == 0 and self.text:
            self.token_count = count_tokens(self.text)

    def __len__(self) -> int:
        return len(self.text)


# ---------------------------------------------------------------------------
# Embedding function protocol (for semantic chunking)
# ---------------------------------------------------------------------------


class EmbedFunction(Protocol):
    """Callable that turns a list of strings into a 2-D numpy array."""

    def __call__(self, texts: list[str]) -> np.ndarray: ...


# ---------------------------------------------------------------------------
# 1. Recursive character splitter
# ---------------------------------------------------------------------------

# Separators ordered from strongest to weakest document boundary.
DEFAULT_SEPARATORS: list[str] = ["\n\n", "\n", ". ", "? ", "! ", "; ", ", ", " "]


def recursive_split(
    text: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    separators: Optional[list[str]] = None,
) -> list[str]:
    """Split *text* recursively using a hierarchy of separators.

    The algorithm attempts to split on the strongest separator first. If any
    resulting piece still exceeds *chunk_size*, it recurses with the next
    separator.  Overlap between consecutive chunks is achieved by prepending
    a suffix of the previous chunk.

    Args:
        text: Source text to split.
        chunk_size: Target maximum chunk size in characters.
        chunk_overlap: Characters of overlap between consecutive chunks.
        separators: Ordered list of separators (strongest first).

    Returns:
        A list of text chunks.
    """
    if separators is None:
        separators = DEFAULT_SEPARATORS

    if len(text) <= chunk_size:
        return [text] if text.strip() else []

    # Find the first separator that actually occurs in the text.
    chosen_sep: Optional[str] = None
    for sep in separators:
        if sep in text:
            chosen_sep = sep
            break

    if chosen_sep is None:
        # No separator works — hard-split at chunk_size.
        chunks: list[str] = []
        for i in range(0, len(text), chunk_size - chunk_overlap):
            piece = text[i : i + chunk_size]
            if piece.strip():
                chunks.append(piece.strip())
        return chunks

    parts = text.split(chosen_sep)

    # Merge small parts into chunks that approach chunk_size.
    merged: list[str] = []
    current = ""
    remaining_seps = separators[separators.index(chosen_sep) + 1 :]

    for part in parts:
        candidate = f"{current}{chosen_sep}{part}" if current else part
        if len(candidate) <= chunk_size:
            current = candidate
        else:
            if current.strip():
                merged.append(current.strip())
            # If the individual part exceeds chunk_size, recurse.
            if len(part) > chunk_size:
                sub_chunks = recursive_split(
                    part, chunk_size, chunk_overlap, remaining_seps
                )
                merged.extend(sub_chunks)
                current = ""
            else:
                current = part

    if current.strip():
        merged.append(current.strip())

    # Apply overlap by prepending a suffix of the previous chunk.
    if chunk_overlap > 0 and len(merged) > 1:
        overlapped: list[str] = [merged[0]]
        for i in range(1, len(merged)):
            prev_suffix = merged[i - 1][-chunk_overlap:]
            overlapped.append(f"{prev_suffix} {merged[i]}")
        return overlapped

    return merged


# ---------------------------------------------------------------------------
# 2. Semantic chunking
# ---------------------------------------------------------------------------

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def semantic_chunk(
    text: str,
    embed_fn: EmbedFunction,
    similarity_threshold: float = 0.75,
    min_chunk_size: int = 200,
    max_chunk_size: int = 2000,
) -> list[str]:
    """Group sentences into chunks based on embedding similarity.

    The algorithm embeds each sentence, then walks through them sequentially.
    When the cosine similarity between the running group centroid and the
    next sentence drops below *similarity_threshold*, a new chunk boundary
    is created.

    Args:
        text: Source text.
        embed_fn: A callable that maps ``list[str]`` to a 2-D numpy array.
        similarity_threshold: Cosine-similarity cutoff for grouping.
        min_chunk_size: Minimum characters per chunk (small sentences are
            merged with the next group).
        max_chunk_size: Maximum characters before a forced split.

    Returns:
        A list of semantically coherent text chunks.
    """
    sentences = _SENTENCE_SPLIT_RE.split(text)
    sentences = [s.strip() for s in sentences if s.strip()]

    if not sentences:
        return []
    if len(sentences) == 1:
        return sentences

    embeddings = embed_fn(sentences)  # shape (n, dim)

    def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))

    chunks: list[str] = []
    current_sentences: list[str] = [sentences[0]]
    current_centroid: np.ndarray = embeddings[0].copy()

    for i in range(1, len(sentences)):
        sim = _cosine_sim(current_centroid, embeddings[i])
        current_text = " ".join(current_sentences)

        # Start a new chunk if similarity drops or size exceeds max.
        if (
            sim < similarity_threshold and len(current_text) >= min_chunk_size
        ) or len(current_text) >= max_chunk_size:
            chunks.append(current_text)
            current_sentences = [sentences[i]]
            current_centroid = embeddings[i].copy()
        else:
            current_sentences.append(sentences[i])
            # Update centroid as running mean.
            n = len(current_sentences)
            current_centroid = (
                current_centroid * (n - 1) + embeddings[i]
            ) / n

    # Flush remaining
    remaining = " ".join(current_sentences)
    if remaining.strip():
        chunks.append(remaining.strip())

    return chunks


# ---------------------------------------------------------------------------
# 3. Section-aware chunking
# ---------------------------------------------------------------------------

# Regex patterns for common 10-K sub-section headers.
_SUB_SECTION_RE = re.compile(
    r"\n(?=[A-Z][A-Za-z\s,&]{5,80}\n)",
    re.MULTILINE,
)


def section_aware_chunk(
    text: str,
    section_name: str = "",
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> list[tuple[str, str]]:
    """Chunk text while respecting sub-section boundaries.

    First splits on detected sub-section headers, then applies recursive
    splitting within each sub-section.  Returns tuples of
    ``(sub_section_name, chunk_text)`` so callers can annotate metadata.

    Args:
        text: Section text to chunk.
        section_name: Parent section name (e.g. "Risk Factors").
        chunk_size: Target chunk size in characters.
        chunk_overlap: Overlap between chunks.

    Returns:
        A list of ``(sub_section_label, chunk_text)`` tuples.
    """
    # Attempt to split on sub-section headers.
    parts = _SUB_SECTION_RE.split(text)

    results: list[tuple[str, str]] = []
    for part in parts:
        part = part.strip()
        if not part:
            continue

        # Try to extract a sub-section label from the first line.
        lines = part.split("\n", 1)
        first_line = lines[0].strip()
        if len(first_line) < 100 and first_line[0:1].isupper():
            sub_label = f"{section_name} > {first_line}" if section_name else first_line
            body = lines[1].strip() if len(lines) > 1 else first_line
        else:
            sub_label = section_name
            body = part

        sub_chunks = recursive_split(body, chunk_size, chunk_overlap)
        for sc in sub_chunks:
            results.append((sub_label, sc))

    return results


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


class DocumentChunker:
    """Orchestrates chunking of ``Document`` objects using the configured strategy.

    Produces a list of ``Chunk`` objects with enriched metadata ready for
    embedding and indexing.

    Args:
        settings: Application configuration.
        embed_fn: Optional embedding function required for semantic chunking.
    """

    def __init__(
        self,
        settings: Settings,
        embed_fn: Optional[EmbedFunction] = None,
    ) -> None:
        self._chunk_size = settings.chunk_size
        self._chunk_overlap = settings.chunk_overlap
        self._strategy = settings.chunking_strategy
        self._embed_fn = embed_fn

        if self._strategy == ChunkingStrategy.SEMANTIC and embed_fn is None:
            logger.warning(
                "Semantic chunking selected but no embed_fn provided; "
                "falling back to recursive strategy."
            )
            self._strategy = ChunkingStrategy.RECURSIVE

    def chunk_document(self, document: Document) -> list[Chunk]:
        """Chunk a single document using the configured strategy.

        Args:
            document: A ``Document`` instance (typically one section of a filing).

        Returns:
            A list of ``Chunk`` objects.
        """
        text = document.text
        base_meta = document.metadata

        if self._strategy == ChunkingStrategy.RECURSIVE:
            return self._recursive(text, base_meta)
        elif self._strategy == ChunkingStrategy.SEMANTIC:
            return self._semantic(text, base_meta)
        elif self._strategy == ChunkingStrategy.SECTION_AWARE:
            return self._section_aware(text, base_meta)
        else:
            logger.warning(
                "Unknown strategy '%s'; defaulting to recursive.", self._strategy
            )
            return self._recursive(text, base_meta)

    def chunk_documents(self, documents: list[Document]) -> list[Chunk]:
        """Chunk a list of documents.

        Args:
            documents: List of ``Document`` objects.

        Returns:
            A flat list of ``Chunk`` objects from all documents.
        """
        all_chunks: list[Chunk] = []
        for doc in documents:
            chunks = self.chunk_document(doc)
            all_chunks.extend(chunks)

        logger.info(
            "Chunked %d documents into %d chunks (strategy=%s)",
            len(documents),
            len(all_chunks),
            self._strategy.value,
        )
        return all_chunks

    # -- Strategy implementations -------------------------------------------

    def _recursive(
        self, text: str, base_meta: DocumentMetadata
    ) -> list[Chunk]:
        raw_chunks = recursive_split(text, self._chunk_size, self._chunk_overlap)
        return self._to_chunks(raw_chunks, text, base_meta)

    def _semantic(
        self, text: str, base_meta: DocumentMetadata
    ) -> list[Chunk]:
        assert self._embed_fn is not None
        raw_chunks = semantic_chunk(
            text,
            embed_fn=self._embed_fn,
            max_chunk_size=self._chunk_size,
        )
        return self._to_chunks(raw_chunks, text, base_meta)

    def _section_aware(
        self, text: str, base_meta: DocumentMetadata
    ) -> list[Chunk]:
        pairs = section_aware_chunk(
            text,
            section_name=base_meta.section,
            chunk_size=self._chunk_size,
            chunk_overlap=self._chunk_overlap,
        )
        chunks: list[Chunk] = []
        offset = 0
        for idx, (sub_label, chunk_text) in enumerate(pairs):
            start = text.find(chunk_text[:50], offset)
            if start == -1:
                start = offset
            end = start + len(chunk_text)
            meta = DocumentMetadata(
                company_name=base_meta.company_name,
                cik=base_meta.cik,
                filing_date=base_meta.filing_date,
                filing_type=base_meta.filing_type,
                section=sub_label,
                source_file=base_meta.source_file,
                accession_number=base_meta.accession_number,
            )
            chunks.append(
                Chunk(
                    text=chunk_text,
                    metadata=meta,
                    chunk_index=idx,
                    start_char=start,
                    end_char=end,
                )
            )
            offset = max(offset, start)
        return chunks

    @staticmethod
    def _to_chunks(
        raw: list[str], source_text: str, base_meta: DocumentMetadata
    ) -> list[Chunk]:
        """Convert raw text fragments into ``Chunk`` objects with offsets."""
        chunks: list[Chunk] = []
        offset = 0
        for idx, text in enumerate(raw):
            start = source_text.find(text[:50], offset)
            if start == -1:
                start = offset
            end = start + len(text)
            chunks.append(
                Chunk(
                    text=text,
                    metadata=base_meta,
                    chunk_index=idx,
                    start_char=start,
                    end_char=end,
                )
            )
            offset = max(offset, start)
        return chunks
