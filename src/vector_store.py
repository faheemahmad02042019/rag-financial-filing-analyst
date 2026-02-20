"""
Vector store management built on ChromaDB.

Provides a high-level interface for:
- Creating and managing persistent collections with metadata.
- Adding, updating, and deleting documents.
- Similarity search with configurable score thresholds.
- Hybrid search combining dense embeddings with BM25 sparse retrieval.
- Metadata-filtered queries (by company, section, date range, etc.).
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import asdict
from pathlib import Path
from typing import Any, Optional

import chromadb
import numpy as np
from chromadb.config import Settings as ChromaSettings

from src.chunking import Chunk
from src.config import Settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data classes for search results
# ---------------------------------------------------------------------------


class SearchResult:
    """A single search result from the vector store.

    Attributes:
        chunk_id: Unique identifier for the stored chunk.
        text: The chunk text content.
        score: Similarity score (higher is more similar, normalised to [0, 1]).
        metadata: Metadata dictionary associated with the chunk.
    """

    __slots__ = ("chunk_id", "text", "score", "metadata")

    def __init__(
        self,
        chunk_id: str,
        text: str,
        score: float,
        metadata: dict[str, Any],
    ) -> None:
        self.chunk_id = chunk_id
        self.text = text
        self.score = score
        self.metadata = metadata

    def __repr__(self) -> str:
        return (
            f"SearchResult(id={self.chunk_id!r}, score={self.score:.4f}, "
            f"section={self.metadata.get('section', '')!r}, "
            f"len={len(self.text)})"
        )


# ---------------------------------------------------------------------------
# BM25 sparse index (lightweight, in-process)
# ---------------------------------------------------------------------------


class BM25Index:
    """In-memory BM25 sparse index for keyword-based retrieval.

    Built on the ``rank_bm25`` library.  Indexes are rebuilt from scratch
    whenever documents are added (suitable for moderate-scale collections).

    Args:
        tokenizer: A callable that splits text into tokens.
    """

    def __init__(self, tokenizer: Optional[Any] = None) -> None:
        self._tokenizer = tokenizer or self._default_tokenize
        self._corpus_tokens: list[list[str]] = []
        self._doc_ids: list[str] = []
        self._doc_texts: list[str] = []
        self._doc_metadata: list[dict[str, Any]] = []
        self._bm25: Any = None

    @staticmethod
    def _default_tokenize(text: str) -> list[str]:
        """Lowercase whitespace tokenizer with basic punctuation stripping."""
        import re

        tokens = re.findall(r"\b\w+\b", text.lower())
        return tokens

    def add_documents(
        self,
        ids: list[str],
        texts: list[str],
        metadatas: list[dict[str, Any]],
    ) -> None:
        """Add documents to the sparse index and rebuild BM25.

        Args:
            ids: Document identifiers.
            texts: Document texts.
            metadatas: Metadata dicts for each document.
        """
        from rank_bm25 import BM25Okapi

        for doc_id, text, meta in zip(ids, texts, metadatas):
            self._doc_ids.append(doc_id)
            self._doc_texts.append(text)
            self._doc_metadata.append(meta)
            self._corpus_tokens.append(self._tokenizer(text))

        self._bm25 = BM25Okapi(self._corpus_tokens)
        logger.debug("BM25 index rebuilt with %d documents", len(self._doc_ids))

    def search(
        self,
        query: str,
        top_k: int = 5,
        metadata_filter: Optional[dict[str, Any]] = None,
    ) -> list[SearchResult]:
        """Query the BM25 index.

        Args:
            query: User query string.
            top_k: Number of results to return.
            metadata_filter: Optional metadata filter (exact match).

        Returns:
            A list of ``SearchResult`` objects sorted by BM25 score descending.
        """
        if self._bm25 is None or not self._doc_ids:
            return []

        query_tokens = self._tokenizer(query)
        scores = self._bm25.get_scores(query_tokens)

        # Normalise scores to [0, 1]
        max_score = scores.max() if scores.max() > 0 else 1.0
        normalised = scores / max_score

        # Build candidates
        candidates: list[tuple[int, float]] = []
        for i, score in enumerate(normalised):
            if metadata_filter:
                if not all(
                    self._doc_metadata[i].get(k) == v
                    for k, v in metadata_filter.items()
                ):
                    continue
            candidates.append((i, float(score)))

        candidates.sort(key=lambda x: x[1], reverse=True)

        results: list[SearchResult] = []
        for idx, score in candidates[:top_k]:
            results.append(
                SearchResult(
                    chunk_id=self._doc_ids[idx],
                    text=self._doc_texts[idx],
                    score=score,
                    metadata=self._doc_metadata[idx],
                )
            )
        return results

    def clear(self) -> None:
        """Remove all documents from the sparse index."""
        self._corpus_tokens.clear()
        self._doc_ids.clear()
        self._doc_texts.clear()
        self._doc_metadata.clear()
        self._bm25 = None


# ---------------------------------------------------------------------------
# ChromaDB vector store
# ---------------------------------------------------------------------------


class VectorStore:
    """ChromaDB-backed vector store with hybrid search support.

    Manages one or more ChromaDB collections and maintains a parallel
    BM25 index for sparse retrieval.

    Args:
        settings: Application configuration.
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._persist_dir = settings.get_chroma_persist_path()

        self._client = chromadb.PersistentClient(
            path=str(self._persist_dir),
            settings=ChromaSettings(anonymized_telemetry=False),
        )

        self._collection = self._client.get_or_create_collection(
            name=settings.chroma_collection_name,
            metadata={"hnsw:space": "cosine"},
        )

        self._bm25 = BM25Index()
        logger.info(
            "VectorStore initialised: collection='%s', persist='%s', count=%d",
            settings.chroma_collection_name,
            self._persist_dir,
            self._collection.count(),
        )

    @property
    def collection_name(self) -> str:
        """Name of the active ChromaDB collection."""
        return self._settings.chroma_collection_name

    @property
    def count(self) -> int:
        """Number of documents in the active collection."""
        return self._collection.count()

    # -- CRUD ---------------------------------------------------------------

    def add_chunks(
        self,
        chunks: list[Chunk],
        embeddings: np.ndarray,
    ) -> list[str]:
        """Add a batch of chunks with pre-computed embeddings.

        Args:
            chunks: List of ``Chunk`` objects to store.
            embeddings: 2-D numpy array of shape ``(len(chunks), dim)``.

        Returns:
            A list of generated IDs for the stored chunks.
        """
        if len(chunks) != embeddings.shape[0]:
            raise ValueError(
                f"Chunk count ({len(chunks)}) does not match "
                f"embedding count ({embeddings.shape[0]})"
            )

        ids: list[str] = []
        texts: list[str] = []
        metadatas: list[dict[str, Any]] = []
        embedding_list: list[list[float]] = []

        for chunk, emb in zip(chunks, embeddings):
            doc_id = str(uuid.uuid4())
            meta = {
                "company_name": chunk.metadata.company_name,
                "cik": chunk.metadata.cik,
                "filing_date": chunk.metadata.filing_date,
                "filing_type": chunk.metadata.filing_type,
                "section": chunk.metadata.section,
                "source_file": chunk.metadata.source_file,
                "chunk_index": chunk.chunk_index,
                "token_count": chunk.token_count,
            }
            ids.append(doc_id)
            texts.append(chunk.text)
            metadatas.append(meta)
            embedding_list.append(emb.tolist())

        # ChromaDB upsert (idempotent)
        self._collection.upsert(
            ids=ids,
            documents=texts,
            metadatas=metadatas,
            embeddings=embedding_list,
        )

        # Parallel BM25 index
        self._bm25.add_documents(ids, texts, metadatas)

        logger.info("Added %d chunks to vector store", len(chunks))
        return ids

    def delete_by_ids(self, ids: list[str]) -> None:
        """Remove documents by their IDs.

        Args:
            ids: List of document IDs to delete.
        """
        self._collection.delete(ids=ids)
        logger.info("Deleted %d documents from collection", len(ids))

    def delete_collection(self) -> None:
        """Delete the entire active collection."""
        self._client.delete_collection(self._settings.chroma_collection_name)
        self._bm25.clear()
        logger.info("Deleted collection '%s'", self._settings.chroma_collection_name)

    # -- Search -------------------------------------------------------------

    def similarity_search(
        self,
        query_embedding: np.ndarray,
        top_k: Optional[int] = None,
        score_threshold: Optional[float] = None,
        metadata_filter: Optional[dict[str, Any]] = None,
    ) -> list[SearchResult]:
        """Dense similarity search using ChromaDB.

        Args:
            query_embedding: 1-D query vector.
            top_k: Number of results (defaults to config value).
            score_threshold: Minimum cosine similarity (defaults to config).
            metadata_filter: ChromaDB ``where`` filter dict.

        Returns:
            A list of ``SearchResult`` objects sorted by score descending.
        """
        k = top_k or self._settings.retrieval_top_k
        threshold = score_threshold or self._settings.retrieval_score_threshold

        where = self._build_where_filter(metadata_filter) if metadata_filter else None

        results = self._collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=k,
            where=where,
            include=["documents", "metadatas", "distances"],
        )

        search_results: list[SearchResult] = []
        if results["ids"] and results["ids"][0]:
            for idx in range(len(results["ids"][0])):
                # ChromaDB returns cosine *distance*; convert to similarity.
                distance = results["distances"][0][idx]
                similarity = 1.0 - distance

                if similarity < threshold:
                    continue

                search_results.append(
                    SearchResult(
                        chunk_id=results["ids"][0][idx],
                        text=results["documents"][0][idx],
                        score=similarity,
                        metadata=results["metadatas"][0][idx],
                    )
                )

        logger.debug(
            "Dense search returned %d results (threshold=%.2f)",
            len(search_results),
            threshold,
        )
        return search_results

    def sparse_search(
        self,
        query: str,
        top_k: Optional[int] = None,
        metadata_filter: Optional[dict[str, Any]] = None,
    ) -> list[SearchResult]:
        """BM25 sparse keyword search.

        Args:
            query: Raw query string.
            top_k: Number of results.
            metadata_filter: Exact-match metadata filter.

        Returns:
            A list of ``SearchResult`` objects sorted by BM25 score descending.
        """
        k = top_k or self._settings.retrieval_top_k
        return self._bm25.search(query, top_k=k, metadata_filter=metadata_filter)

    def hybrid_search(
        self,
        query: str,
        query_embedding: np.ndarray,
        top_k: Optional[int] = None,
        dense_weight: Optional[float] = None,
        sparse_weight: Optional[float] = None,
        metadata_filter: Optional[dict[str, Any]] = None,
    ) -> list[SearchResult]:
        """Reciprocal rank fusion of dense and sparse search.

        Combines similarity search (dense embeddings) and BM25 (sparse) by
        computing a weighted reciprocal rank fusion score.

        Args:
            query: Raw query string (used for BM25).
            query_embedding: Dense query vector.
            top_k: Number of final results.
            dense_weight: Weight for dense retrieval (default from config).
            sparse_weight: Weight for sparse retrieval (default from config).
            metadata_filter: Metadata filter applied to both searches.

        Returns:
            A fused list of ``SearchResult`` objects.
        """
        k = top_k or self._settings.retrieval_top_k
        dw = dense_weight or self._settings.dense_weight
        sw = sparse_weight or self._settings.sparse_weight

        # Fetch more than top_k from each source to improve fusion quality.
        fetch_k = min(k * 3, 50)

        dense_results = self.similarity_search(
            query_embedding, top_k=fetch_k, score_threshold=0.0, metadata_filter=metadata_filter
        )
        sparse_results = self.sparse_search(
            query, top_k=fetch_k, metadata_filter=metadata_filter
        )

        # Reciprocal rank fusion
        rrf_constant = 60  # standard RRF constant
        rrf_scores: dict[str, float] = {}
        result_map: dict[str, SearchResult] = {}

        for rank, sr in enumerate(dense_results):
            rrf_scores[sr.chunk_id] = rrf_scores.get(sr.chunk_id, 0.0) + dw / (
                rrf_constant + rank + 1
            )
            result_map[sr.chunk_id] = sr

        for rank, sr in enumerate(sparse_results):
            rrf_scores[sr.chunk_id] = rrf_scores.get(sr.chunk_id, 0.0) + sw / (
                rrf_constant + rank + 1
            )
            if sr.chunk_id not in result_map:
                result_map[sr.chunk_id] = sr

        # Sort by fused score and return top_k
        sorted_ids = sorted(rrf_scores, key=rrf_scores.get, reverse=True)[:k]

        fused: list[SearchResult] = []
        for doc_id in sorted_ids:
            sr = result_map[doc_id]
            sr.score = rrf_scores[doc_id]
            fused.append(sr)

        logger.info(
            "Hybrid search: %d dense + %d sparse -> %d fused results",
            len(dense_results),
            len(sparse_results),
            len(fused),
        )
        return fused

    # -- Helpers ------------------------------------------------------------

    @staticmethod
    def _build_where_filter(
        metadata_filter: dict[str, Any],
    ) -> dict[str, Any]:
        """Convert a flat metadata dict into a ChromaDB ``where`` filter.

        Supports simple equality filters.  Multiple keys are combined with
        ``$and``.

        Args:
            metadata_filter: Dict of field -> value.

        Returns:
            A ChromaDB-compatible ``where`` expression.
        """
        conditions = []
        for key, value in metadata_filter.items():
            conditions.append({key: {"$eq": value}})

        if len(conditions) == 1:
            return conditions[0]
        return {"$and": conditions}

    def list_collections(self) -> list[str]:
        """List all collection names in the ChromaDB instance."""
        return [c.name for c in self._client.list_collections()]

    def get_collection_stats(self) -> dict[str, Any]:
        """Return summary statistics for the active collection."""
        return {
            "collection_name": self.collection_name,
            "document_count": self.count,
            "persist_directory": str(self._persist_dir),
        }
