"""
Multi-model embedding pipeline with caching and batch processing.

Supports three embedding backends:
- **OpenAI** (``text-embedding-3-small``, ``text-embedding-3-large``)
- **Sentence-Transformers** (any model from the ``sentence-transformers`` hub)
- **Hugging Face** (arbitrary ``transformers`` models via feature extraction)

Embeddings are optionally cached to disk so re-ingesting the same document
does not re-compute vectors, saving both time and API cost.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np
from tqdm import tqdm

from src.config import EmbeddingProvider, Settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------


class EmbeddingCache:
    """Simple file-backed embedding cache keyed by content hash.

    Each entry is stored as a ``.npy`` file whose name is the SHA-256 of
    the input text.  The cache directory is created lazily on first write.

    Args:
        cache_dir: Directory to store cached embeddings.
    """

    def __init__(self, cache_dir: Path) -> None:
        self._dir = cache_dir

    @staticmethod
    def _key(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def get(self, text: str) -> Optional[np.ndarray]:
        """Retrieve a cached embedding, or ``None`` if not cached."""
        path = self._dir / f"{self._key(text)}.npy"
        if path.exists():
            return np.load(path)
        return None

    def put(self, text: str, embedding: np.ndarray) -> None:
        """Store an embedding in the cache."""
        self._dir.mkdir(parents=True, exist_ok=True)
        path = self._dir / f"{self._key(text)}.npy"
        np.save(path, embedding)

    def get_batch(self, texts: list[str]) -> tuple[list[np.ndarray | None], list[int]]:
        """Look up a batch of texts, returning cached values and miss indices.

        Returns:
            A tuple of (results, miss_indices) where *results* has the same
            length as *texts* (``None`` for misses) and *miss_indices*
            contains the indices of texts that need embedding.
        """
        results: list[np.ndarray | None] = []
        misses: list[int] = []
        for i, text in enumerate(texts):
            cached = self.get(text)
            results.append(cached)
            if cached is None:
                misses.append(i)
        return results, misses

    def put_batch(self, texts: list[str], embeddings: np.ndarray) -> None:
        """Cache a batch of embeddings."""
        for text, emb in zip(texts, embeddings):
            self.put(text, emb)

    def clear(self) -> int:
        """Delete all cached embeddings. Returns the number of files removed."""
        if not self._dir.exists():
            return 0
        files = list(self._dir.glob("*.npy"))
        for f in files:
            f.unlink()
        return len(files)


# ---------------------------------------------------------------------------
# Embedding backends
# ---------------------------------------------------------------------------


class OpenAIEmbedder:
    """Embedding backend using the OpenAI API.

    Args:
        api_key: OpenAI API key.
        model: Model name (e.g. ``text-embedding-3-small``).
        dimension: Desired output dimension (must match model capability).
    """

    def __init__(self, api_key: str, model: str, dimension: int) -> None:
        from openai import OpenAI

        self._client = OpenAI(api_key=api_key)
        self._model = model
        self._dimension = dimension

    def embed(self, texts: list[str]) -> np.ndarray:
        """Embed a batch of texts via the OpenAI API.

        Args:
            texts: List of strings to embed.

        Returns:
            A 2-D numpy array of shape ``(len(texts), dimension)``.
        """
        response = self._client.embeddings.create(
            input=texts,
            model=self._model,
            dimensions=self._dimension,
        )
        vectors = [item.embedding for item in response.data]
        return np.array(vectors, dtype=np.float32)


class SentenceTransformerEmbedder:
    """Embedding backend using the ``sentence-transformers`` library.

    Loads the model once and encodes batches on CPU or available GPU.

    Args:
        model_name: Hugging Face model identifier
            (e.g. ``all-MiniLM-L6-v2``).
    """

    def __init__(self, model_name: str) -> None:
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(model_name)
        logger.info("Loaded sentence-transformers model: %s", model_name)

    def embed(self, texts: list[str]) -> np.ndarray:
        """Encode texts using the loaded sentence-transformer model.

        Returns:
            A 2-D numpy array of shape ``(len(texts), dim)``.
        """
        embeddings = self._model.encode(
            texts, show_progress_bar=False, convert_to_numpy=True
        )
        return np.array(embeddings, dtype=np.float32)


class HuggingFaceEmbedder:
    """Embedding backend using arbitrary Hugging Face ``transformers`` models.

    Uses mean-pooling over the last hidden state to produce a fixed-size
    vector per input text.

    Args:
        model_name: Hugging Face model identifier.
    """

    def __init__(self, model_name: str) -> None:
        from transformers import AutoModel, AutoTokenizer
        import torch

        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._model = AutoModel.from_pretrained(model_name)
        self._model.eval()
        self._torch = torch
        logger.info("Loaded HuggingFace model for embeddings: %s", model_name)

    def embed(self, texts: list[str]) -> np.ndarray:
        """Mean-pool the last hidden state to produce embeddings.

        Returns:
            A 2-D numpy array of shape ``(len(texts), hidden_size)``.
        """
        encoded = self._tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )
        with self._torch.no_grad():
            outputs = self._model(**encoded)

        # Mean pooling over non-padding tokens.
        attention_mask = encoded["attention_mask"].unsqueeze(-1).float()
        token_embeddings = outputs.last_hidden_state
        summed = (token_embeddings * attention_mask).sum(dim=1)
        counts = attention_mask.sum(dim=1).clamp(min=1e-9)
        mean_pooled = summed / counts
        return mean_pooled.numpy().astype(np.float32)


# ---------------------------------------------------------------------------
# Unified pipeline
# ---------------------------------------------------------------------------


class EmbeddingPipeline:
    """Unified embedding pipeline with caching, batching, and multi-backend support.

    Instantiates the correct backend based on ``settings.embedding_provider``
    and transparently handles the embedding cache.

    Args:
        settings: Application configuration.
        cache_dir: Override the default cache directory.
    """

    DEFAULT_BATCH_SIZE = 64

    def __init__(
        self,
        settings: Settings,
        cache_dir: Optional[Path] = None,
    ) -> None:
        self._settings = settings
        self._cache = EmbeddingCache(
            cache_dir or Path(".embedding_cache")
        )
        self._backend = self._init_backend(settings)
        self._dimension = settings.embedding_dimension

    @staticmethod
    def _init_backend(
        settings: Settings,
    ) -> OpenAIEmbedder | SentenceTransformerEmbedder | HuggingFaceEmbedder:
        """Create the appropriate embedding backend."""
        provider = settings.embedding_provider

        if provider == EmbeddingProvider.OPENAI:
            if not settings.openai_api_key:
                raise ValueError(
                    "OPENAI_API_KEY is required when embedding_provider is 'openai'."
                )
            return OpenAIEmbedder(
                api_key=settings.openai_api_key,
                model=settings.embedding_model_name,
                dimension=settings.embedding_dimension,
            )

        if provider == EmbeddingProvider.SENTENCE_TRANSFORMERS:
            return SentenceTransformerEmbedder(
                model_name=settings.embedding_model_name
            )

        if provider == EmbeddingProvider.HUGGINGFACE:
            return HuggingFaceEmbedder(
                model_name=settings.embedding_model_name
            )

        raise ValueError(f"Unsupported embedding provider: {provider}")

    @property
    def dimension(self) -> int:
        """Return the configured embedding dimension."""
        return self._dimension

    def embed_texts(
        self,
        texts: list[str],
        batch_size: int = DEFAULT_BATCH_SIZE,
        use_cache: bool = True,
        show_progress: bool = True,
    ) -> np.ndarray:
        """Embed a list of texts with caching and batched API calls.

        Args:
            texts: List of strings to embed.
            batch_size: Number of texts per backend call.
            use_cache: Whether to use the disk cache.
            show_progress: Show a tqdm progress bar.

        Returns:
            A 2-D numpy array of shape ``(len(texts), dimension)``.
        """
        if not texts:
            return np.empty((0, self._dimension), dtype=np.float32)

        # Check cache
        if use_cache:
            cached_results, miss_indices = self._cache.get_batch(texts)
        else:
            cached_results = [None] * len(texts)
            miss_indices = list(range(len(texts)))

        cache_hits = len(texts) - len(miss_indices)
        if cache_hits > 0:
            logger.info(
                "Embedding cache: %d hits, %d misses", cache_hits, len(miss_indices)
            )

        # Embed cache misses in batches
        if miss_indices:
            miss_texts = [texts[i] for i in miss_indices]
            miss_embeddings: list[np.ndarray] = []

            iterator = range(0, len(miss_texts), batch_size)
            if show_progress:
                iterator = tqdm(
                    iterator,
                    desc="Embedding batches",
                    total=(len(miss_texts) + batch_size - 1) // batch_size,
                )

            for start in iterator:
                batch = miss_texts[start : start + batch_size]
                batch_emb = self._backend.embed(batch)
                miss_embeddings.append(batch_emb)

            all_miss_embeddings = np.concatenate(miss_embeddings, axis=0)

            # Write misses back to cache
            if use_cache:
                self._cache.put_batch(miss_texts, all_miss_embeddings)

            # Merge into the results array
            for idx, miss_idx in enumerate(miss_indices):
                cached_results[miss_idx] = all_miss_embeddings[idx]

        result = np.array(cached_results, dtype=np.float32)
        logger.info(
            "Embedded %d texts -> shape %s", len(texts), result.shape
        )
        return result

    def embed_query(self, query: str) -> np.ndarray:
        """Embed a single query string (no caching, for real-time retrieval).

        Args:
            query: The user query.

        Returns:
            A 1-D numpy array of shape ``(dimension,)``.
        """
        result = self._backend.embed([query])
        return result[0]

    def clear_cache(self) -> int:
        """Clear the embedding cache. Returns the number of entries removed."""
        removed = self._cache.clear()
        logger.info("Cleared %d entries from embedding cache", removed)
        return removed
