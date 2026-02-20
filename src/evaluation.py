"""
RAG evaluation pipeline for the Financial Filing Analyst.

Measures both retrieval quality and generation quality with a comprehensive
set of metrics:

**Retrieval metrics:**
- Precision@K — fraction of retrieved documents that are relevant.
- Recall@K — fraction of relevant documents that are retrieved.
- MRR (Mean Reciprocal Rank) — average of 1/rank for the first relevant hit.
- NDCG@K (Normalised Discounted Cumulative Gain) — ranking-aware relevance.

**Generation metrics:**
- Faithfulness — fraction of answer claims grounded in the sources.
- Answer relevancy — semantic similarity between answer and query.
- Context relevancy — fraction of retrieved context that is useful.

**Custom financial metrics:**
- Numerical accuracy — fraction of numbers in the answer that match sources.

All metrics can be logged to MLflow for experiment tracking.
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np

from src.config import Settings
from src.embeddings import EmbeddingPipeline
from src.guardrails import extract_numbers, normalise_number
from src.vector_store import SearchResult

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class EvalQuestion:
    """A single evaluation question with ground-truth annotations.

    Attributes:
        query: The question text.
        ground_truth_answer: Expected answer (for generation evaluation).
        relevant_chunk_ids: IDs of chunks that should be retrieved.
        relevant_sections: Sections that contain the answer.
        expected_numbers: Financial figures that should appear in the answer.
        metadata: Additional context (company, filing date, etc.).
    """

    query: str
    ground_truth_answer: str = ""
    relevant_chunk_ids: list[str] = field(default_factory=list)
    relevant_sections: list[str] = field(default_factory=list)
    expected_numbers: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class EvalResult:
    """Evaluation results for a single question.

    Attributes:
        query: The input question.
        precision_at_k: Precision@K score.
        recall_at_k: Recall@K score.
        mrr: Reciprocal rank of the first relevant result.
        ndcg_at_k: NDCG@K score.
        faithfulness: Fraction of answer claims grounded in sources.
        answer_relevancy: Cosine similarity between answer and query embeddings.
        context_relevancy: Fraction of context that is relevant.
        numerical_accuracy: Fraction of answer numbers verified in sources.
    """

    query: str = ""
    precision_at_k: float = 0.0
    recall_at_k: float = 0.0
    mrr: float = 0.0
    ndcg_at_k: float = 0.0
    faithfulness: float = 0.0
    answer_relevancy: float = 0.0
    context_relevancy: float = 0.0
    numerical_accuracy: float = 0.0


@dataclass
class AggregateMetrics:
    """Averaged metrics across an evaluation dataset.

    Attributes:
        precision_at_k: Mean Precision@K.
        recall_at_k: Mean Recall@K.
        mrr: Mean Reciprocal Rank.
        ndcg_at_k: Mean NDCG@K.
        faithfulness: Mean faithfulness.
        answer_relevancy: Mean answer relevancy.
        context_relevancy: Mean context relevancy.
        numerical_accuracy: Mean numerical accuracy.
        num_questions: Number of questions evaluated.
    """

    precision_at_k: float = 0.0
    recall_at_k: float = 0.0
    mrr: float = 0.0
    ndcg_at_k: float = 0.0
    faithfulness: float = 0.0
    answer_relevancy: float = 0.0
    context_relevancy: float = 0.0
    numerical_accuracy: float = 0.0
    num_questions: int = 0

    def to_dict(self) -> dict[str, float]:
        """Serialize all metrics to a flat dict."""
        return {
            "precision_at_k": self.precision_at_k,
            "recall_at_k": self.recall_at_k,
            "mrr": self.mrr,
            "ndcg_at_k": self.ndcg_at_k,
            "faithfulness": self.faithfulness,
            "answer_relevancy": self.answer_relevancy,
            "context_relevancy": self.context_relevancy,
            "numerical_accuracy": self.numerical_accuracy,
            "num_questions": float(self.num_questions),
        }


# ---------------------------------------------------------------------------
# Retrieval metrics
# ---------------------------------------------------------------------------


def precision_at_k(
    retrieved_ids: list[str], relevant_ids: set[str], k: int
) -> float:
    """Compute Precision@K.

    Args:
        retrieved_ids: Ordered list of retrieved chunk IDs.
        relevant_ids: Set of ground-truth relevant IDs.
        k: Cutoff rank.

    Returns:
        Precision@K in [0, 1].
    """
    if k <= 0:
        return 0.0
    top_k = retrieved_ids[:k]
    relevant_in_top_k = sum(1 for doc_id in top_k if doc_id in relevant_ids)
    return relevant_in_top_k / k


def recall_at_k(
    retrieved_ids: list[str], relevant_ids: set[str], k: int
) -> float:
    """Compute Recall@K.

    Args:
        retrieved_ids: Ordered list of retrieved chunk IDs.
        relevant_ids: Set of ground-truth relevant IDs.
        k: Cutoff rank.

    Returns:
        Recall@K in [0, 1].
    """
    if not relevant_ids:
        return 0.0
    top_k = retrieved_ids[:k]
    relevant_in_top_k = sum(1 for doc_id in top_k if doc_id in relevant_ids)
    return relevant_in_top_k / len(relevant_ids)


def reciprocal_rank(
    retrieved_ids: list[str], relevant_ids: set[str]
) -> float:
    """Compute the reciprocal rank of the first relevant result.

    Args:
        retrieved_ids: Ordered list of retrieved chunk IDs.
        relevant_ids: Set of ground-truth relevant IDs.

    Returns:
        1/rank of the first relevant result, or 0 if none found.
    """
    for rank, doc_id in enumerate(retrieved_ids, start=1):
        if doc_id in relevant_ids:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(
    retrieved_ids: list[str], relevant_ids: set[str], k: int
) -> float:
    """Compute NDCG@K (binary relevance).

    Args:
        retrieved_ids: Ordered list of retrieved chunk IDs.
        relevant_ids: Set of ground-truth relevant IDs.
        k: Cutoff rank.

    Returns:
        NDCG@K in [0, 1].
    """
    if not relevant_ids or k <= 0:
        return 0.0

    # DCG
    dcg = 0.0
    for i, doc_id in enumerate(retrieved_ids[:k]):
        rel = 1.0 if doc_id in relevant_ids else 0.0
        dcg += rel / math.log2(i + 2)  # i+2 because log2(1) = 0

    # Ideal DCG: all relevant docs at the top
    ideal_count = min(len(relevant_ids), k)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_count))

    return dcg / idcg if idcg > 0 else 0.0


# ---------------------------------------------------------------------------
# Generation metrics
# ---------------------------------------------------------------------------


def compute_faithfulness(
    answer: str, source_texts: list[str]
) -> float:
    """Estimate faithfulness by checking sentence-level grounding.

    Splits the answer into sentences and checks what fraction have
    significant key-phrase overlap with the concatenated sources.

    Args:
        answer: Generated answer.
        source_texts: List of source chunk texts.

    Returns:
        Faithfulness score in [0, 1].
    """
    import re as _re

    sentences = _re.split(r"(?<=[.!?])\s+", answer)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 15]

    if not sentences:
        return 1.0  # trivially faithful if no claims

    source_combined = " ".join(source_texts).lower()
    grounded = 0

    for sentence in sentences:
        words = _re.findall(r"\b\w{4,}\b", sentence.lower())
        if not words:
            grounded += 1
            continue
        overlap = sum(1 for w in words if w in source_combined)
        if overlap / len(words) >= 0.3:
            grounded += 1

    return grounded / len(sentences)


def compute_answer_relevancy(
    query_embedding: np.ndarray, answer_embedding: np.ndarray
) -> float:
    """Compute cosine similarity between query and answer embeddings.

    Args:
        query_embedding: 1-D query vector.
        answer_embedding: 1-D answer vector.

    Returns:
        Cosine similarity in [-1, 1] (typically [0, 1] for these models).
    """
    norm_q = np.linalg.norm(query_embedding)
    norm_a = np.linalg.norm(answer_embedding)
    if norm_q == 0 or norm_a == 0:
        return 0.0
    return float(np.dot(query_embedding, answer_embedding) / (norm_q * norm_a))


def compute_context_relevancy(
    query: str, source_texts: list[str]
) -> float:
    """Estimate what fraction of the retrieved context is relevant.

    Uses a heuristic: a source chunk is "relevant" if it shares significant
    key-phrase overlap with the query.

    Args:
        query: User question.
        source_texts: List of retrieved chunk texts.

    Returns:
        Context relevancy in [0, 1].
    """
    import re as _re

    if not source_texts:
        return 0.0

    query_words = set(_re.findall(r"\b\w{4,}\b", query.lower()))
    if not query_words:
        return 0.0

    relevant_count = 0
    for text in source_texts:
        text_words = set(_re.findall(r"\b\w{4,}\b", text.lower()))
        overlap = len(query_words & text_words) / len(query_words)
        if overlap >= 0.2:
            relevant_count += 1

    return relevant_count / len(source_texts)


def compute_numerical_accuracy(
    answer: str, source_texts: list[str], tolerance: float = 0.02
) -> float:
    """Compute the fraction of answer numbers verifiable in sources.

    Args:
        answer: Generated answer.
        source_texts: Source chunk texts.
        tolerance: Relative tolerance for numeric comparison.

    Returns:
        Numerical accuracy in [0, 1]. Returns 1.0 if no numbers are present.
    """
    answer_numbers = extract_numbers(answer)
    if not answer_numbers:
        return 1.0

    source_text = " ".join(source_texts)
    source_numbers = extract_numbers(source_text)
    source_vals = [normalise_number(n) for n in source_numbers]
    source_vals_clean = [v for v in source_vals if v is not None]

    verified = 0
    for num_str in answer_numbers:
        if num_str in source_text:
            verified += 1
            continue
        val = normalise_number(num_str)
        if val is not None and source_vals_clean:
            if any(
                abs(val - sv) / max(abs(sv), 1e-10) < tolerance
                for sv in source_vals_clean
            ):
                verified += 1

    return verified / len(answer_numbers)


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------


class RAGEvaluator:
    """End-to-end RAG evaluation pipeline.

    Takes a pipeline instance, runs it on a set of evaluation questions,
    computes all metrics, and optionally logs results to MLflow.

    Args:
        pipeline: The RAG pipeline (must implement `query` method).
        embedding_pipeline: Embedding pipeline for answer relevancy.
        settings: Application configuration.
    """

    def __init__(
        self,
        pipeline: Any,
        embedding_pipeline: Optional[EmbeddingPipeline] = None,
        settings: Optional[Settings] = None,
    ) -> None:
        self._pipeline = pipeline
        self._embedder = embedding_pipeline
        self._settings = settings or Settings()

    def load_eval_dataset(self, path: str | Path) -> list[EvalQuestion]:
        """Load evaluation questions from a JSON file.

        Expected format::

            [
                {
                    "query": "What was Apple's revenue in 2024?",
                    "ground_truth_answer": "Apple's revenue was $391.0 billion.",
                    "relevant_chunk_ids": ["id1", "id2"],
                    "relevant_sections": ["Financial Statements"],
                    "expected_numbers": ["$391.0 billion"],
                    "metadata": {"company": "Apple Inc."}
                }
            ]

        Args:
            path: Path to the JSON file.

        Returns:
            A list of ``EvalQuestion`` objects.
        """
        file_path = Path(path)
        if not file_path.exists():
            raise FileNotFoundError(f"Evaluation dataset not found: {file_path}")

        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        questions = []
        for item in data:
            questions.append(
                EvalQuestion(
                    query=item["query"],
                    ground_truth_answer=item.get("ground_truth_answer", ""),
                    relevant_chunk_ids=item.get("relevant_chunk_ids", []),
                    relevant_sections=item.get("relevant_sections", []),
                    expected_numbers=item.get("expected_numbers", []),
                    metadata=item.get("metadata", {}),
                )
            )

        logger.info("Loaded %d evaluation questions from %s", len(questions), file_path)
        return questions

    def evaluate_single(
        self,
        question: EvalQuestion,
        k: int = 5,
    ) -> EvalResult:
        """Evaluate a single question.

        Runs the pipeline, computes retrieval and generation metrics.

        Args:
            question: The evaluation question.
            k: Cutoff rank for retrieval metrics.

        Returns:
            An ``EvalResult`` with all computed metrics.
        """
        # Run the pipeline
        pipeline_result = self._pipeline.query(question.query)
        retrieved_ids = [s.chunk_id for s in pipeline_result.sources]
        source_texts = [s.text for s in pipeline_result.sources]
        answer = pipeline_result.answer
        relevant_ids = set(question.relevant_chunk_ids)

        # Retrieval metrics
        prec = precision_at_k(retrieved_ids, relevant_ids, k) if relevant_ids else 0.0
        rec = recall_at_k(retrieved_ids, relevant_ids, k) if relevant_ids else 0.0
        rr = reciprocal_rank(retrieved_ids, relevant_ids) if relevant_ids else 0.0
        ndcg = ndcg_at_k(retrieved_ids, relevant_ids, k) if relevant_ids else 0.0

        # Generation metrics
        faith = compute_faithfulness(answer, source_texts)
        ctx_rel = compute_context_relevancy(question.query, source_texts)
        num_acc = compute_numerical_accuracy(answer, source_texts)

        # Answer relevancy (requires embeddings)
        ans_rel = 0.0
        if self._embedder is not None:
            try:
                q_emb = self._embedder.embed_query(question.query)
                a_emb = self._embedder.embed_query(answer[:500])
                ans_rel = compute_answer_relevancy(q_emb, a_emb)
            except Exception as exc:
                logger.warning("Answer relevancy computation failed: %s", exc)

        return EvalResult(
            query=question.query,
            precision_at_k=prec,
            recall_at_k=rec,
            mrr=rr,
            ndcg_at_k=ndcg,
            faithfulness=faith,
            answer_relevancy=ans_rel,
            context_relevancy=ctx_rel,
            numerical_accuracy=num_acc,
        )

    def evaluate_dataset(
        self,
        path: str | Path,
        k: int = 5,
        log_to_mlflow: bool = False,
    ) -> AggregateMetrics:
        """Evaluate the pipeline on an entire dataset.

        Args:
            path: Path to evaluation questions JSON.
            k: Cutoff rank.
            log_to_mlflow: Whether to log metrics to MLflow.

        Returns:
            Aggregated metrics across all questions.
        """
        questions = self.load_eval_dataset(path)
        results: list[EvalResult] = []

        for question in questions:
            try:
                result = self.evaluate_single(question, k=k)
                results.append(result)
            except Exception as exc:
                logger.error(
                    "Evaluation failed for query '%s': %s",
                    question.query[:80],
                    exc,
                )

        if not results:
            logger.warning("No evaluation results produced.")
            return AggregateMetrics()

        # Aggregate
        metrics = AggregateMetrics(
            precision_at_k=np.mean([r.precision_at_k for r in results]),
            recall_at_k=np.mean([r.recall_at_k for r in results]),
            mrr=np.mean([r.mrr for r in results]),
            ndcg_at_k=np.mean([r.ndcg_at_k for r in results]),
            faithfulness=np.mean([r.faithfulness for r in results]),
            answer_relevancy=np.mean([r.answer_relevancy for r in results]),
            context_relevancy=np.mean([r.context_relevancy for r in results]),
            numerical_accuracy=np.mean([r.numerical_accuracy for r in results]),
            num_questions=len(results),
        )

        logger.info(
            "Evaluation complete: %d questions, P@%d=%.3f, R@%d=%.3f, "
            "MRR=%.3f, NDCG=%.3f, Faith=%.3f, AnsRel=%.3f",
            len(results),
            k,
            metrics.precision_at_k,
            k,
            metrics.recall_at_k,
            metrics.mrr,
            metrics.ndcg_at_k,
            metrics.faithfulness,
            metrics.answer_relevancy,
        )

        if log_to_mlflow:
            self._log_to_mlflow(metrics)

        return metrics

    def _log_to_mlflow(self, metrics: AggregateMetrics) -> None:
        """Log aggregated metrics to MLflow.

        Args:
            metrics: The aggregated evaluation metrics.
        """
        try:
            import mlflow

            mlflow.set_tracking_uri(self._settings.mlflow_tracking_uri)
            mlflow.set_experiment(self._settings.mlflow_experiment_name)

            with mlflow.start_run():
                mlflow.log_metrics(metrics.to_dict())
                mlflow.log_params(
                    {
                        "retrieval_strategy": self._settings.retrieval_strategy.value,
                        "chunking_strategy": self._settings.chunking_strategy.value,
                        "embedding_model": self._settings.embedding_model_name,
                        "llm_model": self._settings.llm_model_name,
                        "chunk_size": self._settings.chunk_size,
                        "top_k": self._settings.retrieval_top_k,
                    }
                )
            logger.info("Metrics logged to MLflow experiment '%s'", self._settings.mlflow_experiment_name)
        except ImportError:
            logger.warning("MLflow not installed; skipping metric logging.")
        except Exception as exc:
            logger.error("Failed to log to MLflow: %s", exc)
