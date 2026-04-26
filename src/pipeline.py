"""
End-to-end RAG pipeline orchestrating all components.

The ``RAGPipeline`` class wires together document loading, chunking,
embedding, vector storage, retrieval, generation, guardrails, and
knowledge graph augmentation into a single, configurable interface.

Usage::

    from src.pipeline import RAGPipeline
    from src.config import Settings

    settings = Settings()
    pipeline = RAGPipeline(settings)
    pipeline.ingest_document("path/to/filing.txt")
    result = pipeline.query("What was total revenue?")
    print(result.answer)
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Generator, Optional

from src.chunking import DocumentChunker
from src.config import Settings
from src.document_loader import Document, SECFilingLoader
from src.embeddings import EmbeddingPipeline
from src.generator import FinancialGenerator, GenerationResult
from src.guardrails import GuardrailPipeline, GuardrailReport
from src.knowledge_graph import FinancialKnowledgeGraph
from src.retriever import (
    ContextualCompressor,
    EnsembleRetriever,
    QueryExpander,
    RetrievalResult,
)
from src.vector_store import SearchResult, VectorStore

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Pipeline result
# ---------------------------------------------------------------------------


@dataclass
class PipelineResult:
    """Complete output from a pipeline query.

    Attributes:
        answer: The generated answer text.
        sources: Retrieved source chunks.
        confidence_score: Guardrail confidence score.
        guardrail_report: Full guardrail validation report.
        retrieval_result: Detailed retrieval output.
        generation_result: Detailed generation output.
        graph_context: Knowledge graph context used for augmentation.
        latency_ms: End-to-end query latency in milliseconds.
        warnings: Any warnings from guardrails or processing.
    """

    answer: str = ""
    sources: list[SearchResult] = field(default_factory=list)
    confidence_score: float = 1.0
    guardrail_report: Optional[GuardrailReport] = None
    retrieval_result: Optional[RetrievalResult] = None
    generation_result: Optional[GenerationResult] = None
    graph_context: str = ""
    latency_ms: float = 0.0
    warnings: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


class RAGPipeline:
    """End-to-end RAG pipeline for SEC 10-K filing analysis.

    Orchestrates:
    1. Document loading and section parsing.
    2. Chunking with the configured strategy.
    3. Embedding generation and vector store indexing.
    4. Multi-strategy retrieval with optional knowledge graph augmentation.
    5. Chain-of-thought LLM generation with source citations.
    6. Guardrail validation and confidence scoring.

    Args:
        settings: Application configuration. If ``None``, loads from
            environment / ``.env`` file.
    """

    def __init__(self, settings: Optional[Settings] = None) -> None:
        self._settings = settings or Settings()
        self._settings.configure_logging()

        logger.info("Initialising RAG pipeline...")
        start = time.time()

        # Core components
        self._loader = SECFilingLoader(self._settings)
        self._embedding_pipeline = EmbeddingPipeline(self._settings)
        self._chunker = DocumentChunker(
            self._settings,
            embed_fn=self._embedding_pipeline.embed_texts,
        )
        self._vector_store = VectorStore(self._settings)
        self._generator = FinancialGenerator(self._settings)
        self._guardrails = GuardrailPipeline(self._settings)
        self._knowledge_graph = FinancialKnowledgeGraph()

        # Retriever with optional query expansion and reranking
        self._query_expander = QueryExpander(
            llm_fn=self._generator.simple_generate,
            num_variants=3,
        )
        self._compressor = ContextualCompressor(
            embedding_pipeline=self._embedding_pipeline,
            top_n=self._settings.retrieval_top_k,
            compress=False,
        )
        self._retriever = EnsembleRetriever(
            vector_store=self._vector_store,
            embedding_pipeline=self._embedding_pipeline,
            settings=self._settings,
            compressor=self._compressor,
            query_expander=self._query_expander,
        )

        elapsed = (time.time() - start) * 1000
        logger.info("RAG pipeline initialised in %.0f ms", elapsed)

    # -- Document ingestion -------------------------------------------------

    def ingest_document(
        self,
        file_path: str | Path,
        company_name: str = "",
        filing_date: str = "",
    ) -> int:
        """Ingest a local 10-K filing into the vector store.

        Loads the file, parses sections, chunks, embeds, and indexes.
        Also builds the knowledge graph from the extracted entities.

        Args:
            file_path: Path to the filing text file.
            company_name: Override company name.
            filing_date: Override filing date.

        Returns:
            The number of chunks indexed.
        """
        logger.info("Ingesting document: %s", file_path)
        start = time.time()

        # Load and parse
        documents = self._loader.load_from_file(
            file_path,
            company_name=company_name,
            filing_date=filing_date,
        )

        return self._ingest_documents(documents, start)

    def ingest_from_edgar(
        self,
        cik: str,
        company_name: str = "",
        count: int = 1,
    ) -> int:
        """Ingest 10-K filings from EDGAR.

        Args:
            cik: Central Index Key.
            company_name: Company name.
            count: Number of filings to download.

        Returns:
            The number of chunks indexed.
        """
        logger.info("Ingesting from EDGAR: CIK=%s, count=%d", cik, count)
        start = time.time()

        documents = self._loader.load_from_edgar(
            cik, company_name=company_name, count=count
        )

        return self._ingest_documents(documents, start)

    def ingest_text(
        self,
        text: str,
        company_name: str = "",
        section: str = "",
        filing_date: str = "",
    ) -> int:
        """Ingest raw text directly (useful for uploaded content).

        Args:
            text: Filing text content.
            company_name: Company name.
            section: Section label.
            filing_date: Filing date.

        Returns:
            The number of chunks indexed.
        """
        from src.document_loader import DocumentMetadata

        logger.info("Ingesting raw text (%d chars)", len(text))
        start = time.time()

        metadata = DocumentMetadata(
            company_name=company_name,
            section=section,
            filing_date=filing_date,
        )
        documents = [Document(text=text, metadata=metadata)]

        return self._ingest_documents(documents, start)

    def _ingest_documents(
        self, documents: list[Document], start_time: float
    ) -> int:
        """Shared ingestion logic: chunk, embed, index, build graph.

        Args:
            documents: Parsed filing sections.
            start_time: Timestamp when ingestion began.

        Returns:
            Number of chunks indexed.
        """
        if not documents:
            logger.warning("No documents to ingest.")
            return 0

        # Chunk
        chunks = self._chunker.chunk_documents(documents)
        if not chunks:
            logger.warning("Chunking produced no chunks.")
            return 0

        # Embed
        texts = [c.text for c in chunks]
        embeddings = self._embedding_pipeline.embed_texts(texts)

        # Index
        self._vector_store.add_chunks(chunks, embeddings)

        # Build knowledge graph
        company_name = documents[0].metadata.company_name if documents else ""
        self._knowledge_graph.build_from_documents(
            documents, company_name=company_name
        )

        elapsed = (time.time() - start_time) * 1000
        logger.info(
            "Ingestion complete: %d chunks indexed in %.0f ms "
            "(graph: %d nodes, %d edges)",
            len(chunks),
            elapsed,
            self._knowledge_graph.num_nodes,
            self._knowledge_graph.num_edges,
        )
        return len(chunks)

    # -- Query --------------------------------------------------------------

    def query(
        self,
        query: str,
        filters: Optional[dict[str, Any]] = None,
        use_graph: bool = True,
        validate: bool = True,
    ) -> PipelineResult:
        """Execute an end-to-end RAG query.

        Args:
            query: The user's question.
            filters: Optional metadata filters (company, section, date).
            use_graph: Whether to augment context with the knowledge graph.
            validate: Whether to run guardrail validation.

        Returns:
            A ``PipelineResult`` with the answer, sources, and metadata.
        """
        start = time.time()
        logger.info("Processing query: %s", query[:100])

        # 1. Retrieve
        retrieval_result = self._retriever.retrieve(
            query=query,
            metadata_filter=filters,
        )
        sources = retrieval_result.results

        if not sources:
            return PipelineResult(
                answer="I could not find relevant information in the indexed "
                "filings to answer this question. Please ensure the "
                "relevant 10-K filing has been ingested.",
                retrieval_result=retrieval_result,
                latency_ms=(time.time() - start) * 1000,
                warnings=["No relevant documents retrieved."],
            )

        # 2. Knowledge graph augmentation
        graph_context = ""
        if use_graph:
            # Try to identify the company from filters or retrieved metadata
            company = ""
            if filters and "company" in filters:
                company = filters["company"]
            elif sources:
                company = sources[0].metadata.get("company_name", "")

            if company:
                graph_context = self._knowledge_graph.get_graph_context(
                    company, depth=2
                )

        # 3. Generate
        generation_result = self._generator.generate(
            query=query,
            results=sources,
            system_prompt=self._build_system_prompt(graph_context),
        )

        # 4. Guardrails
        guardrail_report = None
        confidence = 1.0
        warnings: list[str] = []

        if validate:
            guardrail_report = self._guardrails.validate(
                answer=generation_result.answer,
                sources=sources,
            )
            confidence = guardrail_report.confidence_score
            warnings = guardrail_report.warnings

        latency = (time.time() - start) * 1000
        logger.info(
            "Query processed in %.0f ms (confidence=%.2f, sources=%d)",
            latency,
            confidence,
            len(sources),
        )

        return PipelineResult(
            answer=generation_result.answer,
            sources=sources,
            confidence_score=confidence,
            guardrail_report=guardrail_report,
            retrieval_result=retrieval_result,
            generation_result=generation_result,
            graph_context=graph_context,
            latency_ms=latency,
            warnings=warnings,
        )

    def query_stream(
        self,
        query: str,
        filters: Optional[dict[str, Any]] = None,
        use_graph: bool = True,
    ) -> Generator[str, None, None]:
        """Stream the answer token by token (for interactive UIs).

        Args:
            query: The user's question.
            filters: Optional metadata filters.
            use_graph: Whether to use knowledge graph augmentation.

        Yields:
            Individual text tokens.
        """
        retrieval_result = self._retriever.retrieve(
            query=query, metadata_filter=filters
        )
        sources = retrieval_result.results

        if not sources:
            yield (
                "I could not find relevant information in the indexed "
                "filings to answer this question."
            )
            return

        graph_context = ""
        if use_graph:
            company = ""
            if filters and "company" in filters:
                company = filters["company"]
            elif sources:
                company = sources[0].metadata.get("company_name", "")
            if company:
                graph_context = self._knowledge_graph.get_graph_context(
                    company, depth=2
                )

        yield from self._generator.generate_stream(
            query=query,
            results=sources,
            system_prompt=self._build_system_prompt(graph_context),
        )

    def _build_system_prompt(self, graph_context: str) -> str:
        """Construct the system prompt, optionally with graph context.

        Args:
            graph_context: Textual context from the knowledge graph.

        Returns:
            The full system prompt string.
        """
        from src.generator import SYSTEM_PROMPT

        if graph_context:
            return (
                f"{SYSTEM_PROMPT}\n\n"
                f"Additional context from the knowledge graph:\n"
                f"{graph_context}"
            )
        return SYSTEM_PROMPT

    # -- Utilities ----------------------------------------------------------

    def get_stats(self) -> dict[str, Any]:
        """Return operational statistics for the pipeline.

        Returns:
            A dict with vector store, knowledge graph, and config stats.
        """
        return {
            "vector_store": self._vector_store.get_collection_stats(),
            "knowledge_graph": {
                "nodes": self._knowledge_graph.num_nodes,
                "edges": self._knowledge_graph.num_edges,
            },
            "config": {
                "llm_provider": self._settings.llm_provider.value,
                "llm_model": self._settings.llm_model_name,
                "embedding_model": self._settings.embedding_model_name,
                "chunking_strategy": self._settings.chunking_strategy.value,
                "retrieval_strategy": self._settings.retrieval_strategy.value,
                "top_k": self._settings.retrieval_top_k,
            },
        }

    def reset(self) -> None:
        """Delete all indexed data and reset the knowledge graph.

        This is destructive and cannot be undone.
        """
        logger.warning("Resetting pipeline: deleting all indexed data.")
        self._vector_store.delete_collection()
        self._knowledge_graph = FinancialKnowledgeGraph()
        self._embedding_pipeline.clear_cache()
        logger.info("Pipeline reset complete.")
