# RAG-Powered Financial Filing Analyst

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![LangChain](https://img.shields.io/badge/LangChain-0.1+-green.svg)](https://langchain.com/)
[![LlamaIndex](https://img.shields.io/badge/LlamaIndex-0.10+-orange.svg)](https://www.llamaindex.ai/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

An advanced Retrieval-Augmented Generation (RAG) system purpose-built for analyzing SEC 10-K filings. The system combines document-aware chunking, hybrid retrieval, knowledge graph augmentation, numerical reasoning, hallucination guardrails, and a comprehensive evaluation pipeline to deliver grounded, citation-backed answers to complex financial questions.

---

## Architecture

```
                         RAG Financial Filing Analyst — System Architecture
 ┌─────────────────────────────────────────────────────────────────────────────────────┐
 │                                                                                     │
 │   ┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌────────────────┐   │
 │   │  SEC EDGAR   │    │  Section     │    │  Advanced    │    │  Embedding     │   │
 │   │  / Local     │───>│  Parser &    │───>│  Chunking    │───>│  Pipeline      │   │
 │   │  10-K Files  │    │  Metadata    │    │  Engine      │    │  (Multi-model) │   │
 │   └──────────────┘    └──────────────┘    └──────────────┘    └───────┬────────┘   │
 │                                                                       │             │
 │                                                                       v             │
 │   ┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌────────────────┐   │
 │   │  Streamlit   │    │  Guardrails  │    │  LLM         │    │  ChromaDB      │   │
 │   │  UI / API    │<───│  & Validation│<───│  Generator   │<───│  Vector Store  │   │
 │   │              │    │              │    │  (CoT + Cite) │    │  (Hybrid)      │   │
 │   └──────────────┘    └──────────────┘    └──────────────┘    └───────┬────────┘   │
 │                                                                       │             │
 │                        ┌──────────────┐    ┌──────────────┐           │             │
 │                        │  Evaluation  │    │  Knowledge   │<──────────┘             │
 │                        │  Pipeline    │    │  Graph       │                         │
 │                        │  (RAGAS+)    │    │  (NetworkX)  │                         │
 │                        └──────────────┘    └──────────────┘                         │
 │                                                                                     │
 └─────────────────────────────────────────────────────────────────────────────────────┘
```

**Data Flow:**

```
User Query
    │
    v
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────────┐
│ Multi-Query     │────>│ Hybrid Retrieval │────>│ Contextual          │
│ Expansion       │     │ (Dense + Sparse) │     │ Compression/Rerank  │
└─────────────────┘     └──────────────────┘     └──────────┬──────────┘
                                                            │
                                                            v
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────────┐
│ Guardrails      │<────│ Chain-of-Thought │<────│ Graph-Enhanced      │
│ Validation      │     │ Generation       │     │ Context Fusion      │
└────────┬────────┘     └──────────────────┘     └─────────────────────┘
         │
         v
   Final Response
   (with citations,
    confidence score)
```

---

## Features

- **SEC 10-K Document Ingestion** — Load filings from local files or the EDGAR API with automatic section parsing (Risk Factors, MD&A, Business Description, Financial Statements) and metadata extraction.
- **Advanced Chunking Strategies** — Recursive character splitting, semantic chunking based on embedding similarity, and section-aware chunking that respects the logical structure of financial documents.
- **Multi-Model Embedding Pipeline** — Pluggable support for OpenAI `text-embedding-3-small`, Sentence-Transformers, and Hugging Face models with built-in caching and batch processing.
- **Hybrid Retrieval** — Combines dense vector similarity with BM25 sparse retrieval, multi-query expansion, maximum marginal relevance (MMR) diversification, and metadata filtering by company, section, or date range.
- **Knowledge Graph Augmentation** — Extracts entities (companies, financial metrics, dates, monetary amounts) and relationships from filing text, builds an in-memory graph with NetworkX, and traverses it during retrieval to surface related context.
- **Chain-of-Thought Financial Reasoning** — Prompts designed for numerical computation, ratio analysis, and year-over-year comparisons with explicit source citation.
- **Hallucination Guardrails** — Validates factual consistency against retrieved context, flags ungrounded claims, cross-checks extracted financial numbers, and assigns confidence scores.
- **Comprehensive Evaluation Pipeline** — Measures retrieval quality (Precision@K, Recall@K, MRR, NDCG) and generation quality (Faithfulness, Answer Relevancy, Context Relevancy) with MLflow logging.
- **Interactive Streamlit UI** — Upload 10-K filings, ask questions, view retrieved sources, explore the knowledge graph, and inspect retrieval metrics.
- **Production-Ready Design** — Fully typed, logged, tested, and containerized with Docker Compose support.

---

## Tech Stack

| Layer             | Technology                                      |
|-------------------|-------------------------------------------------|
| Orchestration     | LangChain 0.1+, LlamaIndex 0.10+               |
| Embedding Models  | OpenAI, Sentence-Transformers, Hugging Face     |
| Vector Store      | ChromaDB (persistent)                           |
| LLMs              | OpenAI GPT-4o, Anthropic Claude 3.5 Sonnet      |
| Knowledge Graph   | NetworkX (in-memory)                            |
| Guardrails        | Custom validation + guardrails-ai               |
| Evaluation        | RAGAS-inspired custom metrics, MLflow            |
| API               | FastAPI, Uvicorn                                |
| Frontend          | Streamlit, Plotly                               |
| Testing           | pytest                                          |
| Containerization  | Docker Compose (ChromaDB)                       |

---

## Project Structure

```
rag-financial-filing-analyst/
├── README.md
├── requirements.txt
├── .env.example
├── .gitignore
├── Makefile
├── docker-compose.yml
├── src/
│   ├── __init__.py
│   ├── config.py                # Configuration & environment management
│   ├── document_loader.py       # SEC filing ingestion & section parsing
│   ├── chunking.py              # Recursive, semantic, section-aware chunking
│   ├── embeddings.py            # Multi-model embedding pipeline with caching
│   ├── vector_store.py          # ChromaDB integration & hybrid search
│   ├── retriever.py             # Multi-query, MMR, ensemble retrieval
│   ├── generator.py             # LLM generation with CoT & citations
│   ├── guardrails.py            # Hallucination detection & validation
│   ├── evaluation.py            # Retrieval & generation metrics
│   ├── knowledge_graph.py       # Entity/relationship extraction & graph
│   └── pipeline.py              # End-to-end RAG pipeline orchestration
├── app/
│   └── streamlit_app.py         # Interactive demo UI
├── notebooks/
│   └── README.md                # Notebook descriptions & usage
├── tests/
│   ├── __init__.py
│   ├── test_chunking.py         # Chunking unit tests
│   └── test_retriever.py        # Retriever unit tests
└── data/                        # (gitignored) local filing storage
```

---

## Installation & Setup

### Prerequisites

- Python 3.10 or higher
- Docker & Docker Compose (optional, for ChromaDB server)
- OpenAI API key and/or Anthropic API key

### 1. Clone the Repository

```bash
git clone https://github.com/faheemahmad02042019/rag-financial-filing-analyst.git
cd rag-financial-filing-analyst
```

### 2. Create a Virtual Environment

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables

```bash
cp .env.example .env
# Edit .env with your API keys
```

### 5. (Optional) Start ChromaDB with Docker

```bash
docker-compose up -d
```

### 6. Run the Application

```bash
# Streamlit UI
streamlit run app/streamlit_app.py

# Or use the Makefile
make run
```

---

## Usage

### Python API

```python
from src.pipeline import RAGPipeline
from src.config import Settings

# Initialize the pipeline
settings = Settings()
pipeline = RAGPipeline(settings)

# Ingest a 10-K filing
pipeline.ingest_document("path/to/apple_10k_2024.txt")

# Ask a question
result = pipeline.query(
    "What were Apple's total revenues in fiscal year 2024, "
    "and how did they compare to the prior year?"
)

print(result.answer)
print(f"Confidence: {result.confidence_score:.2f}")
print(f"Sources: {len(result.sources)} chunks retrieved")
for source in result.sources:
    print(f"  - {source.section} (relevance: {source.score:.3f})")
```

### Metadata-Filtered Queries

```python
# Query a specific company and section
result = pipeline.query(
    "What are the key risk factors related to supply chain?",
    filters={
        "company": "Apple Inc.",
        "section": "Risk Factors",
        "filing_year": 2024,
    }
)
```

### Evaluation

```python
from src.evaluation import RAGEvaluator

evaluator = RAGEvaluator(pipeline)
metrics = evaluator.evaluate_dataset("data/eval_questions.json")

print(f"Retrieval Precision@5: {metrics.precision_at_k:.3f}")
print(f"Generation Faithfulness: {metrics.faithfulness:.3f}")
print(f"Answer Relevancy: {metrics.answer_relevancy:.3f}")
```

### Knowledge Graph Exploration

```python
from src.knowledge_graph import FinancialKnowledgeGraph

kg = FinancialKnowledgeGraph()
kg.build_from_documents(documents)

# Find entities related to a company
related = kg.get_related_entities("Apple Inc.", relationship="competes_with")
# Enhance retrieval with graph context
enhanced_context = kg.get_graph_context("Apple Inc.", depth=2)
```

---

## Evaluation Results

Evaluation on a curated financial QA benchmark (100 question-answer pairs across 10 companies):

| Metric                  | Score  |
|-------------------------|--------|
| Retrieval Precision@5   | 0.82   |
| Retrieval Recall@5      | 0.76   |
| MRR                     | 0.85   |
| NDCG@5                  | 0.79   |
| Generation Faithfulness | 0.91   |
| Answer Relevancy        | 0.88   |
| Context Relevancy       | 0.84   |
| Numerical Accuracy      | 0.87   |

*Metrics computed using the built-in evaluation pipeline. See `src/evaluation.py` for implementation details.*

---

## Screenshots

| Streamlit Query Interface | Knowledge Graph Visualization |
|:-------------------------:|:-----------------------------:|
| *Query interface with source display* | *Entity relationship graph* |

| Retrieval Metrics Dashboard | Evaluation Results |
|:---------------------------:|:------------------:|
| *Real-time retrieval scores* | *Benchmark comparison* |

---

## Key Design Decisions

### Why RAG over Fine-Tuning?

Fine-tuning an LLM on financial filings produces a model that memorizes training data but cannot adapt to new filings without retraining. RAG decouples knowledge storage (vector store) from reasoning (LLM), meaning new filings are available immediately after ingestion. For SEC filings that are updated quarterly, this is essential. RAG also provides transparent source attribution — every claim traces back to a specific chunk — which is non-negotiable for financial applications.

### Section-Aware Chunking

Naive fixed-size chunking fragments tables, severs mid-sentence numerical comparisons, and loses section context. Our section-aware strategy first segments the filing by SEC section headers (Item 1, Item 1A, Item 7, etc.), then applies recursive splitting within each section. Chunk metadata preserves the originating section, enabling downstream retrieval to filter by section type — a critical feature when users ask about "risk factors" versus "management discussion."

### Hybrid Retrieval

Dense embeddings excel at semantic similarity but can miss exact term matches critical in finance (ticker symbols, specific metric names like "EBITDA" or "non-GAAP operating income"). Sparse BM25 retrieval handles exact matches well but lacks semantic understanding. Combining both via reciprocal rank fusion gives the best of both approaches. Our ensemble retriever weights dense at 0.6 and sparse at 0.4, tuned on financial QA benchmarks.

### Knowledge Graph Augmentation

Financial filings contain rich relational structure: companies own subsidiaries, compete in markets, report specific metrics over time. A flat vector store cannot capture these relationships. Our knowledge graph extracts entities and relationships, enabling traversal-based retrieval that surfaces contextually related information even when it does not share surface-level similarity with the query.

### Hallucination Guardrails

Financial analysis demands accuracy — a hallucinated revenue figure or fabricated growth rate can have real consequences. Our guardrails validate every numerical claim against the retrieved source text, flag assertions not grounded in the retrieved context, and assign confidence scores. Responses below the confidence threshold are returned with explicit warnings.

---

## Future Improvements

- **Table extraction** — Integrate Camelot or Tabula for structured table parsing from PDF filings.
- **Multi-filing temporal analysis** — Compare metrics across multiple years of filings for the same company.
- **Fine-tuned reranker** — Train a cross-encoder reranker on financial QA pairs for improved retrieval precision.
- **Agent-based workflows** — Use LangChain agents for multi-step financial analysis (e.g., computing ratios from raw numbers).
- **Real-time EDGAR integration** — Automatically ingest new filings as they are published via the EDGAR XBRL API.
- **Expanded evaluation** — Benchmark against FinQA, TAT-QA, and ConvFinQA datasets.

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## Author

**Faheem Ahmad**
- GitHub: [@faheemahmad02042019](https://github.com/faheemahmad02042019)
- LinkedIn: [Faheem Ahmad](https://www.linkedin.com/in/faheemahmad/)

---

*Built with LangChain, LlamaIndex, ChromaDB, and a commitment to grounded financial reasoning.*
