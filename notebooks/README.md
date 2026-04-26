# Notebooks

This directory contains Jupyter notebooks that demonstrate the end-to-end workflow of the RAG Financial Filing Analyst.

## Planned Notebooks

| Notebook | Description |
|----------|-------------|
| `01_data_ingestion.ipynb` | Load, parse, and chunk SEC 10-K filings into the vector store. |
| `02_retrieval_exploration.ipynb` | Compare retrieval strategies (dense, sparse, hybrid, MMR) on sample queries. |
| `03_generation_and_guardrails.ipynb` | Demonstrate chain-of-thought generation with hallucination guardrails. |
| `04_knowledge_graph.ipynb` | Build and visualize the financial knowledge graph from extracted entities. |
| `05_evaluation.ipynb` | Run the full evaluation pipeline and visualize metrics. |

## Usage

```bash
# Start Jupyter from the project root
jupyter notebook notebooks/
```

Each notebook is self-contained and includes setup cells that import from the `src/` package.
