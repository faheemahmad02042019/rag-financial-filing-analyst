"""
Streamlit demo UI for the RAG Financial Filing Analyst.

Provides an interactive interface for:
- Uploading and ingesting 10-K filings.
- Querying the RAG pipeline with natural language questions.
- Viewing generated answers with source citations.
- Exploring the knowledge graph as an interactive visualization.
- Inspecting retrieval metrics and guardrail reports.
"""

from __future__ import annotations

import json
import logging
import sys
import tempfile
from pathlib import Path

import streamlit as st

# Ensure the project root is on the path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.config import Settings
from src.pipeline import RAGPipeline

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="RAG Financial Filing Analyst",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Session state initialization
# ---------------------------------------------------------------------------


def init_session_state() -> None:
    """Initialise Streamlit session state variables."""
    if "pipeline" not in st.session_state:
        st.session_state.pipeline = None
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "documents_ingested" not in st.session_state:
        st.session_state.documents_ingested = 0


init_session_state()


# ---------------------------------------------------------------------------
# Pipeline initialization
# ---------------------------------------------------------------------------


@st.cache_resource
def get_pipeline() -> RAGPipeline:
    """Create and cache the RAG pipeline (survives reruns)."""
    settings = Settings()
    api_warnings = settings.validate_api_keys()
    if api_warnings:
        for warning in api_warnings:
            st.warning(warning)
    return RAGPipeline(settings)


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------


def render_sidebar() -> None:
    """Render the sidebar with configuration and file upload."""
    st.sidebar.title("Configuration")

    # Pipeline status
    pipeline = st.session_state.pipeline
    if pipeline is not None:
        stats = pipeline.get_stats()
        st.sidebar.success("Pipeline active")
        st.sidebar.metric(
            "Indexed Chunks",
            stats["vector_store"]["document_count"],
        )
        st.sidebar.metric(
            "Knowledge Graph Nodes",
            stats["knowledge_graph"]["nodes"],
        )
        st.sidebar.metric(
            "Knowledge Graph Edges",
            stats["knowledge_graph"]["edges"],
        )
    else:
        st.sidebar.info("Pipeline not initialized. Click 'Initialize Pipeline' below.")

    st.sidebar.divider()

    # Initialize pipeline button
    if st.sidebar.button("Initialize Pipeline", type="primary"):
        with st.spinner("Initializing RAG pipeline..."):
            try:
                st.session_state.pipeline = get_pipeline()
                st.sidebar.success("Pipeline initialized successfully.")
                st.rerun()
            except Exception as exc:
                st.sidebar.error(f"Failed to initialize pipeline: {exc}")

    st.sidebar.divider()

    # File upload
    st.sidebar.subheader("Upload 10-K Filing")
    uploaded_file = st.sidebar.file_uploader(
        "Upload a plain-text 10-K filing",
        type=["txt", "html", "htm"],
        help="Upload the full text of a 10-K filing.",
    )

    company_name = st.sidebar.text_input(
        "Company Name",
        placeholder="e.g., Apple Inc.",
    )
    filing_date = st.sidebar.text_input(
        "Filing Date",
        placeholder="e.g., 2024-10-31",
    )

    if uploaded_file is not None and st.sidebar.button("Ingest Filing"):
        if st.session_state.pipeline is None:
            st.sidebar.error("Please initialize the pipeline first.")
        else:
            with st.spinner("Ingesting filing..."):
                # Write to a temp file
                with tempfile.NamedTemporaryFile(
                    mode="w",
                    suffix=".txt",
                    delete=False,
                    encoding="utf-8",
                ) as tmp:
                    content = uploaded_file.read().decode("utf-8", errors="replace")
                    tmp.write(content)
                    tmp_path = tmp.name

                try:
                    num_chunks = st.session_state.pipeline.ingest_document(
                        tmp_path,
                        company_name=company_name,
                        filing_date=filing_date,
                    )
                    st.session_state.documents_ingested += 1
                    st.sidebar.success(
                        f"Ingested {num_chunks} chunks from "
                        f"{uploaded_file.name}"
                    )
                    st.rerun()
                except Exception as exc:
                    st.sidebar.error(f"Ingestion failed: {exc}")

    st.sidebar.divider()

    # Retrieval settings
    st.sidebar.subheader("Retrieval Settings")
    st.sidebar.slider(
        "Top K",
        min_value=1,
        max_value=20,
        value=5,
        key="top_k",
        help="Number of chunks to retrieve.",
    )
    st.sidebar.selectbox(
        "Retrieval Strategy",
        options=["ensemble", "dense", "sparse", "mmr"],
        key="retrieval_strategy",
    )


# ---------------------------------------------------------------------------
# Main content area
# ---------------------------------------------------------------------------


def render_query_interface() -> None:
    """Render the main query interface with chat history."""
    st.title("RAG Financial Filing Analyst")
    st.markdown(
        "Ask questions about SEC 10-K filings. Upload a filing in the "
        "sidebar to get started."
    )

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "sources" in message:
                render_sources(message["sources"])
            if "metrics" in message:
                render_metrics(message["metrics"])

    # Query input
    query = st.chat_input("Ask a question about the filing...")

    if query:
        # Display user message
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)

        # Generate response
        if st.session_state.pipeline is None:
            with st.chat_message("assistant"):
                st.warning(
                    "Please initialize the pipeline and ingest a filing first."
                )
            return

        with st.chat_message("assistant"):
            with st.spinner("Searching and generating answer..."):
                try:
                    result = st.session_state.pipeline.query(
                        query=query,
                        validate=True,
                    )

                    # Display answer
                    st.markdown(result.answer)

                    # Confidence badge
                    confidence = result.confidence_score
                    if confidence >= 0.8:
                        st.success(f"Confidence: {confidence:.0%}")
                    elif confidence >= 0.6:
                        st.warning(f"Confidence: {confidence:.0%}")
                    else:
                        st.error(f"Confidence: {confidence:.0%}")

                    # Warnings
                    for warning in result.warnings:
                        st.warning(warning)

                    # Sources
                    source_data = []
                    if result.sources:
                        render_sources(result.sources)
                        source_data = [
                            {
                                "section": s.metadata.get("section", ""),
                                "score": f"{s.score:.3f}",
                                "preview": s.text[:200],
                            }
                            for s in result.sources
                        ]

                    # Metrics
                    metrics = {
                        "confidence": f"{confidence:.2f}",
                        "sources_retrieved": len(result.sources),
                        "latency_ms": f"{result.latency_ms:.0f}",
                        "strategy": (
                            result.retrieval_result.strategy
                            if result.retrieval_result
                            else "N/A"
                        ),
                    }
                    render_metrics(metrics)

                    # Store in chat history
                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": result.answer,
                            "sources": source_data,
                            "metrics": metrics,
                        }
                    )

                except Exception as exc:
                    error_msg = f"Query failed: {exc}"
                    st.error(error_msg)
                    logger.error(error_msg, exc_info=True)


def render_sources(sources: list) -> None:
    """Render retrieved source chunks in an expandable section."""
    with st.expander(f"Sources ({len(sources)} chunks)", expanded=False):
        for idx, source in enumerate(sources):
            if isinstance(source, dict):
                st.markdown(f"**Source {idx + 1}** | Section: {source.get('section', 'N/A')} | Score: {source.get('score', 'N/A')}")
                st.text(source.get("preview", ""))
            else:
                st.markdown(
                    f"**Source {idx + 1}** | "
                    f"Section: {source.metadata.get('section', 'N/A')} | "
                    f"Score: {source.score:.3f}"
                )
                st.text(source.text[:300])
            st.divider()


def render_metrics(metrics: dict) -> None:
    """Render retrieval and generation metrics."""
    with st.expander("Metrics", expanded=False):
        cols = st.columns(len(metrics))
        for col, (key, value) in zip(cols, metrics.items()):
            col.metric(key.replace("_", " ").title(), value)


# ---------------------------------------------------------------------------
# Knowledge graph tab
# ---------------------------------------------------------------------------


def render_knowledge_graph_tab() -> None:
    """Render the knowledge graph exploration tab."""
    st.header("Knowledge Graph Explorer")

    if st.session_state.pipeline is None:
        st.info("Initialize the pipeline and ingest a filing to explore the knowledge graph.")
        return

    kg = st.session_state.pipeline._knowledge_graph

    if kg.num_nodes == 0:
        st.info("No entities in the knowledge graph yet. Ingest a filing first.")
        return

    st.metric("Nodes", kg.num_nodes)
    st.metric("Edges", kg.num_edges)

    # Entity search
    entity_name = st.text_input(
        "Search for an entity",
        placeholder="e.g., Apple Inc.",
    )

    if entity_name:
        summary = kg.get_entity_summary(entity_name)
        if "error" in summary:
            st.warning(summary["error"])
        else:
            st.json(summary)

    # Graph visualization using Plotly
    st.subheader("Graph Visualization")
    graph_data = kg.to_serializable()

    if graph_data["nodes"]:
        try:
            import plotly.graph_objects as go

            # Build node positions using a simple spring layout
            import networkx as nx

            G = kg._graph
            if G.number_of_nodes() > 0:
                pos = nx.spring_layout(G, k=2, iterations=50, seed=42)

                # Create edge traces
                edge_x, edge_y = [], []
                for edge in G.edges():
                    x0, y0 = pos[edge[0]]
                    x1, y1 = pos[edge[1]]
                    edge_x.extend([x0, x1, None])
                    edge_y.extend([y0, y1, None])

                edge_trace = go.Scatter(
                    x=edge_x,
                    y=edge_y,
                    line=dict(width=0.5, color="#888"),
                    hoverinfo="none",
                    mode="lines",
                )

                # Create node traces
                node_x = [pos[node][0] for node in G.nodes()]
                node_y = [pos[node][1] for node in G.nodes()]
                node_text = [
                    G.nodes[node].get("name", node)
                    for node in G.nodes()
                ]

                node_trace = go.Scatter(
                    x=node_x,
                    y=node_y,
                    mode="markers+text",
                    hoverinfo="text",
                    text=node_text,
                    textposition="top center",
                    textfont=dict(size=8),
                    marker=dict(
                        size=10,
                        color=[
                            G.degree(node) for node in G.nodes()
                        ],
                        colorscale="YlOrRd",
                        showscale=True,
                        colorbar=dict(title="Connections"),
                    ),
                )

                fig = go.Figure(
                    data=[edge_trace, node_trace],
                    layout=go.Layout(
                        title="Financial Knowledge Graph",
                        showlegend=False,
                        hovermode="closest",
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        height=600,
                    ),
                )
                st.plotly_chart(fig, use_container_width=True)

        except ImportError:
            st.warning("Plotly is required for graph visualization.")
            st.json(graph_data)


# ---------------------------------------------------------------------------
# Main layout
# ---------------------------------------------------------------------------


def main() -> None:
    """Main entry point for the Streamlit app."""
    render_sidebar()

    tab1, tab2 = st.tabs(["Query Interface", "Knowledge Graph"])

    with tab1:
        render_query_interface()

    with tab2:
        render_knowledge_graph_tab()


if __name__ == "__main__":
    main()
