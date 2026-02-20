"""
Knowledge graph augmentation for financial filing analysis.

Extracts entities (companies, financial metrics, dates, monetary amounts)
and relationships (owns, reports, competes_with, subsidiary_of) from filing
text. Builds an in-memory graph using NetworkX and provides graph-enhanced
retrieval that traverses relationships to surface contextually related
information.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Optional

import networkx as nx

from src.document_loader import Document

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Entity types and patterns
# ---------------------------------------------------------------------------


class EntityType:
    """Constants for entity types in the financial knowledge graph."""

    COMPANY = "company"
    METRIC = "financial_metric"
    DATE = "date"
    AMOUNT = "monetary_amount"
    PERSON = "person"
    LOCATION = "location"
    PERCENTAGE = "percentage"
    SECTION = "filing_section"


class RelationType:
    """Constants for relationship types."""

    REPORTS = "reports"
    OWNS = "owns"
    SUBSIDIARY_OF = "subsidiary_of"
    COMPETES_WITH = "competes_with"
    REVENUE_OF = "revenue_of"
    EXPENSE_OF = "expense_of"
    FILED_ON = "filed_on"
    MENTIONED_IN = "mentioned_in"
    YEAR_OVER_YEAR = "year_over_year"
    OPERATES_IN = "operates_in"


# Regex patterns for entity extraction
_COMPANY_PATTERNS = [
    r"(?:Inc\.|Corp\.|Corporation|Company|Ltd\.|LLC|L\.P\.|Group|Holdings)",
]

_MONETARY_RE = re.compile(
    r"\$\s*[\d,]+\.?\d*\s*(?:billion|million|thousand|trillion)?",
    re.IGNORECASE,
)

_PERCENTAGE_RE = re.compile(
    r"\d+\.?\d*\s*(?:%|percent)",
    re.IGNORECASE,
)

_DATE_RE = re.compile(
    r"(?:"
    r"(?:January|February|March|April|May|June|July|August|September|"
    r"October|November|December)\s+\d{1,2},?\s+\d{4}"
    r"|"
    r"\d{4}-\d{2}-\d{2}"
    r"|"
    r"(?:fiscal\s+)?(?:year|quarter)\s+(?:ended?\s+)?\d{4}"
    r"|"
    r"(?:FY|Q[1-4])\s*\d{4}"
    r")",
    re.IGNORECASE,
)

_METRIC_KEYWORDS = {
    "revenue",
    "net income",
    "gross profit",
    "operating income",
    "ebitda",
    "total assets",
    "total liabilities",
    "shareholders equity",
    "cash flow",
    "operating cash flow",
    "free cash flow",
    "earnings per share",
    "eps",
    "gross margin",
    "operating margin",
    "net margin",
    "return on equity",
    "return on assets",
    "debt to equity",
    "current ratio",
    "cost of revenue",
    "research and development",
    "selling general and administrative",
    "depreciation",
    "amortization",
    "capital expenditures",
    "dividends",
}

_METRIC_RE = re.compile(
    r"\b(?:" + "|".join(re.escape(m) for m in _METRIC_KEYWORDS) + r")\b",
    re.IGNORECASE,
)

# Relationship patterns
_SUBSIDIARY_RE = re.compile(
    r"(?:subsidiary|subsidiaries|wholly[- ]owned)\s+(?:of\s+)?([A-Z][\w\s,]+?)(?:\.|,|\band\b)",
    re.IGNORECASE,
)

_COMPETES_RE = re.compile(
    r"(?:compet(?:e|es|ing|itor|itors|ition)\s+(?:with|from|include|including))\s+([A-Z][\w\s,]+?)(?:\.|,|\band\b)",
    re.IGNORECASE,
)

_OPERATES_RE = re.compile(
    r"(?:operat(?:e|es|ing)\s+in)\s+([A-Z][\w\s,]+?)(?:\.|,|\band\b)",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class Entity:
    """A named entity extracted from financial text.

    Attributes:
        name: The canonical entity name.
        entity_type: Type from ``EntityType``.
        properties: Additional properties (e.g., value, currency, section).
        source_document: Source document identifier.
    """

    name: str
    entity_type: str
    properties: dict[str, Any] = field(default_factory=dict)
    source_document: str = ""

    @property
    def node_id(self) -> str:
        """Unique node identifier for the graph."""
        return f"{self.entity_type}::{self.name}"


@dataclass
class Relationship:
    """A directed relationship between two entities.

    Attributes:
        source: Source entity node ID.
        target: Target entity node ID.
        relation_type: Relationship type from ``RelationType``.
        properties: Edge properties (e.g., confidence, source text).
    """

    source: str
    target: str
    relation_type: str
    properties: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Entity extractor
# ---------------------------------------------------------------------------


class FinancialEntityExtractor:
    """Extract entities from financial filing text using regex and heuristics.

    For production use, this could be augmented with a spaCy NER model
    fine-tuned on financial text. The regex-based approach here provides
    a solid baseline without requiring model downloads.
    """

    def extract_entities(
        self, text: str, source_id: str = ""
    ) -> list[Entity]:
        """Extract all recognised entities from text.

        Args:
            text: Input text (typically a chunk or section of a filing).
            source_id: Identifier for the source document.

        Returns:
            A list of ``Entity`` objects.
        """
        entities: list[Entity] = []

        # Monetary amounts
        for match in _MONETARY_RE.finditer(text):
            entities.append(
                Entity(
                    name=match.group().strip(),
                    entity_type=EntityType.AMOUNT,
                    properties={"raw": match.group().strip()},
                    source_document=source_id,
                )
            )

        # Percentages
        for match in _PERCENTAGE_RE.finditer(text):
            entities.append(
                Entity(
                    name=match.group().strip(),
                    entity_type=EntityType.PERCENTAGE,
                    properties={"raw": match.group().strip()},
                    source_document=source_id,
                )
            )

        # Dates
        for match in _DATE_RE.finditer(text):
            entities.append(
                Entity(
                    name=match.group().strip(),
                    entity_type=EntityType.DATE,
                    properties={"raw": match.group().strip()},
                    source_document=source_id,
                )
            )

        # Financial metrics
        seen_metrics: set[str] = set()
        for match in _METRIC_RE.finditer(text):
            metric_name = match.group().strip().lower()
            if metric_name not in seen_metrics:
                seen_metrics.add(metric_name)
                entities.append(
                    Entity(
                        name=metric_name,
                        entity_type=EntityType.METRIC,
                        source_document=source_id,
                    )
                )

        # Companies (entities ending with Corp., Inc., etc.)
        for pattern in _COMPANY_PATTERNS:
            for match in re.finditer(
                r"([A-Z][\w\s&.-]+\s+" + pattern + r")", text
            ):
                company_name = match.group().strip()
                if len(company_name) > 3:
                    entities.append(
                        Entity(
                            name=company_name,
                            entity_type=EntityType.COMPANY,
                            source_document=source_id,
                        )
                    )

        logger.debug(
            "Extracted %d entities from source '%s'", len(entities), source_id
        )
        return entities

    def extract_relationships(
        self,
        text: str,
        company_name: str = "",
        source_id: str = "",
    ) -> list[Relationship]:
        """Extract relationships between entities in the text.

        Args:
            text: Input text.
            company_name: The primary company of the filing (used as a default
                source node for relationships).
            source_id: Identifier for provenance tracking.

        Returns:
            A list of ``Relationship`` objects.
        """
        relationships: list[Relationship] = []
        company_node = f"{EntityType.COMPANY}::{company_name}" if company_name else ""

        # Subsidiary relationships
        for match in _SUBSIDIARY_RE.finditer(text):
            sub_name = match.group(1).strip().rstrip(",. ")
            if company_node and len(sub_name) > 2:
                relationships.append(
                    Relationship(
                        source=f"{EntityType.COMPANY}::{sub_name}",
                        target=company_node,
                        relation_type=RelationType.SUBSIDIARY_OF,
                        properties={"source_text": match.group()[:200]},
                    )
                )

        # Competitor relationships
        for match in _COMPETES_RE.finditer(text):
            competitor_text = match.group(1).strip().rstrip(",. ")
            # Split on commas to get individual competitors
            for competitor in re.split(r",\s*", competitor_text):
                competitor = competitor.strip()
                if company_node and len(competitor) > 2:
                    relationships.append(
                        Relationship(
                            source=company_node,
                            target=f"{EntityType.COMPANY}::{competitor}",
                            relation_type=RelationType.COMPETES_WITH,
                            properties={"source_text": match.group()[:200]},
                        )
                    )

        # Operating regions
        for match in _OPERATES_RE.finditer(text):
            region = match.group(1).strip().rstrip(",. ")
            if company_node and len(region) > 2:
                relationships.append(
                    Relationship(
                        source=company_node,
                        target=f"{EntityType.LOCATION}::{region}",
                        relation_type=RelationType.OPERATES_IN,
                        properties={"source_text": match.group()[:200]},
                    )
                )

        # Link financial metrics and amounts to the company
        entities = self.extract_entities(text, source_id)
        for entity in entities:
            if entity.entity_type == EntityType.METRIC and company_node:
                relationships.append(
                    Relationship(
                        source=company_node,
                        target=entity.node_id,
                        relation_type=RelationType.REPORTS,
                        properties={"source_document": source_id},
                    )
                )

        logger.debug(
            "Extracted %d relationships from source '%s'",
            len(relationships),
            source_id,
        )
        return relationships


# ---------------------------------------------------------------------------
# Knowledge graph
# ---------------------------------------------------------------------------


class FinancialKnowledgeGraph:
    """In-memory knowledge graph for financial filing entities and relationships.

    Built on NetworkX, the graph stores entities as nodes and relationships
    as directed edges. Supports traversal-based context enhancement for RAG
    retrieval.
    """

    def __init__(self) -> None:
        self._graph = nx.DiGraph()
        self._extractor = FinancialEntityExtractor()

    @property
    def num_nodes(self) -> int:
        return self._graph.number_of_nodes()

    @property
    def num_edges(self) -> int:
        return self._graph.number_of_edges()

    def add_entity(self, entity: Entity) -> None:
        """Add an entity as a node in the graph.

        Args:
            entity: The entity to add.
        """
        self._graph.add_node(
            entity.node_id,
            name=entity.name,
            entity_type=entity.entity_type,
            source_document=entity.source_document,
            **entity.properties,
        )

    def add_relationship(self, relationship: Relationship) -> None:
        """Add a relationship as a directed edge.

        Args:
            relationship: The relationship to add.
        """
        # Ensure both nodes exist
        if not self._graph.has_node(relationship.source):
            self._graph.add_node(relationship.source)
        if not self._graph.has_node(relationship.target):
            self._graph.add_node(relationship.target)

        self._graph.add_edge(
            relationship.source,
            relationship.target,
            relation_type=relationship.relation_type,
            **relationship.properties,
        )

    def build_from_documents(
        self,
        documents: list[Document],
        company_name: str = "",
    ) -> None:
        """Build the knowledge graph from a list of filing documents.

        Extracts entities and relationships from each document and populates
        the graph.

        Args:
            documents: Filing section documents.
            company_name: Primary company name for relationship anchoring.
        """
        for doc in documents:
            source_id = doc.metadata.source_file or "unknown"
            doc_company = company_name or doc.metadata.company_name

            # Extract and add entities
            entities = self._extractor.extract_entities(doc.text, source_id)
            for entity in entities:
                self.add_entity(entity)

            # Extract and add relationships
            relationships = self._extractor.extract_relationships(
                doc.text, company_name=doc_company, source_id=source_id
            )
            for rel in relationships:
                self.add_relationship(rel)

            # Add section node and link to company
            if doc.metadata.section:
                section_entity = Entity(
                    name=doc.metadata.section,
                    entity_type=EntityType.SECTION,
                    source_document=source_id,
                )
                self.add_entity(section_entity)
                if doc_company:
                    self.add_relationship(
                        Relationship(
                            source=f"{EntityType.COMPANY}::{doc_company}",
                            target=section_entity.node_id,
                            relation_type=RelationType.MENTIONED_IN,
                        )
                    )

        logger.info(
            "Knowledge graph built: %d nodes, %d edges from %d documents",
            self.num_nodes,
            self.num_edges,
            len(documents),
        )

    def get_related_entities(
        self,
        entity_name: str,
        entity_type: str = EntityType.COMPANY,
        relationship: Optional[str] = None,
        depth: int = 1,
    ) -> list[dict[str, Any]]:
        """Find entities related to a given entity.

        Args:
            entity_name: Name of the starting entity.
            entity_type: Type of the starting entity.
            relationship: Optional filter for a specific relationship type.
            depth: Maximum traversal depth (number of hops).

        Returns:
            A list of dicts with keys: node_id, name, entity_type, relation,
            distance.
        """
        node_id = f"{entity_type}::{entity_name}"
        if not self._graph.has_node(node_id):
            logger.warning("Entity '%s' not found in graph", node_id)
            return []

        related: list[dict[str, Any]] = []
        visited: set[str] = {node_id}
        frontier: list[tuple[str, int]] = [(node_id, 0)]

        while frontier:
            current, dist = frontier.pop(0)
            if dist >= depth:
                continue

            # Outgoing edges
            for _, neighbor, edge_data in self._graph.out_edges(
                current, data=True
            ):
                if neighbor in visited:
                    continue
                rel_type = edge_data.get("relation_type", "")
                if relationship and rel_type != relationship:
                    continue
                visited.add(neighbor)
                node_data = self._graph.nodes[neighbor]
                related.append(
                    {
                        "node_id": neighbor,
                        "name": node_data.get("name", neighbor),
                        "entity_type": node_data.get("entity_type", ""),
                        "relation": rel_type,
                        "distance": dist + 1,
                    }
                )
                frontier.append((neighbor, dist + 1))

            # Incoming edges
            for predecessor, _, edge_data in self._graph.in_edges(
                current, data=True
            ):
                if predecessor in visited:
                    continue
                rel_type = edge_data.get("relation_type", "")
                if relationship and rel_type != relationship:
                    continue
                visited.add(predecessor)
                node_data = self._graph.nodes[predecessor]
                related.append(
                    {
                        "node_id": predecessor,
                        "name": node_data.get("name", predecessor),
                        "entity_type": node_data.get("entity_type", ""),
                        "relation": rel_type,
                        "distance": dist + 1,
                    }
                )
                frontier.append((predecessor, dist + 1))

        return related

    def get_graph_context(
        self,
        entity_name: str,
        entity_type: str = EntityType.COMPANY,
        depth: int = 2,
        max_context_items: int = 20,
    ) -> str:
        """Generate a textual context summary from the graph for RAG enhancement.

        Traverses the graph around the given entity and produces a structured
        text summary of discovered relationships.

        Args:
            entity_name: Starting entity name.
            entity_type: Starting entity type.
            depth: Traversal depth.
            max_context_items: Maximum relationships to include.

        Returns:
            A formatted string summarizing graph relationships, suitable for
            inclusion in the LLM prompt.
        """
        related = self.get_related_entities(
            entity_name, entity_type, depth=depth
        )

        if not related:
            return ""

        lines = [f"Knowledge Graph Context for {entity_name}:"]
        for item in related[:max_context_items]:
            lines.append(
                f"  - [{item['relation']}] {item['name']} "
                f"(type: {item['entity_type']}, hops: {item['distance']})"
            )

        context = "\n".join(lines)
        logger.debug(
            "Generated graph context: %d relationships for '%s'",
            len(related[:max_context_items]),
            entity_name,
        )
        return context

    def get_entity_summary(self, entity_name: str, entity_type: str = EntityType.COMPANY) -> dict[str, Any]:
        """Get a structured summary of an entity and its connections.

        Args:
            entity_name: Entity name.
            entity_type: Entity type.

        Returns:
            A dict with entity properties and categorised relationships.
        """
        node_id = f"{entity_type}::{entity_name}"
        if not self._graph.has_node(node_id):
            return {"error": f"Entity '{entity_name}' not found"}

        node_data = dict(self._graph.nodes[node_id])
        outgoing: dict[str, list[str]] = {}
        incoming: dict[str, list[str]] = {}

        for _, target, data in self._graph.out_edges(node_id, data=True):
            rel = data.get("relation_type", "related_to")
            target_name = self._graph.nodes[target].get("name", target)
            outgoing.setdefault(rel, []).append(target_name)

        for source, _, data in self._graph.in_edges(node_id, data=True):
            rel = data.get("relation_type", "related_to")
            source_name = self._graph.nodes[source].get("name", source)
            incoming.setdefault(rel, []).append(source_name)

        return {
            "entity": entity_name,
            "type": entity_type,
            "properties": node_data,
            "outgoing_relationships": outgoing,
            "incoming_relationships": incoming,
            "degree": self._graph.degree(node_id),
        }

    def to_serializable(self) -> dict[str, Any]:
        """Serialize the graph to a JSON-compatible dict.

        Returns:
            A dict with 'nodes' and 'edges' lists suitable for visualization.
        """
        nodes = []
        for node_id, data in self._graph.nodes(data=True):
            nodes.append({"id": node_id, **data})

        edges = []
        for source, target, data in self._graph.edges(data=True):
            edges.append({"source": source, "target": target, **data})

        return {"nodes": nodes, "edges": edges}
