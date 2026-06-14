"""
Core data model for the Data Lineage & Discovery Agent.

LineageCatalogue is the single source of truth written to data/catalogue.json
and read by every other component (graph builder, discovery agent, UI, eval).
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, field_validator


# ---------------------------------------------------------------------------
# Confidence
# ---------------------------------------------------------------------------

HIGH_CONFIDENCE = 0.85
MEDIUM_CONFIDENCE = 0.60


def confidence_tier(score: float) -> str:
    """Compute display tier from a raw confidence score."""
    if score >= HIGH_CONFIDENCE:
        return "high"
    if score >= MEDIUM_CONFIDENCE:
        return "medium"
    return "low"


# ---------------------------------------------------------------------------
# Evidence (attached to RelationshipEdge)
# ---------------------------------------------------------------------------

class EvidenceType(str, Enum):
    name_match    = "name_match"     # column names identical or near-identical
    value_overlap = "value_overlap"  # sampled values from both sides overlap
    semantic      = "semantic"       # Claude judged the columns semantically related


class Evidence(BaseModel):
    type: EvidenceType
    score: float = Field(ge=0.0, le=1.0)
    detail: str  # e.g. "91% of 200 sampled values in order_id match orders.order_id"


# ---------------------------------------------------------------------------
# Column
# ---------------------------------------------------------------------------

class ColumnAnnotation(BaseModel):
    column_name: str
    sql_type: str                        # raw type from information_schema
    sample_values: list[str] = Field(default_factory=list)
    unique_count: Optional[int] = None   # distinct non-null values in profiled rows; used for cardinality inference
    inferred_description: str
    confidence: float = Field(ge=0.0, le=1.0)
    reviewed: bool = False
    human_description: Optional[str] = None  # set when a human overrides the agent

    @property
    def effective_description(self) -> str:
        return self.human_description if self.human_description is not None else self.inferred_description

    @property
    def confidence_tier(self) -> str:
        return confidence_tier(self.confidence)


# ---------------------------------------------------------------------------
# Table
# ---------------------------------------------------------------------------

class TableAnnotation(BaseModel):
    table_name: str
    row_count: int
    inferred_description: str
    confidence: float = Field(ge=0.0, le=1.0)
    reviewed: bool = False
    human_description: Optional[str] = None
    columns: dict[str, ColumnAnnotation] = Field(default_factory=dict)  # keyed by column_name

    @property
    def effective_description(self) -> str:
        return self.human_description if self.human_description is not None else self.inferred_description

    @property
    def confidence_tier(self) -> str:
        return confidence_tier(self.confidence)


# ---------------------------------------------------------------------------
# Relationship edge
# ---------------------------------------------------------------------------

class Cardinality(str, Enum):
    many_to_one  = "many-to-one"
    one_to_one   = "one-to-one"
    many_to_many = "many-to-many"


class RelationshipEdge(BaseModel):
    from_table: str
    from_column: str
    to_table: str
    to_column: str          # always explicit; equals from_column when column names match
    cardinality: Cardinality
    confidence: float = Field(ge=0.0, le=1.0)
    evidence: list[Evidence] = Field(default_factory=list)
    reviewed: bool = False
    confirmed: Optional[bool] = None  # None=unreviewed  True=confirmed  False=rejected

    @property
    def confidence_tier(self) -> str:
        return confidence_tier(self.confidence)

    def to_eval_edge(self) -> dict:
        """Return the dict format expected by eval/scorer.py."""
        return {
            "from_table": self.from_table,
            "from_column": self.from_column,
            "to_table": self.to_table,
            "to_column": self.to_column,
        }


# ---------------------------------------------------------------------------
# Top-level catalogue
# ---------------------------------------------------------------------------

class LineageCatalogue(BaseModel):
    schema_version: str = "1.0"
    tables: dict[str, TableAnnotation] = Field(default_factory=dict)   # keyed by table_name
    edges: list[RelationshipEdge] = Field(default_factory=list)

    # ------------------------------------------------------------------
    # Convenience accessors used by the graph builder and discovery agent
    # ------------------------------------------------------------------

    def unreviewed_edges(self) -> list[RelationshipEdge]:
        return [e for e in self.edges if not e.reviewed]

    def confirmed_edges(self) -> list[RelationshipEdge]:
        return [e for e in self.edges if e.confirmed is True]

    def low_confidence_columns(self) -> list[tuple[str, ColumnAnnotation]]:
        """Return (table_name, col) pairs below the medium confidence threshold."""
        out = []
        for tname, table in self.tables.items():
            for col in table.columns.values():
                if col.confidence < MEDIUM_CONFIDENCE:
                    out.append((tname, col))
        return out

    def to_eval_answer(self, question_id: str, tables: list[str]) -> dict:
        """
        Produce an agent-answer dict for eval/scorer.py from the catalogue's edges,
        filtered to only those connecting the given tables.
        """
        table_set = set(tables)
        relevant_edges = [
            e.to_eval_edge()
            for e in self.edges
            if e.from_table in table_set and e.to_table in table_set
        ]
        return {
            "question_id": question_id,
            "tables": tables,
            "join_path": relevant_edges,
        }
