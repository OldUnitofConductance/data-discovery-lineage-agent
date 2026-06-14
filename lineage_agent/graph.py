"""
Relationship inference and graph builder — Phase 3.

Given a LineageCatalogue with annotated tables and columns (from Phase 2),
infers RelationshipEdge objects using two algorithmic signals:

  name_match    — Jaccard similarity of underscore-split column name tokens
  value_overlap — Jaccard similarity of sampled column values

Cardinality and FK direction are derived from each column's unique_count
relative to its table's row_count. When unique_count is unavailable the
function falls back to a table-name prefix heuristic.

Builds a NetworkX DiGraph for use by the discovery agent (Phase 4).

CLI usage:
    python -m lineage_agent.graph data/catalogue.json --out data/catalogue.json
"""

from __future__ import annotations

import argparse
import sys
from itertools import combinations
from pathlib import Path

import networkx as nx

from lineage_agent.models import (
    Cardinality,
    ColumnAnnotation,
    Evidence,
    EvidenceType,
    LineageCatalogue,
    RelationshipEdge,
    TableAnnotation,
)

# ---------------------------------------------------------------------------
# Tunable thresholds
# ---------------------------------------------------------------------------

MIN_NAME_SIM = 0.50    # Jaccard token overlap to consider a column pair at all
MIN_CONFIDENCE = 0.60  # combined confidence below which no edge is emitted
HIGH_UNIQUE_RATIO = 0.95  # unique_count / sample_rows above which a column is treated as PK
PROFILE_ROWS = 500     # must match MAX_PROFILE_ROWS in infer.py


# ---------------------------------------------------------------------------
# Signal functions
# ---------------------------------------------------------------------------

def name_similarity(col_a: str, col_b: str) -> float:
    """Jaccard similarity of underscore-split tokens. 1.0 for identical names."""
    if col_a == col_b:
        return 1.0
    tokens_a = set(col_a.lower().split("_"))
    tokens_b = set(col_b.lower().split("_"))
    union = tokens_a | tokens_b
    return len(tokens_a & tokens_b) / len(union) if union else 0.0


def value_overlap(samples_a: list[str], samples_b: list[str]) -> float:
    """Jaccard similarity of two sample-value sets."""
    set_a, set_b = set(samples_a), set(samples_b)
    union = set_a | set_b
    return len(set_a & set_b) / len(union) if union else 0.0


# ---------------------------------------------------------------------------
# Confidence
# ---------------------------------------------------------------------------

def compute_confidence(name_sim: float, val_sim: float) -> float:
    """
    Combine name similarity and value overlap into a confidence score.

    Exact name match sets a high baseline (0.80); value overlap adds up to 0.15.
    Partial name matches use a lower baseline; value overlap matters more.
    Value overlap alone (no name match) never exceeds MIN_CONFIDENCE by design —
    it must accompany at least a partial name match to produce an edge.
    """
    if name_sim == 1.0:
        return round(min(0.95, 0.80 + 0.15 * val_sim), 4)
    else:
        return round(min(0.88, 0.50 + 0.40 * name_sim + 0.15 * val_sim), 4)


# ---------------------------------------------------------------------------
# Cardinality and direction
# ---------------------------------------------------------------------------

def _types_compatible(sql_type_a: str, sql_type_b: str) -> bool:
    """
    Returns False when the two column types are fundamentally incompatible as join keys.
    Numeric ↔ string mismatches are never valid FK candidates.
    """
    def _family(t: str) -> str:
        t = t.lower()
        if t.startswith(("int", "float", "uint")):
            return "numeric"
        return "string"

    return _family(sql_type_a) == _family(sql_type_b)


def _uniqueness_ratio(col: ColumnAnnotation, table: TableAnnotation) -> float | None:
    """
    Fraction of rows that are unique for this column, measured within the profiled
    sample. Normalising by min(row_count, PROFILE_ROWS) rather than full row_count
    gives a meaningful signal because unique_count comes from 500-row profiling.
    """
    if col.unique_count is None or not table.row_count:
        return None
    sample_rows = min(table.row_count, PROFILE_ROWS)
    return col.unique_count / sample_rows


def _pk_score(col_name: str, table: TableAnnotation) -> int:
    """
    Heuristic score for how likely this table is the PK (primary entity) table
    for col_name. Used to break ties when both columns have high uniqueness.

    Logic: strip boilerplate tokens from the table name; the table whose remaining
    tokens are fewest beyond the column's entity prefix is the most 'specific' match
    and is treated as the owner of that entity → PK table.

    Example: col_name='order_id', prefix='order'
      'olist_orders_dataset'         → extra tokens: {}          → score 100
      'olist_order_payments_dataset' → extra tokens: {payments}  → score  99
    """
    prefix = col_name.split("_")[0]
    if prefix not in table.table_name:
        return 0
    boilerplate = {"olist", "dataset"}
    extra = set(table.table_name.split("_")) - boilerplate - {prefix, prefix + "s"}
    return 100 - len(extra)


def infer_direction_and_cardinality(
    col_a: ColumnAnnotation,
    table_a: TableAnnotation,
    col_b: ColumnAnnotation,
    table_b: TableAnnotation,
) -> tuple[str, str, str, str, Cardinality]:
    """
    Return (from_table, from_col, to_table, to_col, cardinality).

    'from' = FK side (many), 'to' = PK side (one).

    Priority:
      1. Same-column-name + ratio: high uniqueness = PK (TO), low = FK (FROM).
         When both are high, _pk_score picks the more specific entity table as PK.
      2. Cross-column-name + ratio: higher uniqueness = FROM (referencing side),
         lower uniqueness = TO (dimension side).  The PK threshold does NOT apply
         here — a dimension key like geolocation_zip appears many times per zip
         (low ratio) while the referencing column (customer_zip) appears mostly
         once per row (high ratio); treating customer_zip as PK would reverse the
         direction.
      3. No ratio data, same-column-name: table-name prefix heuristic.
      4. Alphabetical fallback.
    """
    ratio_a = _uniqueness_ratio(col_a, table_a)
    ratio_b = _uniqueness_ratio(col_b, table_b)
    same_name = (col_a.column_name == col_b.column_name)

    if ratio_a is not None and ratio_b is not None:
        if same_name:
            # Same-name joins: high uniqueness unambiguously identifies the PK table.
            a_pk = ratio_a >= HIGH_UNIQUE_RATIO
            b_pk = ratio_b >= HIGH_UNIQUE_RATIO

            if a_pk and not b_pk:
                return table_b.table_name, col_b.column_name, table_a.table_name, col_a.column_name, Cardinality.many_to_one
            if b_pk and not a_pk:
                return table_a.table_name, col_a.column_name, table_b.table_name, col_b.column_name, Cardinality.many_to_one

            if a_pk and b_pk:
                # Both look like PKs (UUID columns often do in 500-row samples).
                # Use table-name specificity to pick which table "owns" this entity.
                score_a = _pk_score(col_a.column_name, table_a)
                score_b = _pk_score(col_b.column_name, table_b)
                if score_a != score_b:
                    if score_a > score_b:
                        return table_b.table_name, col_b.column_name, table_a.table_name, col_a.column_name, Cardinality.many_to_one
                    return table_a.table_name, col_a.column_name, table_b.table_name, col_b.column_name, Cardinality.many_to_one
                # True tie: one-to-one, alphabetical
                if table_a.table_name <= table_b.table_name:
                    return table_a.table_name, col_a.column_name, table_b.table_name, col_b.column_name, Cardinality.one_to_one
                return table_b.table_name, col_b.column_name, table_a.table_name, col_a.column_name, Cardinality.one_to_one
            # Both below threshold (same-name): fall through to ratio comparison.

        # Cross-column-name OR both-low same-name:
        # Higher column uniqueness = referencing (FROM) side.
        if ratio_a != ratio_b:
            if ratio_a > ratio_b:
                return table_a.table_name, col_a.column_name, table_b.table_name, col_b.column_name, Cardinality.many_to_many
            return table_b.table_name, col_b.column_name, table_a.table_name, col_a.column_name, Cardinality.many_to_many
        # Equal ratios → alphabetical
        if table_a.table_name <= table_b.table_name:
            return table_a.table_name, col_a.column_name, table_b.table_name, col_b.column_name, Cardinality.many_to_many
        return table_b.table_name, col_b.column_name, table_a.table_name, col_a.column_name, Cardinality.many_to_many

    # No ratio data: only the same-column-name prefix heuristic is reliable.
    if same_name:
        score_a = _pk_score(col_a.column_name, table_a)
        score_b = _pk_score(col_b.column_name, table_b)
        if score_a > score_b:
            return table_b.table_name, col_b.column_name, table_a.table_name, col_a.column_name, Cardinality.many_to_one
        if score_b > score_a:
            return table_a.table_name, col_a.column_name, table_b.table_name, col_b.column_name, Cardinality.many_to_one

    # Alphabetical fallback
    if table_a.table_name <= table_b.table_name:
        return table_a.table_name, col_a.column_name, table_b.table_name, col_b.column_name, Cardinality.many_to_one
    return table_b.table_name, col_b.column_name, table_a.table_name, col_a.column_name, Cardinality.many_to_one


# ---------------------------------------------------------------------------
# Edge inference
# ---------------------------------------------------------------------------

def infer_edges(
    catalogue: LineageCatalogue,
    min_name_sim: float = MIN_NAME_SIM,
    min_confidence: float = MIN_CONFIDENCE,
) -> list[RelationshipEdge]:
    """
    Infer relationship edges from all cross-table column pairs.

    For each unordered pair of tables, every column pair is scored. Pairs below
    min_name_sim are skipped early. Only edges above min_confidence are kept.

    Dedup rule: per (from_table, from_column) only the highest-confidence
    candidate is retained — one FK column references at most one PK column.
    """
    tables = list(catalogue.tables.values())
    best: dict[tuple[str, str], RelationshipEdge] = {}
    # Separate float score includes a tiny pk_score tiebreaker so equal-confidence
    # edges resolve deterministically toward the most specific PK table.
    best_score: dict[tuple[str, str], float] = {}

    for table_a, table_b in combinations(tables, 2):
        for col_a in table_a.columns.values():
            for col_b in table_b.columns.values():

                name_sim = name_similarity(col_a.column_name, col_b.column_name)
                if name_sim < min_name_sim:
                    continue

                if not _types_compatible(col_a.sql_type, col_b.sql_type):
                    continue

                val_sim = value_overlap(col_a.sample_values, col_b.sample_values)
                confidence = compute_confidence(name_sim, val_sim)
                if confidence < min_confidence:
                    continue

                evidence: list[Evidence] = [
                    Evidence(
                        type=EvidenceType.name_match,
                        score=round(name_sim, 4),
                        detail=(
                            f"Column names identical: {col_a.column_name!r}"
                            if name_sim == 1.0
                            else f"{name_sim:.0%} token overlap: {col_a.column_name!r} ↔ {col_b.column_name!r}"
                        ),
                    )
                ]
                if val_sim > 0:
                    evidence.append(Evidence(
                        type=EvidenceType.value_overlap,
                        score=round(val_sim, 4),
                        detail=(
                            f"{val_sim:.0%} Jaccard overlap of sampled values "
                            f"({table_a.table_name}.{col_a.column_name} ↔ "
                            f"{table_b.table_name}.{col_b.column_name})"
                        ),
                    ))

                ft, fc, tt, tc, card = infer_direction_and_cardinality(
                    col_a, table_a, col_b, table_b
                )

                edge = RelationshipEdge(
                    from_table=ft,
                    from_column=fc,
                    to_table=tt,
                    to_column=tc,
                    cardinality=card,
                    confidence=confidence,
                    evidence=evidence,
                )

                # Tiebreaker: when confidence values are equal, prefer the edge
                # whose TO table is the most specific owner of this entity
                # (e.g. olist_orders_dataset beats olist_order_payments_dataset
                # as the target for any order_id FK).
                pk = _pk_score(tc, catalogue.tables[tt]) if tt in catalogue.tables else 0
                adjusted = confidence + pk * 1e-6

                key = (ft, fc)
                if key not in best or adjusted > best_score[key]:
                    best[key] = edge
                    best_score[key] = adjusted

    candidates = list(best.values())

    # Remove cross-name edges that reverse an existing same-name FK edge.
    # If A→B already appears via identical column names (i.e. a real FK was
    # inferred algorithmically), a cross-name B→A edge is spurious noise
    # (e.g. translation.product_category_name_english→products.product_category_name
    # reverses the correct products→translation same-name edge).
    same_name_pairs: set[tuple[str, str]] = {
        (e.from_table, e.to_table)
        for e in candidates
        if e.from_column == e.to_column
    }
    edges = [
        e for e in candidates
        if e.from_column == e.to_column                      # same-name: always keep
        or (e.to_table, e.from_table) not in same_name_pairs # cross-name: keep only if no reverse same-name edge exists
    ]

    return sorted(edges, key=lambda e: -e.confidence)


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------

def build_graph(edges: list[RelationshipEdge]) -> nx.DiGraph:
    """
    Build a NetworkX DiGraph from a list of RelationshipEdge objects.

    Nodes = table names (all tables reachable via at least one edge).
    Edge attributes: from_column, to_column, confidence, cardinality, edge (full object).
    """
    G = nx.DiGraph()
    for edge in edges:
        G.add_node(edge.from_table)
        G.add_node(edge.to_table)
        G.add_edge(
            edge.from_table,
            edge.to_table,
            from_column=edge.from_column,
            to_column=edge.to_column,
            confidence=edge.confidence,
            cardinality=edge.cardinality.value,
            edge=edge,
        )
    return G


def update_catalogue(catalogue: LineageCatalogue) -> tuple[LineageCatalogue, nx.DiGraph]:
    """Infer edges, attach them to the catalogue, and return both."""
    edges = infer_edges(catalogue)
    catalogue.edges = edges
    graph = build_graph(edges)
    return catalogue, graph


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Infer relationships and build lineage graph from catalogue."
    )
    parser.add_argument("catalogue", type=Path, help="Path to catalogue.json")
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output path (defaults to overwriting input catalogue)",
    )
    args = parser.parse_args()

    if not args.catalogue.exists():
        print(f"Error: {args.catalogue} not found", file=sys.stderr)
        sys.exit(1)

    catalogue = LineageCatalogue.model_validate_json(args.catalogue.read_text())
    catalogue, graph = update_catalogue(catalogue)

    out_path = args.out or args.catalogue
    out_path.write_text(catalogue.model_dump_json(indent=2))

    print(f"Relationship inference complete")
    print(f"  Edges inferred : {len(catalogue.edges)}")
    print(f"  Graph nodes    : {graph.number_of_nodes()}")
    print(f"  Graph edges    : {graph.number_of_edges()}")
    print()
    print(f"{'CONFIDENCE':>10}  {'CARDINALITY':>12}  EDGE")
    print("-" * 72)
    for edge in catalogue.edges:
        print(
            f"  {edge.confidence:.4f}  {edge.cardinality.value:>12}  "
            f"{edge.from_table}.{edge.from_column} → "
            f"{edge.to_table}.{edge.to_column}"
        )
        for ev in edge.evidence:
            print(f"             {ev.type.value:<14} score={ev.score:.2f}  {ev.detail}")
    print()
    print(f"Catalogue written → {out_path}")


if __name__ == "__main__":
    main()
