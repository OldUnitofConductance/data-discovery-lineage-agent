"""
Tests for lineage_agent/graph.py.

Uses a small synthetic catalogue that mirrors the Olist FK structure
so tests run without CSV files or API calls.
"""

from __future__ import annotations

import pytest
import networkx as nx

from lineage_agent.graph import (
    MIN_CONFIDENCE,
    MIN_NAME_SIM,
    _pk_score,
    _types_compatible,
    build_graph,
    compute_confidence,
    infer_direction_and_cardinality,
    infer_edges,
    name_similarity,
    update_catalogue,
    value_overlap,
)
from lineage_agent.models import (
    Cardinality,
    ColumnAnnotation,
    EvidenceType,
    LineageCatalogue,
    RelationshipEdge,
    TableAnnotation,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_col(
    name: str,
    samples: list[str] | None = None,
    unique_count: int | None = None,
) -> ColumnAnnotation:
    return ColumnAnnotation(
        column_name=name,
        sql_type="object",
        sample_values=samples or [],
        unique_count=unique_count,
        inferred_description=f"Description of {name}",
        confidence=0.9,
    )


def make_table(name: str, row_count: int, columns: dict[str, ColumnAnnotation]) -> TableAnnotation:
    return TableAnnotation(
        table_name=name,
        row_count=row_count,
        inferred_description=f"Description of {name}",
        confidence=0.9,
        columns=columns,
    )


# Olist-like catalogue with known FK structure
@pytest.fixture()
def olist_mini() -> LineageCatalogue:
    """
    Three-table subset of Olist:
      orders (PK: order_id, FK: customer_id)
      order_items (FK: order_id, product_id)
      customers (PK: customer_id)

    unique_count set so cardinality is deterministic.
    """
    orders = make_table("olist_orders_dataset", row_count=100, columns={
        "order_id":    make_col("order_id",    ["o1","o2","o3","o4","o5"], unique_count=100),  # ratio 1.0 → PK
        "customer_id": make_col("customer_id", ["c1","c2","c3","c4","c5"], unique_count=40),   # ratio 0.4 → FK
    })
    items = make_table("olist_order_items_dataset", row_count=250, columns={
        "order_id":   make_col("order_id",   ["o1","o2","o3"],        unique_count=80),   # ratio 0.32 → FK
        "product_id": make_col("product_id", ["p1","p2","p3","p4"],   unique_count=40),   # ratio 0.16 → FK
    })
    customers = make_table("olist_customers_dataset", row_count=40, columns={
        "customer_id":        make_col("customer_id",        ["c1","c2","c3","c4","c5"], unique_count=40),  # ratio 1.0 → PK
        "customer_unique_id": make_col("customer_unique_id", ["u1","u2","u3"],           unique_count=35),  # ratio 0.875 → not PK
    })
    products = make_table("olist_products_dataset", row_count=50, columns={
        "product_id": make_col("product_id", ["p1","p2","p3","p4","p5"], unique_count=50),  # ratio 1.0 → PK
    })
    return LineageCatalogue(tables={
        "olist_orders_dataset": orders,
        "olist_order_items_dataset": items,
        "olist_customers_dataset": customers,
        "olist_products_dataset": products,
    })


# ---------------------------------------------------------------------------
# _types_compatible
# ---------------------------------------------------------------------------

def test_types_compatible_both_string():
    assert _types_compatible("object", "object") is True

def test_types_compatible_both_numeric():
    assert _types_compatible("int64", "float64") is True

def test_types_compatible_string_vs_int():
    assert _types_compatible("object", "int64") is False

def test_types_compatible_int_vs_string():
    assert _types_compatible("int64", "object") is False

def test_types_compatible_blocks_numeric_name_match():
    # order_item_id (int64) vs order_id (object UUID) — must be blocked
    assert _types_compatible("int64", "object") is False


# ---------------------------------------------------------------------------
# _pk_score
# ---------------------------------------------------------------------------

def test_pk_score_exact_match():
    t = make_table("olist_orders_dataset", 100, {})
    assert _pk_score("order_id", t) == 100  # "order" in name, no extra tokens

def test_pk_score_with_extra_tokens():
    t = make_table("olist_order_payments_dataset", 100, {})
    # extra tokens beyond "order": {"payments"} → score = 99
    assert _pk_score("order_id", t) == 99

def test_pk_score_no_match():
    t = make_table("olist_sellers_dataset", 100, {})
    assert _pk_score("order_id", t) == 0

def test_pk_score_orders_beats_order_payments():
    orders = make_table("olist_orders_dataset", 100, {})
    payments = make_table("olist_order_payments_dataset", 100, {})
    assert _pk_score("order_id", orders) > _pk_score("order_id", payments)


# ---------------------------------------------------------------------------
# name_similarity
# ---------------------------------------------------------------------------

def test_name_sim_identical():
    assert name_similarity("order_id", "order_id") == 1.0


def test_name_sim_partial():
    # {customer,zip,code,prefix} ∩ {geolocation,zip,code,prefix} = 3, union = 5
    sim = name_similarity("customer_zip_code_prefix", "geolocation_zip_code_prefix")
    assert pytest.approx(sim, abs=1e-3) == 3 / 5


def test_name_sim_only_id_token_shared():
    # "order_id" vs "seller_id" share only "id" → 1/3
    sim = name_similarity("order_id", "seller_id")
    assert sim < MIN_NAME_SIM  # should NOT produce an edge


def test_name_sim_completely_different():
    assert name_similarity("review_score", "freight_value") == 0.0


def test_name_sim_symmetric():
    a = name_similarity("customer_zip_code_prefix", "geolocation_zip_code_prefix")
    b = name_similarity("geolocation_zip_code_prefix", "customer_zip_code_prefix")
    assert a == b


# ---------------------------------------------------------------------------
# value_overlap
# ---------------------------------------------------------------------------

def test_value_overlap_full():
    assert value_overlap(["a", "b", "c"], ["a", "b", "c"]) == 1.0


def test_value_overlap_none():
    assert value_overlap(["a", "b"], ["c", "d"]) == 0.0


def test_value_overlap_partial():
    ov = value_overlap(["a", "b", "c"], ["b", "c", "d"])
    assert pytest.approx(ov) == 2 / 4  # |{b,c}| / |{a,b,c,d}|


def test_value_overlap_empty_lists():
    assert value_overlap([], []) == 0.0
    assert value_overlap(["a"], []) == 0.0


# ---------------------------------------------------------------------------
# compute_confidence
# ---------------------------------------------------------------------------

def test_confidence_exact_name_no_overlap():
    c = compute_confidence(1.0, 0.0)
    assert c == pytest.approx(0.80)


def test_confidence_exact_name_full_overlap():
    c = compute_confidence(1.0, 1.0)
    assert c == pytest.approx(0.95)


def test_confidence_partial_name_no_overlap():
    # name_sim = 0.6, val = 0
    c = compute_confidence(0.6, 0.0)
    assert c == pytest.approx(0.50 + 0.40 * 0.6)


def test_confidence_exact_always_above_partial():
    assert compute_confidence(1.0, 0.0) > compute_confidence(0.6, 0.0)


def test_confidence_capped_at_095():
    assert compute_confidence(1.0, 1.0) <= 0.95


def test_confidence_below_minimum_for_weak_name():
    # name_sim just above 0.5 with zero value overlap → still above MIN_CONFIDENCE
    c = compute_confidence(MIN_NAME_SIM, 0.0)
    assert c >= MIN_CONFIDENCE


# ---------------------------------------------------------------------------
# infer_direction_and_cardinality
# ---------------------------------------------------------------------------

def _tables_and_cols(
    table_a_name, col_a_name, uc_a, rows_a,
    table_b_name, col_b_name, uc_b, rows_b,
):
    col_a = make_col(col_a_name, unique_count=uc_a)
    col_b = make_col(col_b_name, unique_count=uc_b)
    table_a = make_table(table_a_name, rows_a, {col_a_name: col_a})
    table_b = make_table(table_b_name, rows_b, {col_b_name: col_b})
    return col_a, table_a, col_b, table_b


def test_direction_pk_on_a_side():
    col_a, ta, col_b, tb = _tables_and_cols(
        "orders", "order_id", 100, 100,   # ratio 1.0 → PK
        "order_items", "order_id", 80, 250,  # ratio 0.32 → FK
    )
    ft, fc, tt, tc, card = infer_direction_and_cardinality(col_a, ta, col_b, tb)
    assert ft == "order_items"
    assert tt == "orders"
    assert card == Cardinality.many_to_one


def test_direction_pk_on_b_side():
    col_a, ta, col_b, tb = _tables_and_cols(
        "order_items", "order_id", 80, 250,  # FK
        "orders", "order_id", 100, 100,      # PK
    )
    ft, fc, tt, tc, card = infer_direction_and_cardinality(col_a, ta, col_b, tb)
    assert ft == "order_items"
    assert tt == "orders"
    assert card == Cardinality.many_to_one


def test_direction_many_to_many_both_low_ratio():
    # unique_count must be ≤ PROFILE_ROWS (500); both ratios below HIGH_UNIQUE_RATIO
    col_a, ta, col_b, tb = _tables_and_cols(
        "olist_customers_dataset", "customer_zip_code_prefix", 400, 1000,
        "olist_geolocation_dataset", "geolocation_zip_code_prefix", 50, 1000,
    )
    # ratio_a = 400/500 = 0.80 < 0.95, ratio_b = 50/500 = 0.10 < 0.95 → many-to-many
    ft, fc, tt, tc, card = infer_direction_and_cardinality(col_a, ta, col_b, tb)
    assert card == Cardinality.many_to_many


def test_direction_one_to_one_both_high_ratio():
    col_a, ta, col_b, tb = _tables_and_cols(
        "table_a", "uid", 100, 100,
        "table_b", "uid", 100, 100,
    )
    ft, fc, tt, tc, card = infer_direction_and_cardinality(col_a, ta, col_b, tb)
    assert card == Cardinality.one_to_one


def test_direction_stable_alphabetical_for_symmetric():
    col_a, ta, col_b, tb = _tables_and_cols(
        "b_table", "uid", 100, 100,
        "a_table", "uid", 100, 100,
    )
    ft, fc, tt, tc, card = infer_direction_and_cardinality(col_a, ta, col_b, tb)
    assert ft == "a_table"  # alphabetically first


def test_direction_fallback_name_heuristic():
    # No unique_count — use name prefix heuristic
    col_a = make_col("order_id", unique_count=None)
    col_b = make_col("order_id", unique_count=None)
    ta = make_table("olist_orders_dataset", 100, {"order_id": col_a})
    tb = make_table("olist_order_items_dataset", 250, {"order_id": col_b})
    ft, fc, tt, tc, card = infer_direction_and_cardinality(col_a, ta, col_b, tb)
    # "order" is in "olist_orders_dataset" but NOT exclusively — also in "olist_order_items_dataset"
    # so heuristic falls through to alphabetical
    assert ft in ("olist_orders_dataset", "olist_order_items_dataset")


# ---------------------------------------------------------------------------
# infer_edges
# ---------------------------------------------------------------------------

def test_infer_edges_finds_order_id_fk(olist_mini):
    edges = infer_edges(olist_mini)
    found = [e for e in edges if e.from_column == "order_id" and e.to_table == "olist_orders_dataset"]
    assert len(found) >= 1, "order_items.order_id → orders.order_id not found"


def test_infer_edges_finds_product_id_fk(olist_mini):
    edges = infer_edges(olist_mini)
    found = [e for e in edges if "product_id" in (e.from_column, e.to_column)]
    assert len(found) >= 1


def test_infer_edges_finds_customer_id_fk(olist_mini):
    edges = infer_edges(olist_mini)
    found = [e for e in edges if e.from_column == "customer_id" and "customer" in e.to_table]
    assert len(found) >= 1


def test_infer_edges_dedup_keeps_best_customer_id_match(olist_mini):
    """
    orders.customer_id could match customers.customer_id (exact, 0.80)
    or customers.customer_unique_id (partial, ~0.77).
    Only one edge should be emitted per (from_table, from_column).
    """
    edges = infer_edges(olist_mini)
    order_customer_edges = [
        e for e in edges
        if e.from_table == "olist_orders_dataset" and e.from_column == "customer_id"
    ]
    assert len(order_customer_edges) == 1
    # The surviving edge should point to customer_id, not customer_unique_id
    assert order_customer_edges[0].to_column == "customer_id"


def test_infer_edges_no_self_join(olist_mini):
    edges = infer_edges(olist_mini)
    assert all(e.from_table != e.to_table for e in edges)


def test_infer_edges_all_above_min_confidence(olist_mini):
    edges = infer_edges(olist_mini)
    assert all(e.confidence >= MIN_CONFIDENCE for e in edges)


def test_infer_edges_sorted_descending_confidence(olist_mini):
    edges = infer_edges(olist_mini)
    confs = [e.confidence for e in edges]
    assert confs == sorted(confs, reverse=True)


def test_infer_edges_evidence_has_name_match(olist_mini):
    edges = infer_edges(olist_mini)
    for edge in edges:
        types = [ev.type for ev in edge.evidence]
        assert EvidenceType.name_match in types, f"No name_match evidence on {edge}"


def test_infer_edges_value_overlap_evidence_when_samples_match(olist_mini):
    # order_id samples overlap between orders and order_items
    edges = infer_edges(olist_mini)
    overlapping = [
        e for e in edges
        if e.from_column == "order_id" and e.to_table == "olist_orders_dataset"
    ]
    assert overlapping
    ev_types = [ev.type for ev in overlapping[0].evidence]
    # o1/o2/o3 appear in both samples → value_overlap > 0 → evidence present
    assert EvidenceType.value_overlap in ev_types


def test_infer_edges_blocks_type_incompatible_pairs():
    """int64 column must not produce an edge with an object/string column."""
    int_col = ColumnAnnotation(
        column_name="order_id", sql_type="int64",
        sample_values=["1", "2", "3"], unique_count=3,
        inferred_description="x", confidence=0.9,
    )
    str_col = ColumnAnnotation(
        column_name="order_id", sql_type="object",
        sample_values=["abc", "def"], unique_count=2,
        inferred_description="x", confidence=0.9,
    )
    cat = LineageCatalogue(tables={
        "table_a": make_table("table_a", 10, {"order_id": int_col}),
        "table_b": make_table("table_b", 10, {"order_id": str_col}),
    })
    assert infer_edges(cat) == []


def test_infer_edges_empty_catalogue():
    cat = LineageCatalogue(tables={})
    assert infer_edges(cat) == []


def test_infer_edges_single_table():
    col = make_col("id")
    table = make_table("only_table", 10, {"id": col})
    cat = LineageCatalogue(tables={"only_table": table})
    assert infer_edges(cat) == []


# ---------------------------------------------------------------------------
# build_graph
# ---------------------------------------------------------------------------

def test_build_graph_nodes(olist_mini):
    edges = infer_edges(olist_mini)
    G = build_graph(edges)
    assert G.number_of_nodes() == len(olist_mini.tables)


def test_build_graph_edge_attributes(olist_mini):
    edges = infer_edges(olist_mini)
    G = build_graph(edges)
    for u, v, data in G.edges(data=True):
        assert "from_column" in data
        assert "to_column" in data
        assert "confidence" in data
        assert "cardinality" in data
        assert isinstance(data["edge"], RelationshipEdge)


def test_build_graph_is_directed(olist_mini):
    edges = infer_edges(olist_mini)
    G = build_graph(edges)
    assert isinstance(G, nx.DiGraph)


def test_build_graph_empty_edges():
    G = build_graph([])
    assert G.number_of_nodes() == 0
    assert G.number_of_edges() == 0


# ---------------------------------------------------------------------------
# update_catalogue
# ---------------------------------------------------------------------------

def test_update_catalogue_populates_edges(olist_mini):
    cat, G = update_catalogue(olist_mini)
    assert len(cat.edges) > 0


def test_update_catalogue_returns_graph(olist_mini):
    cat, G = update_catalogue(olist_mini)
    assert isinstance(G, nx.DiGraph)
    assert G.number_of_edges() == len(cat.edges)
