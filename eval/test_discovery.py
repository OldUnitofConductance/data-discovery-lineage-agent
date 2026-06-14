"""
Tests for lineage_agent/discovery.py.

Unit tests mock the Anthropic client — no API key or CSV files required.
Integration tests (marked 'integration') need data/catalogue.json and
ANTHROPIC_API_KEY set; skip with: pytest -m "not integration"
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from lineage_agent.discovery import (
    _build_context,
    _build_user_message,
    answer_question,
)
from lineage_agent.models import (
    Cardinality,
    ColumnAnnotation,
    Evidence,
    EvidenceType,
    LineageCatalogue,
    RelationshipEdge,
    TableAnnotation,
)

DATA_DIR = Path(__file__).parent.parent / "data"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _col(name: str, sql_type: str = "object", desc: str = "A column.") -> ColumnAnnotation:
    return ColumnAnnotation(
        column_name=name,
        sql_type=sql_type,
        inferred_description=desc,
        confidence=0.90,
    )


def _table(name: str, rows: int, cols: list[str], desc: str = "A table.") -> TableAnnotation:
    return TableAnnotation(
        table_name=name,
        row_count=rows,
        inferred_description=desc,
        confidence=0.85,
        columns={c: _col(c) for c in cols},
    )


def _edge(ft: str, fc: str, tt: str, tc: str | None = None) -> RelationshipEdge:
    return RelationshipEdge(
        from_table=ft,
        from_column=fc,
        to_table=tt,
        to_column=tc or fc,
        cardinality=Cardinality.many_to_one,
        confidence=0.80,
        evidence=[Evidence(type=EvidenceType.name_match, score=1.0, detail="identical")],
    )


def _mini_catalogue() -> LineageCatalogue:
    """Small catalogue: orders, customers, reviews — 2 edges."""
    cat = LineageCatalogue()
    cat.tables["orders"] = _table("orders", 1000, ["order_id", "customer_id", "status"], "Order facts.")
    cat.tables["customers"] = _table("customers", 500, ["customer_id", "city"], "Customer master.")
    cat.tables["reviews"] = _table("reviews", 800, ["review_id", "order_id", "score"], "Order reviews.")
    cat.edges = [
        _edge("orders", "customer_id", "customers"),
        _edge("reviews", "order_id", "orders"),
    ]
    return cat


def _mock_client(
    tables: list[str] | None = None,
    join_path: list[dict] | None = None,
    explanation: str = "Join orders to customers on customer_id.",
) -> MagicMock:
    """Mock Anthropic client returning a canned discovery answer."""
    if tables is None:
        tables = ["orders", "customers"]
    if join_path is None:
        join_path = [{"from_table": "orders", "from_column": "customer_id", "to_table": "customers"}]

    tool_block = MagicMock()
    tool_block.type = "tool_use"
    tool_block.input = {
        "reasoning": "orders has customer_id FK; customers is the customer dimension.",
        "tables": tables,
        "join_path": join_path,
        "explanation": explanation,
    }

    response = MagicMock()
    response.content = [tool_block]

    client = MagicMock()
    client.messages.create.return_value = response
    return client


# ---------------------------------------------------------------------------
# _build_context
# ---------------------------------------------------------------------------

class TestBuildContext:
    def test_includes_table_names(self):
        cat = _mini_catalogue()
        ctx = _build_context(cat)
        assert "orders" in ctx
        assert "customers" in ctx
        assert "reviews" in ctx

    def test_includes_table_descriptions(self):
        cat = _mini_catalogue()
        ctx = _build_context(cat)
        assert "Order facts." in ctx
        assert "Customer master." in ctx

    def test_includes_row_counts(self):
        cat = _mini_catalogue()
        ctx = _build_context(cat)
        assert "1,000" in ctx  # 1000 formatted

    def test_includes_column_names(self):
        cat = _mini_catalogue()
        ctx = _build_context(cat)
        assert "customer_id" in ctx
        assert "order_id" in ctx

    def test_same_name_join_uses_symmetric_notation(self):
        cat = _mini_catalogue()
        ctx = _build_context(cat)
        # Symmetric ↔ so Claude chooses navigation direction
        assert "↔" in ctx
        assert "orders ↔ customers" in ctx or "customers ↔ orders" in ctx

    def test_cross_name_join_shows_both_columns(self):
        cat = LineageCatalogue()
        cat.tables["customers"] = _table("customers", 100, ["customer_zip_code_prefix"])
        cat.tables["geolocation"] = _table("geolocation", 1000, ["geolocation_zip_code_prefix"])
        cat.edges = [
            RelationshipEdge(
                from_table="customers",
                from_column="customer_zip_code_prefix",
                to_table="geolocation",
                to_column="geolocation_zip_code_prefix",
                cardinality=Cardinality.many_to_many,
                confidence=0.74,
                evidence=[],
            )
        ]
        ctx = _build_context(cat)
        assert "customer_zip_code_prefix" in ctx
        assert "geolocation_zip_code_prefix" in ctx

    def test_no_edges_omits_join_section(self):
        cat = LineageCatalogue()
        cat.tables["solo"] = _table("solo", 10, ["id"])
        ctx = _build_context(cat)
        assert "join" not in ctx.lower() or "Known join" not in ctx

    def test_with_edges_includes_join_section(self):
        cat = _mini_catalogue()
        ctx = _build_context(cat)
        assert "Known join" in ctx


# ---------------------------------------------------------------------------
# _build_user_message
# ---------------------------------------------------------------------------

class TestBuildUserMessage:
    def test_question_appears_first(self):
        cat = _mini_catalogue()
        msg = _build_user_message("Who placed order #1?", cat)
        assert msg.startswith("Question: Who placed order #1?")

    def test_context_follows_question(self):
        cat = _mini_catalogue()
        msg = _build_user_message("Test?", cat)
        assert "Test?" in msg
        assert "Available tables" in msg


# ---------------------------------------------------------------------------
# answer_question
# ---------------------------------------------------------------------------

class TestAnswerQuestion:
    def test_returns_tables_list(self):
        cat = _mini_catalogue()
        client = _mock_client(tables=["orders", "customers"])
        result = answer_question("Who placed order #1?", cat, client)
        assert result["tables"] == ["orders", "customers"]

    def test_returns_join_path(self):
        cat = _mini_catalogue()
        jp = [{"from_table": "orders", "from_column": "customer_id", "to_table": "customers"}]
        client = _mock_client(join_path=jp)
        result = answer_question("Who placed order #1?", cat, client)
        assert result["join_path"] == jp

    def test_returns_explanation(self):
        cat = _mini_catalogue()
        client = _mock_client(explanation="Join orders to customers.")
        result = answer_question("Q?", cat, client)
        assert result["explanation"] == "Join orders to customers."

    def test_includes_question_text(self):
        cat = _mini_catalogue()
        client = _mock_client()
        result = answer_question("Test question?", cat, client)
        assert result["question"] == "Test question?"

    def test_includes_question_id_when_provided(self):
        cat = _mini_catalogue()
        client = _mock_client()
        result = answer_question("Q?", cat, client, question_id="Q1")
        assert result["question_id"] == "Q1"

    def test_question_id_none_when_not_provided(self):
        cat = _mini_catalogue()
        client = _mock_client()
        result = answer_question("Q?", cat, client)
        assert result["question_id"] is None

    def test_makes_exactly_one_api_call(self):
        cat = _mini_catalogue()
        client = _mock_client()
        answer_question("Q?", cat, client)
        assert client.messages.create.call_count == 1

    def test_uses_forced_tool_choice(self):
        cat = _mini_catalogue()
        client = _mock_client()
        answer_question("Q?", cat, client)
        call_kwargs = client.messages.create.call_args.kwargs
        assert call_kwargs["tool_choice"] == {
            "type": "tool",
            "name": "answer_discovery_question",
        }

    def test_passes_question_in_user_message(self):
        cat = _mini_catalogue()
        client = _mock_client()
        answer_question("Who placed order #1?", cat, client)
        call_kwargs = client.messages.create.call_args.kwargs
        user_content = call_kwargs["messages"][0]["content"]
        assert "Who placed order #1?" in user_content

    def test_context_includes_table_names_in_prompt(self):
        cat = _mini_catalogue()
        client = _mock_client()
        answer_question("Q?", cat, client)
        call_kwargs = client.messages.create.call_args.kwargs
        user_content = call_kwargs["messages"][0]["content"]
        assert "orders" in user_content
        assert "customers" in user_content

    def test_context_includes_join_relationships_in_prompt(self):
        cat = _mini_catalogue()
        client = _mock_client()
        answer_question("Q?", cat, client)
        call_kwargs = client.messages.create.call_args.kwargs
        user_content = call_kwargs["messages"][0]["content"]
        assert "↔" in user_content  # symmetric join notation

    def test_join_path_entry_has_required_fields(self):
        cat = _mini_catalogue()
        jp = [{"from_table": "orders", "from_column": "customer_id", "to_table": "customers"}]
        client = _mock_client(join_path=jp)
        result = answer_question("Q?", cat, client)
        edge = result["join_path"][0]
        assert "from_table" in edge
        assert "from_column" in edge
        assert "to_table" in edge

    def test_cross_name_join_path_preserves_to_column(self):
        cat = _mini_catalogue()
        jp = [{
            "from_table": "customers",
            "from_column": "customer_zip_code_prefix",
            "to_table": "geolocation",
            "to_column": "geolocation_zip_code_prefix",
        }]
        client = _mock_client(join_path=jp)
        result = answer_question("Q?", cat, client)
        edge = result["join_path"][0]
        assert edge.get("to_column") == "geolocation_zip_code_prefix"


# ---------------------------------------------------------------------------
# Integration (requires real API + data/catalogue.json)
# ---------------------------------------------------------------------------

@pytest.mark.integration
def test_integration_q1_order_customer_join():
    """Q1: single-hop join from orders to customers via customer_id."""
    import anthropic

    catalogue_path = DATA_DIR / "catalogue.json"
    if not catalogue_path.exists():
        pytest.skip("data/catalogue.json not found — run Phase 3 first")

    catalogue = LineageCatalogue.model_validate_json(catalogue_path.read_text())
    client = anthropic.Anthropic()

    result = answer_question(
        question="Which table tells me who placed a given order?",
        catalogue=catalogue,
        client=client,
        question_id="Q1",
    )

    assert "olist_orders_dataset" in result["tables"]
    assert "olist_customers_dataset" in result["tables"]
    assert len(result["join_path"]) >= 1

    # Check that customer_id is on the join path
    join_cols = [e["from_column"] for e in result["join_path"]]
    assert "customer_id" in join_cols

    # Check explanation is non-empty
    assert len(result["explanation"]) > 10
