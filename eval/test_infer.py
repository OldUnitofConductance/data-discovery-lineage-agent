"""
Tests for lineage_agent/infer.py.

Unit tests mock the Anthropic client — no API key or CSV files required.
The integration test (marked 'integration') needs real CSVs in data/ and
ANTHROPIC_API_KEY set; skip it with:  pytest -m "not integration"
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from lineage_agent.infer import (
    _build_user_message,
    infer_all,
    infer_table,
    profile_csv,
)
from lineage_agent.models import (
    HIGH_CONFIDENCE,
    MEDIUM_CONFIDENCE,
    ColumnAnnotation,
    LineageCatalogue,
    TableAnnotation,
)

DATA_DIR = Path(__file__).parent.parent / "data"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def sample_csv(tmp_path: Path) -> Path:
    """Write a minimal CSV that mimics the Olist orders table."""
    p = tmp_path / "olist_orders_dataset.csv"
    p.write_text(
        "order_id,customer_id,order_status,order_purchase_timestamp\n"
        "abc1,cust1,delivered,2017-10-02 10:56:33\n"
        "abc2,cust2,shipped,2018-03-15 08:22:10\n"
        "abc3,cust1,canceled,2018-07-04 19:45:01\n"
    )
    return p


def _mock_client(
    table_description: str = "Central orders fact table.",
    table_confidence: float = 0.92,
    columns: list[dict] | None = None,
) -> MagicMock:
    """Build a mock Anthropic client that returns a canned tool-use response."""
    if columns is None:
        columns = [
            {"column_name": "order_id", "description": "Unique order identifier.", "confidence": 0.98},
            {"column_name": "customer_id", "description": "FK to customers table.", "confidence": 0.95},
            {"column_name": "order_status", "description": "Current fulfillment status of the order.", "confidence": 0.90},
            {"column_name": "order_purchase_timestamp", "description": "UTC timestamp when the order was placed.", "confidence": 0.93},
        ]

    tool_use_block = MagicMock()
    tool_use_block.type = "tool_use"
    tool_use_block.input = {
        "table_description": table_description,
        "table_confidence": table_confidence,
        "columns": columns,
    }

    response = MagicMock()
    response.content = [tool_use_block]

    client = MagicMock()
    client.messages.create.return_value = response
    return client


# ---------------------------------------------------------------------------
# profile_csv
# ---------------------------------------------------------------------------

def test_profile_csv_shape(sample_csv: Path) -> None:
    profile = profile_csv(sample_csv)
    assert profile["table_name"] == "olist_orders_dataset"
    assert profile["row_count"] == 3
    assert len(profile["columns"]) == 4


def test_profile_csv_column_fields(sample_csv: Path) -> None:
    profile = profile_csv(sample_csv)
    col = next(c for c in profile["columns"] if c["column_name"] == "order_id")
    assert col["sql_type"]
    assert isinstance(col["null_pct"], float)
    assert isinstance(col["unique_count"], int)
    assert isinstance(col["sample_values"], list)
    assert len(col["sample_values"]) <= 10


def test_profile_csv_sample_values_are_strings(sample_csv: Path) -> None:
    profile = profile_csv(sample_csv)
    for col in profile["columns"]:
        assert all(isinstance(v, str) for v in col["sample_values"])


def test_profile_csv_no_more_than_max_sample_values(sample_csv: Path) -> None:
    profile = profile_csv(sample_csv)
    for col in profile["columns"]:
        assert len(col["sample_values"]) <= 10


# ---------------------------------------------------------------------------
# _build_user_message
# ---------------------------------------------------------------------------

def test_build_user_message_contains_table_name(sample_csv: Path) -> None:
    profile = profile_csv(sample_csv)
    msg = _build_user_message(profile)
    assert "olist_orders_dataset" in msg


def test_build_user_message_contains_all_columns(sample_csv: Path) -> None:
    profile = profile_csv(sample_csv)
    msg = _build_user_message(profile)
    for col in profile["columns"]:
        assert col["column_name"] in msg


def test_build_user_message_contains_row_count(sample_csv: Path) -> None:
    profile = profile_csv(sample_csv)
    msg = _build_user_message(profile)
    assert "3" in msg  # row count


# ---------------------------------------------------------------------------
# infer_table
# ---------------------------------------------------------------------------

def test_infer_table_returns_table_annotation(sample_csv: Path) -> None:
    profile = profile_csv(sample_csv)
    client = _mock_client()
    result = infer_table(profile, client)
    assert isinstance(result, TableAnnotation)


def test_infer_table_maps_description(sample_csv: Path) -> None:
    profile = profile_csv(sample_csv)
    client = _mock_client(table_description="Central orders fact table.")
    result = infer_table(profile, client)
    assert result.inferred_description == "Central orders fact table."


def test_infer_table_maps_confidence(sample_csv: Path) -> None:
    profile = profile_csv(sample_csv)
    client = _mock_client(table_confidence=0.92)
    result = infer_table(profile, client)
    assert result.confidence == pytest.approx(0.92)


def test_infer_table_populates_all_columns(sample_csv: Path) -> None:
    profile = profile_csv(sample_csv)
    client = _mock_client()
    result = infer_table(profile, client)
    assert set(result.columns.keys()) == {"order_id", "customer_id", "order_status", "order_purchase_timestamp"}


def test_infer_table_column_confidence_tiers(sample_csv: Path) -> None:
    profile = profile_csv(sample_csv)
    client = _mock_client()
    result = infer_table(profile, client)
    assert result.columns["order_id"].confidence_tier == "high"   # 0.98
    assert result.columns["order_id"].confidence >= HIGH_CONFIDENCE


def test_infer_table_preserves_sample_values(sample_csv: Path) -> None:
    profile = profile_csv(sample_csv)
    client = _mock_client()
    result = infer_table(profile, client)
    assert len(result.columns["order_id"].sample_values) > 0


def test_infer_table_missing_column_in_response_gets_empty_description(sample_csv: Path) -> None:
    # Claude returns annotation for only 3 of 4 columns
    profile = profile_csv(sample_csv)
    partial_columns = [
        {"column_name": "order_id", "description": "Unique order ID.", "confidence": 0.98},
        {"column_name": "customer_id", "description": "Customer FK.", "confidence": 0.95},
        {"column_name": "order_status", "description": "Status.", "confidence": 0.88},
        # order_purchase_timestamp intentionally missing
    ]
    client = _mock_client(columns=partial_columns)
    result = infer_table(profile, client)
    missing = result.columns["order_purchase_timestamp"]
    assert missing.inferred_description == ""
    assert missing.confidence == 0.0


def test_infer_table_calls_api_once(sample_csv: Path) -> None:
    profile = profile_csv(sample_csv)
    client = _mock_client()
    infer_table(profile, client)
    assert client.messages.create.call_count == 1


def test_infer_table_uses_correct_model(sample_csv: Path) -> None:
    profile = profile_csv(sample_csv)
    client = _mock_client()
    infer_table(profile, client)
    call_kwargs = client.messages.create.call_args.kwargs
    assert call_kwargs["model"] == "claude-sonnet-4-6"


def test_infer_table_uses_tool_choice(sample_csv: Path) -> None:
    profile = profile_csv(sample_csv)
    client = _mock_client()
    infer_table(profile, client)
    call_kwargs = client.messages.create.call_args.kwargs
    assert call_kwargs["tool_choice"] == {"type": "tool", "name": "annotate_table"}


# ---------------------------------------------------------------------------
# infer_all
# ---------------------------------------------------------------------------

def test_infer_all_returns_catalogue(tmp_path: Path) -> None:
    (tmp_path / "table_a.csv").write_text("id,name\n1,foo\n2,bar\n")
    (tmp_path / "table_b.csv").write_text("id,value\n1,10\n2,20\n")
    client = _mock_client(columns=[
        {"column_name": "id", "description": "Identifier.", "confidence": 0.95},
        {"column_name": "name", "description": "Name.", "confidence": 0.88},
    ])
    catalogue = infer_all(tmp_path, client)
    assert isinstance(catalogue, LineageCatalogue)
    assert "table_a" in catalogue.tables
    assert "table_b" in catalogue.tables


def test_infer_all_no_csvs_raises(tmp_path: Path) -> None:
    client = _mock_client()
    with pytest.raises(FileNotFoundError, match="No CSV files found"):
        infer_all(tmp_path, client)


def test_infer_all_calls_api_once_per_table(tmp_path: Path) -> None:
    for name in ["orders.csv", "customers.csv", "products.csv"]:
        (tmp_path / name).write_text("id,val\n1,a\n")
    client = _mock_client(columns=[
        {"column_name": "id", "description": "ID.", "confidence": 0.95},
        {"column_name": "val", "description": "Value.", "confidence": 0.88},
    ])
    infer_all(tmp_path, client)
    assert client.messages.create.call_count == 3


def test_infer_all_edges_empty(tmp_path: Path) -> None:
    (tmp_path / "orders.csv").write_text("order_id,status\n1,done\n")
    client = _mock_client(columns=[
        {"column_name": "order_id", "description": "ID.", "confidence": 0.95},
        {"column_name": "status", "description": "Status.", "confidence": 0.88},
    ])
    catalogue = infer_all(tmp_path, client)
    assert catalogue.edges == []  # edges are Phase 3


# ---------------------------------------------------------------------------
# Integration test — skipped unless real CSVs + API key are present
# ---------------------------------------------------------------------------

@pytest.mark.integration
def test_infer_real_olist_table() -> None:
    """Hits the real Claude API. Run with: pytest -m integration"""
    import anthropic as anthropic_lib

    csv_files = list(DATA_DIR.glob("*.csv"))
    if not csv_files:
        pytest.skip("No CSV files in data/ — download Olist dataset first")

    client = anthropic_lib.Anthropic()
    profile = profile_csv(csv_files[0])
    result = infer_table(profile, client)

    assert result.inferred_description
    assert 0.0 <= result.confidence <= 1.0
    assert len(result.columns) == len(profile["columns"])
    for col in result.columns.values():
        assert col.inferred_description
        assert 0.0 <= col.confidence <= 1.0
