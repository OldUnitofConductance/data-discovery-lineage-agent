"""
pytest tests for eval/scorer.py.

Covers:
  - Perfect answers pass all metrics (Q1–Q5)
  - Partial answers produce correct precision/recall values
  - Extra tables/edges hurt precision, missing ones hurt recall
  - Q4 cross-column-name edge matching
  - Unknown question_id raises ValueError
"""

import json
from pathlib import Path

import pytest

from eval.scorer import (
    PASS_PRECISION,
    PASS_RECALL,
    MetricResult,
    _parse_edge,
    _score_sets,
    score_all,
    score_question,
)

GOLDEN_PATH = Path(__file__).parent / "golden_set.json"
_GOLDEN = json.loads(GOLDEN_PATH.read_text())
_QUESTIONS = {q["id"]: q for q in _GOLDEN["questions"]}


# ---------------------------------------------------------------------------
# Unit tests: _score_sets
# ---------------------------------------------------------------------------

def test_score_sets_perfect():
    r = _score_sets({"a", "b"}, {"a", "b"})
    assert r.precision == 1.0
    assert r.recall == 1.0
    assert r.passed


def test_score_sets_extra_agent_item_hurts_precision():
    r = _score_sets({"a", "b", "c"}, {"a", "b"})
    assert r.precision == pytest.approx(2 / 3)
    assert r.recall == 1.0
    assert not r.passed  # precision < 0.80


def test_score_sets_missing_item_hurts_recall():
    r = _score_sets({"a"}, {"a", "b"})
    assert r.precision == 1.0
    assert r.recall == 0.5
    assert not r.passed  # recall < 0.70


def test_score_sets_empty_agent():
    r = _score_sets(set(), {"a", "b"})
    assert r.precision == 0.0
    assert r.recall == 0.0
    assert not r.passed


def test_score_sets_empty_expected():
    r = _score_sets({"a"}, set())
    assert r.precision == 0.0
    assert r.recall == 0.0


# ---------------------------------------------------------------------------
# Unit tests: _parse_edge normalisation
# ---------------------------------------------------------------------------

def test_parse_edge_symmetric_column():
    e = _parse_edge({"from_table": "A", "from_column": "id", "to_table": "B"})
    assert e.to_column == "id"


def test_parse_edge_asymmetric_column():
    e = _parse_edge({
        "from_table": "olist_customers_dataset",
        "from_column": "customer_zip_code_prefix",
        "to_table": "olist_geolocation_dataset",
        "to_column": "geolocation_zip_code_prefix",
    })
    assert e.from_column == "customer_zip_code_prefix"
    assert e.to_column == "geolocation_zip_code_prefix"
    assert e.from_column != e.to_column


# ---------------------------------------------------------------------------
# Perfect-answer tests (one per golden question)
# ---------------------------------------------------------------------------

def _perfect_answer(qid: str) -> dict:
    q = _QUESTIONS[qid]
    return {
        "question_id": qid,
        "tables": q["tables_required"],
        "join_path": q["join_path"],
    }


@pytest.mark.parametrize("qid", ["Q1", "Q2", "Q3", "Q4", "Q5"])
def test_perfect_answer_passes(qid):
    answer = _perfect_answer(qid)
    q = _QUESTIONS[qid]
    result = score_question(answer["tables"], answer["join_path"], q)
    assert result.table_set.precision == 1.0
    assert result.table_set.recall == 1.0
    assert result.join_path.precision == 1.0
    assert result.join_path.recall == 1.0
    assert result.passed


# ---------------------------------------------------------------------------
# score_all: full report with perfect answers
# ---------------------------------------------------------------------------

def test_score_all_perfect_answers_all_pass():
    answers = [_perfect_answer(qid) for qid in ["Q1", "Q2", "Q3", "Q4", "Q5"]]
    report = score_all(answers, GOLDEN_PATH)
    assert report.total == 5
    assert report.passed_count == 5


def test_score_all_unknown_question_id_raises():
    with pytest.raises(ValueError, match="Unknown question_id"):
        score_all([{"question_id": "Q99", "tables": [], "join_path": []}], GOLDEN_PATH)


# ---------------------------------------------------------------------------
# Partial-answer behaviour
# ---------------------------------------------------------------------------

def test_extra_table_hurts_precision_only():
    q = _QUESTIONS["Q1"]
    result = score_question(
        agent_tables=["olist_orders_dataset", "olist_customers_dataset", "olist_products_dataset"],
        agent_edges=q["join_path"],
        golden=q,
    )
    assert result.table_set.recall == 1.0
    assert result.table_set.precision == pytest.approx(2 / 3)
    assert not result.table_set.passed


def test_missing_table_hurts_recall_only():
    q = _QUESTIONS["Q1"]
    result = score_question(
        agent_tables=["olist_orders_dataset"],
        agent_edges=q["join_path"],
        golden=q,
    )
    assert result.table_set.precision == 1.0
    assert result.table_set.recall == 0.5
    assert not result.table_set.passed


def test_extra_edge_hurts_join_precision():
    q = _QUESTIONS["Q3"]
    spurious_edge = {
        "from_table": "olist_orders_dataset",
        "from_column": "order_id",
        "to_table": "olist_products_dataset",
    }
    result = score_question(
        agent_tables=q["tables_required"],
        agent_edges=list(q["join_path"]) + [spurious_edge],
        golden=q,
    )
    assert result.join_path.recall == 1.0
    assert result.join_path.precision == pytest.approx(2 / 3)


def test_missing_edge_hurts_join_recall():
    q = _QUESTIONS["Q2"]
    result = score_question(
        agent_tables=q["tables_required"],
        agent_edges=q["join_path"][:1],  # only first edge
        golden=q,
    )
    assert result.join_path.precision == 1.0
    assert result.join_path.recall == pytest.approx(1 / 3)


# ---------------------------------------------------------------------------
# Q4: cross-column-name edge must match both column names
# ---------------------------------------------------------------------------

def test_q4_wrong_to_column_does_not_match():
    q = _QUESTIONS["Q4"]
    wrong_edge = {
        "from_table": "olist_customers_dataset",
        "from_column": "customer_zip_code_prefix",
        "to_table": "olist_geolocation_dataset",
        # agent omitted to_column — normalises to from_column, which differs from expected
    }
    result = score_question(
        agent_tables=q["tables_required"],
        agent_edges=[wrong_edge],
        golden=q,
    )
    assert result.join_path.matched == 0
    assert result.join_path.recall == 0.0


def test_q4_correct_to_column_matches():
    q = _QUESTIONS["Q4"]
    correct_edge = {
        "from_table": "olist_customers_dataset",
        "from_column": "customer_zip_code_prefix",
        "to_table": "olist_geolocation_dataset",
        "to_column": "geolocation_zip_code_prefix",
    }
    result = score_question(
        agent_tables=q["tables_required"],
        agent_edges=[correct_edge],
        golden=q,
    )
    assert result.join_path.matched == 1
    assert result.join_path.precision == 1.0
    assert result.join_path.recall == 1.0
