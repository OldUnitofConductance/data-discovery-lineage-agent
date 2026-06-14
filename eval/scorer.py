"""
Eval scoring for the Data Lineage & Discovery Agent.

Applies two metrics uniformly to all Q1–Q5 golden-set questions:
  1. table_set  — precision/recall on the set of tables identified
  2. join_path  — edge-level precision/recall on directed join edges

Agent answer format (one dict per question):
    {
        "question_id": "Q1",
        "tables": ["olist_orders_dataset", "olist_customers_dataset"],
        "join_path": [
            {"from_table": "olist_orders_dataset", "from_column": "customer_id", "to_table": "olist_customers_dataset"}
        ]
    }

For cross-column-name joins (e.g. Q4), include "to_column" explicitly:
    {"from_table": "olist_customers_dataset", "from_column": "customer_zip_code_prefix",
     "to_table": "olist_geolocation_dataset", "to_column": "geolocation_zip_code_prefix"}
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

GOLDEN_SET_PATH = Path(__file__).parent / "golden_set.json"
PASS_PRECISION = 0.80
PASS_RECALL = 0.70


@dataclass(frozen=True)
class Edge:
    from_table: str
    from_column: str
    to_table: str
    to_column: str  # normalized to from_column when not explicitly provided


def _parse_edge(raw: dict) -> Edge:
    return Edge(
        from_table=raw["from_table"],
        from_column=raw["from_column"],
        to_table=raw["to_table"],
        to_column=raw.get("to_column", raw["from_column"]),
    )


@dataclass
class MetricResult:
    precision: float
    recall: float
    matched: int
    agent_total: int
    expected_total: int

    @property
    def passed(self) -> bool:
        return self.precision >= PASS_PRECISION and self.recall >= PASS_RECALL

    def __str__(self) -> str:
        return (
            f"precision={self.precision:.4f} recall={self.recall:.4f} "
            f"({'PASS' if self.passed else 'FAIL'}) "
            f"[matched {self.matched}/{self.expected_total} expected, {self.agent_total} agent]"
        )


@dataclass
class QuestionResult:
    question_id: str
    question: str
    table_set: MetricResult
    join_path: MetricResult

    @property
    def passed(self) -> bool:
        return self.table_set.passed and self.join_path.passed


@dataclass
class EvalReport:
    results: list[QuestionResult]

    @property
    def passed_count(self) -> int:
        return sum(1 for r in self.results if r.passed)

    @property
    def total(self) -> int:
        return len(self.results)


def _score_sets(agent: set, expected: set) -> MetricResult:
    matched = len(agent & expected)
    precision = matched / len(agent) if agent else 0.0
    recall = matched / len(expected) if expected else 0.0
    return MetricResult(
        precision=precision,
        recall=recall,
        matched=matched,
        agent_total=len(agent),
        expected_total=len(expected),
    )


def score_question(agent_tables: list[str], agent_edges: list[dict], golden: dict) -> QuestionResult:
    """Score one agent answer against one golden-set question."""
    expected_tables = set(golden["tables_required"])
    expected_edges = {_parse_edge(e) for e in golden["join_path"]}

    table_result = _score_sets(set(agent_tables), expected_tables)
    join_result = _score_sets({_parse_edge(e) for e in agent_edges}, expected_edges)

    return QuestionResult(
        question_id=golden["id"],
        question=golden["question"],
        table_set=table_result,
        join_path=join_result,
    )


def score_all(agent_answers: list[dict], golden_set_path: Path = GOLDEN_SET_PATH) -> EvalReport:
    """Score a full set of agent answers against the golden set."""
    golden = json.loads(golden_set_path.read_text())
    questions = {q["id"]: q for q in golden["questions"]}

    results = []
    for answer in agent_answers:
        qid = answer["question_id"]
        if qid not in questions:
            raise ValueError(f"Unknown question_id in agent answers: {qid!r}")
        results.append(
            score_question(
                agent_tables=answer["tables"],
                agent_edges=answer["join_path"],
                golden=questions[qid],
            )
        )

    return EvalReport(results=results)
