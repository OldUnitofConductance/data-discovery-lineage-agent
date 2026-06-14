"""
CLI runner: score a JSON file of agent answers against the golden set.

Usage:
    python eval/run_eval.py eval/sample_answers.json

The input file must be a JSON array of answer objects (see scorer.py for format).
Exits with code 1 if any question fails both metrics.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from eval.scorer import EvalReport, QuestionResult, score_all


def _print_report(report: EvalReport, agent_answers: list[dict]) -> None:
    answers_by_id = {a["question_id"]: a for a in agent_answers}
    width = 72
    print("=" * width)
    print(f"  EVAL REPORT  —  {report.passed_count}/{report.total} questions passed")
    print("=" * width)

    for r in report.results:
        status = "PASS" if r.passed else "FAIL"
        print(f"\n[{status}] {r.question_id}: {r.question}")
        print(f"  table_set : {r.table_set}")
        print(f"  join_path : {r.join_path}")

        if not r.table_set.passed or not r.join_path.passed:
            _print_mismatches(r, answers_by_id[r.question_id])

    print("\n" + "=" * width)
    overall = "ALL PASS" if report.passed_count == report.total else f"{report.total - report.passed_count} FAILED"
    print(f"  Overall: {overall}")
    print("=" * width)


def _print_mismatches(r: QuestionResult, agent_answer: dict) -> None:
    from eval.scorer import GOLDEN_SET_PATH, _parse_edge
    import json

    golden = json.loads(GOLDEN_SET_PATH.read_text())
    question = next(q for q in golden["questions"] if q["id"] == r.question_id)

    if not r.table_set.passed:
        expected = set(question["tables_required"])
        agent_tables = set(agent_answer["tables"])
        missing = expected - agent_tables
        extra = agent_tables - expected
        if missing:
            print(f"    tables missing from agent : {sorted(missing)}")
        if extra:
            print(f"    tables extra in agent     : {sorted(extra)}")

    if not r.join_path.passed:
        expected_edges = {_parse_edge(e) for e in question["join_path"]}
        agent_edges = {_parse_edge(e) for e in agent_answer["join_path"]}
        missing = expected_edges - agent_edges
        extra = agent_edges - expected_edges
        if missing:
            print(f"    edges missing from agent  : {[str(e) for e in missing]}")
        if extra:
            print(f"    edges extra in agent      : {[str(e) for e in extra]}")


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python eval/run_eval.py <answers.json>", file=sys.stderr)
        sys.exit(2)

    answers_path = Path(sys.argv[1])
    if not answers_path.exists():
        print(f"File not found: {answers_path}", file=sys.stderr)
        sys.exit(2)

    agent_answers = json.loads(answers_path.read_text())
    report = score_all(agent_answers)
    _print_report(report, agent_answers)

    if report.passed_count < report.total:
        sys.exit(1)


if __name__ == "__main__":
    main()
