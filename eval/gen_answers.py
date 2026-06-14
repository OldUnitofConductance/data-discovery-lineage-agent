"""
Generate discovery agent answers for all 5 golden-set questions.

Reads golden_set.json, answers each question using the discovery agent,
writes answers to a JSON file, then runs the eval scorer.

Usage:
    python3 eval/gen_answers.py                        # answers → data/answers.json
    python3 eval/gen_answers.py --out data/my_run.json
    python3 eval/gen_answers.py --question Q1          # single question
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import anthropic
from dotenv import load_dotenv

# repo root on sys.path so imports work without install
sys.path.insert(0, str(Path(__file__).parent.parent))

from lineage_agent.discovery import answer_question
from lineage_agent.models import LineageCatalogue
from eval.scorer import score_all

load_dotenv()

GOLDEN_SET  = Path(__file__).parent / "golden_set.json"
CATALOGUE   = Path("data/catalogue.json")
DEFAULT_OUT = Path("data/answers.json")


def run(only_id: str | None, out_path: Path) -> None:
    if not CATALOGUE.exists():
        print(f"Error: {CATALOGUE} not found — run Phase 3 first.", file=sys.stderr)
        sys.exit(1)

    catalogue   = LineageCatalogue.model_validate_json(CATALOGUE.read_text())
    golden_set  = json.loads(GOLDEN_SET.read_text())
    questions   = golden_set["questions"]

    if only_id:
        questions = [q for q in questions if q["id"] == only_id]
        if not questions:
            print(f"Error: question id {only_id!r} not found in golden set.", file=sys.stderr)
            sys.exit(1)

    client  = anthropic.Anthropic()
    answers = []

    print(f"Answering {len(questions)} question(s) …\n")
    for q in questions:
        print(f"  [{q['id']}] {q['question'][:80]}…" if len(q["question"]) > 80 else f"  [{q['id']}] {q['question']}")
        answer = answer_question(
            question=q["question"],
            catalogue=catalogue,
            client=client,
            question_id=q["id"],
        )
        answers.append(answer)
        print(f"        → tables: {answer['tables']}")
        for edge in answer["join_path"]:
            fc = edge["from_column"]
            tc = edge.get("to_column", fc)
            print(f"        → {edge['from_table']}.{fc} → {edge['to_table']}.{tc}")
        print()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(answers, indent=2))
    print(f"Answers written → {out_path}\n")

    # Score if we generated all 5 answers
    if not only_id:
        import json as _json
        golden_questions = {q["id"]: q for q in json.loads(GOLDEN_SET.read_text())["questions"]}

        print("=" * 60)
        print("EVAL RESULTS")
        print("=" * 60)
        report = score_all(answers, GOLDEN_SET)
        overall_pass = True

        for qr in report.results:
            status = "PASS" if qr.passed else "FAIL"
            print(f"\n[{qr.question_id}] {status}")

            golden = golden_questions[qr.question_id]
            agent_answer = next(a for a in answers if a["question_id"] == qr.question_id)

            for metric, res in [("table_set", qr.table_set), ("join_path", qr.join_path)]:
                flag = "✓" if res.passed else "✗"
                print(
                    f"  {flag} {metric:<12}  "
                    f"precision={res.precision:.2f}  recall={res.recall:.2f}"
                )

            if not qr.table_set.passed:
                expected = set(golden["tables_required"])
                actual = set(agent_answer["tables"])
                missing = expected - actual
                extra = actual - expected
                if missing:
                    print(f"      tables missing: {sorted(missing)}")
                if extra:
                    print(f"      tables extra:   {sorted(extra)}")

            if not qr.join_path.passed:
                from eval.scorer import _parse_edge, Edge
                expected_edges = {_parse_edge(e) for e in golden["join_path"]}
                agent_edges = {_parse_edge(e) for e in agent_answer["join_path"]}
                missing_e = expected_edges - agent_edges
                extra_e = agent_edges - expected_edges
                if missing_e:
                    print(f"      edges missing: {[f'{e.from_table}.{e.from_column}→{e.to_table}' for e in missing_e]}")
                if extra_e:
                    print(f"      edges extra:   {[f'{e.from_table}.{e.from_column}→{e.to_table}' for e in extra_e]}")

            if not qr.passed:
                overall_pass = False

        print()
        passed = report.passed_count
        print(f"Overall: {passed}/{report.total} questions passed")
        sys.exit(0 if overall_pass else 1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate discovery answers for eval.")
    parser.add_argument(
        "--out", type=Path, default=DEFAULT_OUT, help="Output path for answers JSON"
    )
    parser.add_argument(
        "--question", dest="only_id", default=None, help="Run only this question id (e.g. Q1)"
    )
    args = parser.parse_args()
    run(args.only_id, args.out)


if __name__ == "__main__":
    main()
