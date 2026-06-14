"""
Discovery Q&A agent — Phase 4.

Given a natural language question, identifies which tables are needed
and traces the join path across the lineage graph, returning results in
the navigation direction (hub → satellite) expected by the eval harness.

CLI usage:
    python -m lineage_agent.discovery data/catalogue.json "your question"
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import anthropic
from dotenv import load_dotenv

from lineage_agent.models import LineageCatalogue

load_dotenv()

MODEL = "claude-sonnet-4-6"


# ---------------------------------------------------------------------------
# Tool schema — forces structured output with embedded chain-of-thought
# ---------------------------------------------------------------------------

_DISCOVER_TOOL: dict[str, Any] = {
    "name": "answer_discovery_question",
    "description": "Answer a data discovery question about a relational data lake.",
    "input_schema": {
        "type": "object",
        "required": ["reasoning", "tables", "join_path", "explanation"],
        "properties": {
            "reasoning": {
                "type": "string",
                "description": (
                    "Step-by-step reasoning: which tables contain the needed information, "
                    "which columns link them, and what join order makes sense for the question."
                ),
            },
            "tables": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "Exact table names needed to answer the question — no more, no fewer. "
                    "Use names exactly as they appear in the schema context."
                ),
            },
            "join_path": {
                "type": "array",
                "items": {
                    "type": "object",
                    "required": ["from_table", "from_column", "to_table"],
                    "properties": {
                        "from_table": {
                            "type": "string",
                            "description": "The anchor/hub table for this join step.",
                        },
                        "from_column": {
                            "type": "string",
                            "description": "The join column on from_table.",
                        },
                        "to_table": {
                            "type": "string",
                            "description": "The table being joined in.",
                        },
                        "to_column": {
                            "type": "string",
                            "description": (
                                "The join column on to_table. "
                                "Required only when it differs from from_column "
                                "(e.g. customer_zip_code_prefix → geolocation_zip_code_prefix)."
                            ),
                        },
                    },
                },
                "description": (
                    "One edge per JOIN needed. from_table is the table an analyst "
                    "would navigate FROM — the anchor for this step. "
                    "to_table is the table being joined in. "
                    "If both tables join on the same column name, omit to_column."
                ),
            },
            "explanation": {
                "type": "string",
                "description": (
                    "One or two sentences explaining the join path in plain language "
                    "that an analyst could follow directly."
                ),
            },
        },
    },
}

_SYSTEM_PROMPT = """\
You are a data discovery assistant for a Brazilian e-commerce data lake. \
Given a natural language question and a schema context, identify which tables \
are required and how to join them.

Join direction rules:
- from_table is the table an analyst starts from — the anchor or hub for that step.
- to_table is the table being joined in to enrich or filter the anchor.
- Example: if the question centres on orders and needs customer info, the join is \
  orders → customers on customer_id, even though customer_id is an FK column in orders.
- For a hub-and-spoke pattern (e.g. multiple satellite tables hanging off orders), \
  use one join edge per satellite, all starting from the hub.

Only include tables and joins strictly necessary to answer the question. \
Use exact table and column names from the schema context.\
"""


# ---------------------------------------------------------------------------
# Context builder
# ---------------------------------------------------------------------------

def _build_context(catalogue: LineageCatalogue) -> str:
    """
    Format all table/column descriptions and known joins for the discovery prompt.
    Join relationships are presented symmetrically (↔) so Claude chooses
    the navigation direction, not the FK direction stored in the graph.
    """
    lines: list[str] = []

    lines.append("## Available tables\n")
    for tname, table in catalogue.tables.items():
        lines.append(f"**{tname}** ({table.row_count:,} rows)")
        lines.append(f"  {table.effective_description}")
        if table.columns:
            lines.append("  Columns:")
            for col in table.columns.values():
                lines.append(
                    f"    - {col.column_name} ({col.sql_type}): "
                    f"{col.effective_description}"
                )
        lines.append("")

    if catalogue.edges:
        lines.append("## Known join relationships\n")
        for edge in catalogue.edges:
            if edge.from_column == edge.to_column:
                lines.append(
                    f"- {edge.from_table} ↔ {edge.to_table}"
                    f"  on  {edge.from_column}"
                )
            else:
                lines.append(
                    f"- {edge.from_table}.{edge.from_column}"
                    f" ↔ {edge.to_table}.{edge.to_column}"
                )

    return "\n".join(lines)


def _build_user_message(question: str, catalogue: LineageCatalogue) -> str:
    return f"Question: {question}\n\n{_build_context(catalogue)}"


# ---------------------------------------------------------------------------
# Core agent
# ---------------------------------------------------------------------------

def answer_question(
    question: str,
    catalogue: LineageCatalogue,
    client: anthropic.Anthropic,
    question_id: str | None = None,
) -> dict[str, Any]:
    """
    Answer a discovery question using the catalogue context.

    Returns a dict in the eval-compatible format:
      {question_id, question, tables, join_path, explanation}
    """
    response = client.messages.create(
        model=MODEL,
        max_tokens=2048,
        system=_SYSTEM_PROMPT,
        tools=[_DISCOVER_TOOL],
        tool_choice={"type": "tool", "name": "answer_discovery_question"},
        messages=[{
            "role": "user",
            "content": _build_user_message(question, catalogue),
        }],
    )

    tool_input: dict[str, Any] = next(
        block.input for block in response.content if block.type == "tool_use"
    )

    return {
        "question_id": question_id,
        "question": question,
        "tables": tool_input["tables"],
        "join_path": tool_input["join_path"],
        "explanation": tool_input["explanation"],
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Answer a data discovery question using the lineage catalogue."
    )
    parser.add_argument("catalogue", type=Path, help="Path to catalogue.json")
    parser.add_argument("question", type=str, help="Natural language question")
    parser.add_argument(
        "--id",
        dest="question_id",
        default=None,
        help="Optional question ID (e.g. Q1) for eval tracking",
    )
    args = parser.parse_args()

    if not args.catalogue.exists():
        print(f"Error: {args.catalogue} not found", file=sys.stderr)
        sys.exit(1)

    catalogue = LineageCatalogue.model_validate_json(args.catalogue.read_text())
    client = anthropic.Anthropic()

    answer = answer_question(args.question, catalogue, client, args.question_id)

    print(f"\nQuestion: {answer['question']}")
    print(f"\nTables needed:")
    for t in answer["tables"]:
        print(f"  - {t}")
    print(f"\nJoin path:")
    for edge in answer["join_path"]:
        to_col = edge.get("to_column", edge["from_column"])
        print(
            f"  {edge['from_table']}.{edge['from_column']}"
            f" → {edge['to_table']}.{to_col}"
        )
    print(f"\nExplanation: {answer['explanation']}")
    print()
    print(json.dumps(answer, indent=2))


if __name__ == "__main__":
    main()
