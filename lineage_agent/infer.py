"""
Schema inference agent — Phase 2.

Reads CSV files, profiles each table's schema and sample data, then calls
Claude to produce table/column descriptions with confidence scores.
Output is a LineageCatalogue (tables populated, edges empty — Phase 3 adds edges).

CLI usage:
    python -m lineage_agent.infer data/ --out data/catalogue.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import anthropic
import pandas as pd
from dotenv import load_dotenv

from lineage_agent.models import (
    ColumnAnnotation,
    LineageCatalogue,
    TableAnnotation,
)

load_dotenv()

MODEL = "claude-sonnet-4-6"
MAX_SAMPLE_VALUES = 10   # distinct values stored per column for evidence use later
MAX_PROFILE_ROWS = 500   # rows read for type/null/sample profiling


# ---------------------------------------------------------------------------
# Claude tool definition — forces structured output
# ---------------------------------------------------------------------------

_ANNOTATE_TOOL: dict[str, Any] = {
    "name": "annotate_table",
    "description": (
        "Return inferred descriptions and confidence scores for a database table "
        "and each of its columns."
    ),
    "input_schema": {
        "type": "object",
        "required": ["table_description", "table_confidence", "columns"],
        "properties": {
            "table_description": {
                "type": "string",
                "description": (
                    "One or two sentences describing what this table represents "
                    "and its role in the overall data model."
                ),
            },
            "table_confidence": {
                "type": "number",
                "description": "Confidence score 0.0–1.0 for the table description.",
            },
            "columns": {
                "type": "array",
                "items": {
                    "type": "object",
                    "required": ["column_name", "alternative_interpretation", "description", "confidence"],
                    "properties": {
                        "column_name": {"type": "string"},
                        "alternative_interpretation": {
                            "type": "string",
                            "description": (
                                "The most plausible alternative meaning for this column — "
                                "what else could it represent given only the name, type, and "
                                "sample values? Write 'none' only if no credible alternative exists."
                            ),
                        },
                        "description": {
                            "type": "string",
                            "description": "One sentence describing what this column contains.",
                        },
                        "confidence": {
                            "type": "number",
                            "description": (
                                "Confidence score 0.0–1.0 for this column description. "
                                "If alternative_interpretation is a credible alternative "
                                "(not 'none'), this score must be lower to reflect that uncertainty."
                            ),
                        },
                    },
                },
            },
        },
    },
}

_SYSTEM_PROMPT = """\
You are a data catalogue assistant. You will be given the schema and sample data \
for a database table from a Brazilian e-commerce platform and must return concise, \
accurate descriptions and confidence scores via the annotate_table tool.

For every column you must first write the most plausible alternative interpretation \
before committing to a confidence score. Ask yourself: given only the column name, \
data type, and sample values — what else could this column mean to someone who has \
never seen this database? A credible alternative interpretation is one you could \
not rule out without additional context.

Confidence score guide:
  0.90–1.00  No credible alternative exists. Name + type + values are unambiguous.
  0.70–0.89  Primary interpretation is strongly suggested, but a credible alternative exists.
  0.50–0.69  Two or more interpretations are plausible; the primary is only somewhat more likely.
  0.00–0.49  Genuinely ambiguous — a human must review.

If you wrote a real alternative (not 'none'), the confidence score must be ≤0.89. \
High-confidence scores (≥0.90) are reserved for columns where you can articulate \
no credible alternative whatsoever.\
"""


# ---------------------------------------------------------------------------
# Profiling
# ---------------------------------------------------------------------------

def _count_rows(csv_path: Path) -> int:
    """Count data rows without loading the file into memory."""
    with open(csv_path, "rb") as f:
        return sum(1 for _ in f) - 1  # subtract header


def profile_csv(csv_path: Path) -> dict[str, Any]:
    """
    Read a CSV and return a schema profile dict ready to pass to the inference prompt.
    Reads only MAX_PROFILE_ROWS for column stats; counts full rows separately.
    """
    row_count = _count_rows(csv_path)
    df = pd.read_csv(csv_path, nrows=MAX_PROFILE_ROWS, low_memory=False)

    columns = []
    for col_name in df.columns:
        series = df[col_name]
        sample_values = (
            series.dropna()
            .astype(str)
            .drop_duplicates()
            .head(MAX_SAMPLE_VALUES)
            .tolist()
        )
        columns.append({
            "column_name": col_name,
            "sql_type": str(series.dtype),
            "null_pct": round(series.isna().mean() * 100, 1),
            "unique_count": int(series.nunique()),
            "sample_values": sample_values,
        })

    return {
        "table_name": csv_path.stem,
        "row_count": row_count,
        "columns": columns,
    }


# ---------------------------------------------------------------------------
# Prompt assembly
# ---------------------------------------------------------------------------

def _build_user_message(profile: dict[str, Any]) -> str:
    lines = [
        f"Table: {profile['table_name']}",
        f"Row count: {profile['row_count']:,}",
        "",
        "Columns:",
    ]
    for col in profile["columns"]:
        samples = ", ".join(col["sample_values"][:5]) or "(no non-null values)"
        lines += [
            f"  {col['column_name']}  ({col['sql_type']})",
            f"    unique: {col['unique_count']}    null%: {col['null_pct']}",
            f"    sample: {samples}",
        ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def infer_table(profile: dict[str, Any], client: anthropic.Anthropic) -> TableAnnotation:
    """
    Call Claude once per table to generate descriptions + confidence scores.
    Uses tool_choice={"type": "tool"} to guarantee structured JSON back.
    """
    response = client.messages.create(
        model=MODEL,
        max_tokens=2048,
        system=_SYSTEM_PROMPT,
        tools=[_ANNOTATE_TOOL],
        tool_choice={"type": "tool", "name": "annotate_table"},
        messages=[{"role": "user", "content": _build_user_message(profile)}],
    )

    tool_input: dict[str, Any] = next(
        block.input for block in response.content if block.type == "tool_use"
    )

    # Index returned column annotations by name for O(1) lookup
    col_annotations: dict[str, dict] = {
        c["column_name"]: c for c in tool_input["columns"]
    }

    columns: dict[str, ColumnAnnotation] = {}
    for col_profile in profile["columns"]:
        name = col_profile["column_name"]
        ann = col_annotations.get(name, {})
        columns[name] = ColumnAnnotation(
            column_name=name,
            sql_type=col_profile["sql_type"],
            sample_values=col_profile["sample_values"],
            unique_count=col_profile["unique_count"],
            inferred_description=ann.get("description", ""),
            confidence=float(ann.get("confidence", 0.0)),
        )

    return TableAnnotation(
        table_name=profile["table_name"],
        row_count=profile["row_count"],
        inferred_description=tool_input["table_description"],
        confidence=float(tool_input["table_confidence"]),
        columns=columns,
    )


def infer_all(data_dir: Path, client: anthropic.Anthropic) -> LineageCatalogue:
    """
    Infer descriptions for every CSV in data_dir.
    Returns a LineageCatalogue with tables populated and edges empty.
    """
    csv_files = sorted(data_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {data_dir}")

    catalogue = LineageCatalogue()

    for csv_path in csv_files:
        print(f"  {csv_path.name:<45}", end="", flush=True)
        profile = profile_csv(csv_path)
        table = infer_table(profile, client)
        catalogue.tables[table.table_name] = table
        low_conf_cols = [
            c.column_name for c in table.columns.values() if c.confidence < 0.60
        ]
        flag = f"  [{len(low_conf_cols)} cols need review]" if low_conf_cols else ""
        print(f"table={table.confidence:.2f} ({table.confidence_tier}){flag}")

    return catalogue


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run schema inference on a directory of CSV files."
    )
    parser.add_argument("data_dir", type=Path, help="Directory containing CSV files")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("data/catalogue.json"),
        help="Output path for catalogue JSON (default: data/catalogue.json)",
    )
    args = parser.parse_args()

    if not args.data_dir.is_dir():
        print(f"Error: {args.data_dir} is not a directory", file=sys.stderr)
        sys.exit(1)

    client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from env/.env

    print(f"Schema inference — {args.data_dir}\n")
    catalogue = infer_all(args.data_dir, client)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(catalogue.model_dump_json(indent=2))

    print(f"\nCatalogue written → {args.out}")
    print(f"Tables annotated : {len(catalogue.tables)}")

    low_conf = catalogue.low_confidence_columns()
    if low_conf:
        print(f"\nColumns flagged for human review (confidence < 0.60): {len(low_conf)}")
        for tname, col in low_conf:
            print(f"  {tname}.{col.column_name}  {col.confidence:.2f}")
    else:
        print("No columns flagged for review.")


if __name__ == "__main__":
    main()
