# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A **Data Lineage & Discovery Agent** for modern data lake environments (AWS Athena/Glue, Postgres). The core problem: tables have no enforced relationships, no FK constraints, and little documentation — analysts can't find which tables answer a question or how to join them.

**Dataset:** Olist Brazilian E-Commerce (Kaggle) — ~9 related tables, real FK structure used as eval ground truth (not revealed to the agent).

**Primary users:** Analysts, PMs, new data team members needing discovery + join paths.  
**Secondary users:** Data catalogue owners needing AI-assisted first-pass documentation.

## What the Agent Does

1. **Schema inference** — infers table/column meaning with confidence levels; flags ambiguous fields for human review
2. **Relationship inference** — infers joins between tables via shared column names + value overlap (no FKs); builds a lineage graph
3. **Discovery Q&A** — answers natural-language questions by tracing a join path across the graph with explanation

Human corrections feed back into the catalogue.

## Success Metrics

| Metric | Target |
|---|---|
| Description accuracy | ≥75% |
| High-confidence wrong rate | <5% |
| Relationship precision | ≥80% |
| Relationship recall | ≥70% |
| Discovery accuracy | ≥75% |

## Stack

- **Python** — primary language
- **Claude API** (`claude-sonnet-4-6`) — inference, description generation, Q&A
- **NetworkX** — lineage graph representation
- **SQLite or DuckDB** — local data store for Olist tables during development
- **Streamlit** — lightweight UI for human review/correction loop
- **pytest** — evaluation harness

## Project Phases

- **Phase 1 (current):** Project structure + eval/golden-set design (5 test questions with known-correct answers against Olist FK ground truth)
- **Phase 2:** Schema inference agent (column descriptions + confidence)
- **Phase 3:** Relationship inference (join path graph)
- **Phase 4:** Discovery Q&A agent
- **Phase 5:** Human review UI + correction loop

## Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Run the agent pipeline
python -m lineage_agent.main

# Run the eval harness
pytest eval/ -v

# Launch the review UI
streamlit run ui/app.py

# Run a single eval test
pytest eval/test_discovery.py::test_order_customer_join -v
```

## Repository Structure

```
data-lineage-agent/
├── data/               # Olist CSV files + ground truth FK map
│   └── ground_truth.json
├── lineage_agent/      # Core agent package
│   ├── infer.py        # Schema + column description inference
│   ├── graph.py        # Lineage graph (NetworkX)
│   ├── discovery.py    # Discovery Q&A agent
│   └── catalogue.py    # Catalogue store (corrections + state)
├── eval/               # Evaluation harness
│   ├── golden_set.json # 5 test questions + known-correct answers
│   └── test_*.py       # pytest eval tests
├── ui/                 # Streamlit review UI
│   └── app.py
├── requirements.txt
└── CLAUDE.md
```

## Eval / Golden Set Design

The `eval/golden_set.json` contains 5 discovery questions whose correct answers are derivable from Olist's real FK structure (ground truth). Each entry has:
- `question` — natural language query
- `tables_required` — which tables are needed
- `join_path` — the correct join sequence
- `answer` — expected natural-language answer summary

The agent is never told the FK structure; its inferred graph is scored against ground truth at eval time.

## Key Design Constraints

- Agent must infer relationships from column names + value overlap **only** — no FK hints
- Confidence levels must be calibrated (high-confidence predictions must be correct >95% of the time)
- Human corrections via UI must persist and override agent inferences
- Keep inference and graph building stateless and re-runnable from raw CSV
