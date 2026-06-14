# Data Lineage & Discovery Agent

An AI agent that infers what tables and columns mean, discovers relationships between them with no foreign keys declared, and answers "which tables do I need and how do I join them?" - all with confidence scores and a human-in-the-loop review layer.

## The problem

In modern data lake / lake-house environments (e.g. AWS Athena + Glue Catalog), tables have **no enforced relationships** — no foreign keys, no built-in lineage, often no documentation. When an analyst, PM, or new data engineer needs to answer a question, they can't simply "look up" which tables to use or how they connect; that knowledge exists only in the heads of people who built the pipelines.

This creates two compounding problems:
- **Discovery** — "Which table(s) do I need to answer X?" has no reliable answer without asking someone
- **Lineage** — even once you find a relevant table, you don't know how it relates to others, so joining data correctly requires guesswork

## The core idea

Given raw schema metadata + sample data (the same information exposed by `information_schema` in Postgres *or* AWS Glue Catalog/Athena), the agent:

1. **Infers what each table and column means**, with a confidence score — flagging genuinely ambiguous fields rather than guessing silently
2. **Infers relationships between tables** by analyzing column names and value overlap — building a lineage graph from scratch, since none are declared
3. **Answers discovery questions** by tracing a join path across that graph and explaining why — not just naming a table, but showing how to get there

A human can review and correct inferences, and corrections are tracked separately from the AI's original guess — preserving the signal of where the agent was wrong.

## Architecture

<img width="1472" height="1080" alt="image" src="https://github.com/user-attachments/assets/7c4ac555-c6b9-47db-9215-dd5cd98a1828" />
<img width="1472" height="720" alt="image" src="https://github.com/user-attachments/assets/2020f48a-411d-41ef-87f2-e4f155d14129" />



The **lineage catalogue** is the single source of truth — every component reads from and writes to it.

## Status

| Phase | Status |
|---|---|
| 1 — Scope & Research | ✅ Done |
| 2 — Design (data model + architecture) | ✅ Done |
| 3a — Eval harness (golden set + scorer) | ✅ Done |
| 3b — Schema inference agent | ✅ Done |
| 3c — Relationship inference + graph builder | ✅ Done |
| 3d — Discovery agent | ✅ Done |
| 5 — Streamlit UI | 🚧 In progress |

## Eval results

**Schema inference — confidence calibration**

| Tier | Before prompt fix | After prompt fix |
|---|---|---|
| High (≥0.90) | 52/52 (100%) | 5/52 (10%) |
| Medium (0.60–0.89) | 0 | 47/52 (90%) |
| Low (<0.60) | 0 | 0 |

The initial prompt produced uniformly high confidence with no discrimination between clear and ambiguous columns. After adding a reasoning step asking the model to consider alternative interpretations, scores spread meaningfully — e.g. `customer_id` dropped to 0.72, correctly flagging that it's a per-order identifier, not a stable per-person ID.

**Relationship inference — vs. ground truth**

9/9 relationship edges correctly identified, 0 spurious edges, 48/48 unit tests passing.

**Discovery agent — golden set (5 questions)**

| Question | Table-set P/R | Join-path P/R | Status |
|---|---|---|---|
| Q1 — single-hop | 1.00 / 1.00 | 1.00 / 1.00 | PASS |
| Q2 — multi-hop (translation table) | 1.00 / 1.00 | 1.00 / 1.00 | PASS |
| Q3 — parallel joins | 1.00 / 1.00 | 1.00 / 1.00 | PASS |
| Q4 — cross-column geospatial | 1.00 / 1.00 | 1.00 / 1.00 | PASS |
| Q5 — full star schema (7 joins) | 1.00 / 1.00 | 1.00 / 1.00 | PASS |

All success metrics from the original spec (≥75% description accuracy, ≥80%/70% relationship precision/recall, ≥75% discovery accuracy) were exceeded.

## Running locally

1. Download the [Olist Brazilian E-Commerce dataset](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce) and place the CSVs in `data/`
2. Create a `.env` file with `ANTHROPIC_API_KEY=sk-ant-...`
3. Install dependencies: `pip install -r requirements.txt`
4. Run schema inference: `python3 -m lineage_agent.infer data/ --out data/catalogue.json`
5. Run relationship inference: `python3 -m lineage_agent.graph data/catalogue.json --out data/catalogue.json`
6. Run the eval: `python3 eval/gen_answers.py`

## Why this generalizes beyond Olist

The ingestion approach (`information_schema`-style schema + sample data) is identical for Postgres and AWS Athena/Glue — so the same pipeline applies directly to lake-house environments, where lineage genuinely doesn't exist as metadata and can only be inferred.

## Reading order for reviewers

1. This README
2. [`DESIGN-DECISIONS.md`](./DESIGN-DECISIONS.md) — every significant design decision, with reasoning and what would change it
3. [`eval/`](./eval/) — golden set, ground truth, and scorer — the proof behind the results above
4. PRODUCT-DESIGN.md — competitor research, architecture, data model, and UI flow
