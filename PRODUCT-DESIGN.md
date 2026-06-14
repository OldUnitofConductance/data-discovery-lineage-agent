# Product Design

## Competitor Analysis

| Tool | Does well | Falls short |
|---|---|---|
| **Atlan** | Active metadata, AI-assisted descriptions, lineage visualization | Lineage relies on parsing query history — needs it to exist |
| **Select Star** | Automated column-level lineage, popularity-based discovery | Same — depends on existing query/BI tool history |
| **Castor** | Strong documentation-generation UX, business glossary | Augments existing metadata rather than inferring cold-start |
| **OpenMetadata** | Open-source, ingestion connectors, AI description features | Lineage mostly declared via connectors reading existing FK/query info |
| **dbt docs** | Excellent lineage *if* models are written in dbt | Only works if transformations already defined in dbt |

**The gap:** every competitor requires *some* existing metadata or query history to bootstrap lineage. None solve the cold-start case — a lake with zero documentation and zero query history, where lineage must be inferred purely from schema + sample data.

---

## Architecture

```mermaid
flowchart TD
    subgraph Input
        CSV[Olist CSVs]
        GT[Ground Truth]
    end

    subgraph Core[lineage_agent]
        INFER[Schema Inference Agent]
        GRAPH[Relationship Inference]
        CAT[Lineage Catalogue]
        DISC[Discovery Agent]
    end

    subgraph UI[ui]
        CATVIEW[Catalogue View]
        DISCVIEW[Discovery View]
    end

    subgraph Eval[eval]
        SCORER[Eval Scorer]
        GOLDEN[Golden Set]
    end

    CSV --> INFER
    INFER --> CAT
    CAT --> GRAPH
    GRAPH --> CAT
    CAT --> DISC
    CAT <--> CATVIEW
    DISC --> DISCVIEW
    DISC --> SCORER
    GT --> SCORER
    GOLDEN --> SCORER
```

The **lineage catalogue** is the single source of truth — every component reads from and writes to it.

---

## Data Model

```mermaid
erDiagram
    LineageCatalogue ||--o{ TableAnnotation : tables
    LineageCatalogue ||--o{ RelationshipEdge : edges
    TableAnnotation ||--o{ ColumnAnnotation : columns
    RelationshipEdge ||--o{ Evidence : evidence

    LineageCatalogue {
        string schema_version
    }
    TableAnnotation {
        string table_name
        int row_count
        string inferred_description
        float confidence
        bool reviewed
        string human_description
    }
    ColumnAnnotation {
        string column_name
        string sql_type
        list sample_values
        string inferred_description
        float confidence
        bool reviewed
        string human_description
    }
    RelationshipEdge {
        string from_table
        string from_column
        string to_table
        string to_column
        string cardinality
        float confidence
        bool reviewed
        string confirmed
    }
    Evidence {
        string type
        float score
        string detail
    }
```

Every annotation type (table, column, edge) carries the same pattern: an AI-inferred value + confidence, plus a separate field for human corrections — so disagreement is never overwritten, only added.

---

## UI Flow (Phase 5 — in progress)

```mermaid
flowchart LR
    Start([User opens app]) --> Cat[Catalogue View]
    Cat -->|browse tables| TableDetail[Table detail:<br/>columns + confidence badges]
    TableDetail -->|edit description| Correction[Correction saved<br/>to catalogue]
    Cat -->|switch tab| Disc[Discovery View]
    Disc -->|ask a question| Trace[Agent traces join path]
    Trace -->|shows tables, joins,<br/>explanation, confidence| Result[Result view]
    Result -->|click a table| TableDetail
```

**Catalogue View**
- Browse all tables; each shows its inferred description and a confidence badge (high / medium / low)
- Click into a table to see column-level descriptions, sample values, and confidence
- Edit/accept inferred descriptions — corrections are stored separately from the original AI guess

**Discovery View**
- Ask a natural-language question (e.g. the 5 golden-set questions)
- Agent returns the required tables, the join path (with columns), and a plain-English explanation of how to get there
- Each table in the result links back to its Catalogue entry for more context
