# DESIGN-DECISIONS.md

Significant design decisions made during Phases 1–4 of the Data Lineage & Discovery Agent.
Format: Decision → Options considered → Choice → Reasoning → What would change it.

---

## 1. Eval-first approach

**Decision:** Build the golden set (`eval/golden_set.json`), ground truth FK map (`data/ground_truth.json`), and scorer (`eval/scorer.py`) before writing any inference code.

**Options considered:**
- Inference-first: build the agent, then design eval around what it produces.
- Eval-first: define success criteria and ground truth upfront, then build toward them.

**Choice:** Eval-first.

**Reasoning:** Inference-first risks optimising for outputs that look good rather than outputs that are correct. The Olist FK structure is fixed and knowable; defining ground truth upfront means the agent is always scored against an objective standard it has never seen. It also forces precision about what "correct" means (directed edges? table sets? both?) before implementation commits to an answer.

**What would change it:** If the domain had no ground truth (e.g., a genuinely undocumented legacy schema), inference-first would be unavoidable and eval would shift to human-in-the-loop spot-checking.

---

## 2. Uniform scoring rubric across Q1–Q5

**Decision:** Apply identical table-set precision/recall and join-path edge-level precision/recall to every question, including Q5 (full star schema, 7 edges).

**Options considered:**
- Treat Q5 differently: require only partial credit, weight edges by "importance", or use a relaxed threshold.
- Uniform rubric: same pass thresholds (precision ≥ 0.80, recall ≥ 0.70) for every question regardless of complexity.

**Choice:** Uniform rubric.

**Reasoning:** A special case for Q5 would require subjective decisions about which edges matter more. The uniform rubric is self-documenting and directly comparable across questions. Q5's larger edge count naturally makes it harder to pass—that's intentional, not a problem to work around. Partial credit for "almost correct" star schemas would mask real gaps.

**What would change it:** If the project grew to include questions with very different scopes (e.g., one edge vs. 20 edges), weighted scoring or normalised-by-size metrics might be fairer.

---

## 3. Field naming: `from_column` everywhere (Option A)

**Decision:** Standardise on `from_column` across `data/ground_truth.json`, `eval/golden_set.json`, `eval/scorer.py`, and `lineage_agent/models.py`. The alternative was `join_column`.

**Options considered:**
- **Option A:** `from_column` everywhere — makes the from/to directional pair explicit.
- **Option B:** `join_column` everywhere — single symmetric name, direction implied by from/to table order.

**Choice:** Option A (`from_column`).

**Reasoning:** The project models directed edges (FK side → PK side). `from_column` and `to_column` as a pair makes the asymmetry explicit and catches cross-column-name joins (Q4: `customer_zip_code_prefix` → `geolocation_zip_code_prefix`) without a special field. `join_column` obscures which side of the join the column belongs to and requires a separate mechanism for cross-name joins.

**What would change it:** A symmetric join model with no direction would make `join_column` cleaner.

---

## 4. Data model: `LineageCatalogue` and `RelationshipEdge`

### 4a. Evidence as a list per edge

**Choice:** `evidence: list[Evidence]` on `RelationshipEdge`, where each `Evidence` has a type (`name_match`, `value_overlap`, `semantic`) and a score.

**Reasoning:** An edge can be supported by multiple independent signals simultaneously (e.g., identical column names AND 80% value overlap). A single `evidence_score` float would collapse distinct signals into one number, losing the breakdown the human-review UI needs.

**What would change it:** If the UI only ever shows one confidence number, the list is unnecessary overhead.

---

### 4b. Separate `inferred_description` and `human_description` fields

**Choice:** Both fields stored explicitly. `effective_description` is a computed property that returns `human_description` if set, otherwise `inferred_description`.

**Reasoning:** Human corrections must not overwrite AI work. If a correction turns out to be wrong, the agent's original inference should be recoverable. Keeping both fields also lets the UI show "AI said X, human changed to Y" without re-running inference.

**What would change it:** A full audit-log approach (append-only correction history) would make the two-field pattern obsolete.

---

### 4c. `columns` as a dict, not a list

**Choice:** `TableAnnotation.columns: dict[str, ColumnAnnotation]` keyed by column name.

**Reasoning:** Every downstream consumer (graph builder, discovery agent, UI) needs to look up columns by name. A list forces O(n) scans or an extra index. The column name is already a natural unique key within a table.

**What would change it:** Ordered column iteration matters (e.g., preserving original CSV column order for display). In that case a list or `OrderedDict` would be preferable.

---

### 4d. `confirmed: bool | None` as a three-state field

**Choice:** `None` = unreviewed, `True` = human confirmed, `False` = human rejected — on `RelationshipEdge`.

**Reasoning:** A two-state `bool` collapses "not yet reviewed" and "explicitly rejected," making it impossible to distinguish new inferences from ones a human has seen and rejected. The three states map directly to the three UI actions: ignore (leave `None`), approve, or reject.

**What would change it:** If every edge is reviewed before the catalogue is used (batch mode), a two-state field is sufficient.

---

### 4e. `to_column` always explicit on `RelationshipEdge`

**Choice:** `to_column` is a required field, always set — even when it equals `from_column`.

**Reasoning:** Q4 requires matching `customer_zip_code_prefix` to `geolocation_zip_code_prefix`. If `to_column` were optional and defaulted to `from_column`, cross-name joins would need a separate code path. Making it always explicit means edge creation, serialisation, and eval parsing are all uniform. The scorer's `_parse_edge` still normalises missing `to_column` in agent answers (where it's optional for brevity).

**What would change it:** If the schema had no cross-column-name joins, the redundancy of storing `to_column = from_column` on every same-name edge would be the more compelling argument.

---

### 4f. `confidence_tier` as a computed property

**Choice:** `confidence_tier` is `@property` on `ColumnAnnotation`, `TableAnnotation`, and `RelationshipEdge`, derived from the stored `confidence` float. It is not persisted to JSON.

**Reasoning:** Tier is fully determined by the float value; storing it separately would create a sync hazard if the float is ever updated without updating the tier. The thresholds (`HIGH_CONFIDENCE = 0.85`, `MEDIUM_CONFIDENCE = 0.60`) live in one place in `models.py`.

**What would change it:** If tier thresholds needed to be per-table or user-configurable at query time, a stored field with a recalculation step would be needed.

---

### 4g. `to_eval_edge()` adapter on `RelationshipEdge`

**Choice:** A method on `RelationshipEdge` that strips all inference metadata and returns the minimal dict the scorer expects: `{from_table, from_column, to_table, to_column}`.

**Reasoning:** Keeps the rich catalogue model decoupled from the eval wire format. If the eval format changes (e.g., adding a confidence field), only the adapter changes, not the model or the scorer.

**What would change it:** If the eval format stabilised permanently and was identical to the model, the adapter would be dead code.

---

## 5. Confidence calibration: alternative-interpretation reasoning step

**Decision:** Add a required `alternative_interpretation` field to the Claude tool schema for `infer.py`, forcing the model to articulate an alternative before committing to a confidence score.

**Options considered:**
- **Quota:** Require at most N columns at high confidence (e.g., ≤10%).
- **Reasoning step:** Require an explicit alternative interpretation; if it is non-trivial, the system prompt constrains confidence to ≤ 0.89.
- **Accept the current distribution:** Document that Olist's clean naming is the reason everything scores high.

**Choice:** Reasoning step (not a quota).

**Reasoning:** A quota is a mechanical constraint disconnected from actual ambiguity — it would force low confidence on genuinely unambiguous columns. The reasoning step is calibrated: if the model cannot articulate a credible alternative, high confidence is appropriate; if it can, the system prompt rule (`≥0.90 only when no credible alternative whatsoever`) enforces a lower score. The result was 5 high / 47 medium / 0 low across 52 columns. `customer_id` scoring 0.72 is the key validation: it genuinely could be a customer number, a session ID, or a loyalty account ID based on name and UUID values alone.

**What would change it:** A schema with stronger naming conventions (e.g., `customer_pk`, `order_fk`) might make the reasoning step unnecessary because the type-system information alone would calibrate confidence correctly.

---

## 6. Phase 3 relationship inference: direction and dedup bugs

Four bugs were found and fixed when running the graph builder against the real Olist catalogue.

### 6a. Equal-confidence dedup: `order_items.order_id` pointed to `order_payments` instead of `orders`

**Bug:** The dedup dict kept whichever same-name FK edge was processed last when confidence values tied. `order_items.order_id` ended up pointing to `order_payments` (confidence 0.80) instead of `orders` (also 0.80) because iteration order was arbitrary.

**Fix:** Added a `pk_score` tiebreaker: `adjusted = confidence + _pk_score(to_column, to_table) * 1e-6`. `_pk_score` returns 100 for `olist_orders_dataset` ("order" exactly matches table entity, no extra tokens) vs. 99 for `olist_order_payments_dataset` ("payments" is an extra token beyond the entity prefix). The same fix resolved `order_payments.order_id → order_reviews` (same tie scenario).

**What would change it:** A richer PK detection approach (e.g., checking for a true primary key constraint or a uniqueness index) would make the heuristic unnecessary.

---

### 6b. Zip-code direction: `geolocation → customers` instead of `customers → geolocation`

**Bug:** `customers.customer_zip_code_prefix` has uniqueness ratio 0.968 in the 500-row sample (each customer in the sample has a unique zip). The algorithm classified this as a PK column, reversing the direction: `geolocation` (low ratio, 0.10) became the FK side (FROM), and `customers` became the TO side. Ground truth has it the other way.

Additionally, both `customers → geolocation` and `sellers → geolocation` collapsed to the same dedup key `(geolocation, geolocation_zip_code_prefix)` because both cross-name pairs resolved to `from=geolocation` under the old logic. Only one survived.

**Fix:** Added a `same_name` guard: the PK-threshold early-return branches (`a_pk and not b_pk`, etc.) only fire for same-column-name pairs. For cross-column-name joins, the function falls through to a ratio-comparison branch where **higher uniqueness = FROM side** (the referencing table). Geolocation's zip column has ratio 0.10 (many rows per zip — it is the looked-up dimension), while customers and sellers each have ratio > 0.90 for their zip columns (one entry per entity). This correctly gives `customers → geolocation` and `sellers → geolocation` as separate edges with separate dedup keys.

**Reasoning for the convention reversal:** For same-name joins, high uniqueness = PK = TO side (the entity owner). For cross-name joins, high uniqueness on a column means each row of that table has a distinct lookup value — consistent with being the referencing (FROM) side pointing into a dimension with repeated keys.

**What would change it:** Larger sample sizes (e.g., 5,000 rows) would give more reliable uniqueness ratios and might reduce reliance on the same-name guard.

---

### 6c. Spurious cross-name edges after the direction fix

**Bug:** After the `same_name` guard was added, two spurious edges appeared with new, non-colliding dedup keys:
- `customers.customer_unique_id → orders.customer_id` (67% token overlap between `customer_unique_id` and `customer_id`)
- `translation.product_category_name_english → products.product_category_name` (75% token overlap)

Previously these were silently deduped away because they collided with correct same-name FK edges sharing the same `(from_table, from_column)` key. After the fix, cross-name edges got different FROM tables, creating new keys that did not collide.

**Fix:** Post-processing filter in `infer_edges`: remove any cross-name edge whose `(to_table, from_table)` pair is already represented as a same-name FK edge (i.e., the cross-name edge is the exact reverse of a confirmed algorithmic FK). The spurious `translation → products` reverses the correct `products → translation`; the spurious `customers → orders` reverses the correct `orders → customers`.

**What would change it:** A schema where cross-name reverse edges are legitimately meaningful (e.g., a bidirectional lookup) would require a more nuanced filter.

---

**Final graph result:** 9/9 edges correct (precision = 1.00, recall = 1.00) against `data/ground_truth.json`. All 9 Olist FK relationships inferred with correct direction, cardinality, and join columns.

---

## 7. Phase 4: Discovery Q&A agent

### 7a. Q2 golden-set question edited to specify "in English"

**Decision:** Change Q2's question text from "find the product category and seller city" to "find the product category **in English** and seller city."

**Options considered:**
- Leave the question as-is and improve the system prompt to encourage translation-table joins.
- Edit the question to remove the ambiguity.

**Choice:** Edit the question.

**Reasoning:** The original wording was genuinely ambiguous: `products.product_category_name` already contains a category name (in Portuguese), so returning only the products table is a defensible correct answer. The golden set's expected join path, however, requires the translation table — meaning the eval was silently testing for something the question did not ask for. Fixing the question makes the eval self-consistent: the expected answer is now the only reasonable answer. Patching the system prompt to over-specify "always include translation tables" would be the wrong layer to fix an underspecified question.

**What would change it:** If the eval were designed to accept either Portuguese or English category names (awarding partial credit for the shorter path), the original question wording would be fine.

---

### 7b. Symmetric join presentation to Claude

**Decision:** Present join relationships to Claude using `↔` notation (e.g., `orders ↔ customers on customer_id`) rather than the FK-direction notation stored in the graph (`orders.customer_id → customers.customer_id`).

**Options considered:**
- Pass FK-direction edges directly: simpler, but forces Claude to reverse the direction for hub-and-spoke questions.
- Present edges symmetrically and let Claude choose the navigation direction based on the question.

**Choice:** Symmetric presentation.

**Reasoning:** The graph stores edges in FK direction (from = FK table, to = PK table) because that is what the relationship inference algorithm produces. But discovery answers require navigation direction, which depends on the question: Q1 wants `orders → customers` (FK → PK, which happens to match storage), while Q3 wants `orders → reviews` and `orders → payments` (PK → FK, which is reversed from storage). Presenting edges symmetrically means Claude reasons about direction from the question context — which it does correctly — rather than inheriting the storage convention and getting half the hub-and-spoke questions wrong.

**What would change it:** If the graph stored both directions explicitly (undirected), the presentation choice would be moot. Alternatively, if all questions shared the same navigation convention (always FK → PK), passing the storage direction directly would work.

---

### 7c. Phase 4 eval results

All five golden-set questions passed on the first real run after the Q2 question fix, with no prompt tuning beyond the initial system prompt.

| Question | Category | Table P | Table R | Edge P | Edge R | Result |
|---|---|---|---|---|---|---|
| Q1 | Single-hop | 1.00 | 1.00 | 1.00 | 1.00 | PASS |
| Q2 | Multi-hop (translation bridge) | 1.00 | 1.00 | 1.00 | 1.00 | PASS |
| Q3 | Parallel hub-and-spoke | 1.00 | 1.00 | 1.00 | 1.00 | PASS |
| Q4 | Cross-column-name geospatial | 1.00 | 1.00 | 1.00 | 1.00 | PASS |
| Q5 | Full star schema (7 joins) | 1.00 | 1.00 | 1.00 | 1.00 | PASS |

All CLAUDE.md targets exceeded: description accuracy ≥ 75% (Phase 2), relationship precision ≥ 80% and recall ≥ 70% (Phase 3, achieved 100%/100%), discovery accuracy ≥ 75% (Phase 4, achieved 100%). The forced-tool-call pattern with embedded `reasoning` chain-of-thought was sufficient; no multi-turn dialogue or graph traversal code was needed.
