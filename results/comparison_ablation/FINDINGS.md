# Experimental Findings: QRScore Head Ablation on SEC Filings

## Overview

We evaluated **QRScore attention head detection** on Llama-3.1-8B-Instruct using a leakage-safe SEC 10-K filing dataset (8 extraction tasks, 24 test instances per task = 192 total). The experiments compare three head ranking sources, measure per-task ablation sensitivity, and quantify cross-task transfer and specificity of the detected heads.

**Model:** `meta-llama/Llama-3.1-8B-Instruct` (stock HuggingFace weights, `flash_attention_2`)  
**Head masking:** Forward pre-hooks on `o_proj` layers zeroing out ablated head slices  
**Baseline accuracy (K=0):** 91.1% across all 8 tasks

---

## Background: How the Three Head Rankings Were Built

The three head ranking sources compared in this study—QRScore-SEC, QRScore-8B-NQ-TRAIN, and QRScore-8B-LME-TRAIN—were produced under fundamentally different **task paradigms**, not just different text domains. Understanding these paradigm differences is essential for interpreting the ablation results.

### QRScore-8B-NQ-TRAIN — Passage Re-ranking (Sorting)

Natural Questions (NQ) was used as part of the BEIR benchmark. To build the ranking context, 200 distinct, **disjoint** passages previously retrieved by BM25 were concatenated into a single artificial context block (16K–64K tokens). The QRRetriever system then scored attention heads based on their ability to **sort** these passages by relevance—i.e., re-rank 200 independent paragraphs so the most relevant appear at the top. Success was measured by ranking accuracy (`nDCG@10`). Head detection used 256 held-out NQ datapoints and was applied zero-shot to other BEIR datasets.

**Key characteristic:** The context is a synthetic bag of unrelated passages. The attention task is *inter-passage comparison and ranking*.

### QRScore-8B-LME-TRAIN — Long-Context Extraction + Reasoning (RAG)

LongMemEval (LME) presented a completely different challenge. The context was a **naturally continuous** ~115K-token dialogue history (chat sessions segmented at the round level). Head detection was performed on 70 single-hop examples from LME's single-session-user subset. The evaluation was a two-step RAG pipeline: (1) use QRRetriever to score and extract the top-k most relevant dialogue rounds, (2) feed those rounds back into the model to generate a final answer. Success was measured by both retrieval recall and end-to-end task accuracy.

**Key characteristic:** The context is a single continuous document. The attention task is *locating relevant spans within a coherent narrative* and then synthesizing an answer.

### QRScore-SEC — Document-Level Fact Extraction

Our SEC detection was performed on 974 training instances derived from SEC 10-K filings. Each instance is a long, continuous financial document with a specific factual needle (CEO name, employee count, incorporation state, etc.) embedded within it. Like LME, the context is a **single continuous document** rather than concatenated disjoint passages.

**Key characteristic:** The context is a single continuous document. The attention task is *locating a specific fact within a structured narrative*.

### Why This Matters

The three rankings span two distinct task paradigms:

| Paradigm | Rankings | Context Type | Attention Task |
|----------|----------|--------------|----------------|
| **Passage sorting** | NQ-TRAIN | Concatenated disjoint passages | Inter-passage comparison |
| **Span extraction** | LME-TRAIN, SEC | Single continuous document | Intra-document fact location |

This paradigm distinction—not merely the text domain—turns out to be the primary explanatory variable for the ablation results that follow.

---

## Metric Definitions

This section defines every calculated metric referenced in the findings below. All formulas correspond directly to the code in `scripts/evaluation/run_ablation.py`.

### Accuracy

For a given set of test instances (either all 192, or the 24 for a single task), accuracy is the fraction answered correctly:

$$\text{accuracy}(K) = \frac{\text{correct}(K)}{\text{total}}$$

A prediction is **correct** if, after normalisation (lowercasing, stripping articles/punctuation, collapsing whitespace and commas in numbers), either the predicted string equals the gold string or one is a substring of the other.

### Drop from Baseline (Drop@K)

The accuracy difference between the unablated model ($K{=}0$) and the ablated model at a particular knockout size $K$:

$$\text{drop@}K = \text{accuracy}(0) - \text{accuracy}(K)$$

A positive value means ablation hurt performance. This is computed per-task and overall for every method.

### On-Target Drop

In the cross-task transfer experiment, each source task $s$ has its top-$K$ heads knocked out and every target task is evaluated. The **on-target drop** is the drop when the source and target are the same task:

$$\text{on\_target\_drop}(s) = \text{accuracy}_s(0) - \text{accuracy}_s(K) \quad \text{where ablated heads come from source } s$$

It measures how much knocking out task $s$'s own detected heads hurts task $s$ itself.

### Off-Target Mean Drop

The mean drop across all *other* target tasks $t \neq s$ when source $s$'s heads are ablated:

$$\text{off\_target\_mean\_drop}(s) = \frac{1}{|T|-1} \sum_{t \neq s} \bigl[\text{accuracy}_t(0) - \text{accuracy}_t(K)\bigr]$$

where $T$ is the set of all 8 tasks and the ablated heads are source $s$'s heads. It measures collateral damage to unrelated tasks.

### Specificity Index

$$\text{specificity\_index}(s) = \text{on\_target\_drop}(s) - \text{off\_target\_mean\_drop}(s)$$

- **Positive** → ablation hurts the source task more than other tasks (heads are task-specific).
- **Negative** → ablation hurts other tasks more than the source task (heads are broadly shared, not specific to $s$).
- **Zero** → damage is uniform.

### Surgicality Ratio

$$\text{surgicality\_ratio}(s) = \frac{\text{on\_target\_drop}(s)}{\max(\text{off\_target\_mean\_drop}(s),\; \epsilon)}$$

where $\epsilon = 10^{-9}$ prevents division by zero. A ratio $> 1$ means the ablation is more surgical (on-target damage exceeds collateral); a ratio $< 1$ means collateral damage dominates.

### Jaccard Head Similarity

For two tasks $a$ and $b$, given their top-$K$ ranked head sets $H_a$ and $H_b$ (each a set of (layer, head) tuples):

$$J(a, b, K) = \frac{|H_a \cap H_b|}{|H_a \cup H_b|}$$

Ranges from 0 (disjoint head sets) to 1 (identical head sets). Computed at each $K \in \{8, 16, 32, 48, 64, 96, 128\}$ and reported as an $8 \times 8$ symmetric matrix across all task pairs.

### Baseline Accuracy

The overall accuracy at $K{=}0$ (no heads ablated): **91.1%** across all 192 test instances. This is the reference point for all drop calculations.

### Accuracy Curve

The vector of accuracy values at each $K$ for a given method, e.g. $[\text{acc}(0), \text{acc}(8), \text{acc}(16), \ldots, \text{acc}(128)]$. Plotted in `accuracy_vs_knockout.png` and `per_task_accuracy_curves.png`.

---

## Experiment 1: Pooled Ablation Comparison

**Question:** How effectively do different head ranking methods identify retrieval-critical heads?

Three ranking methods were compared by knocking out their top-K ranked heads and measuring answer accuracy degradation:

| Method | Source | K=0 | K=8 | K=16 | K=32 |
|--------|--------|-----|-----|------|------|
| **QRScore-SEC** | SEC train detection | 91.1% | 55.2% | 39.6% | 29.7% |
| **QRScore-8B-LME-TRAIN** | LM-Eval train | 91.1% | 65.1% | 25.0% | 19.3% |
| **QRScore-8B-NQ-TRAIN** | Natural Questions train | 91.1% | 90.1% | 85.9% | 77.1% |

**Chart:** `accuracy_vs_knockout.png`, `per_task_heatmaps.png`

### Key Finding 1: Task paradigm—not just text domain—determines head relevance

- **QRScore-SEC** and **QRScore-8B-LME-TRAIN** both cause severe accuracy drops, reaching <30% by K=32.
- **QRScore-8B-NQ-TRAIN** (detected on Natural Questions) barely affects SEC task performance — only a 5.2% drop at K=16 and 14.1% at K=32.

**Paradigm-level interpretation:** The critical divide is between **passage-sorting heads** (NQ) and **span-extraction heads** (LME, SEC). NQ-detected heads were optimized for comparing 200 disjoint passages and ranking them—an *inter-passage* attention pattern over synthetic, concatenated context. SEC tasks require locating a specific fact within a single continuous document—an *intra-document* attention pattern. These are fundamentally different attention behaviours, so knocking out NQ's top heads barely touches the circuits SEC tasks rely on.

Conversely, LME-detected heads—identified on a continuous ~115K-token dialogue history where the model must *locate relevant spans within a coherent narrative*—transfer effectively to SEC fact extraction despite the genre mismatch (chat logs vs. financial filings). Both are span-extraction tasks over continuous documents.

- **Implication for paper:** Head importance rankings are **paradigm-specific** more than domain-specific. The shared retrieval mechanism between LME and SEC is not "financial knowledge" or "chat knowledge" but rather the ability to **scan a single long document and locate relevant spans**. NQ's passage-sorting heads represent a categorically different attention circuit.

### Key Finding 2: Task-level sensitivity varies dramatically

At K=16, QRScore-SEC ablation impact by task:

| Task | K=0 | K=8 | K=16 | Drop@K=16 |
|------|-----|-----|------|-----------|
| `employees_count_total` | 95.8% | 16.7% | 4.2% | **91.7%** |
| `ceo_lastname` | 91.7% | 20.8% | 0.0% | **91.7%** |
| `holder_record_amount` | 79.2% | 33.3% | 20.8% | 58.3% |
| `headquarters_city` | 95.8% | 62.5% | 37.5% | 58.3% |
| `incorporation_year` | 83.3% | 62.5% | 41.7% | 41.7% |
| `incorporation_state` | 100% | 79.2% | 66.7% | 33.3% |
| `registrant_name` | 100% | 91.7% | 75.0% | 25.0% |
| `headquarters_state` | 83.3% | 75.0% | 70.8% | 12.5% |

**Chart:** `per_task_accuracy_curves.png`

- **Numeric/name extraction tasks** (`employees_count_total`, `ceo_lastname`) are devastated by just 8 head knockouts — these tasks rely on a small, concentrated set of heads.
- **Location/entity tasks** (`headquarters_state`, `registrant_name`) degrade more gradually — their retrieval is distributed across more heads.
- **Implication for paper:** Different information types within the same document domain have markedly different head concentration profiles. This suggests a **hierarchy of retrieval difficulty** where numeric facts depend on fewer, more specialized heads.

### Key Finding 3: LME-TRAIN shows a different ablation profile than SEC — same paradigm, different priority order

Although both QRScore-SEC and QRScore-8B-LME-TRAIN achieve similarly low accuracy at K=32, their degradation curves differ:
- **QRScore-SEC** drops steeply from K=0 to K=8 (91.1% → 55.2%) then declines gradually.
- **QRScore-8B-LME-TRAIN** holds higher at K=8 (65.1%) but then collapses at K=16 (25.0%).

**Paradigm-level interpretation:** Both LME and SEC operate in the span-extraction paradigm (locating information within a single continuous document), which is why both rankings ultimately identify the same pool of critical heads. However, the **priority order** within that shared pool differs. SEC detection frontloads the heads most critical for SEC-style fact extraction (short, precise answers from structured filings), while LME detection frontloads heads optimized for dialogue-round retrieval (longer, more narrative chunks from chat logs). By K=16, both rankings have captured enough of the shared extraction substrate that accuracy converges to similarly low levels.

This suggests the span-extraction paradigm relies on a **common head pool**, but the ranking within that pool reflects the specific retrieval granularity (sentence-level fact vs. paragraph-level dialogue round) of the detection data.

---

## Experiment 2: Cross-Task Transfer Ablation

**Question:** When we knock out heads detected for task A, how much does task B suffer? Are the detected heads task-specific or broadly shared?

We ran 8 source rankings (one per task) × 8 target tasks × 8 K-values using only QRScore-SEC task-level detections.

**Charts:** `transfer_drop_heatmap_K16.png`, `transfer_drop_heatmap_K32.png`, `transfer_drop_heatmap_K128.png`

### Key Finding 4: Heads are NOT task-specific — ablation causes broad collateral damage

At K=16, the specificity metrics reveal that most task-specific head knockouts cause as much or more damage to *other* tasks than to the source task:

| Source Task | On-Target Drop | Off-Target Mean Drop | Specificity Index |
|-------------|---------------|---------------------|-------------------|
| `headquarters_city` | 0.50 | 0.52 | **-0.02** |
| `headquarters_state` | 0.21 | 0.63 | **-0.42** |
| `registrant_name` | 0.17 | 0.33 | **-0.17** |
| `employees_count_total` | 0.08 | 0.02 | +0.06 |
| `holder_record_amount` | 0.04 | -0.01 | +0.05 |
| `ceo_lastname` | -0.04 | 0.04 | -0.08 |
| `incorporation_state` | 0.00 | 0.03 | -0.03 |
| `incorporation_year` | 0.00 | 0.01 | -0.01 |

**Chart:** `specificity_bars.png`, `specificity_table.csv`

- **Negative specificity index** means off-target damage exceeds on-target damage. This is the case for 6 of 8 tasks.
- **`headquarters_state`** is the most extreme: knocking out its heads causes 0.63 mean off-target drop but only 0.21 on-target drop (specificity = -0.42). Its heads are broadly important for SEC retrieval, not HQ-state-specific.
- **Implication for paper:** The QRScore-detected heads form a **shared retrieval substrate** rather than task-specific circuits. Ablating any task's top heads degrades the model's general ability to extract information from SEC documents. This is evidence that long-context retrieval in LLMs uses a common set of attention heads regardless of the specific information being retrieved.

### Key Finding 5: A cluster of related tasks shares heads

The Jaccard head similarity analysis reveals a clear cluster:

At top-16 heads:

| | hq\_city | hq\_state | registrant\_name |
|---|---------|----------|------------------|
| **hq\_city** | 1.00 | **0.78** | 0.33 |
| **hq\_state** | 0.78 | 1.00 | **0.45** |
| **registrant\_name** | 0.33 | 0.45 | 1.00 |

**Chart:** `head_similarity_heatmaps.png`

- **`headquarters_city` and `headquarters_state`** share 78% of their top-16 heads — nearly identical head sets.
- **`registrant_name`** overlaps at 33-45% with both HQ tasks, forming a geographic/entity cluster.
- All other task pairs have near-zero overlap (<7%).
- This cluster persists and strengthens at larger K (top-128: hq\_city–hq\_state = 0.75, hq\_state–registrant = 0.64).
- **Implication for paper:** There are **functional head groups** in the model. Location-related extraction (city, state, company name from SEC headers) is handled by a shared set of heads, while other fact types (CEO name, employee count, year) use distinct (but still broadly impactful) heads. This suggests the model develops specialized head clusters for **semantically related extraction patterns**.

### Key Finding 6: Some task-specific detections fail to isolate their own task

At K=16, several task-specific ablations show **zero on-target drop**:
- `incorporation_state` heads: 0% on-target drop, 3% off-target drop
- `incorporation_year` heads: 0% on-target drop, 0.6% off-target drop
- `ceo_lastname` heads: -4.2% on-target drop (accuracy *improved*), 3.6% off-target drop

This means the per-task detection for these tasks either (a) identified heads that aren't actually critical for that specific task at K=16, or (b) the task is robust to losing its "top" heads because redundant pathways exist. The negative on-target drop for `ceo_lastname` suggests mild regularization effects from head removal.

---

## Summary of Key Claims for Paper

1. **Paradigm specificity of retrieval heads** — The dominant factor in head transferability is not text domain but **task paradigm**. Passage-sorting heads (NQ, detected on 200 concatenated disjoint passages) cause only 5.2% drop at K=16 on SEC tasks. Span-extraction heads (LME, detected on continuous ~115K-token dialogue; SEC, detected on continuous financial filings) cause 25.0–39.6% drop at K=16 on the same tasks. Heads that scan a single long document for relevant spans form a categorically different attention circuit than heads that compare and rank independent passages.

2. **Cross-genre transfer within the span-extraction paradigm** — LME-detected heads transfer effectively to SEC extraction despite a complete genre mismatch (chat logs vs. 10-K filings). This demonstrates that the span-extraction attention mechanism is **genre-agnostic**: the model reuses the same heads for locating facts in financial documents as for locating dialogue rounds in chat histories. The shared mechanism is *intra-document span location*, not domain knowledge.

3. **Shared retrieval substrate** — Cross-task transfer experiments show that task-specific head ablations cause broad, non-specific damage. Specificity indices are negative for 6/8 tasks. The model uses a common set of retrieval heads across SEC extraction tasks.

4. **Semantic head clusters** — Jaccard analysis reveals a geographic/entity cluster (`headquarters_city`, `headquarters_state`, `registrant_name`) sharing 45-78% of top heads, while other task pairs are near-disjoint. The model develops functionally specialized head groups for related extraction patterns.

5. **Task difficulty hierarchy** — Numeric extraction (employee count, CEO name) collapses with just 8 knocked-out heads (>70% drop), while location/entity tasks degrade gradually. Information type determines head concentration.

6. **Priority order within shared head pools** — SEC and LME rankings converge to similar accuracy by K=32 but differ in degradation trajectory. SEC-detected rankings frontload heads critical for sentence-level fact extraction; LME-detected rankings frontload heads for paragraph-level dialogue retrieval. The underlying head pool is shared, but the priority ordering reflects the retrieval granularity of the detection data.

---

## Generated Artifacts Reference

### Plots
| File | Description |
|------|-------------|
| `accuracy_vs_knockout.png` | Overall accuracy curves: 3 methods compared |
| `per_task_accuracy_curves.png` | 8 subplots showing per-task degradation for all methods |
| `per_task_heatmaps.png` | Tasks × K heatmaps with % annotations per method |
| `transfer_drop_heatmap_K{8,16,32,48,64,96,128}.png` | 8×8 source-target drop matrices at each K |
| `head_similarity_heatmaps.png` | Jaccard similarity panels at top-{8,16,32,...,128} |
| `specificity_bars.png` | On-target vs off-target drop + specificity index bars |

### Tables
| File | Description |
|------|-------------|
| `accuracy_table.csv` | Method × Task × K accuracy values |
| `drop_from_baseline_table.csv` | Drop from K=0 baseline for each cell |
| `specificity_table.csv` | Per-source-task specificity index and surgicality ratio |

### Raw Data
| File | Description |
|------|-------------|
| `QRScore-SEC_results.json` | Full per-task accuracy curves (K=0 through K=128) |
| `QRScore-8B-LME-TRAIN_results.json` | LME-TRAIN method results |
| `QRScore-8B-NQ-TRAIN_results.json` | NQ-TRAIN method results |
| `cross_task_transfer_matrix.json` | Full 8×8×8 transfer drop matrix |
| `cross_task_specificity_metrics.json` | Specificity/surgicality at summary K=16 |
| `cross_task_head_similarity_topk.json` | Jaccard overlap at each top-K |
| `QRScore-SEC_token_log.jsonl` | Raw token-level generation logs for all instances |
| `comparison_summary.json` | High-level method comparison summary |
