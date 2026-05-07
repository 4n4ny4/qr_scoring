# Target-Task Sensitivity Report

## Definitions

### The cross-task transfer experiment in plain English

For every model, the QRScore detection step produced a *ranked list of attention heads* for **each of the 8 SEC tasks** (e.g. `ceo_lastname`, `headquarters_city`, …). We then run a cross-task ablation:

> *For every (source task, target task) pair, take the top-K heads detected for the source task, zero them out ("ablate" them), and measure how much accuracy drops on the target task's test set.*

That gives an `8 × 8 × |K|` cube of accuracies, stored in `cross_task_transfer_matrix.json`.

### Variables used in the formulas

| Symbol | Type | Meaning | Example value |
| :--- | :--- | :--- | :--- |
| `s` | task name | **Source task** — the task whose top-K detected heads we ablate. | `ceo_lastname` |
| `t` | task name | **Target task** — the task whose accuracy we *measure* after the ablation. | `employees_count_total` |
| `K` | int ≥ 0 | **Knockout size** — how many top-ranked heads we ablate. `K=0` = unablated baseline. | `16` |
| `accuracy[s][t][K]` | float in [0,1] | Accuracy on target `t`'s test set when source `s`'s top-K heads are ablated. | `0.792` |
| `drop[s][t][K]` | float in [-1,1] | Accuracy lost vs the unablated baseline on `t`. Positive = the ablation hurt `t`. | `0.125` |

> When `s == t` the model is being asked to do task `t` with `t`'s own detected heads removed — this is the **diagonal** of the source × target heatmaps.
> When `s != t` the model is doing task `t` with some *other* task's heads removed — collateral damage, **off-diagonal**.

### Metrics computed in this report

| Metric | Formula | What it asks |
| :--- | :--- | :--- |
| `drop[s][t][K]` | `accuracy[s][t][K=0] − accuracy[s][t][K]` | How much does ablating `s`'s top-K heads hurt task `t`? |
| **`sensitivity[t][K]`** | mean over **all sources `s`** of `drop[s][t][K]` (8 sources, **including** `s = t`) | **How fragile is target task `t` overall?** Averages the whole `t`-th column of the 8×8 drop heatmap. |
| `on_target_drop[t][K]` | `drop[t][t][K]`  *(single diagonal cell)* | How much does task `t` suffer when its **own** heads are removed? |
| `off_target_mean[t][K]` | mean over `s ≠ t` of `drop[s][t][K]` (7 sources) | How much does task `t` suffer purely from **other tasks'** ablations (collateral)? |

### Worked example (real numbers, recomputed at runtime)

Take **`t = employees_count_total`**, **`K = 16`**, **model = Mistral-7B-Instruct-v0.3**. The 8 cells of that drop *column* in the cross-task matrix are:

| Source `s` | `drop[s][t][K]` | Note |
| :--- | ---: | :--- |
| `ceo_lastname` | 0.3750 |  |
| `employees_count_total` | 0.8333 | ← `on_target_drop` (s == t, diagonal) |
| `headquarters_city` | 0.8750 |  |
| `headquarters_state` | 0.8750 |  |
| `holder_record_amount` | 0.2500 |  |
| `incorporation_state` | 0.8333 |  |
| `incorporation_year` | 0.8750 |  |
| `registrant_name` | 0.8333 |  |

So for that (model, target, K):

- `sensitivity      = mean of all 8`              = **0.7188**
- `on_target_drop   = drop[t][t][K]`              = **0.8333**
- `off_target_mean  = mean of the other 7 (s ≠ t)` = **0.7024**

> **`sensitivity` is the column average of the cross-task drop heatmap; `on_target_drop` is its diagonal; `off_target_mean` is the column average with the diagonal cell removed.**

### Why this is *target-centric* (and how it differs from the existing source-centric `specificity_index`)

The repo's `cross_task_specificity_metrics.json` and `FINDINGS.md` use a **source-centric** view (one row of the 8×8 heatmap per source `s`):

```text
source-centric on_target_drop[s]  = drop[s][s][K]                        (same diagonal)
source-centric off_target_mean[s] = mean over t != s of drop[s][t][K]    (row mean, no diagonal)
specificity_index[s]              = on_target - off_target               (row metric)
```

This report uses a **target-centric** view (one *column* of the same heatmap per target `t`):

```text
target-centric on_target_drop[t]  = drop[t][t][K]                        (same diagonal)
target-centric off_target_mean[t] = mean over s != t of drop[s][t][K]    (column mean, no diagonal)
sensitivity[t]                    = mean over ALL s of drop[s][t][K]     (column mean, with diagonal)
```

Both views read from the **same 8 × 8 matrix**; they just slice it along different axes. The diagonal `drop[t][t][K]` is identical between the two views; everything else is different.

All numbers below are derived directly from `cross_task_transfer_matrix.json` and cross-validated against the existing source-centric `cross_task_specificity_metrics.json` / `specificity_table.csv` artifacts (see *Data provenance & validation* below).

## Setup

**Models** (short label → full HuggingFace name):

| Short | Model |
| :--- | :--- |
| Llama | `Llama-3.1-8B-Instruct` |
| Qwen | `Qwen2.5-7B-Instruct` |
| Mistral | `Mistral-7B-Instruct-v0.3` |

| Field | Value |
| :--- | :--- |
| Tasks | 8 — `ceo_lastname`, `employees_count_total`, `headquarters_city`, `headquarters_state`, `holder_record_amount`, `incorporation_state`, `incorporation_year`, `registrant_name` |
| K values | 8, 16, 32, 48, 64, 96, 128 |

## Data provenance & validation

| Model | Matrix path | Git ref | Validated against | Status |
| :--- | :--- | :--- | :--- | :--- |
| Llama-3.1-8B-Instruct | `results/comparison_ablation/cross_task_transfer_matrix.json` | `origin/main` | `specificity_table.csv` (K=16) | OK |
| Qwen2.5-7B-Instruct | `results/runs/qwen_true_detect_2026-04-07/cross_task_transfer_matrix.json` | _working tree_ | `cross_task_specificity_metrics.json` (K=16) | OK |
| Mistral-7B-Instruct-v0.3 | `results/comparison_ablation_mistral/cross_task_transfer_matrix.json` | _working tree_ | `cross_task_specificity_metrics.json` (K=16) | OK |

## Per-K target sensitivity

_Each row = one target task. Cells = mean drop in accuracy on that target when each model's top-K heads (detected for any of the 8 source tasks) are ablated. Tasks sorted by `Mean` column; bold row = most-fragile task at this K._

### K = 8

| Rank | Target task | Llama | Qwen | Mistral | Mean |
| :--- | ---: | ---: | ---: | ---: | ---: |
| **1** | **employees_count_total** | **0.469** | **0.682** | **0.635** | **0.595** |
| 2 | holder_record_amount | 0.448 | 0.432 | 0.245 | 0.375 |
| 3 | incorporation_year | 0.229 | 0.517 | 0.365 | 0.370 |
| 4 | incorporation_state | 0.255 | 0.438 | 0.396 | 0.363 |
| 5 | ceo_lastname | 0.562 | 0.118 | 0.219 | 0.300 |
| 6 | headquarters_city | 0.344 | 0.357 | 0.115 | 0.272 |
| 7 | headquarters_state | 0.120 | 0.278 | 0.146 | 0.181 |
| 8 | registrant_name | 0.120 | 0.095 | 0.036 | 0.084 |

### K = 16

| Rank | Target task | Llama | Qwen | Mistral | Mean |
| :--- | ---: | ---: | ---: | ---: | ---: |
| **1** | **employees_count_total** | **0.812** | **0.833** | **0.719** | **0.788** |
| 2 | incorporation_year | 0.542 | 0.679 | 0.505 | 0.575 |
| 3 | ceo_lastname | 0.865 | 0.425 | 0.396 | 0.562 |
| 4 | incorporation_state | 0.359 | 0.586 | 0.458 | 0.468 |
| 5 | holder_record_amount | 0.536 | 0.479 | 0.250 | 0.422 |
| 6 | headquarters_city | 0.531 | 0.540 | 0.130 | 0.401 |
| 7 | headquarters_state | 0.203 | 0.366 | 0.198 | 0.256 |
| 8 | registrant_name | 0.266 | 0.107 | 0.036 | 0.136 |

### K = 32

| Rank | Target task | Llama | Qwen | Mistral | Mean |
| :--- | ---: | ---: | ---: | ---: | ---: |
| **1** | **employees_count_total** | **0.917** | **0.927** | **0.849** | **0.898** |
| 2 | ceo_lastname | 0.917 | 0.607 | 0.812 | 0.779 |
| 3 | incorporation_year | 0.589 | 0.812 | 0.719 | 0.707 |
| 4 | incorporation_state | 0.453 | 0.789 | 0.625 | 0.622 |
| 5 | headquarters_city | 0.531 | 0.741 | 0.443 | 0.572 |
| 6 | holder_record_amount | 0.552 | 0.531 | 0.250 | 0.444 |
| 7 | headquarters_state | 0.354 | 0.606 | 0.354 | 0.438 |
| 8 | registrant_name | 0.391 | 0.220 | 0.323 | 0.311 |

### K = 48

| Rank | Target task | Llama | Qwen | Mistral | Mean |
| :--- | ---: | ---: | ---: | ---: | ---: |
| **1** | **employees_count_total** | **0.927** | **0.938** | **0.849** | **0.905** |
| 2 | ceo_lastname | 0.901 | 0.857 | 0.870 | 0.876 |
| 3 | incorporation_year | 0.594 | 0.867 | 0.760 | 0.740 |
| 4 | incorporation_state | 0.484 | 0.895 | 0.641 | 0.673 |
| 5 | headquarters_city | 0.526 | 0.871 | 0.448 | 0.615 |
| 6 | registrant_name | 0.578 | 0.506 | 0.516 | 0.533 |
| 7 | headquarters_state | 0.370 | 0.745 | 0.339 | 0.485 |
| 8 | holder_record_amount | 0.568 | 0.542 | 0.245 | 0.451 |

### K = 64

| Rank | Target task | Llama | Qwen | Mistral | Mean |
| :--- | ---: | ---: | ---: | ---: | ---: |
| **1** | **employees_count_total** | **0.927** | **0.927** | **0.870** | **0.908** |
| 2 | ceo_lastname | 0.917 | 0.907 | 0.833 | 0.886 |
| 3 | incorporation_year | 0.698 | 0.867 | 0.729 | 0.765 |
| 4 | incorporation_state | 0.500 | 0.922 | 0.682 | 0.701 |
| 5 | registrant_name | 0.620 | 0.814 | 0.536 | 0.657 |
| 6 | headquarters_city | 0.557 | 0.888 | 0.484 | 0.643 |
| 7 | holder_record_amount | 0.583 | 0.536 | 0.250 | 0.457 |
| 8 | headquarters_state | 0.276 | 0.764 | 0.307 | 0.449 |

### K = 96

| Rank | Target task | Llama | Qwen | Mistral | Mean |
| :--- | ---: | ---: | ---: | ---: | ---: |
| **1** | **ceo_lastname** | **0.911** | **0.929** | **0.875** | **0.905** |
| 2 | employees_count_total | 0.953 | 0.844 | 0.875 | 0.891 |
| 3 | registrant_name | 0.943 | 0.939 | 0.755 | 0.879 |
| 4 | incorporation_year | 0.781 | 0.858 | 0.755 | 0.798 |
| 5 | incorporation_state | 0.615 | 0.879 | 0.740 | 0.744 |
| 6 | headquarters_city | 0.672 | 0.915 | 0.589 | 0.725 |
| 7 | headquarters_state | 0.380 | 0.829 | 0.359 | 0.523 |
| 8 | holder_record_amount | 0.630 | 0.510 | 0.240 | 0.460 |

### K = 128

| Rank | Target task | Llama | Qwen | Mistral | Mean |
| :--- | ---: | ---: | ---: | ---: | ---: |
| **1** | **ceo_lastname** | **0.917** | **0.939** | **0.875** | **0.910** |
| 2 | registrant_name | 0.984 | 0.970 | 0.766 | 0.907 |
| 3 | employees_count_total | 0.953 | 0.891 | 0.875 | 0.906 |
| 4 | incorporation_state | 0.781 | 0.871 | 0.839 | 0.830 |
| 5 | headquarters_city | 0.740 | 0.893 | 0.854 | 0.829 |
| 6 | incorporation_year | 0.812 | 0.858 | 0.792 | 0.821 |
| 7 | headquarters_state | 0.500 | 0.736 | 0.656 | 0.631 |
| 8 | holder_record_amount | 0.635 | 0.484 | 0.229 | 0.450 |

## Aggregate sensitivity across all K

_Mean over K in {8, 16, 32, 48, 64, 96, 128}._

| Rank | Target task | Llama | Qwen | Mistral | Mean |
| :--- | ---: | ---: | ---: | ---: | ---: |
| **1** | **employees_count_total** | **0.851** | **0.863** | **0.810** | **0.842** |
| 2 | ceo_lastname | 0.856 | 0.683 | 0.697 | 0.745 |
| 3 | incorporation_year | 0.606 | 0.780 | 0.661 | 0.682 |
| 4 | incorporation_state | 0.493 | 0.768 | 0.626 | 0.629 |
| 5 | headquarters_city | 0.557 | 0.744 | 0.438 | 0.579 |
| 6 | registrant_name | 0.557 | 0.521 | 0.424 | 0.501 |
| 7 | holder_record_amount | 0.565 | 0.502 | 0.244 | 0.437 |
| 8 | headquarters_state | 0.315 | 0.618 | 0.337 | 0.423 |

## Early sensitivity (low-K collapse)

_Mean over K in {8, 16}._

| Rank | Target task | Llama | Qwen | Mistral | Mean |
| :--- | ---: | ---: | ---: | ---: | ---: |
| **1** | **employees_count_total** | **0.641** | **0.758** | **0.677** | **0.692** |
| 2 | incorporation_year | 0.385 | 0.598 | 0.435 | 0.473 |
| 3 | ceo_lastname | 0.714 | 0.271 | 0.307 | 0.431 |
| 4 | incorporation_state | 0.307 | 0.512 | 0.427 | 0.415 |
| 5 | holder_record_amount | 0.492 | 0.456 | 0.247 | 0.398 |
| 6 | headquarters_city | 0.438 | 0.449 | 0.122 | 0.336 |
| 7 | headquarters_state | 0.161 | 0.322 | 0.172 | 0.218 |
| 8 | registrant_name | 0.193 | 0.101 | 0.036 | 0.110 |

## Sensitivity rank per model

_Rank 1 = most sensitive target task for that (model, K). Each column is independent — rows are not sorted, just listed alphabetically._

### K = 8

| Target task | Llama | Qwen | Mistral |
| :--- | ---: | ---: | ---: |
| ceo_lastname | 1 | 7 | 5 |
| employees_count_total | 2 | 1 | 1 |
| headquarters_city | 4 | 5 | 7 |
| headquarters_state | 7 | 6 | 6 |
| holder_record_amount | 3 | 4 | 4 |
| incorporation_state | 5 | 3 | 2 |
| incorporation_year | 6 | 2 | 3 |
| registrant_name | 8 | 8 | 8 |

### K = 16

| Target task | Llama | Qwen | Mistral |
| :--- | ---: | ---: | ---: |
| ceo_lastname | 1 | 6 | 4 |
| employees_count_total | 2 | 1 | 1 |
| headquarters_city | 5 | 4 | 7 |
| headquarters_state | 8 | 7 | 6 |
| holder_record_amount | 4 | 5 | 5 |
| incorporation_state | 6 | 3 | 3 |
| incorporation_year | 3 | 2 | 2 |
| registrant_name | 7 | 8 | 8 |

### K = 32

| Target task | Llama | Qwen | Mistral |
| :--- | ---: | ---: | ---: |
| ceo_lastname | 2 | 5 | 2 |
| employees_count_total | 1 | 1 | 1 |
| headquarters_city | 5 | 4 | 5 |
| headquarters_state | 8 | 6 | 6 |
| holder_record_amount | 4 | 7 | 8 |
| incorporation_state | 6 | 3 | 4 |
| incorporation_year | 3 | 2 | 3 |
| registrant_name | 7 | 8 | 7 |

### K = 48

| Target task | Llama | Qwen | Mistral |
| :--- | ---: | ---: | ---: |
| ceo_lastname | 2 | 5 | 1 |
| employees_count_total | 1 | 1 | 2 |
| headquarters_city | 6 | 3 | 6 |
| headquarters_state | 8 | 6 | 7 |
| holder_record_amount | 5 | 7 | 8 |
| incorporation_state | 7 | 2 | 4 |
| incorporation_year | 3 | 4 | 3 |
| registrant_name | 4 | 8 | 5 |

### K = 64

| Target task | Llama | Qwen | Mistral |
| :--- | ---: | ---: | ---: |
| ceo_lastname | 2 | 3 | 2 |
| employees_count_total | 1 | 1 | 1 |
| headquarters_city | 6 | 4 | 6 |
| headquarters_state | 8 | 7 | 7 |
| holder_record_amount | 5 | 8 | 8 |
| incorporation_state | 7 | 2 | 4 |
| incorporation_year | 3 | 5 | 3 |
| registrant_name | 4 | 6 | 5 |

### K = 96

| Target task | Llama | Qwen | Mistral |
| :--- | ---: | ---: | ---: |
| ceo_lastname | 3 | 2 | 1 |
| employees_count_total | 1 | 6 | 2 |
| headquarters_city | 5 | 3 | 6 |
| headquarters_state | 8 | 7 | 7 |
| holder_record_amount | 6 | 8 | 8 |
| incorporation_state | 7 | 4 | 5 |
| incorporation_year | 4 | 5 | 3 |
| registrant_name | 2 | 1 | 4 |

### K = 128

| Target task | Llama | Qwen | Mistral |
| :--- | ---: | ---: | ---: |
| ceo_lastname | 3 | 2 | 1 |
| employees_count_total | 2 | 4 | 2 |
| headquarters_city | 6 | 3 | 3 |
| headquarters_state | 8 | 7 | 7 |
| holder_record_amount | 7 | 8 | 8 |
| incorporation_state | 5 | 5 | 4 |
| incorporation_year | 4 | 6 | 5 |
| registrant_name | 1 | 1 | 6 |

## Consistency across (K, model) cells

_How often each task is in the top-N most sensitive across all 7 K-values × 3 models = **21** cells._

| Target task | Top-2 / 21 | Top-4 / 21 |
| :--- | ---: | ---: |
| **employees_count_total** | **19** | **20** |
| ceo_lastname | 12 | 16 |
| incorporation_year | 4 | 16 |
| incorporation_state | 3 | 12 |
| headquarters_city | 0 | 8 |
| registrant_name | 4 | 7 |
| holder_record_amount | 0 | 5 |
| headquarters_state | 0 | 0 |
