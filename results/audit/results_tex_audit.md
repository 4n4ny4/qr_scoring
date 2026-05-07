# `results.tex` audit — quantitative claims vs canonical data

**Date**: 2026-05-05
**Audit target**: `results.tex` (168 lines, 5 subsections, Table 1, 2 figures)
**Auditor's scope**: Verify every numerical claim in the manuscript against the actual experimental artifacts in this repo, pulled from whichever git branch holds the canonical version of each artifact.

---

## tl;dr

| Section | Status | Required action |
|---|---|---|
| §4.1 — common retrieval-head pool | **❌ K is wrong** (says K=64, data is K=128 with confounds excluded); definitions ✓ | Change `K=64` → `K=128`, add "excluding confound pairs" |
| §4.2 — sensitivity is conserved | ✓ all 7 numbers correct | None |
| §4.3 — efficacy not conserved (Table 1) | ✓ R² values all match within ±0.005; ⚠️ "5–10×" framing is wrong | Change "5–10×" → "4–12×" |
| §4.3 — `holder_record_amount` outlier | ✓ baselines correct; ⚠️ Qwen at exact threshold | Loosen `≥0.54` → `≥0.5` |
| §4.4 — high overlap pairs | ❌ "50–90% average" not supported by any K; "75–95% pairs" needs K specified | Restate as range; specify K=128; tighten to 80–95% |
| §4.5 — cross-dataset transfer | ✓ all 5 K=16 numbers correct, both CIs correct | None |

---

## Provenance — which file came from which branch

Before computing anything, I located the canonical source for every artifact the LaTeX claims depend on.

| Artifact | Path | Source branch | Note |
|---|---|---|---|
| Llama cross-task transfer matrix | `results/comparison_ablation/cross_task_transfer_matrix.json` | `origin/main` | Numerically identical (md5 matches at all 512 cells) to `origin/ananya/cross_ablations` and `origin/ananya/gamma_olmo`. Differs in *formatting only* from `origin/ananya/icml2026_submission_cleanup` (same numerical content, different key ordering). |
| Qwen cross-task transfer matrix | `results/runs/qwen_true_detect_2026-04-07/cross_task_transfer_matrix.json` | `mistral_exp` (working tree) | Numerically identical to `origin/ananya/icml2026_submission_cleanup:results/comparison_ablation/Qwen__Qwen2.5-7B-Instruct/...` |
| Mistral cross-task transfer matrix | `results/comparison_ablation_mistral/cross_task_transfer_matrix.json` | `mistral_exp` (working tree) | Only one location; produced by the Mistral pipeline run earlier in this branch. |
| Llama per-task head rankings | `results/detection/long_context_<task>_heads.json` and `results/detection/topk/long_context_<task>_top<K>.json` | `origin/main` | 5 of 8 tasks have only the `_heads.json` (full ranking); 3 have only the `_top<K>.json` (pre-truncated). The Jaccard loader tries both. |
| Qwen per-task head rankings | `results/detection_qwen/long_context_<task>_heads.json` | `mistral_exp` | All 8 tasks present as full rankings. |
| Mistral per-task head rankings | `results/detection_mistral/long_context_<task>_heads.json` | `mistral_exp` | All 8 tasks present as full rankings. |
| Confidence intervals (Llama) | `results/comparison_ablation/confidence_intervals.csv` | `origin/main` | Source for §4.5 K=16 / K=32 / K=0 numbers; produced by `scripts/evaluation/compute_confidence_intervals.py` with n=192, 10k bootstrap iterations. |
| Source-centric specificity metrics (Qwen, Mistral) | `cross_task_specificity_metrics.json` next to each transfer matrix | working tree | Used as the cross-validation anchor — derived target-centric metrics must agree with these on the diagonal. |
| Source-centric specificity table (Llama) | `results/comparison_ablation/specificity_table.csv` | `origin/main` | Same purpose as the JSON above (Llama's was committed in CSV form). |

**How I confirmed each branch is authoritative**: For every artifact I pulled with `git show <ref>:<path>`, I parsed it as JSON/CSV and asserted that:

1. Schema matches the others (same `sources` / `targets` lists, same K grid, same column headers).
2. Numerical content is consistent across alternate locations (md5 of body, or per-cell diff).
3. K=0 baseline is constant across sources (catches accidental schema drift).

Only branches that passed all three checks were used.

---

## Tooling I used / built for the audit

All scripts live in `scripts/evaluation/`. The Jaccard and K-FE scripts didn't exist before the audit; I wrote them specifically to make the verification reproducible.

| Script | Purpose | Key import |
|---|---|---|
| `compute_target_sensitivity.py` | Derives target-centric metrics from any cross-task matrix; cross-validates against existing source-centric specificity files. | — |
| `plot_target_sensitivity.py` | 3-panel sensitivity heatmap. | `compute_target_metrics`, `load_matrix` |
| `plot_jaccard_appendix.py` | Loads per-task top-K head sets across branches; computes 8×8 Jaccard matrix per model; emits PNG + markdown summary. | `SHORT_LABELS`, `short_label` |
| `compute_kfe_correlations.py` | K-residualized cross-model R² for sensitivity and four candidate efficacy definitions; auto-flags the definition that best matches Table 1. | `compute_target_metrics`, `load_matrix` |

The reason I share a loader (`load_matrix` in `compute_target_sensitivity.py`) across all of these is so the figure, the K-FE table, and the audit numbers are physically the same code path — they can't drift.

---

## Assumptions I made (state explicitly)

1. **`s == t` is included in `sensitivity[t][K]`**. The script averages across all 8 sources including the diagonal (matching the user's original snippet). The companion target-centric metric that excludes the diagonal is also computed (`off_target_mean[t][K]`); both are stored in `target_sensitivity_long.csv`. The LaTeX uses the include-diagonal version (verified by spot-checking the aggregate sensitivity numbers in §4.2 against the report).

2. **K=0 is excluded from K-FE correlations**. By construction every cell at K=0 is `drop_from_k0 = 0`, so the residual is zero for every model and any correlation is undefined. The script's `--ks` default is `8,16,32,48,64,96,128` for this reason.

3. **Pearson R²**, not Spearman or partial. The LaTeX caption says "K-fixed-effects R²" without specifying. Pearson on K-residualized vectors is the natural reading and reproduces the published values within ±0.005 — that's strong evidence it's the right definition.

4. **Efficacy = `off_target_mean`** is what the prior agent used. I verified this by computing R² for four candidate definitions (`on_target_drop`, `row_mean_drop`, `off_target_mean`, `specificity_index`) and picking the one with smallest total |Δ| vs the published Table 1. `off_target_mean` wins by an order of magnitude over the next-best (`row_mean_drop`, total |Δ|=0.127 vs winner 0.007). See §"K-FE verification" below for the full deviation table.

5. **Top-K head sets are the first K entries of the sorted ranking.** Both `_heads.json` and `_top<K>.json` are sorted descending by score. For files where only `_heads.json` exists, slicing `raw[:K]` gives the same set as the pre-truncated snapshot would. I verified this by spot-checking pairs where both files exist on `origin/main` — the slice-of-`_heads.json` set equals the `_top<K>.json` set.

6. **"Confound pairs" = `(headquarters_city, headquarters_state)` and `(incorporation_state, incorporation_year)`** because those are the two pairs §4.4 explicitly flags. When I report "no-confound" statistics I exclude both unordered pairs (4 ordered pairs total) from the off-diagonal pool.

7. **Branch authority**: when an artifact existed on multiple branches, I preferred `origin/main` for Llama and the working-tree (`mistral_exp`) for Qwen/Mistral. Numerical equivalence to other branches was always verified before relying on `origin/main`.

---

## §4.1 — "Recall tasks share a common retrieval-head pool"

### Claims
> *"Within each model, these sets substantially overlap across tasks: pairwise top-$K=64$ Jaccard similarity ranges from 31--80\% in Llama, 44--83\% in Qwen, and 59--77\% in Mistral."*

> *"target sensitivity (the column mean: a target's mean drop across all source-task ablations, including its own) and source efficacy (the off-diagonal row mean: a source's mean drop on the seven other target tasks)."*

### Verification — definitions ✓
The two definitions match the actual code in `compute_target_sensitivity.py:155-165` and `compute_kfe_correlations.py:build_efficacy_vector(definition="off_target_mean")` exactly. No issue.

### Verification — Jaccard ranges ❌

I exhaustively swept K ∈ {8, 16, 32, 48, 64, 96, 128}, computed pairwise off-diagonal Jaccard for the 28 unordered task pairs, and reported `(min, p25, p50, p75, max)` per (K, model). Then I repeated with the two confound pairs excluded.

**Code I ran:**

```python
import json, subprocess, numpy as np
from itertools import combinations

def load_heads(spec_list, task, k, ref):
    """Return top-K (layer, head) set, trying each candidate path."""
    for tmpl in spec_list:
        path = tmpl.format(task=task, k=k)
        try:
            if ref:
                b = subprocess.check_output(
                    ["git","show",f"{ref}:{path}"], stderr=subprocess.DEVNULL)
            else:
                b = open(path,"rb").read()
        except Exception:
            continue
        raw = json.loads(b)
        return {(int(it[0].split("-")[0]), int(it[0].split("-")[1])) for it in raw[:k]}
    raise FileNotFoundError(f"No path for {task} K={k}")

models = {
    "Llama":   (["results/detection/long_context_{task}_heads.json",
                 "results/detection/topk/long_context_{task}_top{k}.json"], "origin/main"),
    "Qwen":    (["results/detection_qwen/long_context_{task}_heads.json"], None),
    "Mistral": (["results/detection_mistral/long_context_{task}_heads.json"], None),
}
tasks = [...]  # the 8 SEC tasks
CONFOUNDS = {("headquarters_city","headquarters_state"),
             ("incorporation_state","incorporation_year")}

for k in [8,16,32,48,64,96,128]:
    head_sets = {label:{t:load_heads(spec,t,k,ref) for t in tasks}
                 for label,(spec,ref) in models.items()}
    for label in models:
        all_pairs, no_confound = [], []
        for a,b in combinations(tasks,2):
            j = len(head_sets[label][a] & head_sets[label][b]) / \
                len(head_sets[label][a] | head_sets[label][b])
            all_pairs.append(j)
            if (a,b) not in CONFOUNDS and (b,a) not in CONFOUNDS:
                no_confound.append(j)
        # Compare min/max against PAPER ranges...
```

**Result table** (compact):

| K | Model | All-pairs (min, max) | No-confound (min, max) | **Paper claim** |
|---:|---|---|---|---|
| 64 | Llama | 0.219, 0.910 | 0.219, 0.778 | 0.31, 0.80 |
| 64 | Qwen | 0.267, 0.882 | 0.267, 0.684 | 0.44, 0.83 |
| 64 | Mistral | 0.471, 0.910 | 0.471, 0.684 | 0.59, 0.77 |
| **128** | **Llama** | 0.313, 0.855 | **0.313, 0.803** | **0.31, 0.80** ✓ |
| **128** | **Qwen** | 0.438, 0.925 | **0.438, 0.829** | **0.44, 0.83** ✓ |
| **128** | **Mistral** | 0.590, 0.954 | **0.590, 0.766** | **0.59, 0.77** ✓ |

Six bounds match exactly at **K=128 with the two confound pairs excluded**. None of the K=64 configurations match. This is a real factual error in the paper: the claim says K=64 but the data is K=128.

### Medians (for the user's "median X% across pairs" request)

| K | Model | Median (all-pairs) | Median (no-confound) |
|---:|---|---:|---:|
| 64 | Llama | 0.45 | 0.44 |
| 64 | Qwen | 0.44 | 0.44 |
| 64 | Mistral | 0.55 | 0.54 |
| **128** | **Llama** | 0.49 | **0.48** |
| **128** | **Qwen** | 0.62 | **0.61** |
| **128** | **Mistral** | 0.66 | **0.65** |

So the addition: *"with median 48–65% across pairs"* (using K=128, no-confound).

### Recommended fix to §4.1

```latex
% Replace lines 9-11 with:
pairwise top-$K{=}128$ Jaccard similarity (excluding the two
detection-protocol confound pairs identified in
Section~\ref{sec:result-overlap}) ranges from 31\%--80\% in Llama,
44\%--83\% in Qwen, and 59\%--77\% in Mistral, with median
48\%--65\% across pairs.
```

---

## §4.2 — "Target-task sensitivity is conserved across model families"

### Claims
- Top-4 most sensitive tasks (aggregated across models and K): `employees_count_total` 0.842, `ceo_lastname` 0.745, `incorporation_year` 0.682, `incorporation_state` 0.629
- Bottom-4 within drop range 0.42–0.58
- `employees_count_total` in top-2 sensitivity in 19/21 (K, model) cells
- `headquarters_state` in top-4 sensitivity in 0/21 cells

### Verification ✓

All seven values are independently regenerated each run by `compute_target_sensitivity.py`:

```bash
python scripts/evaluation/compute_target_sensitivity.py
# writes results/target_sensitivity/target_sensitivity_report.md
```

Comparing the report against the LaTeX:

| Claim | Derived value | Source |
|---|---:|---|
| `employees_count_total` agg sensitivity | **0.842** | `target_sensitivity_report.md` "Aggregate sensitivity" table |
| `ceo_lastname` agg | **0.745** | same |
| `incorporation_year` agg | **0.682** | same |
| `incorporation_state` agg | **0.629** | same |
| Bottom-4 sensitivities | **0.423, 0.437, 0.501, 0.579** (range 0.42–0.58 ✓) | same |
| `employees_count_total` top-2 cells | **19/21** | "Consistency" table |
| `headquarters_state` top-4 cells | **0/21** | "Consistency" table |

Cross-validation passes for all 3 models against their existing `cross_task_specificity_metrics.json` / `specificity_table.csv` (the script aborts if any cell disagrees beyond ±1e-6 for JSON / ±5e-4 for CSV).

**No change required to §4.2.**

---

## §4.3 — "Source-head efficacy is not conserved"

### Claims
- Table 1 K-FE R² values: Llama–Qwen 0.18/0.02, Llama–Mistral 0.47/0.04, Qwen–Mistral 0.59/0.15
- *"target sensitivity is 5–10× more conserved than source efficacy"*
- Sensitivity correlates 0.18–0.59; efficacy correlates 0.02–0.15

### Verification — Table 1 ✓

The K-FE protocol from the LaTeX caption: *"computed by residualizing each measure against within-K task means before correlation."*

I implemented this in `compute_kfe_correlations.py`:

```python
def k_residualize(arr):  # arr shape: (|tasks|, |K|)
    return arr - arr.mean(axis=0, keepdims=True)

def pearson_r2(a, b):
    a, b = a.ravel() - a.mean(), b.ravel() - b.mean()
    return float((a * b).sum() / np.sqrt((a**2).sum() * (b**2).sum()))**2
```

For sensitivity the formula is unambiguous (column-mean drop, target-centric). For efficacy I tested four plausible source-centric definitions because the original code wasn't preserved anywhere in the repo:

| Efficacy definition | Total |Δ| vs Table 1 | Best fit? |
|---|---:|---|
| `on_target_drop` = `drop[s][s][K]` | 0.456 | no |
| `row_mean_drop` = mean over t of `drop[s][t][K]` | 0.127 | no |
| **`off_target_mean` = mean over t≠s of `drop[s][t][K]`** | **0.007** | ✓ |
| `specificity_index` = `on - off_mean` | 0.823 | no |

`off_target_mean` wins by ~18× over the next-best definition. Per-pair derived values:

| Pair | Sensitivity (derived → Table 1) | Efficacy (derived → Table 1) |
|---|---|---|
| Llama–Qwen | **0.184** → 0.18 | **0.018** → 0.02 |
| Llama–Mistral | **0.472** → 0.47 | **0.039** → 0.04 |
| Qwen–Mistral | **0.593** → 0.59 | **0.153** → 0.15 |

All six values match within ±0.005. Table 1 is **correct as printed**.

### Verification — "5–10× more conserved" ⚠️

Computed ratios:

| Pair | Sens R² | Eff R² | Ratio |
|---|---:|---:|---:|
| Llama–Qwen | 0.184 | 0.018 | **10.2×** |
| Llama–Mistral | 0.472 | 0.039 | **12.1×** |
| Qwen–Mistral | 0.593 | 0.153 | **3.9×** |

The Qwen–Mistral ratio is below 5×. The accurate range is **4–12×**. Recommended fix:

```latex
% line 49-50 — change "5--10\times" to:
target sensitivity is $4$--$12\times$ more conserved than source
efficacy (Table~\ref{tab:kfe}).
```

Or, if you want to weight by the strongest evidence (which is what readers will remember anyway):

```latex
target sensitivity is at least $4\times$ more conserved than source
efficacy and reaches $12\times$ for the most distant pair
(Llama--Mistral) (Table~\ref{tab:kfe}).
```

### Verification — sensitivity 0.18–0.59, efficacy 0.02–0.15 ✓

Direct read-off of the per-pair table above: 0.184–0.593 → "0.18–0.59" ✓; 0.018–0.153 → "0.02–0.15" ✓.

### `holder_record_amount` outlier paragraph (line 89–100)

**Claims:**
- Mistral baseline = 0.25
- Llama and Qwen baselines ≥ 0.54

**Verification — actual K=0 accuracies per task per model** (read from `cross_task_transfer_matrix.json["results"][s][t]["by_k"]["0"]["accuracy"]`, which is the same for all `s` for fixed `t`):

| Task | Llama | Qwen | Mistral |
|---|---:|---:|---:|
| ceo_lastname | 0.917 | 0.943 | 0.875 |
| employees_count_total | 0.958 | 0.958 | 0.875 |
| headquarters_city | 0.958 | 0.929 | 1.000 |
| headquarters_state | 0.833 | 0.852 | 0.958 |
| **holder_record_amount** | **0.792** | **0.542** | **0.250** |
| incorporation_state | 1.000 | 0.969 | 0.958 |
| incorporation_year | 0.833 | 0.867 | 0.792 |
| registrant_name | 1.000 | 0.976 | 0.917 |

- Llama 0.792 ≥ 0.54 ✓
- Qwen **0.542** ≥ 0.54 — passes by 0.002 (i.e. exactly 0.5417 rounded to 0.54)
- Mistral 0.250 ✓

The Qwen value is sitting at the threshold. Not a bug, but if anything in the data ever shifts, the inequality flips. Recommended fix for robustness:

```latex
% line 99-100 — change "\geq 0.54" to:
The same low-sensitivity pattern holds in Llama and Qwen, where the
baseline is $\geq 0.5$, so the outlier status is not solely a
baseline artifact.
```

---

## §4.4 — "High overlap pairs reflect detection-protocol confounds"

### Claims
- *"the average top-K Jaccard overlap among head sets across the eight tasks is between 50 and 90 percent"*
- *"two task pairs have a top-K overlap of 75 to 95 percent: HQ_city↔HQ_state, inc_state↔inc_year"*

### Verification — "average ... 50–90%" ❌

The K isn't specified. At every K I tested, the **mean** off-diagonal Jaccard is much narrower than 50–90%:

| K | Llama mean | Qwen mean | Mistral mean |
|---:|---:|---:|---:|
| 16 | 0.443 | 0.479 | 0.539 |
| 64 | 0.514 | 0.491 | 0.586 |
| 128 | 0.567 | 0.654 | 0.677 |

The means are 44–68% across (K, model) — **never reaches 90%**. The "50–90%" claim is not supported as a statement about the average.

It could be defensible as a **range statement** (the value of any given pair falls in [low, high]). At K=128:

| Model | Range across pairs |
|---|---|
| Llama | 31–86% |
| Qwen | 44–93% |
| Mistral | 59–95% |

Across all three models combined: 31–95%. So the actual range is wider than 50–90 on the low end and as wide on the high end.

### Verification — "75–95% for two pairs" ⚠️

Without K specified, the claim's truth depends on K. At K=128, both pairs exceed 80% in all three models; at K=64, the Llama HQ pair drops to 68% and the Llama inc pair was at 91%; at K=16, the Llama inc pair is only 60%.

**Per-K, per-model values for the two flagged pairs:**

| Pair | K | Llama | Qwen | Mistral |
|---|---:|---:|---:|---:|
| HQ_city ↔ HQ_state | 16 | 0.78 | 0.88 | 0.88 |
| HQ_city ↔ HQ_state | 64 | 0.68 | 0.88 | 0.88 |
| HQ_city ↔ HQ_state | **128** | **0.86** | **0.93** | **0.95** |
| inc_state ↔ inc_year | 16 | 0.60 | 0.78 | 0.88 |
| inc_state ↔ inc_year | 64 | 0.91 | 0.83 | 0.91 |
| inc_state ↔ inc_year | **128** | **0.80** | **0.83** | **0.95** |

K=128 is the only value where **both pairs exceed 80% across all three models** — uniformly true claim. The current "75–95%" range needs to be tightened to **80–95%** at K=128.

### Recommended fix to §4.4

```latex
% Replace lines 105-117 with:
While the off-diagonal pairwise top-$K{=}128$ Jaccard overlap across
the eight tasks ranges from 31\% to 95\% across models (per-model
means 56\%--68\%), two task pairs exceed 80\%--95\% overlap:
\texttt{headquarters\_city} with \texttt{headquarters\_state}, and
\texttt{incorporation\_state} with \texttt{incorporation\_year}. In
each of these cases, the two tasks share the same evidence sentences
(for example, ``headquartered in Malvern, Pennsylvania'' contains
the answer to both \texttt{headquarters\_city} and
\texttt{headquarters\_state}). The high observed overlap therefore
reflects the structure of the detection setup, in which QRScore
converges on the same heads when the evidence span is identical,
rather than any sharing of head sets within the model itself. We
flag these pairs as confounds in the overlap analysis; our
sensitivity claims, which are derived from column-mean structure
across all eight tasks, are unaffected.
```

This unifies the K choice with §4.1 (also K=128) and ensures both quantitative claims in §4.4 are uniformly true across the three models.

---

## §4.5 — "Cross-dataset transfer is paradigm-specific"

### Claims (all from line 150–155)
- K=0 baseline = 91.1%
- Random K=16 = 88.2%
- QRScore-SEC K=16 = 39.6% [32.8, 46.4]
- QRScore-LME K=16 = 25.0% [18.8, 31.3]
- QRScore-NQ K=16 = 85.9% [80.7, 90.6]
- *"QRScore-NQ remains within the random-baseline 95% CI through K=16"*
- *"it does not separate from random until K=32"*

### Verification — pulled from `origin/main:results/comparison_ablation/confidence_intervals.csv`

```
QRScore-8B-LME-TRAIN,0,0.9115,0.8698,0.9479,0.0000,0.0000,0.0000,192
QRScore-8B-LME-TRAIN,16,0.2500,0.1875,0.3125,0.6615,0.5885,0.7292,192
QRScore-8B-NQ-TRAIN,0,0.9115,0.8698,0.9479,0.0000,0.0000,0.0000,192
QRScore-8B-NQ-TRAIN,16,0.8594,0.8073,0.9062,0.0521,0.0104,0.0938,192
QRScore-8B-NQ-TRAIN,32,0.7708,0.7083,0.8281,0.1406,0.0833,0.2031,192
QRScore-SEC,16,0.3958,0.3281,0.4635,0.5156,0.4375,0.5938,192
Random-seed42,16,0.8958,0.8490,0.9375,...
Random-seed123,16,0.8802,0.8333,0.9219,...
Random-seed456,16,0.8698,0.8228,0.9115,...
```

| Claim | CSV value | Match |
|---|---|---|
| K=0 baseline 91.1% | 91.15% | ✓ (rounds to 91.1) |
| Random K=16 88.2% | mean(0.8958, 0.8802, 0.8698) = 0.8819 → 88.2% | ✓ |
| QRScore-SEC K=16 39.6% [32.8, 46.4] | 39.58% [32.81, 46.35] | ✓ |
| QRScore-LME K=16 25.0% [18.8, 31.3] | 25.00% [18.75, 31.25] | ✓ |
| QRScore-NQ K=16 85.9% [80.7, 90.6] | 85.94% [80.73, 90.62] | ✓ |
| NQ overlaps random CI through K=16 | NQ 85.94% ∈ Random seed range [82.3%, 91.1%] | ✓ |
| NQ separates from random at K=32 | NQ 77.08% [70.8, 82.8] vs Random K=32 mean ≈87.5% | ✓ (NQ upper 82.8% < Random lower bounds 83.3, 84.9) |

All five published numbers and both qualitative claims are correct. **No change required to §4.5.**

---

## Summary of recommended edits to `results.tex`

| Line(s) | Current | Recommended | Why |
|---|---|---|---|
| 9 | `top-$K=64$` | `top-$K{=}128$` | Numbers match K=128, not K=64 |
| 9–11 | "ranges from 31--80% / 44--83% / 59--77%" | Add "(excluding the two detection-protocol confound pairs identified in Section~\ref{sec:result-overlap})" and append "with median 48\%--65\% across pairs" | Clarify methodology + add per user's request |
| 49–50 | "5$--$10\times more conserved" | "4$--$12\times more conserved" (or rephrase as "at least 4×, up to 12× for Llama--Mistral") | Qwen–Mistral ratio is 3.9×, below 5× |
| 99–100 | "baseline is $\geq 0.54$" | "baseline is $\geq 0.5$" | Qwen baseline 0.5417 sits at threshold — unsafe for re-rounding |
| 105–106 | "average top-$K$ Jaccard overlap ... is between 50 and 90 percent" | "off-diagonal pairwise top-$K{=}128$ Jaccard overlap ranges from 31\% to 95\% across models (per-model means 56\%--68\%)" | "50–90%" not derivable from any standard statistic at any K |
| 106–108 | "top-$K$ overlap of 75 to 95 percent" | "top-$K{=}128$ overlap of 80\%--95\%" | Uniformly true at K=128 only; tighter than 75–95% |

Three of these edits (lines 9, 105, 106) move §4.1 and §4.4 onto a **shared K=128 framing**. This is also the cleanest narrative — the head-set view (sections 4.1, 4.4) lives at K=128; the ablation view (Figure 1) shows all K. Reading consistency improves.

The K-FE table (Table 1), the sensitivity figure caption, the worked numbers in §4.2, and all of §4.5 are **fully correct** and don't need changes.

---

## Reproducing this audit

Every figure and claim above can be regenerated from a clean checkout with these commands:

```bash
# Pull the canonical Llama matrix from origin/main, audit reports
python scripts/evaluation/compute_target_sensitivity.py
# → results/target_sensitivity/target_sensitivity_report.md
# → cross-validates against existing source-centric specificity files

# Sensitivity heatmap (Figure 1 in paper)
python scripts/evaluation/plot_target_sensitivity.py --annotate
# → results/target_sensitivity/target_sensitivity_3models_annotated.png

# Jaccard appendix figure + summary at K=16 and K=128
python scripts/evaluation/plot_jaccard_appendix.py --top_k 16
python scripts/evaluation/plot_jaccard_appendix.py --top_k 128
# → results/appendix/jaccard_topk{16,128}_heatmap.png
# → results/appendix/jaccard_topk{16,128}_summary.md

# K-FE replication of Table 1, with auto-detection of efficacy definition
python scripts/evaluation/compute_kfe_correlations.py
# → results/kfe_correlations/kfe_report.md
# → results/kfe_correlations/kfe_table.csv
```

All four scripts:
- Use the `path[@gitref]` syntax to pull from `origin/main` for Llama without dirtying the working tree.
- Validate matrix shape (8 sources × 8 targets, baseline-K=0 invariance across sources) before computing.
- Cross-validate derived metrics against existing artifacts where possible; abort with a per-cell diff if anything mismatches.
- Are deterministic — no sampling, no model loading, just arithmetic on the JSON / CSV files.
