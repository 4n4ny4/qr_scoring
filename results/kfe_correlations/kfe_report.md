# K-FE cross-model correlation report

## Method

Replicates `Table 1` of `results.tex`. For every model we build a `(|tasks|, |K|) = (8, 7)` matrix, K-residualize (subtract the within-K mean across tasks from each entry), flatten to a 56-element vector, and report the squared Pearson correlation between two models' residuals.

Two measures are computed:

- **Sensitivity** (target-centric, unambiguous): `sensitivity[t][K] = mean over sources s of drop[s][t][K]`
- **Efficacy** (source-centric, ambiguous): four candidate definitions are computed because the original analysis code wasn't preserved in this repo:
    - `on_target_drop`    = `drop[s][s][K]`
    - `row_mean_drop`     = mean over `t` of `drop[s][t][K]`
    - `off_target_mean`   = mean over `t ≠ s` of `drop[s][t][K]`
    - `specificity_index` = `on_target_drop − off_target_mean`

Each derived value is compared to the value printed in `results.tex` Table 1, and the efficacy definition with the smallest total absolute deviation is flagged as the best fit.

K values used: [8, 16, 32, 48, 64, 96, 128]  (K=0 always excluded — its drop is 0 by construction).

## Sensitivity K-FE R²

| Model A | Model B | Derived | Published (`results.tex`) | |Δ| |
| :--- | :--- | ---: | ---: | ---: |
| Llama | Qwen | **0.184** | 0.180 | 0.004 |
| Llama | Mistral | **0.472** | 0.470 | 0.002 |
| Qwen | Mistral | **0.593** | 0.590 | 0.003 |

Total |Δ| (sensitivity): **0.008**

## Efficacy K-FE R² — four candidate definitions

Smaller total |Δ| is closer to the published Table 1.

| Definition | Total |Δ| vs published |
| :--- | ---: |
| `on_target_drop` | 0.456 |
| `row_mean_drop` | 0.127 |
| `off_target_mean` ← **best fit** | 0.007 |
| `specificity_index` | 0.823 |

### efficacy = `on_target_drop`

| Model A | Model B | Derived | Published (`results.tex`) | |Δ| |
| :--- | :--- | ---: | ---: | ---: |
| Llama | Qwen | **0.031** | 0.020 | 0.011 |
| Llama | Mistral | **0.332** | 0.040 | 0.292 |
| Qwen | Mistral | **0.302** | 0.150 | 0.152 |

### efficacy = `row_mean_drop`

| Model A | Model B | Derived | Published (`results.tex`) | |Δ| |
| :--- | :--- | ---: | ---: | ---: |
| Llama | Qwen | **0.054** | 0.020 | 0.034 |
| Llama | Mistral | **0.002** | 0.040 | 0.038 |
| Qwen | Mistral | **0.096** | 0.150 | 0.054 |

### efficacy = `off_target_mean`  — **best fit**

| Model A | Model B | Derived | Published (`results.tex`) | |Δ| |
| :--- | :--- | ---: | ---: | ---: |
| Llama | Qwen | **0.018** | 0.020 | 0.002 |
| Llama | Mistral | **0.039** | 0.040 | 0.001 |
| Qwen | Mistral | **0.153** | 0.150 | 0.003 |

### efficacy = `specificity_index`

| Model A | Model B | Derived | Published (`results.tex`) | |Δ| |
| :--- | :--- | ---: | ---: | ---: |
| Llama | Qwen | **0.133** | 0.020 | 0.113 |
| Llama | Mistral | **0.466** | 0.040 | 0.426 |
| Qwen | Mistral | **0.434** | 0.150 | 0.284 |

## Verdict

- Best-matching efficacy definition: **`off_target_mean`** (total |Δ| = 0.007).
- Sensitivity values reproduce the published table to within total |Δ| = 0.008. **Confirmed.**
- Best efficacy fit reproduces the published table to within 0.007. **Confirmed.**
