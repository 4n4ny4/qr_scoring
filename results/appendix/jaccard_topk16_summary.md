# Per-task top-16 Jaccard overlap

Top-K detected heads per task were loaded from each model's QRScore-SEC detection output (see *Provenance* below).

## Provenance

Per-task head files actually used for each model:

| Model | Resolved file | Git ref | Coverage |
| :--- | :--- | :--- | :--- |
| Llama-3.1-8B-Instruct | `results/detection/long_context_headquarters_city_heads.json` | `origin/main` | 1/8 tasks |
| Llama-3.1-8B-Instruct | `results/detection/long_context_headquarters_state_heads.json` | `origin/main` | 1/8 tasks |
| Llama-3.1-8B-Instruct | `results/detection/long_context_registrant_name_heads.json` | `origin/main` | 1/8 tasks |
| Llama-3.1-8B-Instruct | `results/detection/topk/long_context_ceo_lastname_top16.json` | `origin/main` | 1/8 tasks |
| Llama-3.1-8B-Instruct | `results/detection/topk/long_context_employees_count_total_top16.json` | `origin/main` | 1/8 tasks |
| Llama-3.1-8B-Instruct | `results/detection/topk/long_context_holder_record_amount_top16.json` | `origin/main` | 1/8 tasks |
| Llama-3.1-8B-Instruct | `results/detection/topk/long_context_incorporation_state_top16.json` | `origin/main` | 1/8 tasks |
| Llama-3.1-8B-Instruct | `results/detection/topk/long_context_incorporation_year_top16.json` | `origin/main` | 1/8 tasks |
| Qwen2.5-7B-Instruct | `results/detection_qwen/long_context_ceo_lastname_heads.json` | _working tree_ | 1/8 tasks |
| Qwen2.5-7B-Instruct | `results/detection_qwen/long_context_employees_count_total_heads.json` | _working tree_ | 1/8 tasks |
| Qwen2.5-7B-Instruct | `results/detection_qwen/long_context_headquarters_city_heads.json` | _working tree_ | 1/8 tasks |
| Qwen2.5-7B-Instruct | `results/detection_qwen/long_context_headquarters_state_heads.json` | _working tree_ | 1/8 tasks |
| Qwen2.5-7B-Instruct | `results/detection_qwen/long_context_holder_record_amount_heads.json` | _working tree_ | 1/8 tasks |
| Qwen2.5-7B-Instruct | `results/detection_qwen/long_context_incorporation_state_heads.json` | _working tree_ | 1/8 tasks |
| Qwen2.5-7B-Instruct | `results/detection_qwen/long_context_incorporation_year_heads.json` | _working tree_ | 1/8 tasks |
| Qwen2.5-7B-Instruct | `results/detection_qwen/long_context_registrant_name_heads.json` | _working tree_ | 1/8 tasks |
| Mistral-7B-Instruct-v0.3 | `results/detection_mistral/long_context_ceo_lastname_heads.json` | _working tree_ | 1/8 tasks |
| Mistral-7B-Instruct-v0.3 | `results/detection_mistral/long_context_employees_count_total_heads.json` | _working tree_ | 1/8 tasks |
| Mistral-7B-Instruct-v0.3 | `results/detection_mistral/long_context_headquarters_city_heads.json` | _working tree_ | 1/8 tasks |
| Mistral-7B-Instruct-v0.3 | `results/detection_mistral/long_context_headquarters_state_heads.json` | _working tree_ | 1/8 tasks |
| Mistral-7B-Instruct-v0.3 | `results/detection_mistral/long_context_holder_record_amount_heads.json` | _working tree_ | 1/8 tasks |
| Mistral-7B-Instruct-v0.3 | `results/detection_mistral/long_context_incorporation_state_heads.json` | _working tree_ | 1/8 tasks |
| Mistral-7B-Instruct-v0.3 | `results/detection_mistral/long_context_incorporation_year_heads.json` | _working tree_ | 1/8 tasks |
| Mistral-7B-Instruct-v0.3 | `results/detection_mistral/long_context_registrant_name_heads.json` | _working tree_ | 1/8 tasks |

Tasks (8): `ceo_lastname`, `employees_count_total`, `headquarters_city`, `headquarters_state`, `holder_record_amount`, `incorporation_state`, `incorporation_year`, `registrant_name`

## Per-model average off-diagonal Jaccard (top-16)

| Model | Mean | Min | Max |
| :--- | ---: | ---: | ---: |
| Llama-3.1-8B-Instruct | 0.443 | 0.143 | 0.778 |
| Qwen2.5-7B-Instruct | 0.479 | 0.143 | 0.882 |
| Mistral-7B-Instruct-v0.3 | 0.539 | 0.280 | 0.882 |

> _The LaTeX states the average top-K Jaccard overlap is between 50% and 90%. Compare the **Mean** column above to verify._

## Highlighted task pairs (top-16)

These are the pairs explicitly discussed in `results.tex` Section 4.4 as having 75-95% top-K overlap. The values below let a reviewer verify that claim directly.

| Task A | Task B | Llama | Qwen | Mistral |
| :--- | :--- | ---: | ---: | ---: |
| `headquarters_city` | `headquarters_state` | 0.778 | 0.882 | 0.882 |
| `incorporation_state` | `incorporation_year` | 0.600 | 0.778 | 0.882 |

## Full matrix — Llama-3.1-8B-Instruct

| Task | ceo_lastname | employees_count_total | headquarters_city | headquarters_state | holder_record_amount | incorporation_state | incorporation_year | registrant_name |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ceo_lastname | 1.000 | 0.524 | 0.391 | 0.455 | 0.524 | 0.391 | 0.455 | 0.333 |
| employees_count_total | 0.524 | 1.000 | 0.455 | 0.600 | 0.684 | 0.333 | 0.524 | 0.391 |
| headquarters_city | 0.391 | 0.455 | 1.000 | 0.778 | 0.391 | 0.333 | 0.391 | 0.333 |
| headquarters_state | 0.455 | 0.600 | 0.778 | 1.000 | 0.524 | 0.391 | 0.455 | 0.455 |
| holder_record_amount | 0.524 | 0.684 | 0.391 | 0.524 | 1.000 | 0.455 | 0.524 | 0.333 |
| incorporation_state | 0.391 | 0.333 | 0.333 | 0.391 | 0.455 | 1.000 | 0.600 | 0.143 |
| incorporation_year | 0.455 | 0.524 | 0.391 | 0.455 | 0.524 | 0.600 | 1.000 | 0.231 |
| registrant_name | 0.333 | 0.391 | 0.333 | 0.455 | 0.333 | 0.143 | 0.231 | 1.000 |

## Full matrix — Qwen2.5-7B-Instruct

| Task | ceo_lastname | employees_count_total | headquarters_city | headquarters_state | holder_record_amount | incorporation_state | incorporation_year | registrant_name |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ceo_lastname | 1.000 | 0.280 | 0.280 | 0.333 | 0.280 | 0.231 | 0.231 | 0.143 |
| employees_count_total | 0.280 | 1.000 | 0.524 | 0.524 | 0.455 | 0.391 | 0.524 | 0.391 |
| headquarters_city | 0.280 | 0.524 | 1.000 | 0.882 | 0.524 | 0.684 | 0.882 | 0.524 |
| headquarters_state | 0.333 | 0.524 | 0.882 | 1.000 | 0.524 | 0.600 | 0.778 | 0.524 |
| holder_record_amount | 0.280 | 0.455 | 0.524 | 0.524 | 1.000 | 0.524 | 0.524 | 0.280 |
| incorporation_state | 0.231 | 0.391 | 0.684 | 0.600 | 0.524 | 1.000 | 0.778 | 0.333 |
| incorporation_year | 0.231 | 0.524 | 0.882 | 0.778 | 0.524 | 0.778 | 1.000 | 0.455 |
| registrant_name | 0.143 | 0.391 | 0.524 | 0.524 | 0.280 | 0.333 | 0.455 | 1.000 |

## Full matrix — Mistral-7B-Instruct-v0.3

| Task | ceo_lastname | employees_count_total | headquarters_city | headquarters_state | holder_record_amount | incorporation_state | incorporation_year | registrant_name |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ceo_lastname | 1.000 | 0.280 | 0.391 | 0.391 | 0.391 | 0.524 | 0.455 | 0.455 |
| employees_count_total | 0.280 | 1.000 | 0.455 | 0.524 | 0.333 | 0.524 | 0.455 | 0.391 |
| headquarters_city | 0.391 | 0.455 | 1.000 | 0.882 | 0.455 | 0.778 | 0.778 | 0.684 |
| headquarters_state | 0.391 | 0.524 | 0.882 | 1.000 | 0.391 | 0.778 | 0.778 | 0.684 |
| holder_record_amount | 0.391 | 0.333 | 0.455 | 0.391 | 1.000 | 0.391 | 0.391 | 0.455 |
| incorporation_state | 0.524 | 0.524 | 0.778 | 0.778 | 0.391 | 1.000 | 0.882 | 0.600 |
| incorporation_year | 0.455 | 0.455 | 0.778 | 0.778 | 0.391 | 0.882 | 1.000 | 0.600 |
| registrant_name | 0.455 | 0.391 | 0.684 | 0.684 | 0.455 | 0.600 | 0.600 | 1.000 |

