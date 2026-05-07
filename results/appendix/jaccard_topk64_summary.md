# Per-task top-64 Jaccard overlap

Top-K detected heads per task were loaded from each model's QRScore-SEC detection output (see *Provenance* below).

## Provenance

Per-task head files actually used for each model:

| Model | Resolved file | Git ref | Coverage |
| :--- | :--- | :--- | :--- |
| Llama-3.1-8B-Instruct | `results/detection/long_context_headquarters_city_heads.json` | `origin/main` | 1/8 tasks |
| Llama-3.1-8B-Instruct | `results/detection/long_context_headquarters_state_heads.json` | `origin/main` | 1/8 tasks |
| Llama-3.1-8B-Instruct | `results/detection/long_context_registrant_name_heads.json` | `origin/main` | 1/8 tasks |
| Llama-3.1-8B-Instruct | `results/detection/topk/long_context_ceo_lastname_top64.json` | `origin/main` | 1/8 tasks |
| Llama-3.1-8B-Instruct | `results/detection/topk/long_context_employees_count_total_top64.json` | `origin/main` | 1/8 tasks |
| Llama-3.1-8B-Instruct | `results/detection/topk/long_context_holder_record_amount_top64.json` | `origin/main` | 1/8 tasks |
| Llama-3.1-8B-Instruct | `results/detection/topk/long_context_incorporation_state_top64.json` | `origin/main` | 1/8 tasks |
| Llama-3.1-8B-Instruct | `results/detection/topk/long_context_incorporation_year_top64.json` | `origin/main` | 1/8 tasks |
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

## Per-model average off-diagonal Jaccard (top-64)

| Model | Mean | Min | Max |
| :--- | ---: | ---: | ---: |
| Llama-3.1-8B-Instruct | 0.514 | 0.219 | 0.910 |
| Qwen2.5-7B-Instruct | 0.491 | 0.267 | 0.882 |
| Mistral-7B-Instruct-v0.3 | 0.586 | 0.471 | 0.910 |

> _The LaTeX states the average top-K Jaccard overlap is between 50% and 90%. Compare the **Mean** column above to verify._

## Highlighted task pairs (top-64)

These are the pairs explicitly discussed in `results.tex` Section 4.4 as having 75-95% top-K overlap. The values below let a reviewer verify that claim directly.

| Task A | Task B | Llama | Qwen | Mistral |
| :--- | :--- | ---: | ---: | ---: |
| `headquarters_city` | `headquarters_state` | 0.684 | 0.882 | 0.882 |
| `incorporation_state` | `incorporation_year` | 0.910 | 0.829 | 0.910 |

## Full matrix — Llama-3.1-8B-Instruct

| Task | ceo_lastname | employees_count_total | headquarters_city | headquarters_state | holder_record_amount | incorporation_state | incorporation_year | registrant_name |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ceo_lastname | 1.000 | 0.641 | 0.422 | 0.407 | 0.620 | 0.684 | 0.641 | 0.280 |
| employees_count_total | 0.641 | 1.000 | 0.438 | 0.438 | 0.753 | 0.730 | 0.753 | 0.280 |
| headquarters_city | 0.422 | 0.438 | 1.000 | 0.684 | 0.362 | 0.438 | 0.438 | 0.455 |
| headquarters_state | 0.407 | 0.438 | 0.684 | 1.000 | 0.362 | 0.455 | 0.422 | 0.542 |
| holder_record_amount | 0.620 | 0.753 | 0.362 | 0.362 | 1.000 | 0.753 | 0.778 | 0.219 |
| incorporation_state | 0.684 | 0.730 | 0.438 | 0.455 | 0.753 | 1.000 | 0.910 | 0.255 |
| incorporation_year | 0.641 | 0.753 | 0.438 | 0.422 | 0.778 | 0.910 | 1.000 | 0.243 |
| registrant_name | 0.280 | 0.280 | 0.455 | 0.542 | 0.219 | 0.255 | 0.243 | 1.000 |

## Full matrix — Qwen2.5-7B-Instruct

| Task | ceo_lastname | employees_count_total | headquarters_city | headquarters_state | holder_record_amount | incorporation_state | incorporation_year | registrant_name |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ceo_lastname | 1.000 | 0.376 | 0.362 | 0.347 | 0.422 | 0.293 | 0.267 | 0.293 |
| employees_count_total | 0.376 | 1.000 | 0.506 | 0.471 | 0.422 | 0.438 | 0.438 | 0.488 |
| headquarters_city | 0.362 | 0.506 | 1.000 | 0.882 | 0.407 | 0.684 | 0.662 | 0.561 |
| headquarters_state | 0.347 | 0.471 | 0.882 | 1.000 | 0.391 | 0.684 | 0.641 | 0.524 |
| holder_record_amount | 0.422 | 0.422 | 0.407 | 0.391 | 1.000 | 0.438 | 0.422 | 0.333 |
| incorporation_state | 0.293 | 0.438 | 0.684 | 0.684 | 0.438 | 1.000 | 0.829 | 0.561 |
| incorporation_year | 0.267 | 0.438 | 0.662 | 0.641 | 0.422 | 0.829 | 1.000 | 0.600 |
| registrant_name | 0.293 | 0.488 | 0.561 | 0.524 | 0.333 | 0.561 | 0.600 | 1.000 |

## Full matrix — Mistral-7B-Instruct-v0.3

| Task | ceo_lastname | employees_count_total | headquarters_city | headquarters_state | holder_record_amount | incorporation_state | incorporation_year | registrant_name |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ceo_lastname | 1.000 | 0.620 | 0.524 | 0.506 | 0.620 | 0.524 | 0.561 | 0.542 |
| employees_count_total | 0.620 | 1.000 | 0.524 | 0.524 | 0.561 | 0.561 | 0.580 | 0.471 |
| headquarters_city | 0.524 | 0.524 | 1.000 | 0.882 | 0.506 | 0.684 | 0.684 | 0.580 |
| headquarters_state | 0.506 | 0.524 | 0.882 | 1.000 | 0.488 | 0.684 | 0.684 | 0.542 |
| holder_record_amount | 0.620 | 0.561 | 0.506 | 0.488 | 1.000 | 0.542 | 0.580 | 0.471 |
| incorporation_state | 0.524 | 0.561 | 0.684 | 0.684 | 0.542 | 1.000 | 0.910 | 0.506 |
| incorporation_year | 0.561 | 0.580 | 0.684 | 0.684 | 0.580 | 0.910 | 1.000 | 0.542 |
| registrant_name | 0.542 | 0.471 | 0.580 | 0.542 | 0.471 | 0.506 | 0.542 | 1.000 |

