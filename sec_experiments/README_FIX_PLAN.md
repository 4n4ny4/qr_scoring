# Fix QRHead Ablation Pipeline

## Root Cause: Why Ablation Shows No Accuracy Drop

Three compounding problems make the current ablation unable to show any effect:

### Problem 1: Head Detection is Broken (All Scores Negative)

Every single one of the 1024 attention heads has a **negative** QRScore across all 8 tasks. This means the null query ("N/A") directs MORE attention to the gold document than the actual question does -- the opposite of what should happen:

- `registrant_name`: max score = -0.000049, min = -0.000229
- `ceo_lastname`: max score = -0.000169, min = -0.000363

The paper's QRHeads for Llama-3.1-8B-Instruct are in **middle layers (8-17)** (from `src/qrretriever/configs/Llama-3.1-8B-Instruct_qr_head_LME.yaml`):

```
13-18, 13-21, 8-11, 14-13, 17-29, 13-1, 13-13, 14-29, 14-31, ...
```

Our detected heads are nearly all in **layers 0-2** (from `results/ablation/configs/registrant_name_qr_head_top16.yaml`):

```
0-21, 0-14, 0-0, 0-26, 0-11, 0-24, 2-2, 2-19, 2-22, ...
```

**Root cause**: Context is ~3K tokens per instance (6-13 docs x 400 words). The paper detects QRHeads at 32K-128K tokens. At 3K tokens, the model attends to everything easily without specialized retrieval heads, so detection picks up noise in early layers.

### Problem 2: The 400-Word Window is Counter-Productive

- Truncating gold sections to 400 words **loses the answer** for many instances. For `employees_count_total`, 93% of instances have the needle_value absent from the gold doc because the CSV stores "29900" but the text says "29,900"
- More importantly, short documents make retrieval trivially easy (or irrelevant), preventing QRHeads from being needed
- The user correctly notes: the needle_sentence already identifies the exact sentence with the answer

### Problem 3: Ablation Only Affects Retrieval Scoring, Not Generation

Current pipeline:

1. QRRetriever scores documents using selected attention heads
2. Top-1 document is fed to a **completely standard** `AutoModelForCausalLM` for generation

Step 2 uses **all attention heads** regardless of the knockout. So even if retrieval changes (which it barely does -- only 3-14% of top-1 docs change), the LM can still answer correctly if it gets the right document.

The paper's NIAH test (Figure 1) masks heads **during generation** and shows the model fails to copy the needle.

---

## Proposed Fix

### Phase 1: Build NIAH-style test data

Instead of the current short-context retrieval approach, build long-context prompts directly from `haystack_plan.csv`:

- Use `haystack_text` (~5000 words of distractor SEC sections) as the haystack
- Insert `needle_sentence` at a controlled position within the haystack
- Total context: ~5000-7000 words (~7000-9000 tokens)
- Use `needle_value` as ground truth
- **Drop the 400-word windowing entirely** for the ablation evaluation

This makes the task a true NIAH test: can the model find the answer sentence among thousands of words of irrelevant SEC filing text?

### Phase 2: Use paper's pre-computed QRHead ranking + re-run detection

- **Primary**: Use the paper's 16 QRHeads from `src/qrretriever/configs/Llama-3.1-8B-Instruct_qr_head_LME.yaml` as the verified top-16
- **Extended ranking**: Re-run detection on the NIAH-style long-context data (Phase 1 train split) to get a proper full 1024-head ranking for the sweep beyond 16 heads
- Validate: check if the re-detected top heads overlap with the paper's LME/NQ heads (they should now be in layers 8-17)

### Phase 3: Implement head masking during generation

Modify the generation pipeline to support zeroing out specific attention heads during inference. Two options:

- **Option A (simpler)**: Use PyTorch `register_forward_hook` on each `LlamaAttention` layer to zero out masked heads' attention outputs after softmax but before value projection
- **Option B (cleaner)**: Add a `masked_heads` parameter to the custom `src/qrretriever/custom_modeling_llama.py`'s `LlamaAttention.forward` that zeros out specified heads

This replaces the current approach where masking only affects retrieval scoring.

### Phase 4: Run ablation sweep

For K = 0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100:

- Mask the top K heads (by QRScore rank) during generation
- Generate answers from the full NIAH-style context
- Compare answer to `needle_value` using normalized matching (handle commas in numbers, case-insensitive, containment)
- Record accuracy at each level

Expected result: accuracy should decline as more QRHeads are masked, with the steepest drop in the 0-16 range (the paper's verified QRHeads).

### Phase 5: Visualize and report

- Plot accuracy vs number of knocked-out heads
- Statistical significance testing (paired bootstrap) between K=0 and each K level
- Update dashboard

---

## Key Files to Modify/Create

- `sec_experiments/build_niah_data.py` -- new: build NIAH-style test prompts from haystack_plan.csv
- `sec_experiments/run_niah_ablation.py` -- new: generation with head masking + sweep
- `src/qrretriever/custom_modeling_llama.py` -- modify: add head masking support to attention forward pass
- `sec_experiments/convert_haystack_to_json.py` -- modify: generate long-context detection data (for Phase 2 re-detection)

## Implementation Status

| Phase | Status | Notes |
|-------|--------|-------|
| Phase 1: NIAH data | Done | `build_niah_data.py` creates train/test splits in `data/niah_input/` |
| Phase 2: Detection | Done | Re-ran on NIAH data; top heads still in layers 0-2 (context too short for proper detection). Using paper's pre-computed heads as primary. |
| Phase 3: Head masking | Done | Added `_masked_head_indices` to `LlamaAttention`, `LlamaFlashAttention2`, `LlamaSdpaAttention`. Added `set_head_mask()` to `LlamaModel` and `LlamaForCausalLM`. Also added `_num_logits_to_keep` optimization to avoid OOM on logits. |
| Phase 4: Ablation sweep | In progress | `run_niah_ablation.py` written. OOM issues with 40GB A100 at 8K tokens resolved by truncating to 4K and optimizing logits. |
| Phase 5: Visualization | Pending | |
