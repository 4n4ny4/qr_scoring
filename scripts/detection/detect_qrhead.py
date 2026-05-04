import argparse
try:
    from tqdm import tqdm
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "Missing dependency 'tqdm'. Install project dependencies with `python -m pip install -e .` "
        "(or quick fix: `python -m pip install tqdm`)."
    ) from exc
from itertools import product
import json
import os
import sys
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(os.path.dirname(SCRIPT_DIR))
SRC_DIR = os.path.join(PROJECT_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from qrretriever.config import load_config
from qrretriever.detection_backends import build_full_head_retriever
from qrretriever.model_runtime import (
    resolve_detection_dir,
    resolve_model_spec,
)

DEFAULT_EXPORT_TOP_K = [8, 16, 32, 48, 64, 96, 128]


def lme_eval(retrieval_results, data_instances):
    """
    retrieval_results: a dict of qid -> {doc_id -> score}, retrieval results from a specific head
    data_instances: a list of dicts, each dict represents an instance
    """
    all_score_over_gold = []

    for data in data_instances:
        qid = data['idx']
        gt_docs = data["gt_docs"] # a list of doc ids

        doc_id2score = retrieval_results[qid] # doc_id -> score

        if len(gt_docs) == 0:
            score_over_gold = 0
        else:
            score_over_gold = np.sum([doc_id2score[doc_id] for doc_id in gt_docs])
            sorted_docs_ids = sorted(doc_id2score.items(), key=lambda x: x[1], reverse=True)
            sorted_docs_ids = [doc_id for doc_id, _ in sorted_docs_ids]

        all_score_over_gold.append(score_over_gold)

    mean_score_over_gold = np.mean(all_score_over_gold)
    return mean_score_over_gold # QRScore for a specific head


def get_doc_scores_per_head(full_head_retriever, data_instances, truncate_by_space=0, trust_remote_code=False):
    """
    data_instances: a list of dicts, each dict represents an instance
    """
    doc_scores_per_head = {} # qid -> {doc_id -> score tensor with shape (n_layers, n_heads)}
    for i, data in enumerate(tqdm(data_instances)):

        query = data["question"]
        docs = data["paragraphs"]
        
        for p in docs:

            paragraph_text = p['paragraph_text'].strip()

            if truncate_by_space > 0:
                # Truncate each paragraph by space.
                if len(paragraph_text.split(' ')) > truncate_by_space:
                    print('number of words being truncated: ', len(paragraph_text.split(' ')) - truncate_by_space, flush=True)

                p['paragraph_text'] = ' '.join(paragraph_text.split(' ')[:truncate_by_space])

            else:
                p['paragraph_text'] = paragraph_text

        retrieval_scores = full_head_retriever.score_docs_per_head_for_detection(query, docs) # doc_id -> score tensor with shape (n_layers, n_heads)
        doc_scores_per_head[data['idx']] = retrieval_scores

    return doc_scores_per_head



def score_heads(doc_scores_per_head, data_instances):
    """
    doc_scores_per_head: a dict of dicts, outer dict key is question idx, inner dict key is doc idx, value is a (n_layers, n_heads) tensor
    """

    # pick first qid
    first_qid = next(iter(doc_scores_per_head))
    first_doc_id = next(iter(doc_scores_per_head[first_qid]))
    example_tensor = doc_scores_per_head[first_qid][first_doc_id]
    num_layers, num_heads = example_tensor.shape

    # score by head
    layer_head = product(range(num_layers), range(num_heads))
    head_scores = {}

    for layer, head in tqdm(layer_head, total=num_layers * num_heads):
        retrieval_results = {} # get new retrieval results for this head, qid -> {doc_id -> score}

        for qid, per_doc_score_tensors in doc_scores_per_head.items():
            # per_doc_score_tensors: a dict of doc_id -> (n_layers, n_heads) tensor
            doc_id2score = {}
            for doc_id, score_tensor in per_doc_score_tensors.items():
                score = score_tensor[layer][head]
                doc_id2score[doc_id] = score.item()

            retrieval_results[qid] = doc_id2score
            
        head_score = lme_eval(retrieval_results, data_instances) # QRScore for this head
        head_scores[(layer, head)] = head_score

    # replace key with layer-head
    head_scores_list = [(f"{layer}-{head}", score) for (layer, head), score in head_scores.items()]
    # sort heads by scores
    head_scores_list.sort(key=lambda x: x[1], reverse=True)

    return head_scores_list # a list of tuples (head, score)


def export_top_k_files(
    head_scores_list,
    export_dir,
    export_prefix,
    top_ks,
    source_file,
    output_file,
    model_metadata,
):
    os.makedirs(export_dir, exist_ok=True)
    manifest = {
        "source_file": source_file,
        "output_file": output_file,
        "export_prefix": export_prefix,
        "top_k_values": top_ks,
        "model_name": model_metadata["model_name"],
        "model_slug": model_metadata["model_slug"],
        "model_family": model_metadata["model_family"],
        "exports": {},
    }

    for k in top_ks:
        out_path = os.path.join(export_dir, f"{export_prefix}_top{k}.json")
        payload = head_scores_list[: min(k, len(head_scores_list))]
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        manifest["exports"][str(k)] = out_path

    manifest_path = os.path.join(export_dir, f"{export_prefix}_heads_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"Top-K exports saved under: {export_dir}")
    print(f"Manifest: {manifest_path}")





if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, required=True, help="Path to the input JSON file to find QRHead.")
    parser.add_argument("--output_file", type=str, default=None, help="Path to the output JSON file to save scores for each head.")
    parser.add_argument("--detection_dir", type=str, default=None, help="Optional model-specific detection output directory.")

    parser.add_argument("--truncate_by_space", type=int, default=0, help="Truncate paragraphs by number of words. Default is 0 (no truncation).")

    parser.add_argument("--config_or_config_path", type=str, default=None, help="Path to the configuration file or a configuration string. If not provided, defaults will be used.")
    parser.add_argument("--model_name_or_path", type=str, default=None, help="Path to the model directory or model name.")
    parser.add_argument("--tokenizer_name_or_path", type=str, default=None, help="Optional tokenizer override.")
    parser.add_argument("--model_slug", type=str, default=None, help="Optional slug override for per-model output directories.")
    parser.add_argument("--trust_remote_code", action="store_true", help="Allow Hugging Face remote code when required by the model family.")
    parser.add_argument("--model_load_in_8bit", action="store_true", help="Load model in 8-bit mode.")
    parser.add_argument("--task_name", type=str, default=None, help="Optional task label used in export file naming.")
    parser.add_argument("--export_dir", type=str, default=None, help="Optional directory for top-K export files.")
    parser.add_argument(
        "--export_top_k",
        nargs="+",
        type=int,
        default=DEFAULT_EXPORT_TOP_K,
        help="Top-K sizes to export into separate files.",
    )

    args = parser.parse_args()

    config = load_config(args.config_or_config_path) if args.config_or_config_path else {}
    resolved_model_name = (
        args.model_name_or_path
        or config.get("model_name_or_path")
        or config.get("model_base_class")
    )
    if resolved_model_name is None:
        raise ValueError(
            "Detection requires --model_name_or_path or a config that provides model_name_or_path/model_base_class."
        )
    model_spec = resolve_model_spec(
        model_name=resolved_model_name,
        model_slug=args.model_slug,
        tokenizer_name=args.tokenizer_name_or_path,
        trust_remote_code=args.trust_remote_code,
    )
    detection_dir = resolve_detection_dir(PROJECT_DIR, model_spec, args.detection_dir)
    if args.output_file is not None:
        output_file = args.output_file
    else:
        if not args.task_name:
            raise ValueError("Provide --output_file or --task_name so detection can resolve a per-model output path.")
        output_file = os.path.join(detection_dir, f"{args.task_name}_heads.json")
    export_dir = args.export_dir or os.path.join(detection_dir, "topk")

    os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
    print(f"Resolved project_dir={PROJECT_DIR}", flush=True)
    print(f"Resolved detection_dir={detection_dir}", flush=True)
    print(f"Resolved output_file={output_file}", flush=True)
    print(f"Resolved export_dir={export_dir}", flush=True)
    print(f"Resolved model={model_spec.model_name} ({model_spec.model_family})", flush=True)

    full_head_retriever = build_full_head_retriever(
        config_or_config_path=args.config_or_config_path,
        model_name_or_path=args.model_name_or_path,
        tokenizer_name_or_path=args.tokenizer_name_or_path,
        trust_remote_code=args.trust_remote_code,
        model_load_in_8bit=args.model_load_in_8bit,
    )

    # read input file
    print(f"Reading input file: {args.input_file}", flush=True)
    with open(args.input_file, "r") as f:
        data_instances = json.load(f)

    doc_scores_per_head = get_doc_scores_per_head(full_head_retriever, data_instances, truncate_by_space=args.truncate_by_space) # qid -> {doc_id -> score tensor with shape (n_layers, n_heads)}
    head_scores_list = score_heads(doc_scores_per_head, data_instances)

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(head_scores_list, f, indent=4)

    export_prefix = args.task_name or os.path.splitext(os.path.basename(output_file))[0]
    top_ks = sorted(set(k for k in args.export_top_k if k > 0))
    export_top_k_files(
        head_scores_list=head_scores_list,
        export_dir=export_dir,
        export_prefix=export_prefix,
        top_ks=top_ks,
        source_file=args.input_file,
        output_file=output_file,
        model_metadata=model_spec.as_metadata(),
    )
