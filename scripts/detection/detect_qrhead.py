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
import numpy as np
from qrretriever.attn_retriever import FullHeadRetriever

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


def get_doc_scores_per_head(full_head_retriever, data_instances, truncate_by_space=0):
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


def export_top_k_files(head_scores_list, export_dir, export_prefix, top_ks, source_file, output_file):
    os.makedirs(export_dir, exist_ok=True)
    manifest = {
        "source_file": source_file,
        "output_file": output_file,
        "export_prefix": export_prefix,
        "top_k_values": top_ks,
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
    parser.add_argument("--output_file", type=str, required=True, help="Path to the output JSON file to save scores for each head.")

    parser.add_argument("--truncate_by_space", type=int, default=0, help="Truncate paragraphs by number of words. Default is 0 (no truncation).")

    parser.add_argument("--config_or_config_path", type=str, default=None, help="Path to the configuration file or a configuration string. If not provided, defaults will be used.")
    parser.add_argument("--model_name_or_path", type=str, default=None, help="Path to the model directory or model name.")
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

    full_head_retriever = FullHeadRetriever(
        config_or_config_path=args.config_or_config_path,
        model_name_or_path=args.model_name_or_path,
    )

    # read input file
    print(f"Reading input file: {args.input_file}", flush=True)
    with open(args.input_file, "r") as f:
        data_instances = json.load(f)

    doc_scores_per_head = get_doc_scores_per_head(full_head_retriever, data_instances, truncate_by_space=args.truncate_by_space) # qid -> {doc_id -> score tensor with shape (n_layers, n_heads)}
    head_scores_list = score_heads(doc_scores_per_head, data_instances)

    with open(args.output_file, "w", encoding="utf-8") as f:
        json.dump(head_scores_list, f, indent=4)

    export_dir = args.export_dir or os.path.dirname(os.path.abspath(args.output_file))
    export_prefix = args.task_name or os.path.splitext(os.path.basename(args.output_file))[0]
    top_ks = sorted(set(k for k in args.export_top_k if k > 0))
    export_top_k_files(
        head_scores_list=head_scores_list,
        export_dir=export_dir,
        export_prefix=export_prefix,
        top_ks=top_ks,
        source_file=args.input_file,
        output_file=args.output_file,
    )