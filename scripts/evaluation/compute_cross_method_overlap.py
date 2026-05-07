#!/usr/bin/env python3
"""Compute Jaccard overlap between SEC, LME, and NQ head rankings.

The original helper was hard-coded to the Llama submission artifacts. This
version is model-aware so we can compare SEC-vs-NQ overlap for Qwen, OLMo,
Mistral, or any future model as soon as the relevant detection directory lands.

Examples:
  python scripts/evaluation/compute_cross_method_overlap.py
  python scripts/evaluation/compute_cross_method_overlap.py \
    --model-slugs Qwen__Qwen2.5-7B-Instruct allenai__OLMo-7B \
    --pairs SEC:NQ --allow-missing
  python scripts/evaluation/compute_cross_method_overlap.py \
    --pairs SEC:NQ --include-transferred-external
  python scripts/evaluation/compute_cross_method_overlap.py \
    --detection-dirs results/detection/mistralai__Mistral-7B-Instruct-v0.3
"""

import argparse
import csv
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


PROJECT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_K_VALUES = [8, 16, 32, 48, 64, 96, 128]
DEFAULT_PAIRS = ["SEC:LME", "SEC:NQ", "LME:NQ"]

# Query-head intervention counts for common models in this project.
# Used only for the expected-random Jaccard reference line.
TOTAL_HEADS_BY_SLUG = {
    "meta-llama__Llama-3.1-8B-Instruct": 1024,
    "meta-llama/Llama-3.1-8B-Instruct": 1024,
    "Qwen__Qwen2.5-7B-Instruct": 784,
    "Qwen/Qwen2.5-7B-Instruct": 784,
    "allenai__OLMo-7B": 1024,
    "allenai__OLMo-2-1124-7B-Instruct": 1024,
    "google__gemma-7b": 448,
    "mistralai__Mistral-7B-Instruct-v0.3": 1024,
    "Mistral-7B-Instruct-v0.3": 1024,
}


Head = Tuple[int, int]


@dataclass
class Ranking:
    label: str
    path: Path
    heads: List[Head]
    source_file: Optional[str] = None
    note: Optional[str] = None


def parse_head(row) -> Head:
    """Parse a head row from list/tuple or dict JSON formats."""
    head_str = row[0] if isinstance(row, list) else row.get("head")
    if not head_str:
        raise ValueError(f"Unable to parse head row: {row!r}")
    layer, head = map(int, str(head_str).split("-"))
    return layer, head


def load_heads(path: Path) -> List[Head]:
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    return [parse_head(row) for row in data]


def jaccard(set_a: set, set_b: set) -> float:
    union = set_a | set_b
    return len(set_a & set_b) / len(union) if union else 0.0


def overlap_coefficient(set_a: set, set_b: set) -> float:
    denom = min(len(set_a), len(set_b))
    return len(set_a & set_b) / denom if denom else 0.0


def expected_random_jaccard(k: int, n: int) -> float:
    """Expected Jaccard for two random subsets of size k drawn from n items."""
    return k / (2 * n - k) if n > 0 else 0.0


def as_project_path(path_value: str, manifest_path: Optional[Path] = None) -> Path:
    """Resolve stale absolute or repo-relative paths found in old manifests."""
    candidate = Path(path_value)
    if candidate.is_absolute() and candidate.exists():
        return candidate
    if not candidate.is_absolute():
        repo_candidate = PROJECT_DIR / candidate
        if repo_candidate.exists():
            return repo_candidate
    if manifest_path is not None:
        sibling = manifest_path.parent / candidate.name
        if sibling.exists():
            return sibling
    return candidate


def resolve_manifest_export(manifest_path: Path, max_k: int) -> Tuple[Optional[Path], Dict]:
    with manifest_path.open(encoding="utf-8") as f:
        manifest = json.load(f)

    exports = manifest.get("exports", {})
    if not exports:
        return None, manifest

    export_ks = sorted(int(k) for k in exports)
    eligible = [k for k in export_ks if k >= max_k]
    chosen_k = eligible[0] if eligible else export_ks[-1]
    export_path = as_project_path(exports[str(chosen_k)], manifest_path)
    if not export_path.exists():
        return None, manifest
    return export_path, manifest


def ranking_candidates(label: str, detection_dir: Path, max_k: int) -> List[Path]:
    label = label.upper()
    is_qwen = "Qwen" in str(detection_dir) or "qwen" in str(detection_dir)
    if label == "SEC":
        return [
            detection_dir / "long_context_combined_heads.json",
            detection_dir / "combined_heads.json",
            detection_dir / "topk" / "long_context_combined_heads_manifest.json",
            detection_dir / "topk" / f"long_context_combined_top{max_k}.json",
        ]
    if label == "NQ":
        candidates = []
        if is_qwen:
            for directory in ["Qwen-2.5-7B-Instruct", "Qwen2.5-7B-Instruct", "Qwen__Qwen2.5-7B-Instruct"]:
                candidates.extend(
                    [
                        PROJECT_DIR / directory / "nq_TRAIN_qwen.json",
                        PROJECT_DIR / directory / "NQ_TRAIN_qwen.json",
                        PROJECT_DIR / directory / "nq_train_qwen.json",
                        Path("/") / directory / "nq_TRAIN_qwen.json",
                        Path("/") / directory / "NQ_TRAIN_qwen.json",
                        Path("/") / directory / "nq_train_qwen.json",
                    ]
                )
        candidates.extend(
            [
                detection_dir / "nq_train_heads.json",
                detection_dir / "NQ_train_heads.json",
                detection_dir / "long_context_nq_train_heads.json",
            ]
        )
        return candidates
    if label == "LME":
        candidates = []
        if is_qwen:
            for directory in ["Qwen-2.5-7B-Instruct", "Qwen2.5-7B-Instruct", "Qwen__Qwen2.5-7B-Instruct"]:
                candidates.extend(
                    [
                        PROJECT_DIR / directory / "lme_TRAIN_qwen.json",
                        PROJECT_DIR / directory / "LME_TRAIN_qwen.json",
                        PROJECT_DIR / directory / "lme_train_qwen.json",
                        Path("/") / directory / "lme_TRAIN_qwen.json",
                        Path("/") / directory / "LME_TRAIN_qwen.json",
                        Path("/") / directory / "lme_train_qwen.json",
                    ]
                )
        candidates.extend(
            [
                detection_dir / "lme_train_heads.json",
                detection_dir / "LME_train_heads.json",
                detection_dir / "long_context_lme_train_heads.json",
            ]
        )
        return candidates
    raise ValueError(f"Unsupported ranking label `{label}`. Expected SEC, NQ, or LME.")


def transferred_external_candidates(label: str, detection_dir: Path, max_k: int) -> List[Path]:
    """Legacy cross-model exports, included only when explicitly requested."""
    label = label.upper()
    if label == "NQ":
        return [
            detection_dir / "topk" / "external" / "nq_train_heads_manifest.json",
            detection_dir / "topk" / "external" / f"nq_train_top{max_k}.json",
        ]
    if label == "LME":
        return [
            detection_dir / "topk" / "external" / "lme_train_heads_manifest.json",
            detection_dir / "topk" / "external" / f"lme_train_top{max_k}.json",
        ]
    return []


def resolve_ranking(
    label: str,
    detection_dir: Path,
    max_k: int,
    include_transferred_external: bool = False,
) -> Optional[Ranking]:
    candidates = ranking_candidates(label, detection_dir, max_k)
    if include_transferred_external:
        candidates = candidates + transferred_external_candidates(label, detection_dir, max_k)

    for candidate in candidates:
        if not candidate.exists():
            continue
        note = None
        source_file = None
        resolved_path = candidate
        if candidate.name.endswith("_heads_manifest.json"):
            resolved_path, manifest = resolve_manifest_export(candidate, max_k)
            if resolved_path is None:
                continue
            source_file = manifest.get("source_file")
            note = f"resolved from manifest {candidate}"
        elif "topk/external" in str(candidate):
            note = "legacy transferred/external ranking"

        heads = load_heads(resolved_path)
        return Ranking(
            label=label.upper(),
            path=resolved_path,
            heads=heads,
            source_file=source_file,
            note=note,
        )
    return None


def discover_detection_dirs(detection_root: Path) -> List[Path]:
    """Find canonical immediate per-model dirs with SEC combined rankings."""
    if not detection_root.exists():
        return []
    dirs = []
    for child in sorted(detection_root.iterdir()):
        if not child.is_dir():
            continue
        if (child / "long_context_combined_heads.json").exists():
            dirs.append(child)
    return dirs


def detection_dirs_from_args(args) -> List[Path]:
    detection_root = PROJECT_DIR / "results" / "detection"
    if args.detection_dirs:
        return [as_project_path(path) for path in args.detection_dirs]
    if args.model_slugs:
        return [detection_root / slug for slug in args.model_slugs]
    return discover_detection_dirs(detection_root)


def model_slug_from_dir(detection_dir: Path) -> str:
    try:
        return str(detection_dir.relative_to(PROJECT_DIR / "results" / "detection"))
    except ValueError:
        return detection_dir.name


def infer_total_heads(model_slug: str, rankings: Dict[str, Ranking], override: Optional[int]) -> int:
    if override is not None:
        return override
    if model_slug in TOTAL_HEADS_BY_SLUG:
        return TOTAL_HEADS_BY_SLUG[model_slug]
    slug_tail = model_slug.split("/")[-1]
    if slug_tail in TOTAL_HEADS_BY_SLUG:
        return TOTAL_HEADS_BY_SLUG[slug_tail]

    max_layer = 0
    max_head = 0
    for ranking in rankings.values():
        if not ranking.heads:
            continue
        max_layer = max(max_layer, max(layer for layer, _ in ranking.heads))
        max_head = max(max_head, max(head for _, head in ranking.heads))
    return (max_layer + 1) * (max_head + 1)


def parse_pairs(pair_args: Sequence[str]) -> List[Tuple[str, str]]:
    pairs = []
    for pair in pair_args:
        if ":" in pair:
            left, right = pair.split(":", 1)
        elif "-" in pair:
            left, right = pair.split("-", 1)
        else:
            raise ValueError(f"Pair `{pair}` must look like SEC:NQ or SEC-NQ.")
        pairs.append((left.upper(), right.upper()))
    return pairs


def compute_pair_metrics(heads_a: List[Head], heads_b: List[Head], k_values: Iterable[int]) -> Dict:
    metrics = {}
    for k in k_values:
        set_a = set(heads_a[: min(k, len(heads_a))])
        set_b = set(heads_b[: min(k, len(heads_b))])
        overlap_count = len(set_a & set_b)
        union_size = len(set_a | set_b)
        metrics[str(k)] = {
            "jaccard": round(jaccard(set_a, set_b), 4),
            "overlap_coefficient": round(overlap_coefficient(set_a, set_b), 4),
            "overlap_count": overlap_count,
            "union_size": union_size,
            "left_size": len(set_a),
            "right_size": len(set_b),
        }
    return metrics


def write_per_model_json(model_slug: str, payload: Dict) -> None:
    out_dir = PROJECT_DIR / "results" / "comparison_ablation" / model_slug
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "cross_method_head_overlap.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"  Wrote per-model overlap: {out_path}")


def write_combined_csv(path: Path, combined: Dict) -> None:
    rows = []
    for model_slug, model_payload in combined["models"].items():
        random_expected = model_payload.get("random_expected", {})
        rankings = model_payload.get("rankings", {})
        for pair_key, by_k in model_payload.get("pairs", {}).items():
            left, right = pair_key.split("-")
            for k_str, metrics in by_k.items():
                rows.append(
                    {
                        "model_slug": model_slug,
                        "pair": pair_key,
                        "left": left,
                        "right": right,
                        "K": k_str,
                        "jaccard": metrics["jaccard"],
                        "overlap_coefficient": metrics["overlap_coefficient"],
                        "overlap_count": metrics["overlap_count"],
                        "union_size": metrics["union_size"],
                        "left_size": metrics["left_size"],
                        "right_size": metrics["right_size"],
                        "random_expected_jaccard": random_expected.get(k_str),
                        "left_path": rankings.get(left, {}).get("path"),
                        "right_path": rankings.get(right, {}).get("path"),
                        "left_note": rankings.get(left, {}).get("note"),
                        "right_note": rankings.get(right, {}).get("note"),
                        "left_source_file": rankings.get(left, {}).get("source_file"),
                        "right_source_file": rankings.get(right, {}).get("source_file"),
                    }
                )

    fieldnames = [
        "model_slug",
        "pair",
        "left",
        "right",
        "K",
        "jaccard",
        "overlap_coefficient",
        "overlap_count",
        "union_size",
        "left_size",
        "right_size",
        "random_expected_jaccard",
        "left_path",
        "right_path",
        "left_note",
        "right_note",
        "left_source_file",
        "right_source_file",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def print_summary(combined: Dict, pairs: Sequence[Tuple[str, str]], k_values: Sequence[int]) -> None:
    for left, right in pairs:
        pair_key = f"{left}-{right}"
        rows = []
        for model_slug, payload in combined["models"].items():
            if pair_key in payload.get("pairs", {}):
                rows.append((model_slug, payload["pairs"][pair_key]))
        if not rows:
            continue
        print(f"\n{pair_key} Jaccard")
        print(f"{'Model':45s}", end="")
        for k in k_values:
            print(f"  K={k:<3d}", end="")
        print()
        for model_slug, by_k in rows:
            print(f"{model_slug[:45]:45s}", end="")
            for k in k_values:
                print(f"  {by_k[str(k)]['jaccard']:.4f}", end="")
            print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute cross-method head overlap by model.")
    parser.add_argument("--model-slugs", nargs="+", default=None, help="Model slugs under results/detection.")
    parser.add_argument("--detection-dirs", nargs="+", default=None, help="Explicit detection directories.")
    parser.add_argument("--output-dir", default=os.path.join(PROJECT_DIR, "results", "comparison_ablation"))
    parser.add_argument("--k-values", nargs="+", type=int, default=DEFAULT_K_VALUES)
    parser.add_argument("--pairs", nargs="+", default=DEFAULT_PAIRS, help="Pairs like SEC:NQ or SEC-LME.")
    parser.add_argument("--num-total-heads", type=int, default=None, help="Override expected-random total heads.")
    parser.add_argument(
        "--allow-missing",
        action="store_true",
        help="Skip models/pairs with missing rankings instead of raising.",
    )
    parser.add_argument(
        "--include-transferred-external",
        action="store_true",
        help=(
            "Allow legacy topk/external LME/NQ rankings when same-model ranking files "
            "are absent. Keep this off for same-model SEC-NQ Jaccard tables."
        ),
    )
    parser.add_argument(
        "--no-per-model-json",
        action="store_true",
        help="Only write the combined JSON/CSV outputs.",
    )
    args = parser.parse_args()

    k_values = sorted(set(k for k in args.k_values if k > 0))
    max_k = max(k_values)
    pairs = parse_pairs(args.pairs)
    needed_labels = sorted({label for pair in pairs for label in pair})
    detection_dirs = detection_dirs_from_args(args)

    if not detection_dirs:
        raise FileNotFoundError("No detection directories found. Use --model-slugs or --detection-dirs.")

    combined = {
        "k_values": k_values,
        "pairs_requested": [f"{left}-{right}" for left, right in pairs],
        "models": {},
        "missing": {},
    }

    print("Computing head-set overlap")
    for detection_dir in detection_dirs:
        model_slug = model_slug_from_dir(detection_dir)
        print(f"\nModel: {model_slug}")
        print(f"  Detection dir: {detection_dir}")
        rankings: Dict[str, Ranking] = {}
        missing_labels = []
        for label in needed_labels:
            ranking = resolve_ranking(
                label,
                detection_dir,
                max_k,
                include_transferred_external=args.include_transferred_external,
            )
            if ranking is None:
                missing_labels.append(label)
                print(f"  {label}: missing")
            else:
                rankings[label] = ranking
                note = f" ({ranking.note})" if ranking.note else ""
                print(f"  {label}: {len(ranking.heads)} heads from {ranking.path}{note}")

        model_missing = {}
        pair_results = {}
        for left, right in pairs:
            pair_key = f"{left}-{right}"
            if left not in rankings or right not in rankings:
                model_missing[pair_key] = {
                    "missing_rankings": [label for label in (left, right) if label not in rankings]
                }
                if not args.allow_missing:
                    raise FileNotFoundError(
                        f"Missing rankings for {model_slug} pair {pair_key}: "
                        f"{model_missing[pair_key]['missing_rankings']}"
                    )
                continue
            pair_results[pair_key] = compute_pair_metrics(rankings[left].heads, rankings[right].heads, k_values)

        if not pair_results:
            combined["missing"][model_slug] = model_missing or {"rankings": missing_labels}
            continue

        num_total_heads = infer_total_heads(model_slug, rankings, args.num_total_heads)
        model_payload = {
            "model_slug": model_slug,
            "detection_dir": str(detection_dir),
            "num_total_heads": num_total_heads,
            "rankings": {
                label: {
                    "path": str(ranking.path),
                    "num_heads": len(ranking.heads),
                    "source_file": ranking.source_file,
                    "note": ranking.note,
                }
                for label, ranking in rankings.items()
            },
            "pairs": pair_results,
            "random_expected": {
                str(k): round(expected_random_jaccard(k, num_total_heads), 4)
                for k in k_values
            },
        }
        if model_missing:
            model_payload["missing"] = model_missing
            combined["missing"][model_slug] = model_missing

        combined["models"][model_slug] = model_payload
        if not args.no_per_model_json:
            write_per_model_json(model_slug, model_payload)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "cross_model_head_overlap.json"
    csv_path = output_dir / "cross_model_head_overlap.csv"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(combined, f, indent=2)
    write_combined_csv(csv_path, combined)

    print_summary(combined, pairs, k_values)
    print(f"\nWrote combined overlap JSON: {json_path}")
    print(f"Wrote combined overlap CSV:  {csv_path}")

    if combined["missing"]:
        print("\nMissing artifacts:")
        for model_slug, missing in combined["missing"].items():
            print(f"  {model_slug}: {missing}")


if __name__ == "__main__":
    main()
