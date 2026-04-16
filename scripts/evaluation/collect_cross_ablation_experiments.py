#!/usr/bin/env python3
"""Assemble workshop-ready cross-ablation artifacts into a canonical layout.

This script is meant for the "several experiment branches, one submission repo"
workflow. It copies committed artifacts from legacy branch layouts into the
current canonical structure:

  results/detection/<model_slug>/
  results/comparison_ablation/<model_slug>/

It also enriches copied JSON payloads with model metadata when older branches
did not store it, and writes a repo-level index describing which model
artifacts were found.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent.parent
SRC_DIR = PROJECT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from qrretriever.model_runtime import infer_model_family, resolve_model_slug


INDEX_PATH = PROJECT_DIR / "results" / "cross_ablation_index.json"


@dataclass(frozen=True)
class ExperimentSource:
    model_name: str
    source_kind: str
    detection_candidates: tuple[str, ...]
    ablation_candidates: tuple[str, ...]
    source_ref: Optional[str] = None
    notes: tuple[str, ...] = ()
    included_in_submission: bool = True

    @property
    def model_slug(self) -> str:
        return resolve_model_slug(self.model_name)

    @property
    def model_family(self) -> str:
        return infer_model_family(self.model_name)


EXPERIMENT_SOURCES = (
    ExperimentSource(
        model_name="meta-llama/Llama-3.1-8B-Instruct",
        source_kind="filesystem",
        detection_candidates=("results/detection",),
        ablation_candidates=("results/comparison_ablation",),
        notes=(
            "Llama artifacts are migrated from the legacy root-level results layout.",
        ),
    ),
    ExperimentSource(
        model_name="Qwen/Qwen2.5-7B-Instruct",
        source_kind="git",
        source_ref="origin/ananya/qwen_exp",
        detection_candidates=("results/detection_qwen",),
        ablation_candidates=(
            "results/runs/qwen_true_detect_2026-04-07",
            "results/comparison_ablation_qwen",
        ),
        notes=(
            "Qwen artifacts are imported from the historical qwen experiment branch.",
        ),
    ),
    ExperimentSource(
        model_name="allenai/OLMo-7B",
        source_kind="git",
        source_ref="origin/ananya/gamma_olmo",
        detection_candidates=(
            "results/detection/allenai__OLMo-7B",
            "results/detection_olmo",
        ),
        ablation_candidates=(
            "results/comparison_ablation/allenai__OLMo-7B",
            "results/runs/olmo_true_detect_2026-04-07",
            "results/comparison_ablation_olmo",
        ),
        notes=(
            "OLMo code support is retained, but OLMo result artifacts are excluded from this submission-cleanup branch.",
        ),
        included_in_submission=False,
    ),
)


def run_git_bytes(*args: str) -> bytes:
    proc = subprocess.run(
        ["git", *args],
        cwd=PROJECT_DIR,
        check=True,
        capture_output=True,
    )
    return proc.stdout


def run_git_text(*args: str) -> str:
    return run_git_bytes(*args).decode("utf-8")


def list_git_files(ref: str, prefix: str) -> list[str]:
    try:
        output = run_git_text("ls-tree", "-r", "--name-only", ref, prefix)
    except subprocess.CalledProcessError:
        return []
    return [line.strip() for line in output.splitlines() if line.strip() and not line.endswith(".DS_Store")]


def find_git_source_dir(ref: str, candidates: Iterable[str]) -> Optional[str]:
    for candidate in candidates:
        if list_git_files(ref, candidate):
            return candidate
    return None


def list_local_files(prefix: Path, *, excluded_first_segments: set[str]) -> list[Path]:
    if not prefix.exists():
        return []
    files = []
    for path in prefix.rglob("*"):
        if not path.is_file() or path.name == ".DS_Store":
            continue
        rel = path.relative_to(prefix)
        if rel.parts and rel.parts[0] in excluded_first_segments:
            continue
        files.append(path)
    return files


def copy_local_tree(src_dir: Path, dst_dir: Path, *, excluded_first_segments: set[str]) -> int:
    files = list_local_files(src_dir, excluded_first_segments=excluded_first_segments)
    for path in files:
        rel = normalize_output_rel_path(path.relative_to(src_dir))
        out_path = dst_dir / rel
        out_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, out_path)
    return len(files)


def copy_git_tree(ref: str, src_prefix: str, dst_dir: Path) -> int:
    files = list_git_files(ref, src_prefix)
    for path_str in files:
        rel = normalize_output_rel_path(Path(path_str).relative_to(src_prefix))
        out_path = dst_dir / rel
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(run_git_bytes("show", f"{ref}:{path_str}"))
    return len(files)


def load_json(path: Path) -> Optional[dict]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _remap_source_file(path_str: Optional[str]) -> Optional[str]:
    if not path_str:
        return path_str
    normalized = path_str.replace("\\", "/")
    marker = "/data/"
    if marker in normalized:
        suffix = normalized.split(marker, 1)[1]
        return str(PROJECT_DIR / "data" / suffix)
    return path_str


def normalize_output_rel_path(rel: Path) -> Path:
    parts = rel.parts
    if len(parts) >= 2 and parts[0] == "topk" and parts[1] == "8b_external":
        return Path("topk") / "external" / Path(*parts[2:])
    return rel


def enrich_json_metadata(path: Path, metadata: dict) -> None:
    payload = load_json(path)
    if payload is None:
        return
    changed = False
    for key, value in metadata.items():
        if payload.get(key) is None:
            payload[key] = value
            changed = True
    if changed:
        save_json(path, payload)


def enrich_detection_manifests(detection_dir: Path, metadata: dict) -> None:
    for path in detection_dir.rglob("*_heads_manifest.json"):
        payload = load_json(path)
        if payload is None:
            continue
        for key, value in metadata.items():
            if payload.get(key) is None:
                payload[key] = value
        export_prefix = payload.get("export_prefix")
        if export_prefix:
            payload["output_file"] = str(detection_dir / f"{export_prefix}_heads.json")
            payload["source_file"] = _remap_source_file(payload.get("source_file"))
            topk_dir = path.parent
            payload["exports"] = {
                str(k): str(topk_dir / f"{export_prefix}_top{k}.json")
                for k in payload.get("top_k_values", [])
            }
        save_json(path, payload)


def enrich_ablation_jsons(ablation_dir: Path, metadata: dict) -> None:
    for pattern in (
        "*_results.json",
        "comparison_summary.json",
        "cross_task_transfer_matrix.json",
        "cross_task_specificity_metrics.json",
        "cross_task_head_similarity_topk.json",
    ):
        for path in ablation_dir.glob(pattern):
            enrich_json_metadata(path, metadata)


def build_summary_from_results(ablation_dir: Path, metadata: dict) -> Optional[dict]:
    methods = {}
    for path in sorted(ablation_dir.glob("*_results.json")):
        payload = load_json(path)
        if payload is None:
            continue
        method_name = payload.get("method") or path.stem.replace("_results", "")
        curve = payload.get("accuracy_curve", {})
        baseline = curve.get("0")
        k16 = curve.get("16")
        drop_at_16 = None
        if baseline is not None and k16 is not None:
            drop_at_16 = baseline - k16
        methods[method_name] = {
            "accuracy_curve": curve,
            "baseline_accuracy": baseline,
            "accuracy_at_k16": k16,
            "drop_at_k16": drop_at_16,
        }

    if not methods:
        return None

    first_payload = None
    for path in sorted(ablation_dir.glob("*_results.json")):
        first_payload = load_json(path)
        if first_payload is not None:
            break

    summary = {
        "model": metadata["model_name"],
        "model_name": metadata["model_name"],
        "model_slug": metadata["model_slug"],
        "model_family": metadata["model_family"],
        "methods": methods,
    }
    if first_payload is not None:
        summary["knockout_sizes"] = first_payload.get("knockout_sizes", [])
        num_instances = None
        details = first_payload.get("details", {})
        if isinstance(details, dict):
            baseline_details = details.get("0")
            if isinstance(baseline_details, list):
                num_instances = len(baseline_details)
        if num_instances is not None:
            summary["num_instances"] = num_instances
    return summary


def count_files(root: Path) -> int:
    if not root.exists():
        return 0
    return sum(1 for path in root.rglob("*") if path.is_file() and path.name != ".DS_Store")


def relpath_or_none(path: Path) -> Optional[str]:
    try:
        return str(path.relative_to(PROJECT_DIR))
    except ValueError:
        return None


def collect_one(source: ExperimentSource, *, excluded_first_segments: set[str]) -> dict:
    metadata = {
        "model_name": source.model_name,
        "model_slug": source.model_slug,
        "model_family": source.model_family,
    }
    detection_out = PROJECT_DIR / "results" / "detection" / source.model_slug
    ablation_out = PROJECT_DIR / "results" / "comparison_ablation" / source.model_slug

    detection_found = None
    ablation_found = None
    copied_detection_files = 0
    copied_ablation_files = 0

    if source.source_kind == "filesystem":
        for candidate in source.detection_candidates:
            candidate_path = PROJECT_DIR / candidate
            if list_local_files(candidate_path, excluded_first_segments=excluded_first_segments):
                detection_found = candidate
                copied_detection_files = copy_local_tree(
                    candidate_path,
                    detection_out,
                    excluded_first_segments=excluded_first_segments,
                )
                break

        for candidate in source.ablation_candidates:
            candidate_path = PROJECT_DIR / candidate
            if list_local_files(candidate_path, excluded_first_segments=excluded_first_segments):
                ablation_found = candidate
                copied_ablation_files = copy_local_tree(
                    candidate_path,
                    ablation_out,
                    excluded_first_segments=excluded_first_segments,
                )
                break
    else:
        if source.source_ref is None:
            raise ValueError(f"git-backed source for {source.model_name} is missing source_ref")
        detection_found = find_git_source_dir(source.source_ref, source.detection_candidates)
        if detection_found is not None:
            copied_detection_files = copy_git_tree(source.source_ref, detection_found, detection_out)
        ablation_found = find_git_source_dir(source.source_ref, source.ablation_candidates)
        if ablation_found is not None:
            copied_ablation_files = copy_git_tree(source.source_ref, ablation_found, ablation_out)

    if copied_detection_files:
        enrich_detection_manifests(detection_out, metadata)
    if copied_ablation_files:
        enrich_ablation_jsons(ablation_out, metadata)
        summary = build_summary_from_results(ablation_out, metadata)
        if summary is not None:
            save_json(ablation_out / "comparison_summary.json", summary)

    detection_count = count_files(detection_out)
    ablation_count = count_files(ablation_out)
    if detection_found is None and detection_count:
        detection_found = relpath_or_none(detection_out)
    if ablation_found is None and ablation_count:
        ablation_found = relpath_or_none(ablation_out)

    experiment_manifest = {
        **metadata,
        "source_kind": source.source_kind,
        "source_ref": source.source_ref,
        "included_in_submission": source.included_in_submission,
        "source_detection_dir": detection_found,
        "source_ablation_dir": ablation_found,
        "notes": list(source.notes),
        "detection_file_count": detection_count,
        "ablation_file_count": ablation_count,
    }

    if not source.included_in_submission:
        status = "excluded"
    elif experiment_manifest["detection_file_count"] and experiment_manifest["ablation_file_count"]:
        status = "complete"
    elif experiment_manifest["detection_file_count"] or experiment_manifest["ablation_file_count"]:
        status = "partial"
    else:
        status = "missing"
    experiment_manifest["status"] = status

    summary_path = ablation_out / "comparison_summary.json"
    methods = []
    if summary_path.exists():
        summary_payload = load_json(summary_path) or {}
        methods = sorted(summary_payload.get("methods", {}).keys())
    experiment_manifest["methods"] = methods
    experiment_manifest["has_transfer_matrix"] = (ablation_out / "cross_task_transfer_matrix.json").exists()

    if status not in {"missing", "excluded"}:
        save_json(ablation_out / "experiment_manifest.json", experiment_manifest)

    return experiment_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect cross-ablation artifacts into canonical per-model folders.")
    parser.add_argument(
        "--index_path",
        default=str(INDEX_PATH),
        help="Path for the repo-level experiment index JSON.",
    )
    args = parser.parse_args()

    known_slugs = {source.model_slug for source in EXPERIMENT_SOURCES}
    experiments = [collect_one(source, excluded_first_segments=known_slugs) for source in EXPERIMENT_SOURCES]

    index_payload = {
        "generated_from_repo": str(PROJECT_DIR),
        "experiments": experiments,
    }
    save_json(Path(args.index_path), index_payload)
    print(f"Wrote experiment index to {args.index_path}")
    for experiment in experiments:
        print(
            f"- {experiment['model_slug']}: status={experiment['status']} "
            f"detection_files={experiment['detection_file_count']} "
            f"ablation_files={experiment['ablation_file_count']}"
        )


if __name__ == "__main__":
    main()
