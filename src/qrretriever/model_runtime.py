import importlib
import os
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import torch
import transformers


SUPPORTED_MODEL_NAMES = {
    "meta-llama/Llama-3.1-8B-Instruct": {
        "family": "llama",
        "allow_external_rankings": True,
        "requires_trust_remote_code": False,
    },
    "Qwen/Qwen2.5-7B-Instruct": {
        "family": "qwen",
        "allow_external_rankings": True,
        "requires_trust_remote_code": False,
    },
    "google/gemma-7b": {
        "family": "gemma",
        "allow_external_rankings": False,
        "requires_trust_remote_code": False,
    },
    "allenai/OLMo-7B": {
        "family": "olmo",
        "allow_external_rankings": True,
        "requires_trust_remote_code": True,
    },
}

FAMILY_MODELING_MODULES = {
    "gemma": "transformers.models.gemma.modeling_gemma",
    "olmo": "transformers.models.olmo.modeling_olmo",
}


@dataclass(frozen=True)
class ModelSpec:
    model_name: str
    model_slug: str
    model_family: str
    tokenizer_name: str
    trust_remote_code: bool
    supports_stock_head_masking: bool
    detection_backend_available: bool
    allow_external_rankings: bool
    requires_trust_remote_code: bool

    def as_metadata(self):
        return asdict(self)


def slugify_model_name(model_name: str) -> str:
    name = model_name.strip().strip("/")
    name = name.replace("/", "__")
    return re.sub(r"[^A-Za-z0-9._-]+", "-", name).strip("-")


def infer_model_family(model_name: str) -> str:
    normalized = model_name.lower()
    if "llama-3.1-8b-instruct" in normalized:
        return "llama"
    if "qwen2.5-7b-instruct" in normalized:
        return "qwen"
    if "gemma-7b" in normalized:
        return "gemma"
    if "olmo-7b" in normalized:
        return "olmo"
    raise ValueError(
        "Unsupported model_name. Supported examples: "
        "`meta-llama/Llama-3.1-8B-Instruct`, `Qwen/Qwen2.5-7B-Instruct`, "
        "`google/gemma-7b`, `allenai/OLMo-7B`."
    )


def resolve_model_slug(model_name: str, model_slug: Optional[str] = None) -> str:
    return model_slug or slugify_model_name(model_name)


def resolve_model_spec(
    model_name: str,
    model_slug: Optional[str] = None,
    tokenizer_name: Optional[str] = None,
    trust_remote_code: bool = False,
) -> ModelSpec:
    family = infer_model_family(model_name)
    defaults = SUPPORTED_MODEL_NAMES.get(model_name, {})
    requires_trust_remote_code = defaults.get("requires_trust_remote_code", family == "olmo")
    return ModelSpec(
        model_name=model_name,
        model_slug=resolve_model_slug(model_name, model_slug),
        model_family=family,
        tokenizer_name=tokenizer_name or model_name,
        trust_remote_code=trust_remote_code,
        supports_stock_head_masking=family in {"llama", "qwen", "gemma", "olmo"},
        detection_backend_available=family in {"llama", "qwen", "gemma", "olmo"},
        allow_external_rankings=defaults.get("allow_external_rankings", family == "llama"),
        requires_trust_remote_code=requires_trust_remote_code,
    )


def resolve_detection_dir(project_dir: str, model_spec: ModelSpec, override: Optional[str] = None) -> str:
    return override or os.path.join(project_dir, "results", "detection", model_spec.model_slug)


def resolve_ablation_dir(project_dir: str, model_spec: ModelSpec, override: Optional[str] = None) -> str:
    return override or os.path.join(project_dir, "results", "comparison_ablation", model_spec.model_slug)


def _has_local_editable_torch() -> bool:
    try:
        torch_path = Path(torch.__file__).resolve()
    except Exception:
        return False
    path_str = str(torch_path)
    return "site-packages" not in path_str and "dist-packages" not in path_str


def _build_runtime_error(model_spec: ModelSpec, exc: Exception) -> RuntimeError:
    lines = [
        f"Runtime preflight failed for `{model_spec.model_name}` ({model_spec.model_family}).",
        f"Underlying import error: {exc}",
    ]
    exc_text = repr(exc)
    if "torchvision" in exc_text or "nms" in exc_text:
        lines.append(
            "Your current `torch` / `torchvision` runtime looks incompatible. "
            "Use a released `torch` build with a matching `torchvision` install."
        )
    if _has_local_editable_torch():
        lines.append(
            f"Detected a local editable torch build at `{torch.__file__}`. "
            "That setup can break `transformers` model imports for Gemma/OLMo."
        )
    lines.append(
        "This repo does not require `torchvision` directly, but this `transformers` "
        "runtime may import it transitively for some model families."
    )
    return RuntimeError("\n".join(lines))


def preflight_model_environment(model_spec: ModelSpec) -> None:
    if model_spec.requires_trust_remote_code and not model_spec.trust_remote_code:
        raise RuntimeError(
            f"`{model_spec.model_name}` requires `--trust_remote_code` in this repo's "
            "supported workflow."
        )

    module_name = FAMILY_MODELING_MODULES.get(model_spec.model_family)
    if module_name is None:
        return

    try:
        importlib.import_module(module_name)
    except Exception as exc:
        raise _build_runtime_error(model_spec, exc) from exc


def load_tokenizer(model_spec: ModelSpec):
    preflight_model_environment(model_spec)
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_spec.tokenizer_name,
        trust_remote_code=model_spec.trust_remote_code,
    )
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_stock_causal_lm(
    model_spec: ModelSpec,
    resolved_device: str,
    *,
    for_detection: bool = False,
    load_in_8bit: bool = False,
):
    preflight_model_environment(model_spec)

    load_kwargs = {
        "low_cpu_mem_usage": True,
        "trust_remote_code": model_spec.trust_remote_code,
    }
    if load_in_8bit:
        load_kwargs["load_in_8bit"] = True
        # Not all models support load_in_8bit; Qwen2 is one example.
        # We can guard this by checking the model config, but for now, a
        # targeted exclusion is sufficient.
        if "qwen2" in model_spec.model_name.lower():
            del load_kwargs["load_in_8bit"]

    preferred_attn = None

    if resolved_device == "cuda":
        if not load_in_8bit:
            load_kwargs["device_map"] = "auto"
        load_kwargs["torch_dtype"] = "auto" if model_spec.model_family == "olmo" else torch.float16
        preferred_attn = "eager" if for_detection else "flash_attention_2"
    elif resolved_device == "mps":
        load_kwargs["torch_dtype"] = "auto" if model_spec.model_family == "olmo" else torch.float16
    elif model_spec.model_family == "olmo":
        load_kwargs["torch_dtype"] = "auto"

    if preferred_attn is not None:
        try:
            model = transformers.AutoModelForCausalLM.from_pretrained(
                model_spec.model_name,
                attn_implementation=preferred_attn,
                **load_kwargs,
            )
        except Exception:
            model = transformers.AutoModelForCausalLM.from_pretrained(
                model_spec.model_name,
                **load_kwargs,
            )
    else:
        model = transformers.AutoModelForCausalLM.from_pretrained(
            model_spec.model_name,
            **load_kwargs,
        )

    if resolved_device != "cuda" and not load_in_8bit:
        model = model.to(resolved_device)
    elif resolved_device == "cuda" and not load_in_8bit:
        # device_map="auto" handles this, so we don't need to move it.
        pass
    elif load_in_8bit:
        # 8-bit models loaded with device_map="auto" (the default for 8-bit)
        # should not be moved again.
        pass
    else:
        model = model.to(resolved_device)

    return model
