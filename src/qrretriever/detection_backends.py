import os
from typing import Dict, List, Optional, Tuple, Union

import torch

from .config import load_config
from .model_runtime import (
    load_stock_causal_lm,
    load_tokenizer,
    preflight_model_environment,
    resolve_model_spec,
)


DETECTION_PROMPT_PREFIX = "Here are some paragraphs:"
DETECTION_QUERY_BLOCK_TEMPLATE = (
    "Please find information that are relevant to the following query in the paragraphs above.\n"
    "Query: {query}"
)


class StockFullHeadRetriever:
    def __init__(
        self,
        model_name_or_path: str,
        tokenizer_name_or_path: Optional[str] = None,
        trust_remote_code: bool = False,
        device: Optional[str] = None,
        load_in_8bit: bool = False,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_spec = resolve_model_spec(
            model_name=model_name_or_path,
            tokenizer_name=tokenizer_name_or_path,
            trust_remote_code=trust_remote_code,
        )
        preflight_model_environment(self.model_spec)
        self.tokenizer = load_tokenizer(self.model_spec)
        if not getattr(self.tokenizer, "is_fast", False):
            raise RuntimeError(
                f"Detection backend for `{self.model_spec.model_name}` requires a fast tokenizer "
                "with offset mappings."
            )
        self.llm = load_stock_causal_lm(
            self.model_spec,
            self.device,
            for_detection=True,
            load_in_8bit=load_in_8bit,
        )
        if self.llm.config.pad_token_id is None:
            self.llm.config.pad_token_id = self.tokenizer.pad_token_id or self.llm.config.eos_token_id
        self.llm.eval()

    def _build_docs_block(self, docs: List[Dict]) -> str:
        pieces = [DETECTION_PROMPT_PREFIX]
        for idx, doc in enumerate(docs, start=1):
            paragraph_text = doc["paragraph_text"]
            if doc.get("title"):
                paragraph_text = f"{doc['title']}\n{paragraph_text}"
            pieces.append(f"[{idx}] {paragraph_text}")
        return "\n\n".join(pieces)

    def _build_query_block(self, query: str) -> str:
        return DETECTION_QUERY_BLOCK_TEMPLATE.format(query=query)

    def _render_prompt(self, docs_block: str, query_block: str) -> str:
        user_content = f"{docs_block}\n\n{query_block}"
        if hasattr(self.tokenizer, "apply_chat_template") and self.tokenizer.chat_template:
            return self.tokenizer.apply_chat_template(
                [{"role": "user", "content": user_content}],
                add_generation_prompt=True,
                tokenize=False,
            )
        return user_content

    def _build_char_to_token_map(self, offset_mapping):
        mapping = {}
        for token_idx, (start, end) in enumerate(offset_mapping):
            for char_idx in range(start, end):
                mapping[char_idx] = token_idx
        return mapping

    def _find_content_span(self, prompt_text: str, char_offset_to_token_idx: dict, content: str) -> Tuple[int, int]:
        if content not in prompt_text:
            raise ValueError("Content not found in rendered detection prompt.")
        start_char = prompt_text.index(content)
        end_char = start_char + len(content) - 1
        return char_offset_to_token_idx[start_char], char_offset_to_token_idx[end_char]

    def compose_scoring_prompt(self, query: str, docs: List[Dict]):
        docs_block = self._build_docs_block(docs)
        query_block = self._build_query_block(query)
        prompt_text = self._render_prompt(docs_block, query_block)
        tokenization = self.tokenizer(prompt_text, return_offsets_mapping=True, return_tensors="pt")
        char_to_token = self._build_char_to_token_map(tokenization["offset_mapping"][0].tolist())

        doc_spans = []
        for idx, doc in enumerate(docs, start=1):
            paragraph_text = doc["paragraph_text"]
            if doc.get("title"):
                paragraph_text = f"{doc['title']}\n{paragraph_text}"
            doc_text = f"[{idx}] {paragraph_text}"
            doc_spans.append(self._find_content_span(prompt_text, char_to_token, doc_text))

        query_span = self._find_content_span(prompt_text, char_to_token, query_block)
        return prompt_text, tokenization, query_span, doc_spans

    def _score_query_attentions(self, tokenization, query_span: Tuple[int, int]):
        input_ids = tokenization["input_ids"].to(self.llm.device)
        attention_mask = tokenization["attention_mask"].to(self.llm.device)
        query_start, query_end = query_span
        query_len = query_end - query_start + 1
        prefix_len = query_start

        prefix_outputs = None
        if prefix_len > 0:
            prefix_outputs = self.llm(
                input_ids=input_ids[:, :prefix_len],
                attention_mask=attention_mask[:, :prefix_len],
                use_cache=True,
                return_dict=True,
            )

        query_kwargs = {
            "input_ids": input_ids[:, query_start : query_end + 1],
            "attention_mask": attention_mask[:, : query_end + 1],
            "past_key_values": None if prefix_outputs is None else prefix_outputs.past_key_values,
            "use_cache": True,
            "output_attentions": True,
            "return_dict": True,
        }
        cache_position = torch.arange(query_start, query_start + query_len, device=input_ids.device)
        try:
            query_outputs = self.llm(cache_position=cache_position, **query_kwargs)
        except TypeError as e:
            # OLMo doesn't support output_attentions, so we have to handle it differently
            if "output_attentions is not yet supported" in str(e) or (
                "unexpected keyword argument" in str(e) and "output_attentions" in str(e)
            ):
                del query_kwargs["output_attentions"]
                query_outputs = self.llm(**query_kwargs)
                # Manually get attentions from the cache if possible
                if hasattr(query_outputs, "past_key_values") and query_outputs.past_key_values:
                    # This is a model-specific hack and might not be stable.
                    # The location of attention scores in the cache can vary.
                    # This is a guess for OLMo based on common patterns.
                    attentions = [layer[2] for layer in query_outputs.past_key_values]
                    query_outputs.attentions = tuple(attentions)
                else:
                    raise ValueError("Could not retrieve attention scores from OLMo model.")
            else:
                query_outputs = self.llm(**query_kwargs)
        except ValueError as e:
            if "output_attentions is not yet supported" in str(e):
                del query_kwargs["output_attentions"]
                query_outputs = self.llm(**query_kwargs)
                if hasattr(query_outputs, "past_key_values") and query_outputs.past_key_values:
                    attentions = [layer[2] for layer in query_outputs.past_key_values] # Heuristic
                    query_outputs.attentions = tuple(attentions)
                else:
                    raise ValueError("Could not retrieve attention scores from OLMo model after fallback.")
            else:
                raise e

        per_layer = []
        if not hasattr(query_outputs, "attentions") or not query_outputs.attentions:
            raise ValueError("Model output does not contain attention scores.")

    def score_docs_per_head_for_detection(self, query: str, docs: List[Dict]) -> Dict[str, torch.Tensor]:
        prompt_text, tokenized_prompt, query_span, doc_spans = self.compose_scoring_prompt(query, docs)
        null_prompt_text, tokenized_null_prompt, null_query_span, _ = self.compose_scoring_prompt("N/A", docs)

        if query_span[0] != null_query_span[0]:
            raise ValueError("Query start token changed between real and null prompts.")

        doc_scores = self._score_query_attentions(tokenized_prompt, query_span)
        null_doc_scores = self._score_query_attentions(tokenized_null_prompt, null_query_span)

        min_length = min(doc_scores.shape[-1], null_doc_scores.shape[-1])
        per_token_scores_cal = doc_scores[:, :, :min_length] - null_doc_scores[:, :, :min_length]

        results = {}
        for doc, (start_idx, end_idx) in zip(docs, doc_spans):
            end_idx = min(end_idx, min_length - 1)
            curr = per_token_scores_cal[:, :, start_idx : end_idx + 1]
            threshold = curr.mean(dim=-1) - 2 * curr.std(dim=-1)
            tok_mask = curr > threshold.unsqueeze(-1)
            results[doc["idx"]] = curr.masked_fill(~tok_mask, 0.0).sum(dim=-1)

        return results


def _resolve_detection_config(
    config_or_config_path: Optional[Union[Dict, str]],
    model_name_or_path: Optional[str],
    model_base_class: Optional[str],
):
    if config_or_config_path is None:
        return None
    if isinstance(config_or_config_path, dict):
        return config_or_config_path
    return load_config(config_or_config_path)


def build_full_head_retriever(
    config_or_config_path: Optional[Union[Dict, str]] = None,
    model_name_or_path: Optional[str] = None,
    model_base_class: Optional[str] = None,
    device: Optional[str] = None,
    trust_remote_code: bool = False,
    tokenizer_name_or_path: Optional[str] = None,
    model_load_in_8bit: bool = False,
):
    config = _resolve_detection_config(config_or_config_path, model_name_or_path, model_base_class)
    resolved_model_name = model_name_or_path
    resolved_model_base = model_base_class
    if config is not None:
        resolved_model_name = resolved_model_name or config.get("model_name_or_path")
        resolved_model_base = resolved_model_base or config.get("model_base_class")

    if not resolved_model_name and not resolved_model_base:
        raise ValueError("Detection requires model_name_or_path, model_base_class, or a config file.")

    model_hint = resolved_model_name or resolved_model_base
    model_spec = resolve_model_spec(
        model_name=model_hint,
        tokenizer_name=tokenizer_name_or_path,
        trust_remote_code=trust_remote_code,
    )

    if model_spec.model_family == "llama":
        from .attn_retriever import FullHeadRetriever as LlamaFullHeadRetriever

        return LlamaFullHeadRetriever(
            config_or_config_path=config_or_config_path,
            model_name_or_path=model_name_or_path,
            model_base_class=model_base_class,
            device=device,
        )

    return StockFullHeadRetriever(
        model_name_or_path=model_spec.model_name,
        tokenizer_name_or_path=tokenizer_name_or_path,
        trust_remote_code=trust_remote_code,
        device=device,
        load_in_8bit=model_load_in_8bit,
    )
