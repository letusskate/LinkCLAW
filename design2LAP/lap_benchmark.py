import ast
import datetime
import json
import os
import string
import sys
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import regex
import torch
from safetensors import safe_open
from torchtune.models.convert_weights import tune_to_hf
from tqdm.auto import tqdm as auto_tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig, LlamaForCausalLM

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DESIGN1_LSA_ROOT = os.environ.get("DESIGN1_LSA_ROOT", os.path.join(PROJECT_ROOT, "design1LSA"))
if DESIGN1_LSA_ROOT not in sys.path:
	sys.path.insert(0, DESIGN1_LSA_ROOT)

from src.common import move_to_target_device
from src.data.titan_preprocessor import make_segment_mask

from jsonl_loader import load_jsonl_dataset


DEFAULT_BASE_MODEL = "/home/dell/wyb/my_models/models/LLM-Research/Llama-3___2-1B-Instruct"
DEFAULT_LLM_JUDGE_MODEL = "/home/dell/wyb/my_models/LLM-Research/LLM-Research/Meta-Llama-3___1-8B-Instruct"
SYSTEM_PROMPT = (
	"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
	"You are an intelligent AI assistant. Answer based on the conversation history.\n"
	"<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
)
ASSISTANT_PROMPT = "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"


@dataclass
class Fragment:
	token_ids: List[int]
	segment_ids: List[int]


def normalize_answer(text: str) -> str:
	def remove_articles(value: str) -> str:
		return regex.sub(r"\b(a|an|the)\b", " ", value)

	def white_space_fix(value: str) -> str:
		return " ".join(value.split())

	def remove_punc(value: str) -> str:
		exclude = set(string.punctuation)
		return "".join(char for char in value if char not in exclude)

	return white_space_fix(remove_articles(remove_punc(text.lower())))


def best_subspan_em(prediction: str, ground_truths: List[str]) -> float:
	normalized_prediction = normalize_answer(prediction)
	for ground_truth in ground_truths:
		normalized_ground_truth = normalize_answer(ground_truth)
		if normalized_ground_truth and normalized_ground_truth in normalized_prediction:
			return 1.0
	return 0.0


_JUDGE_IGNORED_ATOMS = {
	"",
	"none",
	"null",
	"true",
	"false",
	"ok",
	"error",
	"response",
	"results",
	"data",
	"status",
	"success",
	"failed",
	"unknown",
}


def split_answer_units(text: str) -> List[str]:
	raw_units = regex.split(r"\n{2,}|(?<=[.!?])\s+|[;；]\s*", text or "")
	units: List[str] = []
	for raw_unit in raw_units:
		normalized_unit = normalize_answer(raw_unit)
		if len(normalized_unit) >= 8:
			units.append(normalized_unit)
	if not units:
		normalized_text = normalize_answer(text or "")
		if normalized_text:
			units.append(normalized_text)
	return list(dict.fromkeys(units))


def score_answer_units(prediction: str, gold_answer: str) -> Dict[str, float]:
	normalized_prediction = normalize_answer(prediction)
	units = split_answer_units(gold_answer)
	if not units:
		return {"matched": 0.0, "total": 0.0, "score": 0.0}
	matched = sum(1 for unit in units if unit in normalized_prediction)
	return {
		"matched": float(matched),
		"total": float(len(units)),
		"score": matched / len(units),
	}


def _collect_tool_evidence(value: Any, evidence: List[str], limit: int = 24):
	if len(evidence) >= limit or value is None:
		return
	if isinstance(value, dict):
		for nested in value.values():
			_collect_tool_evidence(nested, evidence, limit=limit)
			if len(evidence) >= limit:
				return
		return
	if isinstance(value, list):
		for nested in value[:limit]:
			_collect_tool_evidence(nested, evidence, limit=limit)
			if len(evidence) >= limit:
				return
		return
	if isinstance(value, (int, float, bool)):
		normalized_scalar = normalize_answer(str(value))
		if normalized_scalar and normalized_scalar not in _JUDGE_IGNORED_ATOMS:
			evidence.append(normalized_scalar)
		return
	for unit in split_answer_units(str(value)):
		if unit in _JUDGE_IGNORED_ATOMS:
			continue
		if len(unit) < 4 and not unit.isdigit():
			continue
		evidence.append(unit)
		if len(evidence) >= limit:
			return


def score_tool_coverage(prediction: str, logical_tools: List[Dict[str, Any]]) -> Dict[str, Any]:
	normalized_prediction = normalize_answer(prediction)
	if not logical_tools:
		return {"matched": 0.0, "total": 0.0, "score": 0.0, "details": []}

	covered_tools = 0
	details = []
	for tool in logical_tools:
		evidence: List[str] = []
		_collect_tool_evidence(tool.get("output"), evidence)
		unique_evidence = list(dict.fromkeys(evidence))[:12]
		if not unique_evidence:
			details.append(
				{
					"tool_name": tool.get("tool_name", ""),
					"matched_evidence": 0,
					"evidence_total": 0,
					"covered": False,
				}
			)
			continue

		matched_evidence = sum(1 for item in unique_evidence if item in normalized_prediction)
		threshold = 1 if len(unique_evidence) <= 3 else 2
		covered = matched_evidence >= min(threshold, len(unique_evidence))
		covered_tools += int(covered)
		details.append(
			{
				"tool_name": tool.get("tool_name", ""),
				"matched_evidence": matched_evidence,
				"evidence_total": len(unique_evidence),
				"covered": covered,
			}
		)

	return {
		"matched": float(covered_tools),
		"total": float(len(logical_tools)),
		"score": covered_tools / len(logical_tools),
		"details": details,
	}


def simple_partial_judge(prediction: str, parsed_sample: Dict[str, Any]) -> Dict[str, Any]:
	answer_unit_score = score_answer_units(prediction, parsed_sample["final_answer"])
	tool_coverage_score = score_tool_coverage(prediction, parsed_sample["logical_tools"])
	denominator = answer_unit_score["total"] + tool_coverage_score["total"]
	combined_score = 0.0 if denominator == 0 else (answer_unit_score["matched"] + tool_coverage_score["matched"]) / denominator
	return {
		"answer_unit_matched": answer_unit_score["matched"],
		"answer_unit_total": answer_unit_score["total"],
		"answer_unit_score": answer_unit_score["score"],
		"tool_coverage_matched": tool_coverage_score["matched"],
		"tool_coverage_total": tool_coverage_score["total"],
		"tool_coverage_score": tool_coverage_score["score"],
		"simple_judge_score": combined_score,
		"tool_coverage_details": tool_coverage_score["details"],
	}


def decomposed_pass_from_judge(simple_judge: Dict[str, Any]) -> float:
	return 1.0 if (simple_judge["answer_unit_matched"] > 0 or simple_judge["tool_coverage_matched"] > 0) else 0.0


def truncate_for_judge(text: str, limit: int = 700) -> str:
	cleaned = (text or "").strip()
	if len(cleaned) <= limit:
		return cleaned
	return cleaned[: limit - 3] + "..."


def build_tool_evidence_summary(parsed_sample: Dict[str, Any], max_tools: int = 6) -> str:
	lines = []
	for tool in parsed_sample.get("logical_tools", [])[:max_tools]:
		tool_name = tool.get("tool_name", "")
		output_text = truncate_for_judge(_serialize_payload(tool.get("output")), limit=350)
		lines.append(f"- {tool_name}: {output_text}")
	return "\n".join(lines) if lines else "- None"


def build_reference_points(parsed_sample: Dict[str, Any], max_points: int = 8) -> str:
	units = split_answer_units(parsed_sample.get("final_answer", ""))[:max_points]
	if not units:
		return "- None"
	return "\n".join(f"- {unit}" for unit in units)


def build_llm_judge_prompt(parsed_sample: Dict[str, Any], response: str) -> str:
	return (
		"Evaluate whether the candidate answer correctly addresses the user's request using the available tool evidence.\n\n"
		"User request:\n"
		f"{truncate_for_judge(parsed_sample.get('user_request', ''), limit=500)}\n\n"
		"Reference final answer:\n"
		f"{truncate_for_judge(parsed_sample.get('final_answer', ''), limit=700)}\n\n"
		"Reference answer points:\n"
		f"{build_reference_points(parsed_sample)}\n\n"
		"Tool evidence:\n"
		f"{build_tool_evidence_summary(parsed_sample)}\n\n"
		"Candidate answer:\n"
		f"{truncate_for_judge(response, limit=900)}\n\n"
		"Scoring rubric:\n"
		"- score = 1.0 if the candidate is substantially correct and covers most major requested results.\n"
		"- score = 0.5 if the candidate is partially correct, faithfully using at least one major tool-backed result but missing others.\n"
		"- score = 0.0 if the candidate is mostly irrelevant, fabricated, or refuses the task.\n"
		"Set pass = 1 if score >= 0.5, else pass = 0.\n"
		"Return JSON only with keys: score, pass, label, matched_points, missing_points."
	)


def parse_llm_judge_output(raw_output: str, fallback_judge: Dict[str, Any]) -> Dict[str, Any]:
	start = raw_output.find("{")
	end = raw_output.rfind("}")
	parsed: Optional[Dict[str, Any]] = None
	if start != -1 and end != -1 and end > start:
		candidate = raw_output[start : end + 1]
		for loader in (json.loads, ast.literal_eval):
			try:
				loaded = loader(candidate)
				if isinstance(loaded, dict):
					parsed = loaded
					break
			except Exception:
				continue

	if parsed is None:
		fallback_score = float(fallback_judge["simple_judge_score"])
		return {
			"score": fallback_score,
			"pass": int(fallback_score >= 0.5),
			"label": "fallback_parse_failed",
			"matched_points": [],
			"missing_points": [],
			"raw_output": raw_output,
		}

	score = float(parsed.get("score", fallback_judge["simple_judge_score"]))
	judge_pass = int(score >= 0.5)
	return {
		"score": score,
		"pass": judge_pass,
		"label": str(parsed.get("label", "partial")),
		"matched_points": parsed.get("matched_points", []),
		"missing_points": parsed.get("missing_points", []),
		"raw_output": raw_output,
	}


def run_llm_judge(
	parsed_samples: List[Dict[str, Any]],
	res_list: List[Dict[str, Any]],
	judge_model_path: str,
	max_new_tokens: int,
) -> List[Dict[str, Any]]:
	judge_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
	judge_tokenizer = AutoTokenizer.from_pretrained(judge_model_path, use_fast=True)
	if judge_tokenizer.pad_token_id is None:
		judge_tokenizer.pad_token = judge_tokenizer.eos_token
	judge_model = AutoModelForCausalLM.from_pretrained(
		judge_model_path,
		torch_dtype=judge_dtype,
		device_map="auto" if torch.cuda.is_available() else None,
	)
	judge_model.eval()
	judge_device = next(judge_model.parameters()).device

	results: List[Dict[str, Any]] = []
	progress = auto_tqdm(range(len(res_list)), desc="LLM judge")
	for index in progress:
		parsed_sample = parsed_samples[index]
		entry = res_list[index]
		prompt = build_llm_judge_prompt(parsed_sample, entry.get("response_for_scoring", entry["response"]))
		messages = [
			{"role": "system", "content": "You are a precise evaluation judge. Output JSON only."},
			{"role": "user", "content": prompt},
		]
		input_ids = judge_tokenizer.apply_chat_template(
			messages,
			add_generation_prompt=True,
			return_tensors="pt",
		).to(judge_device)
		attention_mask = torch.ones_like(input_ids)
		with torch.no_grad():
			outputs = judge_model.generate(
				input_ids=input_ids,
				attention_mask=attention_mask,
				max_new_tokens=max_new_tokens,
				do_sample=False,
				pad_token_id=judge_tokenizer.pad_token_id,
				eos_token_id=judge_tokenizer.eos_token_id,
			)
		decoded = judge_tokenizer.decode(outputs[0, input_ids.shape[1] :], skip_special_tokens=True)
		results.append(parse_llm_judge_output(decoded, entry["simple_judge"]))

	del judge_model
	del judge_tokenizer
	if torch.cuda.is_available():
		torch.cuda.empty_cache()
	return results


def resolve_primary_accuracy(
	primary_metric: str,
	strict_em_accuracy: float,
	decomposed_pass_accuracy: float,
	llm_judge_accuracy: Optional[float],
) -> Tuple[str, float]:
	if primary_metric == "llm_judge_pass":
		if llm_judge_accuracy is None:
			raise ValueError("primary_metric=llm_judge_pass requires a judge model path.")
		return "llm_judge_pass", llm_judge_accuracy
	if primary_metric == "decomposed_pass":
		return "decomposed_pass", decomposed_pass_accuracy
	return "strict_em", strict_em_accuracy


def _clean_user_request(text: str) -> str:
	cleaned = (text or "").strip()
	if cleaned.endswith("Begin!"):
		cleaned = cleaned[: -len("Begin!")].rstrip()
	return cleaned


def _serialize_payload(payload: Any) -> str:
	if payload is None:
		return "None"
	if isinstance(payload, str):
		return payload
	try:
		return json.dumps(payload, ensure_ascii=False, sort_keys=True)
	except TypeError:
		return str(payload)


def parse_async_sample(example: Dict[str, Any]) -> Dict[str, Any]:
	user_request = _clean_user_request(example.get("user_request", ""))
	final_answer = (example.get("final_answer") or "").strip()
	planning_segments = sorted(
		example.get("planning_segments", []),
		key=lambda item: (item.get("start_time", 0.0), item.get("end_time", 0.0)),
	)

	logical_tools: List[Dict[str, Any]] = []
	call_id_to_tool: Dict[str, Dict[str, Any]] = {}
	for fallback_index, tool in enumerate(example.get("tool_calls", []), start=1):
		logical_index = int(tool.get("logical_index", fallback_index))
		logical_tool = {
			"seg_id": logical_index,
			"logical_index": logical_index,
			"tool_name": tool.get("tool_name", ""),
			"arguments": tool.get("arguments", {}),
			"output": tool.get("output"),
			"call_id": tool.get("call_id", f"logical_{logical_index}"),
			"start_time": tool.get("start_time", 0.3),
			"end_time": tool.get("end_time", 0.3),
		}
		logical_tools.append(logical_tool)
		call_id_to_tool[logical_tool["call_id"]] = logical_tool

	logical_tools = sorted(logical_tools, key=lambda item: item["logical_index"])

	tool_returns: List[Dict[str, Any]] = []
	for event in sorted(example.get("event_list", []), key=lambda item: item.get("timestamp", 0.0)):
		if event.get("type") != "tool_return":
			continue
		call_id = event.get("call_id", "")
		base_tool = call_id_to_tool.get(call_id)
		if base_tool is None:
			continue
		tool_returns.append(
			{
				**base_tool,
				"output": event.get("output", base_tool.get("output")),
				"timestamp": event.get("timestamp", base_tool.get("end_time", 0.0)),
				"return_order": event.get("return_order", len(tool_returns)),
			}
		)

	if not tool_returns:
		tool_returns = sorted(
			[{**tool, "timestamp": tool.get("end_time", 0.0)} for tool in logical_tools],
			key=lambda item: (item["timestamp"], item["logical_index"]),
		)

	return {
		"user_request": user_request,
		"planning_segments": planning_segments,
		"final_answer": final_answer,
		"logical_tools": logical_tools,
		"tool_returns": tool_returns,
	}


def build_static_prefix(parsed_sample: Dict[str, Any], tokenizer) -> Fragment:
	planning_texts = [segment.get("text", "").strip() for segment in parsed_sample["planning_segments"] if segment.get("text")]
	pieces = [SYSTEM_PROMPT, "[User Request]\n", parsed_sample["user_request"], "\n\n"]
	if planning_texts:
		pieces.append("[Planning]\n")
		pieces.append("\n\n".join(planning_texts))
		pieces.append("\n\n")
	token_ids = tokenizer("".join(pieces), add_special_tokens=False)["input_ids"]
	return Fragment(token_ids=token_ids, segment_ids=[0] * len(token_ids))


def build_tool_block(tool: Dict[str, Any], tokenizer, seg_id: int) -> Fragment:
	tool_text = (
		f"[Tool {seg_id}: {tool.get('tool_name', '')}]\n"
		f"Arguments: {_serialize_payload(tool.get('arguments', {}))}\n"
		f"Output: {_serialize_payload(tool.get('output'))}\n"
	)
	token_ids = tokenizer(tool_text, add_special_tokens=False)["input_ids"]
	return Fragment(token_ids=token_ids, segment_ids=[seg_id] * len(token_ids))


def build_link_fragment(seg_id: int, reencode_num: int, special_token_start: int) -> Fragment:
	link_ids = [special_token_start + reencode_num * (seg_id - 1) + offset for offset in range(reencode_num)]
	return Fragment(token_ids=link_ids, segment_ids=[0] * len(link_ids))


def build_tool_with_link(tool: Dict[str, Any], tokenizer, seg_id: int, reencode_num: int, special_token_start: int) -> Fragment:
	block_fragment = build_tool_block(tool, tokenizer, seg_id)
	link_fragment = build_link_fragment(seg_id, reencode_num, special_token_start)
	return Fragment(
		token_ids=block_fragment.token_ids + link_fragment.token_ids,
		segment_ids=block_fragment.segment_ids + link_fragment.segment_ids,
	)


def merge_fragments(fragments: List[Fragment]) -> Fragment:
	merged_token_ids: List[int] = []
	merged_segment_ids: List[int] = []
	for fragment in fragments:
		merged_token_ids.extend(fragment.token_ids)
		merged_segment_ids.extend(fragment.segment_ids)
	return Fragment(token_ids=merged_token_ids, segment_ids=merged_segment_ids)


def get_planning_end_time(parsed_sample: Dict[str, Any]) -> float:
	planning_segments = parsed_sample.get("planning_segments", [])
	if not planning_segments:
		return 0.0
	return max(float(segment.get("end_time", 0.0)) for segment in planning_segments)


def compute_overlap_aware_metrics(
	static_release_time: float,
	static_duration: float,
	append_events: List[Dict[str, Any]],
	last_tool_time: float,
	generation_duration: float,
) -> Dict[str, Any]:
	tasks = [
		{
			"phase": "static",
			"release_time": static_release_time,
			"duration": static_duration,
		}
	]
	tasks.extend(append_events)

	current_time = 0.0
	append_compute_before_last_tool = 0.0
	total_append_compute = 0.0
	static_compute_before_last_tool = 0.0

	for task in tasks:
		release_time = float(task.get("release_time", 0.0))
		duration = float(task.get("duration", 0.0))
		start_time = max(current_time, release_time)
		finish_time = start_time + duration
		overlap_before_last_tool = max(0.0, min(finish_time, last_tool_time) - start_time)

		if task.get("phase") == "append":
			total_append_compute += duration
			append_compute_before_last_tool += overlap_before_last_tool
		elif task.get("phase") == "static":
			static_compute_before_last_tool += overlap_before_last_tool

		current_time = finish_time

	prefill_ready_time = current_time
	generation_start_time = max(prefill_ready_time, last_tool_time)
	overlap_total_time = generation_start_time + generation_duration
	remaining_prefill_after_last_tool = max(0.0, prefill_ready_time - last_tool_time)
	overlap_after_last_tool = overlap_total_time - last_tool_time
	append_overlap_ratio = 0.0 if total_append_compute == 0.0 else append_compute_before_last_tool / total_append_compute
	visible_compute_without_overlap = static_duration + total_append_compute + generation_duration
	hidden_prefill_time = static_compute_before_last_tool + append_compute_before_last_tool
	overlap_time_saved = hidden_prefill_time

	return {
		"overlap_aware_total_time": overlap_total_time,
		"overlap_aware_after_last_tool": overlap_after_last_tool,
		"remaining_prefill_after_last_tool": remaining_prefill_after_last_tool,
		"append_overlap_ratio": append_overlap_ratio,
		"append_compute_hidden_before_last_tool": append_compute_before_last_tool,
		"static_compute_hidden_before_last_tool": static_compute_before_last_tool,
		"hidden_prefill_time": hidden_prefill_time,
		"visible_compute_without_overlap": visible_compute_without_overlap,
		"overlap_time_saved": overlap_time_saved,
	}


class IncrementalPrefillRunner:
	def __init__(self, model, tokenizer, device: torch.device, attn_type: str):
		self.model = model
		self.tokenizer = tokenizer
		self.device = device
		self.attn_type = attn_type
		self.timings: Dict[str, float] = {}

	def _record_time(self, key: str, value: float):
		self.timings[key] = self.timings.get(key, 0.0) + value

	def _record_count(self, key: str, value: int = 1):
		self.timings[key] = self.timings.get(key, 0.0) + float(value)

	def prefill_static(self, fragment: Fragment, position_ids: Optional[List[int]] = None):
		total_start = time.perf_counter()
		self._record_count("static_prefill_calls", 1)
		self._record_count("static_prefill_tokens", len(fragment.token_ids))
		input_ids = torch.LongTensor([fragment.token_ids])
		input_ids = move_to_target_device(input_ids, self.device)
		position_ids_tensor = None
		if position_ids is not None:
			position_ids_tensor = torch.LongTensor([position_ids])
			position_ids_tensor = move_to_target_device(position_ids_tensor, self.device)

		mask_start = time.perf_counter()
		if self.attn_type == "blocked":
			segment_ids = torch.LongTensor([fragment.segment_ids])
			attention_mask = make_segment_mask(
				source_segments=segment_ids,
				target_segments=segment_ids,
				add_causal_lm_mask=True,
			).unsqueeze(1)
			attention_mask = move_to_target_device(attention_mask, self.device)
			if attention_mask.dtype != self.model.dtype:
				attention_mask = attention_mask.to(self.model.dtype)
		else:
			attention_mask = torch.ones_like(input_ids)
			attention_mask = move_to_target_device(attention_mask, self.device)
		mask_elapsed = time.perf_counter() - mask_start
		self._record_time("static_mask_time", mask_elapsed)

		forward_start = time.perf_counter()
		with torch.no_grad():
			outputs = self.model(
				input_ids=input_ids,
				attention_mask=attention_mask,
				position_ids=position_ids_tensor,
				use_cache=True,
			)
		forward_elapsed = time.perf_counter() - forward_start
		self._record_time("static_forward_time", forward_elapsed)
		total_elapsed = time.perf_counter() - total_start
		return outputs.past_key_values, {
			"wall_time": total_elapsed,
			"mask_time": mask_elapsed,
			"forward_time": forward_elapsed,
		}

	def append_fragment(
		self,
		fragment: Fragment,
		accumulated_segment_ids: List[int],
		past_key_values,
		position_ids: Optional[List[int]] = None,
	):
		total_start = time.perf_counter()
		self._record_count("append_calls", 1)
		self._record_count("append_tokens", len(fragment.token_ids))
		self._record_count("append_context_tokens", len(accumulated_segment_ids))
		fragment_input_ids = torch.LongTensor([fragment.token_ids])
		fragment_input_ids = move_to_target_device(fragment_input_ids, self.device)
		position_ids_tensor = None
		if position_ids is not None:
			position_ids_tensor = torch.LongTensor([position_ids])
			position_ids_tensor = move_to_target_device(position_ids_tensor, self.device)

		mask_start = time.perf_counter()
		if self.attn_type == "blocked":
			full_segment_ids = torch.LongTensor([accumulated_segment_ids])
			full_segment_ids = move_to_target_device(full_segment_ids, self.device)
			full_attention_mask = make_segment_mask(
				source_segments=full_segment_ids,
				target_segments=full_segment_ids,
				add_causal_lm_mask=True,
			).unsqueeze(1)
			full_attention_mask = move_to_target_device(full_attention_mask, self.device)
			if full_attention_mask.dtype != self.model.dtype:
				full_attention_mask = full_attention_mask.to(self.model.dtype)
			query_length = fragment_input_ids.size(1)
			attention_mask = full_attention_mask[:, :, -query_length:, :]
		else:
			attention_mask = torch.ones((1, len(accumulated_segment_ids)), dtype=torch.long)
			attention_mask = move_to_target_device(attention_mask, self.device)
		mask_elapsed = time.perf_counter() - mask_start
		self._record_time("append_mask_time", mask_elapsed)

		forward_start = time.perf_counter()
		with torch.no_grad():
			outputs = self.model(
				input_ids=fragment_input_ids,
				attention_mask=attention_mask,
				position_ids=position_ids_tensor,
				past_key_values=past_key_values,
				use_cache=True,
			)
		forward_elapsed = time.perf_counter() - forward_start
		self._record_time("append_forward_time", forward_elapsed)
		total_elapsed = time.perf_counter() - total_start
		return outputs.past_key_values, {
			"wall_time": total_elapsed,
			"mask_time": mask_elapsed,
			"forward_time": forward_elapsed,
		}

	def generate_response(self, accumulated_ids: List[int], assistant_ids: torch.Tensor, generation_cfg: GenerationConfig, past_key_values):
		full_input_ids = torch.LongTensor([accumulated_ids])
		full_input_ids = move_to_target_device(full_input_ids, self.device)
		full_attention_mask = torch.ones_like(full_input_ids)

		generation_prefix = assistant_ids.unsqueeze(0)
		generation_input_ids = torch.cat([full_input_ids, generation_prefix], dim=1)
		generation_attention_mask = torch.cat([full_attention_mask, torch.ones_like(generation_prefix)], dim=1)
		self._record_count("generation_calls", 1)
		self._record_count("generation_prompt_tokens", generation_input_ids.size(1))

		ttft_start = time.time()
		with torch.no_grad():
			outputs = self.model.generate(
				input_ids=generation_input_ids,
				attention_mask=generation_attention_mask,
				use_cache=True,
				generation_config=generation_cfg,
				past_key_values=past_key_values,
				tokenizer=self.tokenizer,
			)
		ttft = time.time() - ttft_start
		self._record_time("generate_wall_time", ttft)

		generated_seq = self.tokenizer.decode(outputs[0, full_input_ids.size(1):].tolist())
		response = generated_seq.split("<|start_header_id|>assistant<|end_header_id|>")[-1].strip().split("<|eot_id|>")[0]
		return response, ttft


class BaseSchedule:
	requires_blocked = False
	canonical_interleaved_eval = False
	canonical_interleaved_link_policy: Optional[str] = None

	def __init__(self, tokenizer, reencode_num: int, special_token_start: int):
		self.tokenizer = tokenizer
		self.reencode_num = reencode_num
		self.special_token_start = special_token_start

	def on_arrival(self, tool: Dict[str, Any]) -> List[Fragment]:
		raise NotImplementedError

	def finalize(self) -> List[Fragment]:
		return []


def build_canonical_interleaved_fragments(
	logical_tools: List[Dict[str, Any]],
	tokenizer,
	reencode_num: int,
	special_token_start: int,
) -> List[Fragment]:
	fragments: List[Fragment] = []
	for tool in sorted(logical_tools, key=lambda item: item["seg_id"]):
		fragments.append(
			build_tool_with_link(
				tool,
				tokenizer,
				tool["seg_id"],
				reencode_num,
				special_token_start,
			)
		)
	return fragments


def build_canonical_fragment_plan(
	logical_tools: List[Dict[str, Any]],
	tokenizer,
	reencode_num: int,
	special_token_start: int,
	static_length: int,
) -> Dict[str, Any]:
	block_fragments: Dict[int, Fragment] = {}
	link_fragments: Dict[int, Fragment] = {}
	block_position_ids: Dict[int, List[int]] = {}
	link_position_ids: Dict[int, List[int]] = {}
	canonical_token_ids: List[int] = []
	canonical_segment_ids: List[int] = []
	offset = static_length

	for tool in sorted(logical_tools, key=lambda item: item["seg_id"]):
		seg_id = tool["seg_id"]
		block_fragment = build_tool_block(tool, tokenizer, seg_id)
		link_fragment = build_link_fragment(seg_id, reencode_num, special_token_start)
		block_fragments[seg_id] = block_fragment
		link_fragments[seg_id] = link_fragment
		block_position_ids[seg_id] = list(range(offset, offset + len(block_fragment.token_ids)))
		offset += len(block_fragment.token_ids)
		link_position_ids[seg_id] = list(range(offset, offset + len(link_fragment.token_ids)))
		offset += len(link_fragment.token_ids)
		canonical_token_ids.extend(block_fragment.token_ids)
		canonical_token_ids.extend(link_fragment.token_ids)
		canonical_segment_ids.extend(block_fragment.segment_ids)
		canonical_segment_ids.extend(link_fragment.segment_ids)

	return {
		"block_fragments": block_fragments,
		"link_fragments": link_fragments,
		"block_position_ids": block_position_ids,
		"link_position_ids": link_position_ids,
		"canonical_token_ids": canonical_token_ids,
		"canonical_segment_ids": canonical_segment_ids,
	}


def concat_past_key_values(cache_list: List[Any]):
	valid_caches = [cache for cache in cache_list if cache is not None]
	if not valid_caches:
		return None
	num_layers = len(valid_caches[0])
	combined = []
	for layer_idx in range(num_layers):
		layer_parts = [cache[layer_idx] for cache in valid_caches]
		combined.append(
			tuple(torch.cat([part[tensor_idx] for part in layer_parts], dim=-2) for tensor_idx in range(len(layer_parts[0])))
		)
	return tuple(combined)


def generate_online_canonical_interleaved_response(
	model,
	tokenizer,
	device: torch.device,
	attn_type: str,
	parsed_sample: Dict[str, Any],
	assistant_ids: torch.Tensor,
	generation_cfg: GenerationConfig,
	reencode_num: int,
	special_token_start: int,
	link_policy: str,
) -> str:
	replay_runner = IncrementalPrefillRunner(model, tokenizer, device, attn_type)
	static_fragment = build_static_prefix(parsed_sample, tokenizer)
	static_position_ids = list(range(len(static_fragment.token_ids)))
	static_cache, _ = replay_runner.prefill_static(static_fragment, position_ids=static_position_ids)
	fragment_plan = build_canonical_fragment_plan(
		parsed_sample["logical_tools"],
		tokenizer,
		reencode_num,
		special_token_start,
		static_length=len(static_fragment.token_ids),
	)

	block_caches: Dict[int, Any] = {}
	link_caches: Dict[int, Any] = {}
	arrived_blocks: Set[int] = set()
	next_link = 1
	logical_tools_by_seg = {tool["seg_id"]: tool for tool in parsed_sample["logical_tools"]}

	def compute_block(seg_id: int):
		if seg_id in block_caches:
			return
		block_cache, _ = replay_runner.prefill_static(
			fragment_plan["block_fragments"][seg_id],
			position_ids=fragment_plan["block_position_ids"][seg_id],
		)
		block_caches[seg_id] = block_cache

	def compute_link(seg_id: int):
		prefix_caches = [static_cache]
		prefix_segment_ids = list(static_fragment.segment_ids)
		for prev_seg_id in range(1, seg_id):
			prefix_caches.append(block_caches[prev_seg_id])
			prefix_segment_ids.extend(fragment_plan["block_fragments"][prev_seg_id].segment_ids)
			prefix_caches.append(link_caches[prev_seg_id])
			prefix_segment_ids.extend(fragment_plan["link_fragments"][prev_seg_id].segment_ids)
		prefix_caches.append(block_caches[seg_id])
		prefix_segment_ids.extend(fragment_plan["block_fragments"][seg_id].segment_ids)
		prefix_cache = concat_past_key_values(prefix_caches)
		full_segment_ids = prefix_segment_ids + fragment_plan["link_fragments"][seg_id].segment_ids
		link_cache, _ = replay_runner.append_fragment(
			fragment_plan["link_fragments"][seg_id],
			full_segment_ids,
			prefix_cache,
			position_ids=fragment_plan["link_position_ids"][seg_id],
		)
		link_caches[seg_id] = link_cache

	for tool_return in parsed_sample["tool_returns"]:
		seg_id = tool_return["seg_id"]
		arrived_blocks.add(seg_id)
		compute_block(seg_id)
		if link_policy == "link_chain":
			while next_link in arrived_blocks and next_link not in link_caches:
				compute_link(next_link)
				next_link += 1

	if link_policy == "link_last":
		for seg_id in sorted(logical_tools_by_seg):
			if seg_id not in block_caches:
				compute_block(seg_id)
			compute_link(seg_id)
	else:
		for seg_id in sorted(logical_tools_by_seg):
			if seg_id not in link_caches:
				compute_link(seg_id)

	final_caches = [static_cache]
	for seg_id in sorted(logical_tools_by_seg):
		final_caches.append(block_caches[seg_id])
		final_caches.append(link_caches[seg_id])
	final_cache = concat_past_key_values(final_caches)
	canonical_input_ids = list(static_fragment.token_ids) + fragment_plan["canonical_token_ids"]
	response, _ = replay_runner.generate_response(canonical_input_ids, assistant_ids, generation_cfg, final_cache)
	return response


def generate_canonical_interleaved_response(
	model,
	tokenizer,
	device: torch.device,
	attn_type: str,
	parsed_sample: Dict[str, Any],
	assistant_ids: torch.Tensor,
	generation_cfg: GenerationConfig,
	reencode_num: int,
	special_token_start: int,
	link_policy: Optional[str] = None,
) -> str:
	if link_policy in {"link_last", "link_chain"}:
		return generate_online_canonical_interleaved_response(
			model=model,
			tokenizer=tokenizer,
			device=device,
			attn_type=attn_type,
			parsed_sample=parsed_sample,
			assistant_ids=assistant_ids,
			generation_cfg=generation_cfg,
			reencode_num=reencode_num,
			special_token_start=special_token_start,
			link_policy=link_policy,
		)

	replay_runner = IncrementalPrefillRunner(model, tokenizer, device, attn_type)
	static_fragment = build_static_prefix(parsed_sample, tokenizer)
	past_key_values, _ = replay_runner.prefill_static(static_fragment)
	accumulated_ids = list(static_fragment.token_ids)
	accumulated_segment_ids = list(static_fragment.segment_ids)

	for fragment in build_canonical_interleaved_fragments(
		parsed_sample["logical_tools"],
		tokenizer,
		reencode_num,
		special_token_start,
	):
		accumulated_ids.extend(fragment.token_ids)
		accumulated_segment_ids.extend(fragment.segment_ids)
		past_key_values, _ = replay_runner.append_fragment(fragment, accumulated_segment_ids, past_key_values)

	response, _ = replay_runner.generate_response(accumulated_ids, assistant_ids, generation_cfg, past_key_values)
	return response


class SyncSchedule(BaseSchedule):
	def __init__(self, tokenizer, reencode_num: int, special_token_start: int):
		super().__init__(tokenizer, reencode_num, special_token_start)
		self.arrived: Dict[int, Dict[str, Any]] = {}

	def on_arrival(self, tool: Dict[str, Any]) -> List[Fragment]:
		self.arrived[tool["seg_id"]] = tool
		return []

	def finalize(self) -> List[Fragment]:
		fragments = []
		for seg_id in sorted(self.arrived):
			fragments.append(
				build_tool_with_link(
					self.arrived[seg_id],
					self.tokenizer,
					seg_id,
					self.reencode_num,
					self.special_token_start,
				)
			)
		return fragments


class ChunkSchedule(BaseSchedule):
	def __init__(self, tokenizer, reencode_num: int, special_token_start: int):
		super().__init__(tokenizer, reencode_num, special_token_start)
		self.arrived: Dict[int, Dict[str, Any]] = {}
		self.next_expected = 1

	def on_arrival(self, tool: Dict[str, Any]) -> List[Fragment]:
		self.arrived[tool["seg_id"]] = tool
		fragments = []
		while self.next_expected in self.arrived:
			current_tool = self.arrived.pop(self.next_expected)
			fragments.append(
				build_tool_with_link(
					current_tool,
					self.tokenizer,
					self.next_expected,
					self.reencode_num,
					self.special_token_start,
				)
			)
			self.next_expected += 1
		return fragments


class BlockThenLinksSchedule(BaseSchedule):
	requires_blocked = True

	def __init__(self, tokenizer, reencode_num: int, special_token_start: int):
		super().__init__(tokenizer, reencode_num, special_token_start)
		self.logical_tools: Dict[int, Dict[str, Any]] = {}

	def on_arrival(self, tool: Dict[str, Any]) -> List[Fragment]:
		self.logical_tools[tool["seg_id"]] = tool
		return [build_tool_block(tool, self.tokenizer, tool["seg_id"])]

	def finalize(self) -> List[Fragment]:
		link_fragments = [
			build_link_fragment(seg_id, self.reencode_num, self.special_token_start)
			for seg_id in sorted(self.logical_tools)
		]
		return [merge_fragments(link_fragments)] if link_fragments else []


class BlockAndLinkChainSchedule(BaseSchedule):
	requires_blocked = True

	def __init__(self, tokenizer, reencode_num: int, special_token_start: int):
		super().__init__(tokenizer, reencode_num, special_token_start)
		self.block_done = set()
		self.next_link = 1

	def on_arrival(self, tool: Dict[str, Any]) -> List[Fragment]:
		seg_id = tool["seg_id"]
		fragments = [build_tool_block(tool, self.tokenizer, seg_id)]
		self.block_done.add(seg_id)

		while self.next_link in self.block_done:
			fragments.append(build_link_fragment(self.next_link, self.reencode_num, self.special_token_start))
			self.next_link += 1
		return [merge_fragments(fragments)]


class BlockThenLinksCanonicalInterleavedSchedule(BlockThenLinksSchedule):
	canonical_interleaved_eval = True
	canonical_interleaved_link_policy = "link_last"


class BlockAndLinkChainCanonicalInterleavedSchedule(BlockAndLinkChainSchedule):
	canonical_interleaved_eval = True
	canonical_interleaved_link_policy = "link_chain"


SCHEDULE_REGISTRY = {
	"sync": SyncSchedule,
	"chunk": ChunkSchedule,
	"block_then_links": BlockThenLinksSchedule,
	"block_link_chain": BlockAndLinkChainSchedule,
	"block_then_links_canonical": BlockThenLinksCanonicalInterleavedSchedule,
	"block_link_chain_canonical": BlockAndLinkChainCanonicalInterleavedSchedule,
}


def load_model_weights(ckpt_path: str):
	safe_tensor_file = os.path.join(ckpt_path, "model.safetensors")
	if os.path.exists(safe_tensor_file):
		state_dict = {}
		with safe_open(safe_tensor_file, framework="pt", device="cpu") as handle:
			for key in handle.keys():
				state_dict[key] = handle.get_tensor(key)
		return state_dict

	state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=False)
	if isinstance(state_dict, dict) and "model" in state_dict:
		state_dict = state_dict["model"]
	if "tok_embeddings.weight" in state_dict and "output.weight" not in state_dict:
		state_dict["output.weight"] = state_dict["tok_embeddings.weight"]

	converted_state_dict = tune_to_hf(
		state_dict=state_dict,
		num_heads=32,
		num_kv_heads=8,
		dim=2048,
	)
	return converted_state_dict


def load_tokenizer_and_model(model_path: Optional[str], hf: bool, device: torch.device):
	tokenizer = AutoTokenizer.from_pretrained(DEFAULT_BASE_MODEL, use_fast=True)
	tokenizer.pad_token_id = 128004
	tokenizer.pad_token = "<|finetune_right_pad_id|>"

	if model_path and hf:
		model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16)
	else:
		model = LlamaForCausalLM.from_pretrained(DEFAULT_BASE_MODEL, torch_dtype=torch.bfloat16)
		if model_path:
			state_dict = load_model_weights(model_path)
			model.load_state_dict(state_dict, strict=False)

	model = model.to(device)
	model.eval()
	return tokenizer, model


def build_generation_config(tokenizer) -> Tuple[GenerationConfig, torch.Tensor]:
	eot_id = tokenizer.convert_tokens_to_ids("<|eot_id|>")
	generation_cfg = GenerationConfig(
		do_sample=False,
		num_beams=1,
		max_new_tokens=200,
		stop_strings=["<|end_of_text|>", "<|eot_id|>"],
		pad_token_id=tokenizer.pad_token_id,
		eos_token_id=eot_id,
	)
	assistant_ids = tokenizer(ASSISTANT_PROMPT, add_special_tokens=False)["input_ids"]
	return generation_cfg, torch.LongTensor(assistant_ids)


def run_benchmark(args, schedule_name: str, display_name: str, output_prefix: str):
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	special_token_start = 128011

	dataset_list = load_jsonl_dataset(args.dataset_path, args.num_samples)
	parsed_samples = [parse_async_sample(item) for item in dataset_list]
	parsed_samples = [item for item in parsed_samples if item["user_request"] and item["final_answer"] and item["logical_tools"]]
	if not parsed_samples:
		raise ValueError("No valid samples with non-empty final answers were found in the dataset.")

	schedule_cls = SCHEDULE_REGISTRY[schedule_name]
	if schedule_cls.requires_blocked and args.attn_type != "blocked":
		raise ValueError(f"{display_name} only supports blocked attention.")

	tokenizer, model = load_tokenizer_and_model(args.model_path, args.hf, device)
	generation_cfg, assistant_ids = build_generation_config(tokenizer)
	assistant_ids = move_to_target_device(assistant_ids, device)

	print("=" * 80)
	print(display_name)
	print(f"Samples: {len(parsed_samples)} | Attention: {args.attn_type}")
	print("=" * 80)

	runner = IncrementalPrefillRunner(model, tokenizer, device, args.attn_type)
	timing_keys = [
		"static_mask_time",
		"static_forward_time",
		"append_mask_time",
		"append_forward_time",
		"generate_wall_time",
		"append_calls",
		"append_tokens",
		"append_context_tokens",
	]
	timing_lists: Dict[str, List[float]] = {key: [] for key in timing_keys}

	correct_num = 0
	res_list: List[Dict[str, Any]] = []
	total_compute_time = 0.0
	total_time_with_wait = 0.0
	total_wait_time = 0.0
	ttft_list: List[float] = []
	real_ttft_list: List[float] = []
	ttft_after_last_tool_list: List[float] = []
	prefill_time_list: List[float] = []
	simple_judge_scores: List[float] = []
	answer_unit_scores: List[float] = []
	tool_coverage_scores: List[float] = []
	overlap_aware_total_time_list: List[float] = []
	overlap_aware_after_last_tool_list: List[float] = []
	remaining_prefill_after_last_tool_list: List[float] = []
	append_overlap_ratio_list: List[float] = []
	hidden_prefill_time_list: List[float] = []
	decomposed_pass_list: List[float] = []

	progress = auto_tqdm(range(len(parsed_samples)))
	for parsed_sample in parsed_samples:
		sample_timing_start = dict(runner.timings)
		sample_start = time.time()
		schedule = schedule_cls(tokenizer, args.reencode_num, special_token_start)
		static_fragment = build_static_prefix(parsed_sample, tokenizer)
		past_key_values, static_prefill_meta = runner.prefill_static(static_fragment)
		accumulated_ids = list(static_fragment.token_ids)
		accumulated_segment_ids = list(static_fragment.segment_ids)
		append_events: List[Dict[str, Any]] = []

		prefill_start = time.time()
		simulated_experiment_time = 0.0
		for tool_return in parsed_sample["tool_returns"]:
			tool_arrival_time = tool_return.get("timestamp", 0.0)
			wait_time = max(tool_arrival_time - simulated_experiment_time, 0.0)
			total_wait_time += wait_time
			simulated_experiment_time = max(simulated_experiment_time, tool_arrival_time)

			for fragment in schedule.on_arrival(tool_return):
				accumulated_ids.extend(fragment.token_ids)
				accumulated_segment_ids.extend(fragment.segment_ids)
				past_key_values, append_meta = runner.append_fragment(fragment, accumulated_segment_ids, past_key_values)
				append_events.append(
					{
						"phase": "append",
						"release_time": tool_arrival_time,
						"duration": append_meta["wall_time"],
						"token_count": len(fragment.token_ids),
					}
				)

		for fragment in schedule.finalize():
			accumulated_ids.extend(fragment.token_ids)
			accumulated_segment_ids.extend(fragment.segment_ids)
			past_key_values, append_meta = runner.append_fragment(fragment, accumulated_segment_ids, past_key_values)
			append_events.append(
				{
					"phase": "append",
					"release_time": simulated_experiment_time,
					"duration": append_meta["wall_time"],
					"token_count": len(fragment.token_ids),
				}
			)

		prefill_time = time.time() - prefill_start
		prefill_time_list.append(prefill_time)

		ttft_after_last_start = time.time()
		response, ttft = runner.generate_response(accumulated_ids, assistant_ids, generation_cfg, past_key_values)
		response_for_scoring = response
		canonical_interleaved_response = None
		if schedule.canonical_interleaved_eval:
			canonical_interleaved_response = generate_canonical_interleaved_response(
				model=model,
				tokenizer=tokenizer,
				device=device,
				attn_type=args.attn_type,
				parsed_sample=parsed_sample,
				assistant_ids=assistant_ids,
				generation_cfg=generation_cfg,
				reencode_num=args.reencode_num,
				special_token_start=special_token_start,
			)
			response_for_scoring = canonical_interleaved_response
		ttft_after_last = time.time() - ttft_after_last_start
		overlap_metrics = compute_overlap_aware_metrics(
			static_release_time=get_planning_end_time(parsed_sample),
			static_duration=static_prefill_meta["wall_time"],
			append_events=append_events,
			last_tool_time=simulated_experiment_time,
			generation_duration=ttft,
		)

		ttft_list.append(ttft)
		ttft_after_last_tool_list.append(ttft_after_last)
		real_ttft_list.append(simulated_experiment_time + ttft_after_last)
		overlap_aware_total_time_list.append(overlap_metrics["overlap_aware_total_time"])
		overlap_aware_after_last_tool_list.append(overlap_metrics["overlap_aware_after_last_tool"])
		remaining_prefill_after_last_tool_list.append(overlap_metrics["remaining_prefill_after_last_tool"])
		append_overlap_ratio_list.append(overlap_metrics["append_overlap_ratio"])
		hidden_prefill_time_list.append(overlap_metrics["hidden_prefill_time"])

		sample_timing = {
			key: runner.timings.get(key, 0.0) - sample_timing_start.get(key, 0.0)
			for key in timing_keys
		}
		for key, value in sample_timing.items():
			timing_lists[key].append(value)

		gold_answers = [parsed_sample["final_answer"]]
		score = best_subspan_em(response_for_scoring, gold_answers)
		judge = simple_partial_judge(response_for_scoring, parsed_sample)
		decomposed_pass = decomposed_pass_from_judge(judge)
		correct_num += int(score)
		simple_judge_scores.append(judge["simple_judge_score"])
		answer_unit_scores.append(judge["answer_unit_score"])
		tool_coverage_scores.append(judge["tool_coverage_score"])
		decomposed_pass_list.append(decomposed_pass)
		res_list.append(
			{
				"response": response,
				"response_for_scoring": response_for_scoring,
				"canonical_interleaved_response": canonical_interleaved_response,
				"gold_answer": gold_answers,
				"score": score,
				"strict_em_score": score,
				"simple_judge": judge,
				"decomposed_pass": decomposed_pass,
				"prefill_time": prefill_time,
				"last_tool_time": simulated_experiment_time,
				"overlap_metrics": overlap_metrics,
				"timing_breakdown": sample_timing,
			}
		)

		compute_time = time.time() - sample_start
		total_compute_time += compute_time
		total_time_with_wait += simulated_experiment_time + compute_time
		progress.update(1)

	num_samples = len(parsed_samples)
	strict_em_accuracy = correct_num / num_samples if num_samples else 0.0
	avg_compute_time = total_compute_time / num_samples if num_samples else 0.0
	avg_total_time = total_time_with_wait / num_samples if num_samples else 0.0
	avg_prefill_time = float(np.mean(prefill_time_list)) if prefill_time_list else 0.0
	avg_ttft = float(np.mean(ttft_list)) if ttft_list else 0.0
	avg_real_ttft = float(np.mean(real_ttft_list)) if real_ttft_list else 0.0
	avg_ttft_after_last = float(np.mean(ttft_after_last_tool_list)) if ttft_after_last_tool_list else 0.0
	avg_static_mask_time = float(np.mean(timing_lists["static_mask_time"])) if timing_lists["static_mask_time"] else 0.0
	avg_static_forward_time = float(np.mean(timing_lists["static_forward_time"])) if timing_lists["static_forward_time"] else 0.0
	avg_append_mask_time = float(np.mean(timing_lists["append_mask_time"])) if timing_lists["append_mask_time"] else 0.0
	avg_append_forward_time = float(np.mean(timing_lists["append_forward_time"])) if timing_lists["append_forward_time"] else 0.0
	avg_generate_wall_time = float(np.mean(timing_lists["generate_wall_time"])) if timing_lists["generate_wall_time"] else 0.0
	avg_append_calls = float(np.mean(timing_lists["append_calls"])) if timing_lists["append_calls"] else 0.0
	avg_append_tokens = float(np.mean(timing_lists["append_tokens"])) if timing_lists["append_tokens"] else 0.0
	avg_append_context_tokens = float(np.mean(timing_lists["append_context_tokens"])) if timing_lists["append_context_tokens"] else 0.0
	avg_simple_judge_score = float(np.mean(simple_judge_scores)) if simple_judge_scores else 0.0
	avg_answer_unit_score = float(np.mean(answer_unit_scores)) if answer_unit_scores else 0.0
	avg_tool_coverage_score = float(np.mean(tool_coverage_scores)) if tool_coverage_scores else 0.0
	avg_overlap_aware_total_time = float(np.mean(overlap_aware_total_time_list)) if overlap_aware_total_time_list else 0.0
	avg_overlap_aware_after_last_tool = float(np.mean(overlap_aware_after_last_tool_list)) if overlap_aware_after_last_tool_list else 0.0
	avg_remaining_prefill_after_last_tool = float(np.mean(remaining_prefill_after_last_tool_list)) if remaining_prefill_after_last_tool_list else 0.0
	avg_append_overlap_ratio = float(np.mean(append_overlap_ratio_list)) if append_overlap_ratio_list else 0.0
	avg_hidden_prefill_time = float(np.mean(hidden_prefill_time_list)) if hidden_prefill_time_list else 0.0
	decomposed_pass_accuracy = float(np.mean(decomposed_pass_list)) if decomposed_pass_list else 0.0

	judge_model_path = getattr(args, "judge_model_path", None)
	if not judge_model_path and getattr(args, "primary_metric", "strict_em") == "llm_judge_pass" and os.path.exists(DEFAULT_LLM_JUDGE_MODEL):
		judge_model_path = DEFAULT_LLM_JUDGE_MODEL

	llm_judge_accuracy: Optional[float] = None
	avg_llm_judge_score: Optional[float] = None
	if judge_model_path:
		del runner
		del model
		del tokenizer
		if torch.cuda.is_available():
			torch.cuda.empty_cache()
		llm_judge_results = run_llm_judge(
			parsed_samples=parsed_samples,
			res_list=res_list,
			judge_model_path=judge_model_path,
			max_new_tokens=getattr(args, "judge_max_new_tokens", 96),
		)
		for entry, judge_result in zip(res_list, llm_judge_results):
			entry["llm_judge"] = judge_result
		llm_judge_accuracy = float(np.mean([item["pass"] for item in llm_judge_results])) if llm_judge_results else 0.0
		avg_llm_judge_score = float(np.mean([item["score"] for item in llm_judge_results])) if llm_judge_results else 0.0

	primary_metric_name, accuracy = resolve_primary_accuracy(
		getattr(args, "primary_metric", "strict_em"),
		strict_em_accuracy,
		decomposed_pass_accuracy,
		llm_judge_accuracy,
	)

	print(f"\nFINAL RESULTS ({display_name}):")
	print(f"Primary accuracy ({primary_metric_name}): {accuracy:.4f}")
	print(f"Strict EM accuracy: {strict_em_accuracy:.4f}")
	print(f"Decomposed pass accuracy: {decomposed_pass_accuracy:.4f}")
	if avg_llm_judge_score is not None:
		print(f"LLM judge score: {avg_llm_judge_score:.4f}")
		print(f"LLM judge accuracy: {llm_judge_accuracy:.4f}")
	print(f"Simple judge score: {avg_simple_judge_score:.4f}")
	print(f"Answer-unit score: {avg_answer_unit_score:.4f}")
	print(f"Tool-coverage score: {avg_tool_coverage_score:.4f}")
	print(f"Avg compute time/sample: {avg_compute_time:.4f}s")
	print(f"Avg total time/sample: {avg_total_time:.4f}s")
	print(f"Avg overlap-aware total latency/sample: {avg_overlap_aware_total_time:.4f}s")
	print(f"Avg overlap-aware latency after last tool/sample: {avg_overlap_aware_after_last_tool:.4f}s")
	print(f"Avg remaining prefill after last tool/sample: {avg_remaining_prefill_after_last_tool:.4f}s")
	print(f"Avg append overlap ratio/sample: {avg_append_overlap_ratio:.2%}")
	print(f"Avg hidden prefill time/sample: {avg_hidden_prefill_time:.4f}s")
	print(f"Avg prefill time/sample: {avg_prefill_time:.4f}s")
	print(f"Avg TTFT (compute only): {avg_ttft:.4f}s")
	print(f"Avg REAL TTFT (from start): {avg_real_ttft:.4f}s")
	print(f"Avg TTFT AFTER LAST TOOL ARRIVED: {avg_ttft_after_last:.4f}s")
	print(f"Avg static make_segment_mask time/sample: {avg_static_mask_time:.4f}s")
	print(f"Avg append make_segment_mask time/sample: {avg_append_mask_time:.4f}s")
	print(f"Avg static forward time/sample: {avg_static_forward_time:.4f}s")
	print(f"Avg append forward time/sample: {avg_append_forward_time:.4f}s")
	print(f"Avg generate() wall time/sample: {avg_generate_wall_time:.4f}s")
	print(f"Avg append calls/sample: {avg_append_calls:.2f} (append_fragment() invocations, not time)")

	os.makedirs(args.output_dir, exist_ok=True)
	result_file = os.path.join(args.output_dir, f"{output_prefix}_{args.attn_type}_accuracy_{accuracy:.4f}.jsonl")
	with open(result_file, "w", encoding="utf-8") as handle:
		for entry in res_list:
			handle.write(json.dumps(entry, ensure_ascii=False) + "\n")

	stats_file = os.path.join(args.output_dir, f"{output_prefix}_{args.attn_type}_stats.json")
	with open(stats_file, "w", encoding="utf-8") as handle:
		json.dump(
			{
				"accuracy": accuracy,
				"primary_metric": primary_metric_name,
				"canonical_interleaved_eval": schedule_cls.canonical_interleaved_eval,
				"strict_em_accuracy": strict_em_accuracy,
				"decomposed_pass_accuracy": decomposed_pass_accuracy,
				"avg_llm_judge_score": avg_llm_judge_score,
				"llm_judge_accuracy": llm_judge_accuracy,
				"avg_simple_judge_score": avg_simple_judge_score,
				"avg_answer_unit_score": avg_answer_unit_score,
				"avg_tool_coverage_score": avg_tool_coverage_score,
				"avg_time": avg_compute_time,
				"avg_prefill_time": avg_prefill_time,
				"avg_overlap_aware_total_time": avg_overlap_aware_total_time,
				"avg_overlap_aware_after_last_tool": avg_overlap_aware_after_last_tool,
				"avg_remaining_prefill_after_last_tool": avg_remaining_prefill_after_last_tool,
				"avg_append_overlap_ratio": avg_append_overlap_ratio,
				"avg_hidden_prefill_time": avg_hidden_prefill_time,
				"avg_ttft": avg_ttft,
				"avg_generate_wall_time": avg_generate_wall_time,
				"avg_real_ttft": avg_real_ttft,
				"avg_ttft_after_last_tool": avg_ttft_after_last,
				"avg_static_mask_time": avg_static_mask_time,
				"avg_append_mask_time": avg_append_mask_time,
				"avg_static_forward_time": avg_static_forward_time,
				"avg_append_forward_time": avg_append_forward_time,
				"avg_append_calls": avg_append_calls,
				"avg_append_tokens": avg_append_tokens,
				"avg_append_context_tokens": avg_append_context_tokens,
				"total_time_compute": total_compute_time,
				"total_wait_time": total_wait_time,
				"total_time_total": total_time_with_wait,
				"num_samples": num_samples,
				"attn_type": args.attn_type,
				"description": display_name,
				"metric_note": "accuracy reports the selected primary metric. strict_em_accuracy is the original whole-answer subspan EM. decomposed_pass_accuracy marks a sample correct if at least one gold answer unit or one tool-backed result is correctly covered. If enabled, llm_judge_accuracy uses an external instruction model to score semantic correctness against the user request, reference answer, and tool outputs. If canonical_interleaved_eval is true, accuracy is computed from an additional canonical block-link-block-link replay response, while overlap/timing metrics still reflect the online schedule itself. avg_ttft and avg_ttft_after_last_tool currently measure full generate() wall time, not true first-token latency. avg_append_calls is the average number of append_fragment() invocations per sample, not a time metric. overlap-aware metrics replay static prefill, incremental append prefill, and generation on a single compute timeline using each fragment's measured wall time and release time, so earlier append work can be hidden under tool wait.",
				"timestamp": datetime.datetime.now().isoformat(),
			},
			handle,
			indent=2,
			ensure_ascii=False,
		)

	print(f"Results saved to {result_file}")
	print(f"Stats saved to {stats_file}")
