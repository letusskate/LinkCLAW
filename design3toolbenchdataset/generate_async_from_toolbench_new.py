import argparse
import ast
import datetime
import json
import math
import os
import random
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple


DEFAULT_LLM_DAG_MODEL = "/home/dell/wyb/my_models/Qwen/Qwen/Qwen2___5-0___5B-Instruct"

LATENCY_PROFILE_PRESETS: Dict[str, Dict[str, Any]] = {
    "balanced": {
        "planning_time": 0.3,
        "reaction_time": 0.2,
        "base_latency": 5.0,
        "latency_variance": 2.0,
        "min_latency": 1.0,
        "latency_distribution": "gaussian",
        "root_start_spread": 0.0,
    },
    "wide_gap": {
        "planning_time": 0.3,
        "reaction_time": 0.2,
        "base_latency": 12.0,
        "latency_variance": 7.0,
        "min_latency": 1.0,
        "latency_distribution": "gaussian",
        "root_start_spread": 0.0,
    },
    "heavy_tail": {
        "planning_time": 0.3,
        "reaction_time": 0.2,
        "base_latency": 8.0,
        "latency_variance": 8.0,
        "min_latency": 0.8,
        "latency_distribution": "lognormal",
        "root_start_spread": 0.5,
    },
    "staggered": {
        "planning_time": 0.3,
        "reaction_time": 0.2,
        "base_latency": 6.0,
        "latency_variance": 3.0,
        "min_latency": 0.8,
        "latency_distribution": "gaussian",
        "root_start_spread": 1.5,
    },
}


@dataclass
class ToolCall:
    tool_name: str
    arguments: Dict[str, Any]
    output: Any
    logical_index: int
    call_id: str
    thought: str = ""
    dependencies: List[str] = field(default_factory=list)
    start_time: float = 0.0
    end_time: float = 0.0


@dataclass
class PlanningSegment:
    text: str
    start_time: float
    end_time: float


@dataclass
class AsyncToolBenchSample:
    user_request: str
    user_request_time: float
    planning_segments: List[PlanningSegment]
    tool_calls: List[ToolCall]
    final_answer: str
    final_answer_time: float
    event_list: List[Dict[str, Any]]
    linearized_transcript: str
    metadata: Dict[str, Any]


def _safe_json_loads(value: str) -> Any:
    try:
        return json.loads(value)
    except Exception:
        return None


def _parse_jsonish(value: str) -> Any:
    parsed = _safe_json_loads(value)
    if parsed is not None:
        return parsed
    try:
        return ast.literal_eval(value)
    except Exception:
        return value


def _truncate_text(text: Any, limit: int) -> str:
    serialized = str(text or "").strip()
    if len(serialized) <= limit:
        return serialized
    return serialized[: max(0, limit - 3)] + "..."


def _serialize_payload(value: Any, max_chars: Optional[int] = None) -> str:
    if isinstance(value, str):
        serialized = value
    else:
        try:
            serialized = json.dumps(value, ensure_ascii=False, sort_keys=True)
        except TypeError:
            serialized = str(value)
    if max_chars is None:
        return serialized
    return _truncate_text(serialized, max_chars)


def _extract_finish_answer(message: str) -> str:
    if "Action Input:" not in message:
        return ""
    raw_action_input = message.split("Action Input:", 1)[1].strip()
    parsed = _parse_jsonish(raw_action_input)
    if isinstance(parsed, dict):
        if parsed.get("return_type") == "give_up_and_restart":
            return ""
        for key in ("final_answer", "answer", "response"):
            value = parsed.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        return json.dumps(parsed, ensure_ascii=False)
    return str(parsed).strip()


def _normalize_tool_output(value: Any) -> Any:
    parsed = _parse_jsonish(value) if isinstance(value, str) else value
    if isinstance(parsed, dict):
        normalized = dict(parsed)
        response_value = normalized.get("response")
        if isinstance(response_value, str):
            reparsed = _parse_jsonish(response_value)
            if reparsed is not response_value:
                normalized["response"] = reparsed
        return normalized
    return parsed


def _thought_requires_sequential_dependency(thought: str) -> bool:
    lowered = (thought or "").lower()
    if not lowered:
        return False
    trigger_phrases = (
        "based on the previous action",
        "based on the previous actions",
        "based on the previous result",
        "based on the previous response",
        "based on the current state",
        "based on the current state and the previous actions",
        "previously called",
        "as a result of",
        "to proceed",
        "now, to proceed",
        "now to proceed",
        "using the information",
        "using this information",
        "with the information from",
        "after retrieving",
        "after receiving",
        "i previously called",
        "try a different approach",
    )
    return any(phrase in lowered for phrase in trigger_phrases)


def _extract_action(message: str) -> Tuple[Optional[str], Any, str]:
    thought = ""
    action_name = None
    action_input: Any = {}

    thought_match = re.search(r"Thought:(.*?)(?:Action:|$)", message, re.DOTALL)
    if thought_match:
        thought = thought_match.group(1).strip()

    action_match = re.search(r"Action:(.*?)(?:Action Input:|$)", message, re.DOTALL)
    if action_match:
        action_name = action_match.group(1).strip()

    if "Action Input:" in message:
        raw_action_input = message.split("Action Input:", 1)[1].strip()
        action_input = _parse_jsonish(raw_action_input)

    return action_name, action_input, thought


def _is_valid_user_request(text: str) -> bool:
    cleaned = (text or "").strip()
    if not cleaned:
        return False
    if "This is not the first time you try this task" in cleaned:
        return False
    return True


def _normalize_user_request(text: str) -> str:
    cleaned = (text or "").strip()
    if cleaned.endswith("Begin!"):
        cleaned = cleaned[: -len("Begin!")].rstrip()
    return cleaned


def _collect_atoms(value: Any) -> Set[str]:
    atoms: Set[str] = set()
    if value is None:
        return atoms
    if isinstance(value, dict):
        for nested in value.values():
            atoms.update(_collect_atoms(nested))
        return atoms
    if isinstance(value, list):
        for nested in value:
            atoms.update(_collect_atoms(nested))
        return atoms
    if isinstance(value, (int, float)):
        atoms.add(str(value))
        return atoms
    text = str(value).strip()
    if not text:
        return atoms
    atoms.add(text)
    for part in re.findall(r"[A-Za-z0-9_:\-./]{3,}", text):
        atoms.add(part)
    return atoms


def _count_parallel_roots(tool_calls: List[ToolCall], dependencies: Dict[str, Set[str]]) -> int:
    return sum(1 for call in tool_calls if not dependencies.get(call.call_id))


def _dependency_edge_count(dependencies: Dict[str, Set[str]]) -> int:
    return sum(len(dep_list) for dep_list in dependencies.values())


def _longest_dependency_depth(tool_calls: List[ToolCall], dependencies: Dict[str, Set[str]]) -> int:
    memo: Dict[str, int] = {}

    def depth(call_id: str) -> int:
        if call_id in memo:
            return memo[call_id]
        direct_deps = dependencies.get(call_id, set())
        if not direct_deps:
            memo[call_id] = 1
        else:
            memo[call_id] = 1 + max(depth(dep_id) for dep_id in direct_deps)
        return memo[call_id]

    if not tool_calls:
        return 0
    return max(depth(call.call_id) for call in tool_calls)


def _build_dependencies_from_index_map(
    tool_calls: List[ToolCall], dependency_index_map: Dict[int, List[int]]
) -> Dict[str, Set[str]]:
    id_by_index = {call.logical_index: call.call_id for call in tool_calls}
    dependencies: Dict[str, Set[str]] = {}
    for call in tool_calls:
        dep_indices = dependency_index_map.get(call.logical_index, [])
        dependencies[call.call_id] = {
            id_by_index[dep_index]
            for dep_index in dep_indices
            if dep_index in id_by_index and dep_index < call.logical_index
        }
    return dependencies


def _dependency_index_map_from_dependencies(
    tool_calls: List[ToolCall], dependencies: Dict[str, Set[str]]
) -> Dict[int, List[int]]:
    index_by_id = {call.call_id: call.logical_index for call in tool_calls}
    dependency_index_map: Dict[int, List[int]] = {}
    for call in tool_calls:
        dep_indices = sorted(index_by_id[dep_id] for dep_id in dependencies.get(call.call_id, set()) if dep_id in index_by_id)
        dependency_index_map[call.logical_index] = dep_indices
    return dependency_index_map


def _extract_source_sample_index(sample: Dict[str, Any]) -> Optional[int]:
    metadata = sample.get("metadata", {})
    sample_index = metadata.get("source_sample_index")
    if isinstance(sample_index, int):
        return sample_index
    for tool in sample.get("tool_calls", []):
        call_id = str(tool.get("call_id", ""))
        match = re.match(r"sample(\d+)_call\d+", call_id)
        if match:
            return int(match.group(1))
    return None


def _index_in_range(index: int, start_index: int, end_index: Optional[int]) -> bool:
    if index < start_index:
        return False
    if end_index is not None and index >= end_index:
        return False
    return True


class ToolBenchParser:
    def parse_conversations(self, raw_sample: Dict[str, Any], sample_index: int) -> Optional[Dict[str, Any]]:
        conversations = raw_sample.get("conversations", [])
        if not isinstance(conversations, list) or not conversations:
            return None

        user_request = ""
        for message in conversations:
            if message.get("from") == "user":
                candidate = message.get("value", "")
                if _is_valid_user_request(candidate):
                    user_request = _normalize_user_request(candidate)
                    if "Begin!" in candidate:
                        break

        if not user_request:
            return None

        planning_text = ""
        final_answer = ""
        pending_call: Optional[Dict[str, Any]] = None
        tool_calls: List[Dict[str, Any]] = []

        for message in conversations:
            role = message.get("from")
            content = (message.get("value") or "").strip()

            if role == "assistant":
                action_name, action_input, thought = _extract_action(content)
                if thought and not planning_text:
                    planning_text = thought
                if action_name:
                    if action_name == "Finish":
                        finish_answer = _extract_finish_answer(content)
                        if finish_answer:
                            final_answer = finish_answer
                        pending_call = None
                        continue
                    pending_call = {
                        "tool_name": action_name,
                        "arguments": action_input if isinstance(action_input, dict) else {"raw": action_input},
                        "output": None,
                        "thought": thought,
                    }
                elif content and not final_answer and not pending_call:
                    final_answer = content

            elif role == "function" and pending_call is not None:
                pending_call["output"] = _normalize_tool_output(content)
                tool_calls.append(pending_call)
                pending_call = None

        if pending_call is not None:
            return None
        if len(tool_calls) < 2:
            return None
        if not final_answer:
            return None

        return {
            "sample_index": sample_index,
            "user_request": user_request,
            "planning": planning_text or "I will call the necessary tools and combine their results.",
            "tool_calls": tool_calls,
            "final_answer": final_answer,
        }


class DependencyExtractor:
    policy_name = "balanced_dag"

    def extract_dependencies(
        self, tool_calls: List[ToolCall], parsed_sample: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Set[str]]:
        dependencies: Dict[str, Set[str]] = {}
        output_atoms = {call.call_id: _collect_atoms(call.output) for call in tool_calls}

        for current_index, call in enumerate(tool_calls):
            deps: Set[str] = set()
            argument_atoms = _collect_atoms(call.arguments)
            for previous_call in tool_calls[:current_index]:
                shared = argument_atoms & output_atoms[previous_call.call_id]
                shared = {value for value in shared if value and value not in {"true", "false", "null", "None"}}
                if shared:
                    deps.add(previous_call.call_id)
            if not deps and current_index > 0 and _thought_requires_sequential_dependency(call.thought):
                deps.add(tool_calls[current_index - 1].call_id)
            dependencies[call.call_id] = deps
        return dependencies

    def count_parallel_roots(self, tool_calls: List[ToolCall], dependencies: Dict[str, Set[str]]) -> int:
        return _count_parallel_roots(tool_calls, dependencies)


class AllParallelDependencyExtractor:
    policy_name = "all_parallel"

    def extract_dependencies(
        self, tool_calls: List[ToolCall], parsed_sample: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Set[str]]:
        return {call.call_id: set() for call in tool_calls}

    def count_parallel_roots(self, tool_calls: List[ToolCall], dependencies: Dict[str, Set[str]]) -> int:
        return _count_parallel_roots(tool_calls, dependencies)


class LLMDependencyExtractor:
    policy_name = "llm_dag"

    def __init__(self, args):
        self.args = args
        self.model_path = args.llm_model_path or DEFAULT_LLM_DAG_MODEL
        self.cache_path = args.llm_cache_path or self._build_default_cache_path()
        self.max_new_tokens = args.llm_max_new_tokens
        self.max_user_request_chars = args.llm_user_request_chars
        self.max_planning_chars = args.llm_planning_chars
        self.max_thought_chars = args.llm_thought_chars
        self.max_output_chars = args.llm_output_chars
        self.fallback_policy = args.llm_fallback_policy
        self.fallback_extractor = DependencyExtractor()
        self.cache: Dict[int, Dict[str, Any]] = {}
        self.model = None
        self.tokenizer = None
        self.torch = None
        self.device = None
        self._load_cache()

    def _build_default_cache_path(self) -> str:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        cache_dir = os.path.join(script_dir, "async_agent_bench", "dependency_cache")
        input_name = os.path.splitext(os.path.basename(self.args.input or "toolbench"))[0]
        model_name = os.path.basename(os.path.normpath(self.model_path))
        safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", f"{input_name}_{model_name}_llm_dag")
        return os.path.join(cache_dir, f"{safe_name}.jsonl")

    def _load_cache(self) -> None:
        if not self.cache_path or not os.path.isfile(self.cache_path):
            return
        with open(self.cache_path, "r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except Exception:
                    continue
                sample_index = record.get("sample_index")
                if isinstance(sample_index, int):
                    self.cache[sample_index] = record

    def _write_cache_record(self, record: Dict[str, Any]) -> None:
        os.makedirs(os.path.dirname(self.cache_path) or ".", exist_ok=True)
        with open(self.cache_path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        sample_index = record.get("sample_index")
        if isinstance(sample_index, int):
            self.cache[sample_index] = record

    def _ensure_model(self) -> None:
        if self.model is not None and self.tokenizer is not None:
            return
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        if torch.cuda.is_available() and not self.args.llm_cpu:
            self.device = torch.device("cuda")
            dtype = torch.bfloat16
        else:
            self.device = torch.device("cpu")
            dtype = torch.float32

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, use_fast=True, local_files_only=True)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            dtype=dtype,
            local_files_only=True,
        )
        self.model = self.model.to(self.device)
        self.model.eval()
        generation_config = getattr(self.model, "generation_config", None)
        if generation_config is not None:
            for attr in ("temperature", "top_p", "top_k"):
                if hasattr(generation_config, attr):
                    setattr(generation_config, attr, None)

    def _format_calls(self, tool_calls: List[ToolCall]) -> str:
        lines: List[str] = []
        for call in tool_calls:
            lines.append(f"Call {call.logical_index}")
            lines.append(f"Tool: {call.tool_name}")
            lines.append(f"Arguments: {_serialize_payload(call.arguments, max_chars=220)}")
            if call.thought:
                lines.append(f"Thought: {_truncate_text(call.thought, self.max_thought_chars)}")
            lines.append(f"Observed output: {_serialize_payload(call.output, max_chars=self.max_output_chars)}")
            lines.append("")
        return "\n".join(lines).strip()

    def _build_messages(self, parsed_sample: Dict[str, Any], tool_calls: List[ToolCall]) -> List[Dict[str, str]]:
        system_prompt = (
            "You infer directed dependencies between tool calls in an agent trajectory. "
            "For each call i, decide which earlier calls j < i must finish before i can be issued. "
            "Call 1 must always have an empty dependency list. "
            "Add an edge only when the current call semantically depends on an earlier result or explicitly uses previous state. "
            "If a later call argument reuses an identifier, entity, or value that comes from an earlier output, you must add that edge. "
            "If two calls answer independent parts of the user request, keep them parallel. "
            "Be conservative: if a dependency is plausibly required, add the edge. "
            "Return valid JSON only in the format {\"dependencies\": {\"1\": [], \"2\": [1]}} and include every call index exactly once."
        )
        user_prompt = (
            f"User request: {_truncate_text(parsed_sample['user_request'], self.max_user_request_chars)}\n"
            f"Planning: {_truncate_text(parsed_sample['planning'], self.max_planning_chars)}\n\n"
            "Tool calls in logical order:\n"
            f"{self._format_calls(tool_calls)}\n\n"
            "Return JSON only."
        )
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

    def _render_prompt(self, messages: List[Dict[str, str]]) -> str:
        if hasattr(self.tokenizer, "apply_chat_template"):
            try:
                return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            except Exception:
                pass
        prompt_lines: List[str] = []
        for message in messages:
            prompt_lines.append(f"{message['role'].upper()}: {message['content']}")
        prompt_lines.append("ASSISTANT:")
        return "\n\n".join(prompt_lines)

    def _parse_response(self, response_text: str, tool_calls: List[ToolCall]) -> Dict[int, List[int]]:
        parsed = _safe_json_loads(response_text)
        if parsed is None:
            match = re.search(r"\{.*\}", response_text, re.DOTALL)
            if match:
                parsed = _safe_json_loads(match.group(0))
        if parsed is None:
            raise ValueError("LLM dependency parser did not return valid JSON")

        dependency_block = parsed.get("dependencies", parsed) if isinstance(parsed, dict) else parsed
        if not isinstance(dependency_block, dict):
            raise ValueError("LLM dependency parser JSON must contain an object of dependencies")

        sanitized: Dict[int, List[int]] = {}
        for call in tool_calls:
            sanitized[call.logical_index] = []

        for raw_key, raw_value in dependency_block.items():
            key_match = re.search(r"\d+", str(raw_key))
            if not key_match:
                continue
            current_index = int(key_match.group(0))
            if current_index not in sanitized:
                continue
            if raw_value is None:
                dep_indices: List[int] = []
            elif isinstance(raw_value, list):
                dep_indices = []
                for item in raw_value:
                    dep_match = re.search(r"\d+", str(item))
                    if dep_match:
                        dep_indices.append(int(dep_match.group(0)))
            else:
                dep_indices = []
                dep_match = re.search(r"\d+", str(raw_value))
                if dep_match:
                    dep_indices.append(int(dep_match.group(0)))
            sanitized[current_index] = sorted({dep for dep in dep_indices if 1 <= dep < current_index})
        return sanitized

    def _infer_dependency_index_map(self, parsed_sample: Dict[str, Any], tool_calls: List[ToolCall]) -> Tuple[Dict[int, List[int]], str]:
        self._ensure_model()
        messages = self._build_messages(parsed_sample, tool_calls)
        prompt = self._render_prompt(messages)
        encoded = self.tokenizer(prompt, return_tensors="pt")
        encoded = {key: value.to(self.device) for key, value in encoded.items()}
        with self.torch.inference_mode():
            outputs = self.model.generate(
                **encoded,
                do_sample=False,
                num_beams=1,
                max_new_tokens=self.max_new_tokens,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        generated_tokens = outputs[0, encoded["input_ids"].shape[1] :]
        raw_response = self.tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
        return self._parse_response(raw_response, tool_calls), raw_response

    def extract_dependencies(
        self, tool_calls: List[ToolCall], parsed_sample: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Set[str]]:
        if parsed_sample is None:
            raise ValueError("LLM-DAG extraction requires parsed_sample context")
        sample_index = parsed_sample["sample_index"]

        cached = self.cache.get(sample_index)
        if cached is not None:
            dependency_index_map = {
                int(key): [int(item) for item in value]
                for key, value in cached.get("dependencies_by_index", {}).items()
            }
            return _build_dependencies_from_index_map(tool_calls, dependency_index_map)

        raw_response = ""
        dependency_index_map: Optional[Dict[int, List[int]]] = None
        status = "ok"
        error_message = None
        heuristic_dependencies = self.fallback_extractor.extract_dependencies(tool_calls, parsed_sample)
        try:
            dependency_index_map, raw_response = self._infer_dependency_index_map(parsed_sample, tool_calls)
            llm_dependencies = _build_dependencies_from_index_map(tool_calls, dependency_index_map)
            merged_dependencies: Dict[str, Set[str]] = {}
            for call in tool_calls:
                merged_dependencies[call.call_id] = llm_dependencies.get(call.call_id, set()) | heuristic_dependencies.get(call.call_id, set())
            dependency_index_map = _dependency_index_map_from_dependencies(tool_calls, merged_dependencies)
        except Exception as exc:
            error_message = str(exc)
            if self.fallback_policy == "balanced":
                dependency_index_map = _dependency_index_map_from_dependencies(tool_calls, heuristic_dependencies)
                status = "fallback_balanced"
            else:
                dependency_index_map = {call.logical_index: [] for call in tool_calls}
                status = "fallback_none"

        record = {
            "sample_index": sample_index,
            "status": status,
            "model_path": self.model_path,
            "dependencies_by_index": {str(key): value for key, value in dependency_index_map.items()},
            "raw_response": raw_response,
            "error": error_message,
            "timestamp": datetime.datetime.now().isoformat(),
        }
        self._write_cache_record(record)
        return _build_dependencies_from_index_map(tool_calls, dependency_index_map)

    def count_parallel_roots(self, tool_calls: List[ToolCall], dependencies: Dict[str, Set[str]]) -> int:
        return _count_parallel_roots(tool_calls, dependencies)


class LatencyScheduler:
    def __init__(
        self,
        planning_time: float,
        reaction_time: float,
        base_latency: float,
        variance: float,
        min_latency: float,
        seed: int,
        distribution: str = "gaussian",
        root_start_spread: float = 0.0,
    ):
        self.planning_time = planning_time
        self.reaction_time = reaction_time
        self.base_latency = base_latency
        self.variance = variance
        self.min_latency = min_latency
        self.distribution = distribution
        self.root_start_spread = root_start_spread
        self.random = random.Random(seed)

    def _sample_latency(self) -> float:
        if self.distribution == "lognormal":
            mean = max(self.base_latency, self.min_latency)
            std = max(self.variance, 1e-6)
            sigma_sq = math.log(1.0 + (std * std) / (mean * mean))
            sigma = math.sqrt(sigma_sq)
            mu = math.log(mean) - sigma_sq / 2.0
            latency = self.random.lognormvariate(mu, sigma)
        else:
            latency = self.random.gauss(self.base_latency, self.variance)
        return max(latency, self.min_latency)

    def assign_timestamps(self, tool_calls: List[ToolCall], dependencies: Dict[str, Set[str]]) -> None:
        finish_times: Dict[str, float] = {}
        for call in tool_calls:
            call.dependencies = sorted(dependencies.get(call.call_id, set()))
            if call.dependencies:
                start_time = max(finish_times[dep] for dep in call.dependencies) + self.reaction_time
            else:
                start_time = self.planning_time + self.random.uniform(0.0, self.root_start_spread)
            call.start_time = start_time
            call.end_time = start_time + self._sample_latency()
            finish_times[call.call_id] = call.end_time


def resolve_latency_config(args) -> Dict[str, Any]:
    profile_name = args.latency_profile or "balanced"
    base_config = dict(LATENCY_PROFILE_PRESETS["balanced"])
    if profile_name != "custom":
        base_config.update(LATENCY_PROFILE_PRESETS[profile_name])

    overrides = {
        "planning_time": args.planning_time,
        "reaction_time": args.reaction_time,
        "base_latency": args.base_latency,
        "latency_variance": args.latency_variance,
        "min_latency": args.min_latency,
        "latency_distribution": args.latency_distribution,
        "root_start_spread": args.root_start_spread,
    }
    for key, value in overrides.items():
        if value is not None:
            base_config[key] = value

    base_config["latency_profile"] = profile_name
    return base_config


class TranscriptSynthesizer:
    def build_event_list(self, sample: AsyncToolBenchSample) -> List[Dict[str, Any]]:
        events: List[Dict[str, Any]] = [
            {
                "type": "user_input",
                "timestamp": sample.user_request_time,
                "content": sample.user_request,
            }
        ]

        for segment in sample.planning_segments:
            events.append(
                {
                    "type": "planning_start",
                    "timestamp": segment.start_time,
                    "content": segment.text[:80],
                }
            )
            events.append(
                {
                    "type": "planning_end",
                    "timestamp": segment.end_time,
                    "content": segment.text,
                }
            )

        for call in sample.tool_calls:
            events.append(
                {
                    "type": "tool_invocation",
                    "timestamp": call.start_time,
                    "tool": call.tool_name,
                    "arguments": call.arguments,
                    "call_id": call.call_id,
                    "logical_index": call.logical_index,
                    "dependencies": call.dependencies,
                }
            )

        returned_calls = sorted(sample.tool_calls, key=lambda item: (item.end_time, item.logical_index))
        for return_order, call in enumerate(returned_calls):
            events.append(
                {
                    "type": "tool_return",
                    "timestamp": call.end_time,
                    "tool": call.tool_name,
                    "output": call.output,
                    "call_id": call.call_id,
                    "logical_index": call.logical_index,
                    "return_order": return_order,
                }
            )

        events.append(
            {
                "type": "final_response",
                "timestamp": sample.final_answer_time,
                "content": sample.final_answer,
            }
        )
        return sorted(events, key=lambda item: (item["timestamp"], item["type"]))

    def synthesize_transcript(self, sample: AsyncToolBenchSample) -> str:
        lines = [f"[User Request @ {sample.user_request_time:.2f}s]", sample.user_request, ""]
        for segment in sample.planning_segments:
            lines.extend(
                [
                    f"[Planning @ {segment.start_time:.2f}s - {segment.end_time:.2f}s]",
                    segment.text,
                    "",
                ]
            )

        returned_calls = sorted(sample.tool_calls, key=lambda item: (item.end_time, item.logical_index))
        for call in returned_calls:
            lines.extend(
                [
                    f"[Tool Invoked @ {call.start_time:.2f}s] {call.tool_name} (logical_index={call.logical_index})",
                    f"  Arguments: {json.dumps(call.arguments, ensure_ascii=False, indent=2)}",
                    f"  Dependencies: {call.dependencies}",
                    f"[Tool Returned @ {call.end_time:.2f}s]",
                    f"  Output: {json.dumps(call.output, ensure_ascii=False)}",
                    "",
                ]
            )

        lines.extend([f"[Final Answer @ {sample.final_answer_time:.2f}s]", sample.final_answer])
        return "\n".join(lines)


class AsyncToolBenchGenerator:
    def __init__(self, args):
        self.args = args
        self.latency_config = resolve_latency_config(args)
        self.parser = ToolBenchParser()
        self.dependency_extractor = self._build_dependency_extractor(args)
        self.scheduler = LatencyScheduler(
            planning_time=self.latency_config["planning_time"],
            reaction_time=self.latency_config["reaction_time"],
            base_latency=self.latency_config["base_latency"],
            variance=self.latency_config["latency_variance"],
            min_latency=self.latency_config["min_latency"],
            seed=args.seed,
            distribution=self.latency_config["latency_distribution"],
            root_start_spread=self.latency_config["root_start_spread"],
        )
        self.synthesizer = TranscriptSynthesizer()

    def _build_dependency_extractor(self, args):
        if args.dependency_policy == "balanced_dag":
            return DependencyExtractor()
        if args.dependency_policy == "all_parallel":
            return AllParallelDependencyExtractor()
        if args.dependency_policy == "llm_dag":
            return LLMDependencyExtractor(args)
        raise ValueError(f"Unsupported dependency policy: {args.dependency_policy}")

    def load_toolbench_data(self, input_path: str) -> List[Dict[str, Any]]:
        samples: List[Dict[str, Any]] = []
        if os.path.isfile(input_path):
            with open(input_path, "r", encoding="utf-8") as handle:
                data = json.load(handle)
                if isinstance(data, list):
                    samples.extend(data)
                else:
                    samples.append(data)
            return samples

        for root, _, files in os.walk(input_path):
            for filename in files:
                if not filename.endswith(".json"):
                    continue
                full_path = os.path.join(root, filename)
                with open(full_path, "r", encoding="utf-8") as handle:
                    data = json.load(handle)
                    if isinstance(data, list):
                        samples.extend(data)
                    else:
                        samples.append(data)
        return samples

    def _normalize_tool_calls(self, parsed_sample: Dict[str, Any]) -> List[ToolCall]:
        normalized: List[ToolCall] = []
        for logical_index, tool in enumerate(parsed_sample["tool_calls"], start=1):
            normalized.append(
                ToolCall(
                    tool_name=tool.get("tool_name", "unknown"),
                    arguments=tool.get("arguments", {}),
                    output=tool.get("output"),
                    logical_index=logical_index,
                    call_id=f"sample{parsed_sample['sample_index']}_call{logical_index}",
                    thought=tool.get("thought", ""),
                )
            )
        return normalized

    def generate_sample(self, parsed_sample: Dict[str, Any]) -> Optional[AsyncToolBenchSample]:
        normalized_calls = self._normalize_tool_calls(parsed_sample)
        dependencies = self.dependency_extractor.extract_dependencies(normalized_calls, parsed_sample)
        parallel_roots = self.dependency_extractor.count_parallel_roots(normalized_calls, dependencies)
        if parallel_roots < self.args.min_parallel_roots:
            return None

        self.scheduler.assign_timestamps(normalized_calls, dependencies)
        planning_segment = PlanningSegment(
            text=parsed_sample["planning"],
            start_time=0.0,
            end_time=self.latency_config["planning_time"],
        )

        final_answer_time = max(call.end_time for call in normalized_calls) + self.latency_config["reaction_time"]
        sample = AsyncToolBenchSample(
            user_request=parsed_sample["user_request"],
            user_request_time=0.0,
            planning_segments=[planning_segment],
            tool_calls=normalized_calls,
            final_answer=parsed_sample["final_answer"],
            final_answer_time=final_answer_time,
            event_list=[],
            linearized_transcript="",
            metadata={
                "source_sample_index": parsed_sample["sample_index"],
                "dependency_policy": self.dependency_extractor.policy_name,
                "parallel_root_count": parallel_roots,
                "tool_call_count": len(normalized_calls),
                "dependency_edge_count": _dependency_edge_count(dependencies),
                "longest_dependency_depth": _longest_dependency_depth(normalized_calls, dependencies),
                "planning_time": self.latency_config["planning_time"],
                "reaction_time": self.latency_config["reaction_time"],
                "base_latency": self.latency_config["base_latency"],
                "latency_variance": self.latency_config["latency_variance"],
                "min_latency": self.latency_config["min_latency"],
                "latency_distribution": self.latency_config["latency_distribution"],
                "root_start_spread": self.latency_config["root_start_spread"],
                "latency_profile": self.latency_config["latency_profile"],
            },
        )
        if self.dependency_extractor.policy_name == "llm_dag":
            sample.metadata["llm_dependency_model"] = self.args.llm_model_path or DEFAULT_LLM_DAG_MODEL
            sample.metadata["llm_dependency_cache_path"] = getattr(self.dependency_extractor, "cache_path", "")
        sample.event_list = self.synthesizer.build_event_list(sample)
        sample.linearized_transcript = self.synthesizer.synthesize_transcript(sample)
        return sample

    def _load_processed_indices(self, output_path: str) -> Set[int]:
        processed_indices: Set[int] = set()
        if not self.args.resume or not os.path.isfile(output_path):
            return processed_indices
        with open(output_path, "r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    sample = json.loads(line)
                except Exception:
                    continue
                sample_index = _extract_source_sample_index(sample)
                if sample_index is not None and _index_in_range(sample_index, self.args.start_index, self.args.end_index):
                    processed_indices.add(sample_index)
        return processed_indices

    def generate_dataset(self, input_path: str, output_path: str, num_samples: int) -> Dict[str, Any]:
        raw_samples = self.load_toolbench_data(input_path)
        processed_indices = self._load_processed_indices(output_path)
        total_written = len(processed_indices)
        generated_count = 0
        first_sample: Optional[Dict[str, Any]] = None
        filtered_count = 0
        limit = None if num_samples <= 0 else num_samples

        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        file_mode = "a" if self.args.resume and os.path.isfile(output_path) else "w"
        with open(output_path, file_mode, encoding="utf-8") as handle:
            for index, raw_sample in enumerate(raw_samples):
                if not _index_in_range(index, self.args.start_index, self.args.end_index):
                    continue
                if index in processed_indices:
                    continue
                if limit is not None and total_written >= limit:
                    break

                parsed_sample = self.parser.parse_conversations(raw_sample, sample_index=index)
                if parsed_sample is None:
                    continue
                if len(parsed_sample["tool_calls"]) < self.args.min_tool_calls:
                    continue

                generated = self.generate_sample(parsed_sample)
                if generated is None:
                    filtered_count += 1
                    continue

                sample_dict = asdict(generated)
                handle.write(json.dumps(sample_dict, ensure_ascii=False) + "\n")
                handle.flush()
                if first_sample is None:
                    first_sample = sample_dict
                generated_count += 1
                total_written += 1

        return {
            "generated_count": generated_count,
            "total_written": total_written,
            "already_present": len(processed_indices),
            "filtered_count": filtered_count,
            "first_sample": first_sample,
            "output_path": output_path,
        }


def build_parser() -> argparse.ArgumentParser:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_input = os.path.join(script_dir, "data", "data", "toolllama_G123_dfs_train.json")

    parser = argparse.ArgumentParser(description="Generate dependency-policy variants of AsyncBench from ToolBench.")
    parser.add_argument("--input", type=str, default=default_input, help="Input ToolBench JSON file or directory.")
    parser.add_argument("--output", type=str, default=None, help="Output JSONL path. If omitted, a policy-specific path is used.")
    parser.add_argument("--num_samples", type=int, default=100, help="Maximum number of samples to generate. Use 0 or a negative value for all samples.")
    parser.add_argument("--min_tool_calls", type=int, default=2, help="Minimum number of tool calls per sample.")
    parser.add_argument("--min_parallel_roots", type=int, default=2, help="Minimum number of dependency-free tools launched together.")
    parser.add_argument(
        "--dependency_policy",
        type=str,
        choices=["balanced_dag", "llm_dag", "all_parallel"],
        default="balanced_dag",
        help="Dependency policy used to build the sample DAG.",
    )
    parser.add_argument(
        "--latency_profile",
        type=str,
        choices=["balanced", "wide_gap", "heavy_tail", "staggered", "custom"],
        default="balanced",
        help="Preset latency profile. Use custom to rely only on explicit override values.",
    )
    parser.add_argument("--planning_time", type=float, default=None, help="Override planning duration in seconds.")
    parser.add_argument("--reaction_time", type=float, default=None, help="Override delay between dependency satisfaction and next tool launch.")
    parser.add_argument("--base_latency", type=float, default=None, help="Override mean tool latency in seconds.")
    parser.add_argument("--latency_variance", type=float, default=None, help="Override latency spread in seconds.")
    parser.add_argument("--min_latency", type=float, default=None, help="Override minimum tool latency in seconds.")
    parser.add_argument(
        "--latency_distribution",
        type=str,
        choices=["gaussian", "lognormal"],
        default=None,
        help="Override latency sampling distribution.",
    )
    parser.add_argument("--root_start_spread", type=float, default=None, help="Override extra random launch spread for dependency-free root tools.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--resume", action="store_true", help="Resume generation by appending to an existing output file.")
    parser.add_argument("--start_index", type=int, default=0, help="Inclusive raw ToolBench sample index to start from.")
    parser.add_argument("--end_index", type=int, default=None, help="Exclusive raw ToolBench sample index to stop at.")
    parser.add_argument("--llm_model_path", type=str, default=DEFAULT_LLM_DAG_MODEL, help="Local model path used for llm_dag dependency parsing.")
    parser.add_argument("--llm_cache_path", type=str, default=None, help="Optional JSONL cache path for llm_dag dependency decisions.")
    parser.add_argument(
        "--llm_fallback_policy",
        type=str,
        choices=["balanced", "none"],
        default="balanced",
        help="Fallback policy used when llm_dag output cannot be parsed.",
    )
    parser.add_argument("--llm_max_new_tokens", type=int, default=96, help="Maximum response tokens for llm_dag dependency parsing.")
    parser.add_argument("--llm_user_request_chars", type=int, default=700, help="Maximum user-request characters shown to the llm_dag parser.")
    parser.add_argument("--llm_planning_chars", type=int, default=260, help="Maximum planning characters shown to the llm_dag parser.")
    parser.add_argument("--llm_thought_chars", type=int, default=220, help="Maximum per-call thought characters shown to the llm_dag parser.")
    parser.add_argument("--llm_output_chars", type=int, default=320, help="Maximum per-call output characters shown to the llm_dag parser.")
    parser.add_argument("--llm_cpu", action="store_true", help="Force llm_dag parsing on CPU instead of CUDA.")
    return parser


def _resolve_default_output_path(args) -> str:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    default_filename = f"async_agent_bench_{args.dependency_policy}_{args.latency_profile}_{timestamp}.jsonl"
    return os.path.join(script_dir, "async_agent_bench", default_filename)


def main():
    args = build_parser().parse_args()
    if not args.output:
        args.output = _resolve_default_output_path(args)
    generator = AsyncToolBenchGenerator(args)
    summary = generator.generate_dataset(args.input, args.output, args.num_samples)
    print(f"Dependency policy: {generator.dependency_extractor.policy_name}")
    print(f"Generated {summary['generated_count']} new samples to {summary['output_path']}")
    print(f"Total samples currently written: {summary['total_written']}")
    print(f"Existing samples skipped by resume: {summary['already_present']}")
    print(f"Filtered samples in this run: {summary['filtered_count']}")
    print(f"Latency profile: {generator.latency_config['latency_profile']}")
    print(
        "Effective latency config: "
        f"planning={generator.latency_config['planning_time']:.2f}s, "
        f"reaction={generator.latency_config['reaction_time']:.2f}s, "
        f"mean={generator.latency_config['base_latency']:.2f}s, "
        f"spread={generator.latency_config['latency_variance']:.2f}s, "
        f"min={generator.latency_config['min_latency']:.2f}s, "
        f"distribution={generator.latency_config['latency_distribution']}, "
        f"root_start_spread={generator.latency_config['root_start_spread']:.2f}s"
    )
    if summary["first_sample"]:
        first_sample = summary["first_sample"]
        root_calls = [call for call in first_sample["tool_calls"] if not call.get("dependencies")]
        print(f"First generated sample root parallelism: {len(root_calls)}")
        for call in first_sample["tool_calls"]:
            print(
                f"  logical_index={call['logical_index']} tool={call['tool_name']} "
                f"start={call['start_time']:.2f}s end={call['end_time']:.2f}s deps={call['dependencies']}"
            )


if __name__ == "__main__":
    main()
