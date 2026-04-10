import json
import random
import math
from typing import List, Dict, Any, Tuple, Set
from dataclasses import dataclass, asdict
from collections import defaultdict
import argparse
import os
import datetime


def box_muller_transform() -> float:
    u1 = random.random()
    u2 = random.random()
    z0 = math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)
    return z0


def normal_variate(mean: float, std: float) -> float:
    return mean + std * box_muller_transform()


@dataclass
class ToolCall:
    tool_name: str
    arguments: Dict[str, Any]
    output: Any = None
    start_time: float = 0.0
    end_time: float = 0.0
    call_id: str = ""


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


class ToolBenchParser:
    @staticmethod
    def parse_conversations(data: Dict[str, Any]) -> Dict[str, Any]:
        conversations = data.get("conversations", [])
        
        tool_calls = []
        final_answer = ""
        planning_text = ""
        user_request = ""
        current_tool_call = None
        
        # First, find the REAL user request - it's the FIRST user message that has "Begin!"
        # Skip any later user messages that have "This is not the first time"
        found_real_user_request = False
        for conv in conversations:
            msg_from = conv.get("from", "")
            msg_value = conv.get("value", "").strip()
            
            if msg_from == "user" and not found_real_user_request:
                if "Begin!" in msg_value or "begin!" in msg_value:
                    user_request = msg_value
                    found_real_user_request = True
                    break
        
        # If we didn't find a "Begin!" message, just take the first user message
        if not found_real_user_request:
            for conv in conversations:
                msg_from = conv.get("from", "")
                msg_value = conv.get("value", "").strip()
                if msg_from == "user":
                    user_request = msg_value
                    break
        
        # Now parse the rest of the conversation
        for conv in conversations:
            msg_from = conv.get("from", "")
            msg_value = conv.get("value", "").strip()
            
            # Skip the user messages we already processed
            if msg_from == "user":
                continue
            
            elif msg_from == "assistant":
                if "Thought:" in msg_value and "Action:" in msg_value:
                    thought_part = ""
                    action_name = ""
                    action_input = {}
                    
                    if "Thought:" in msg_value:
                        thought_idx = msg_value.find("Thought:")
                        after_thought = msg_value[thought_idx + len("Thought:"):].strip()
                        
                        if "Action:" in after_thought:
                            action_idx = after_thought.find("Action:")
                            thought_part = after_thought[:action_idx].strip()
                            after_action = after_thought[action_idx + len("Action:"):].strip()
                            
                            if "Action Input:" in after_action:
                                ai_idx = after_action.find("Action Input:")
                                action_name = after_action[:ai_idx].strip()
                                ai_str = after_action[ai_idx + len("Action Input:"):].strip()
                                try:
                                    action_input = json.loads(ai_str)
                                except:
                                    action_input = {"raw": ai_str}
                            
                            if action_name and action_name != "Finish":
                                current_tool_call = {
                                    "tool_name": action_name,
                                    "arguments": action_input,
                                    "output": None
                                }
                                if thought_part and not planning_text:
                                    planning_text = thought_part
                
                elif "Action: Finish" in msg_value or "Finish" in msg_value:
                    try:
                        if "Action Input:" in msg_value:
                            ai_idx = msg_value.find("Action Input:")
                            ai_str = msg_value[ai_idx + len("Action Input:"):].strip()
                            try:
                                finish_input = json.loads(ai_str)
                                if isinstance(finish_input, dict):
                                    final_answer = finish_input.get("final_answer", "")
                                    if not final_answer:
                                        final_answer = str(finish_input)
                                else:
                                    final_answer = str(finish_input)
                            except:
                                final_answer = ai_str
                    except:
                        pass
            
            elif msg_from == "function" and current_tool_call:
                try:
                    func_result = json.loads(msg_value)
                    output = func_result.get("response", func_result)
                    current_tool_call["output"] = output
                    tool_calls.append(current_tool_call)
                    current_tool_call = None
                except:
                    current_tool_call["output"] = msg_value
                    tool_calls.append(current_tool_call)
                    current_tool_call = None
        
        return {
            "user_request": user_request,
            "planning": planning_text,
            "tool_calls": tool_calls,
            "final_answer": final_answer
        }


class ToolBenchNormalizer:
    def __init__(self):
        pass

    def normalize_arguments(self, arguments: Any) -> Dict[str, Any]:
        if isinstance(arguments, str):
            try:
                return json.loads(arguments)
            except:
                return {"raw": arguments}
        elif isinstance(arguments, dict):
            return arguments
        else:
            return {"value": str(arguments)}

    def normalize_tool_call(self, raw_call: Dict[str, Any]) -> ToolCall:
        tool_name = raw_call.get("tool_name", raw_call.get("name", "unknown"))
        raw_args = raw_call.get("arguments", raw_call.get("params", {}))
        arguments = self.normalize_arguments(raw_args)
        output = raw_call.get("output", raw_call.get("result", None))
        
        return ToolCall(
            tool_name=tool_name,
            arguments=arguments,
            output=output,
            call_id=str(random.randint(10000, 99999))
        )

    def select_multi_tool_samples(self, samples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        multi_tool_samples = []
        for sample in samples:
            tool_calls = sample.get("tool_calls", [])
            if len(tool_calls) >= 2:
                multi_tool_samples.append(sample)
        return multi_tool_samples


class DependencyExtractor:
    def __init__(self):
        pass

    def extract_dependencies(self, tool_calls: List[ToolCall]) -> Dict[str, Set[str]]:
        dependencies = defaultdict(set)
        
        for idx, call in enumerate(tool_calls):
            current_id = call.call_id
            if current_id not in dependencies:
                dependencies[current_id] = set()
            
            args_str = json.dumps(call.arguments, sort_keys=True)
            
            for prev_idx, prev_call in enumerate(tool_calls[:idx]):
                prev_id = prev_call.call_id
                prev_output_str = json.dumps(prev_call.output, sort_keys=True) if prev_call.output else ""
                
                if prev_output_str and prev_output_str in args_str:
                    dependencies[current_id].add(prev_id)
        
        return dict(dependencies)

    def get_parallel_groups(self, tool_calls: List[ToolCall], dependencies: Dict[str, Set[str]]) -> List[List[str]]:
        call_ids = [call.call_id for call in tool_calls]
        completed = set()
        groups = []
        
        while len(completed) < len(call_ids):
            current_group = []
            for call_id in call_ids:
                if call_id not in completed:
                    deps = dependencies.get(call_id, set())
                    if deps.issubset(completed):
                        current_group.append(call_id)
            
            if current_group:
                groups.append(current_group)
                completed.update(current_group)
            else:
                break
        
        return groups


class LatencyInjector:
    def __init__(self, base_latency: float = 5.0, variance: float = 3.0):
        self.base_latency = base_latency
        self.variance = variance

    def generate_latency(self, tool_name: str) -> float:
        latency = normal_variate(self.base_latency, self.variance)
        return max(latency, 2.0)

    def assign_timestamps(self, tool_calls: List[ToolCall], 
                         parallel_groups: List[List[str]],
                         planning_time: float = 0.5) -> Tuple[List[ToolCall], float]:
        call_id_to_call = {call.call_id: call for call in tool_calls}
        
        current_time = planning_time
        
        for group in parallel_groups:
            max_latency = 0.0
            group_latencies = {}
            
            for call_id in group:
                call = call_id_to_call[call_id]
                latency = self.generate_latency(call.tool_name)
                group_latencies[call_id] = latency
                if latency > max_latency:
                    max_latency = latency
            
            for call_id in group:
                call = call_id_to_call[call_id]
                call.start_time = current_time
                call.end_time = current_time + group_latencies[call_id]
            
            current_time += max_latency
        
        return tool_calls, current_time


class TranscriptSynthesizer:
    def __init__(self):
        pass

    def build_event_list(self, sample: AsyncToolBenchSample) -> List[Dict[str, Any]]:
        events = []
        
        events.append({
            "type": "user_input",
            "timestamp": sample.user_request_time,
            "content": sample.user_request
        })
        
        for segment in sample.planning_segments:
            events.append({
                "type": "planning_start",
                "timestamp": segment.start_time,
                "content": segment.text[:50] + "..."
            })
            events.append({
                "type": "planning_end",
                "timestamp": segment.end_time,
                "content": segment.text
            })
        
        for call in sample.tool_calls:
            events.append({
                "type": "tool_invocation",
                "timestamp": call.start_time,
                "tool": call.tool_name,
                "arguments": call.arguments,
                "call_id": call.call_id
            })
        
        sorted_returns = sorted(sample.tool_calls, key=lambda x: x.end_time)
        for idx, call in enumerate(sorted_returns):
            events.append({
                "type": "tool_return",
                "timestamp": call.end_time,
                "tool": call.tool_name,
                "output": call.output,
                "call_id": call.call_id,
                "return_order": idx
            })
        
        events.append({
            "type": "final_response",
            "timestamp": sample.final_answer_time,
            "content": sample.final_answer
        })
        
        events.sort(key=lambda x: x["timestamp"])
        return events

    def synthesize_transcript(self, sample: AsyncToolBenchSample) -> str:
        transcript_parts = []
        
        transcript_parts.append(f"[User Request @ {sample.user_request_time:.2f}s]")
        transcript_parts.append(sample.user_request)
        transcript_parts.append("")
        
        for segment in sample.planning_segments:
            transcript_parts.append(f"[Planning @ {segment.start_time:.2f}s - {segment.end_time:.2f}s]")
            transcript_parts.append(segment.text)
            transcript_parts.append("")
        
        sorted_returns = sorted(sample.tool_calls, key=lambda x: x.end_time)
        
        for call in sorted_returns:
            transcript_parts.append(f"[Tool Invoked @ {call.start_time:.2f}s] {call.tool_name}")
            transcript_parts.append(f"  Arguments: {json.dumps(call.arguments, indent=2)}")
            transcript_parts.append(f"[Tool Returned @ {call.end_time:.2f}s]")
            transcript_parts.append(f"  Output: {json.dumps(call.output, indent=2) if call.output else 'None'}")
            transcript_parts.append("")
        
        transcript_parts.append(f"[Final Answer @ {sample.final_answer_time:.2f}s]")
        transcript_parts.append(sample.final_answer)
        
        return "\n".join(transcript_parts)


class AsyncToolBenchGenerator:
    def __init__(self):
        self.parser = ToolBenchParser()
        self.normalizer = ToolBenchNormalizer()
        self.dependency_extractor = DependencyExtractor()
        self.latency_injector = LatencyInjector()
        self.synthesizer = TranscriptSynthesizer()

    def load_toolbench_data(self, input_path: str) -> List[Dict[str, Any]]:
        samples = []
        
        if os.path.isfile(input_path):
            with open(input_path, 'r', encoding='utf-8') as f:
                try:
                    data = json.load(f)
                    if isinstance(data, list):
                        for item in data:
                            samples.append(item)
                    else:
                        samples.append(data)
                except:
                    pass
        
        elif os.path.isdir(input_path):
            for root, _, files in os.walk(input_path):
                for filename in files:
                    if filename.endswith('.json'):
                        filepath = os.path.join(root, filename)
                        try:
                            with open(filepath, 'r', encoding='utf-8') as f:
                                data = json.load(f)
                                if isinstance(data, list):
                                    for item in data:
                                        samples.append(item)
                                else:
                                    samples.append(data)
                        except:
                            pass
        
        return samples

    def generate_sample(self, parsed_data: Dict[str, Any]) -> AsyncToolBenchSample:
        raw_tool_calls = parsed_data.get("tool_calls", [])
        normalized_calls = [self.normalizer.normalize_tool_call(rc) for rc in raw_tool_calls]
        
        dependencies = self.dependency_extractor.extract_dependencies(normalized_calls)
        parallel_groups = self.dependency_extractor.get_parallel_groups(normalized_calls, dependencies)
        
        planning_time = 0.3
        normalized_calls, end_time = self.latency_injector.assign_timestamps(
            normalized_calls, parallel_groups, planning_time
        )
        
        user_request = parsed_data.get("user_request", "")
        final_answer = parsed_data.get("final_answer", "")
        
        planning_segment = PlanningSegment(
            text=parsed_data.get("planning", "I will call the necessary tools to answer this question."),
            start_time=0.0,
            end_time=planning_time
        )
        
        sample = AsyncToolBenchSample(
            user_request=user_request,
            user_request_time=0.0,
            planning_segments=[planning_segment],
            tool_calls=normalized_calls,
            final_answer=final_answer,
            final_answer_time=end_time + 0.2,
            event_list=[],
            linearized_transcript=""
        )
        
        sample.event_list = self.synthesizer.build_event_list(sample)
        sample.linearized_transcript = self.synthesizer.synthesize_transcript(sample)
        
        return sample

    def generate_dataset(self, input_path: str, output_path: str, num_samples: int = None):
        raw_samples = self.load_toolbench_data(input_path)
        
        parsed_samples = []
        valid_count = 0
        for idx, raw_sample in enumerate(raw_samples):
            parsed = self.parser.parse_conversations(raw_sample)
            if parsed["tool_calls"] and parsed["user_request"]:
                # Skip samples with "This is not the first time" in user_request
                if "This is not the first time" in parsed["user_request"]:
                    continue
                parsed_samples.append(parsed)
                valid_count += 1
                if num_samples and valid_count >= num_samples:
                    break
        
        print(f"Found {valid_count} valid samples with real user requests")
        
        multi_tool_samples = self.normalizer.select_multi_tool_samples(parsed_samples)
        print(f"Selected {len(multi_tool_samples)} samples with multiple tools")
        
        if num_samples:
            multi_tool_samples = multi_tool_samples[:num_samples]
        
        generated_samples = []
        for idx, raw_sample in enumerate(multi_tool_samples):
            print(f"Generating sample {idx + 1}/{len(multi_tool_samples)}...")
            try:
                sample = self.generate_sample(raw_sample)
                generated_samples.append(asdict(sample))
            except Exception as e:
                print(f"Error generating sample {idx}: {e}")
        
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            for sample in generated_samples:
                f.write(json.dumps(sample, ensure_ascii=False) + '\n')
        
        print(f"Generated {len(generated_samples)} samples to {output_path}")
        return generated_samples


def main():
    # Get default paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_input = os.path.join(script_dir, "data", "data", "toolllama_G123_dfs_train.json")
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    default_output = os.path.join(script_dir, "async_agent_bench", f"async_agent_bench_{timestamp}.jsonl")
    
    parser = argparse.ArgumentParser(description="Generate Asynchronous ToolBench Dataset from ToolBench data")
    parser.add_argument("--input", type=str, default=default_input, help=f"Path to input ToolBench data (json file or directory) [default: {default_input}]")
    parser.add_argument("--output", type=str, default=default_output, help=f"Path to output dataset [default: {default_output}]")
    parser.add_argument("--num_samples", type=int, default=100, help="Number of samples to generate [default: 100]")
    
    args = parser.parse_args()
    
    generator = AsyncToolBenchGenerator()
    generator.generate_dataset(args.input, args.output, args.num_samples)


if __name__ == "__main__":
    main()
