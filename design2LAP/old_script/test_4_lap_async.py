"""
Test LAP (Link-guided Asynchronous Prefill) with "first-come-first-prefill" execution.
Based on chunk_sparse.py but modified for async tool return order.

Usage:
python test_lap_async.py \
    --dataset_path ../design3toolbenchdataset/async_toolbench_test.jsonl \
    --num_samples 2 \
    --attn_type blocked
"""
import argparse
import datetime
import json
import os
import string
import time
import heapq
import sys
from typing import Dict, List, Any

# Add the project root directory to the Python path
sys.path.insert(0, "/home/dell/wyb/point1/model_trunk_sparse/LocalClaw")

import datasets
import numpy as np
import regex
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm as auto_tqdm
from transformers import AutoTokenizer, GenerationConfig, LlamaForCausalLM, AutoModelForCausalLM

from src.common import move_to_target_device
from src.data.titan_preprocessor import make_segment_mask
from jsonl_loader import load_jsonl_dataset


parser = argparse.ArgumentParser(description="Test LAP async prefill on async ToolBench dataset.")
parser.add_argument(
    "--dataset_path",
    type=str,
    required=True,
    help="Path to async ToolBench dataset (JSONL format)."
)
parser.add_argument(
    "--model_path",
    type=str,
    default=None,
    help="Path to model checkpoint (optional, uses base model if not provided)."
)
parser.add_argument(
    "--num_samples",
    type=int,
    default=100,
    help="Number of samples to evaluate."
)
parser.add_argument(
    "--attn_type",
    type=str,
    default="blocked",
    choices=["standard", "blocked"],
    help="Attention type."
)
parser.add_argument(
    "--reencode_num",
    type=int,
    default=5,
    help="Number of link tokens."
)
parser.add_argument(
    "--hf",
    type=bool,
    default=False,
    help="Use HuggingFace checkpoints."
)
parser.add_argument("--output_dir", type=str, required=True, help="Output directory for results.")

args = parser.parse_args()


def normalize_answer(s: str) -> str:
    """Normalization from the SQuAD evaluation script."""
    def remove_articles(text):
        return regex.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def best_subspan_em(prediction: str, ground_truths: List[str]) -> float:
    normalized_prediction = normalize_answer(prediction)
    for ground_truth in ground_truths:
        normalized_gt = normalize_answer(ground_truth)
        if normalized_gt.lower() in normalized_prediction.lower():
            return 1.0
    return 0.0


def parse_async_sample(example: Dict[str, Any]) -> Dict[str, Any]:
    """Parse async ToolBench sample and extract event timeline."""
    user_request = example.get("user_request", "")
    final_answer = example.get("final_answer", "")
    event_list = example.get("event_list", [])
    tool_calls = example.get("tool_calls", [])
    
    static_parts = []
    tool_returns = []
    
    for event in event_list:
        event_type = event.get("type", "")
        timestamp = event.get("timestamp", 0.0)
        
        if event_type == "user_input":
            static_parts.append({
                "type": "user",
                "content": event.get("content", ""),
                "timestamp": timestamp
            })
        elif event_type == "planning_end":
            static_parts.append({
                "type": "planning",
                "content": event.get("content", ""),
                "timestamp": timestamp
            })
        elif event_type == "tool_return":
            tool_idx = event.get("return_order", 0)
            tool_returns.append({
                "type": "tool_return",
                "tool_name": event.get("tool", ""),
                "output": event.get("output", ""),
                "timestamp": timestamp,
                "call_id": event.get("call_id", ""),
                "return_order": tool_idx
            })
    
    tool_returns.sort(key=lambda x: x["timestamp"])
    
    return {
        "user_request": user_request,
        "final_answer": final_answer,
        "static_parts": static_parts,
        "tool_returns": tool_returns,
        "event_list": event_list,
        "tool_calls": tool_calls,
        "linearized_transcript": example.get("linearized_transcript", "")
    }


def build_tool_fragment_with_link(tool_return: Dict[str, Any], tokenizer, seg_id: int, reencode_num: int, special_token_start: int):
    """Build a single tool return fragment with link tokens."""
    tool_name = tool_return.get("tool_name", "")
    output = tool_return.get("output", "")
    
    fragment = f"[Tool: {tool_name}]\n{json.dumps(output, ensure_ascii=False) if output else 'None'}\n"
    tool_ids = tokenizer(fragment, add_special_tokens=False)["input_ids"]
    segment_ids = [seg_id] * len(tool_ids)
    
    # Add link tokens for this segment
    link_ids = [special_token_start + reencode_num * (seg_id - 1) + sub_idx for sub_idx in range(reencode_num)]
    tool_ids.extend(link_ids)
    segment_ids.extend([0] * len(link_ids))
    
    return tool_ids, segment_ids


class DataCollatorForAsyncPrefill():
    def __init__(self, pad_id: int):
        self.pad_id = pad_id
    
    def __call__(self, batch):
        return batch


def main():
    dataset_path = args.dataset_path
    model_path = args.model_path
    num_samples = args.num_samples
    attn_type = args.attn_type
    reencode_num = args.reencode_num
    hf = args.hf
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    mem_start = 128254
    mem_end = 128255
    special_token_start = 128011
    
    print("=" * 80)
    print(f"LAP Async Prefill Test with {attn_type.upper()} ATTENTION (First-Come-First-Prefill)")
    print("=" * 80)
    
    # Load dataset using jsonl_loader instead of datasets library
    dataset_list = load_jsonl_dataset(dataset_path, num_samples)
    print(f"Dataset loaded: {len(dataset_list)} samples")
    
    # Convert to list for processing
    dataset = dataset_list
    
    tokenizer = AutoTokenizer.from_pretrained(
        "/home/dell/wyb/my_models/models/LLM-Research/Llama-3___2-1B-Instruct",
        use_fast=True
    )
    tokenizer.pad_token_id = 128004
    tokenizer.pad_token = "<|finetune_right_pad_id|>"
    
    print("Loading model...")
    model = LlamaForCausalLM.from_pretrained(
        "/home/dell/wyb/my_models/models/LLM-Research/Llama-3___2-1B-Instruct",
        torch_dtype=torch.bfloat16
    )
    
    if model_path:
        if hf:
            model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16)
        else:
            state_dict = torch.load(model_path, map_location="cpu")
            model.load_state_dict(state_dict, strict=False)
    
    model = model.to(device)
    model.eval()
    
    eot_id = tokenizer.convert_tokens_to_ids("<|eot_id|>")
    generation_cfg = GenerationConfig(
        do_sample=False,
        num_beams=1,
        max_new_tokens=200,
        stop_strings=["<|end_of_text|>", "<|eot_id|>"],
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=eot_id,
    )
    
    generation_prompt = "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    generation_token_ids = tokenizer(generation_prompt, add_special_tokens=False)["input_ids"]
    generation_token_ids = torch.LongTensor(generation_token_ids)
    generation_token_ids = move_to_target_device(generation_token_ids, device)
    
    correct_num = 0
    res_list = []
    total_time = 0.0
    ttft_list = []
    real_ttft_list = []  # TTFT from experiment start (planning end + 0.3s)
    ttft_after_last_tool_list = []  # KEY METRIC: TTFT after LAST tool arrival
    prefill_time_list = []
    total_wait_time = 0.0
    
    prog_bar = auto_tqdm(range(len(dataset)))
    
    for sample_idx in range(len(dataset)):
        example = dataset[sample_idx]
        parsed_sample = parse_async_sample(example)
        
        gold_answer = parsed_sample["final_answer"]
        static_parts = parsed_sample["static_parts"]
        tool_returns = parsed_sample["tool_returns"]
        tool_calls = parsed_sample.get("tool_calls", [])
        
        print(f"\n{'='*80}")
        print(f"Sample {sample_idx + 1}/{len(dataset)}")
        print(f"{'='*80}")
        print(f"User request: {parsed_sample['user_request'][:100]}...")
        print(f"Tool returns: {len(tool_returns)} tools")
        
        sample_start_time = time.time()
        past_key_values = None
        
        try:
            # Timeline simulation setup
            experiment_start = time.time()
            planning_end = 0.3  # Planning takes 0.3 seconds
            
            # Build static prefix with link mechanism
            system = (
                "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
                "You are an intelligent AI assistant. Answer based on the conversation history.\n"
                "<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
            )
            system_ids = tokenizer(system, add_special_tokens=False)["input_ids"]
            static_ids = system_ids[:]
            static_segment_ids = [0] * len(static_ids)
            
            # Static user/planning prefix
            linearized = parsed_sample.get("linearized_transcript", "")
            linearized_ids = tokenizer(linearized, add_special_tokens=False)["input_ids"]
            static_ids.extend(linearized_ids)
            static_segment_ids.extend([0] * len(linearized_ids))
            
            # Sort all tools by their LOGICAL order (not arrival order) to assign segment ids
            # This ensures segment ids are consistent between LAP and standard
            sorted_tools_logical = sorted(tool_calls, key=lambda x: x.get("end_time", 0))
            tool_to_seg_id = {tool.get("call_id", ""): i+1 for i, tool in enumerate(sorted_tools_logical)}
            
            # Build a priority queue of tool returns sorted by ARRIVAL time
            tool_queue = []
            for idx, tr in enumerate(tool_returns):
                # Use index as tiebreaker to avoid dict comparison
                heapq.heappush(tool_queue, (tr["timestamp"], idx, tr))
            
            print(f"  [1/3] Prefilling static prefix ({len(static_ids)} tokens)...")
            static_input_ids = torch.LongTensor([static_ids])
            static_input_ids = move_to_target_device(static_input_ids, device)
            static_segment_ids_tensor = torch.LongTensor([static_segment_ids])
            
            if attn_type == "blocked":
                static_attention_mask_4d = make_segment_mask(
                    source_segments=static_segment_ids_tensor,
                    target_segments=static_segment_ids_tensor,
                    add_causal_lm_mask=True
                ).unsqueeze(1)
                static_attention_mask_4d = move_to_target_device(static_attention_mask_4d, device)
                if static_attention_mask_4d.dtype != model.dtype:
                    static_attention_mask_4d = static_attention_mask_4d.to(model.dtype)
                
                with torch.no_grad():
                    static_outputs = model(
                        input_ids=static_input_ids,
                        attention_mask=static_attention_mask_4d,
                        use_cache=True
                    )
            else:
                static_attention_mask = torch.ones_like(static_input_ids)
                with torch.no_grad():
                    static_outputs = model(
                        input_ids=static_input_ids,
                        attention_mask=static_attention_mask,
                        use_cache=True
                    )
            
            past_key_values = static_outputs.past_key_values
            accumulated_ids = static_ids.copy()
            accumulated_segment_ids = static_segment_ids.copy()
            
            print(f"  [2/3] Incremental prefill for tool returns with link tokens...")
            prefill_start = time.time()
            tool_count = 0
            
            simulated_experiment_time = 0.0
            while tool_queue:
                # Get next tool to arrive
                tool_arrival_time, _, tool_return = heapq.heappop(tool_queue)
                tool_count += 1
                
                # Simulate waiting without actual sleeping
                wait_time = max(tool_arrival_time - simulated_experiment_time, 0)
                
                if wait_time > 0:
                    print(f"    Simulated waiting {wait_time:.2f}s for tool {tool_count}...")
                    total_wait_time += wait_time
                    simulated_experiment_time = tool_arrival_time
                
                print(f"    Prefilling tool {tool_count}/{len(tool_returns)}...")
                
                # Get logical segment id for this tool
                call_id = tool_return.get("call_id", "")
                seg_id = tool_to_seg_id.get(call_id, tool_count)
                
                # Build tool fragment + link tokens
                tool_ids, tool_segment_ids = build_tool_fragment_with_link(
                    tool_return, tokenizer, seg_id, reencode_num, special_token_start
                )
                
                # Add to accumulated
                accumulated_ids.extend(tool_ids)
                accumulated_segment_ids.extend(tool_segment_ids)
                
                # Build input for this increment
                tool_input_ids = torch.LongTensor([tool_ids])
                tool_input_ids = move_to_target_device(tool_input_ids, device)
                
                if attn_type == "blocked":
                    # Build full segment ids for mask computation
                    full_segment_ids = torch.LongTensor([accumulated_segment_ids])
                    full_segment_ids = move_to_target_device(full_segment_ids, device)
                    
                    # Compute attention mask for the current query tokens (tool + link)
                    full_attention_mask_4d = make_segment_mask(
                        source_segments=full_segment_ids,
                        target_segments=full_segment_ids,
                        add_causal_lm_mask=True
                    ).unsqueeze(1)
                    full_attention_mask_4d = move_to_target_device(full_attention_mask_4d, device)
                    if full_attention_mask_4d.dtype != model.dtype:
                        full_attention_mask_4d = full_attention_mask_4d.to(model.dtype)
                    
                    q_len = tool_input_ids.size(1)
                    attention_mask_slice = full_attention_mask_4d[:, :, -q_len:, :]
                    
                    # Prefill this increment (tool + link tokens)
                    with torch.no_grad():
                        tool_outputs = model(
                            input_ids=tool_input_ids,
                            attention_mask=attention_mask_slice,
                            past_key_values=past_key_values,
                            use_cache=True
                        )
                else:
                    # Use STANDARD full attention
                    full_attention_mask = torch.ones((1, len(accumulated_ids)), dtype=torch.long)
                    full_attention_mask = move_to_target_device(full_attention_mask, device)
                    
                    with torch.no_grad():
                        tool_outputs = model(
                            input_ids=tool_input_ids,
                            attention_mask=full_attention_mask,
                            past_key_values=past_key_values,
                            use_cache=True
                        )
                
                past_key_values = tool_outputs.past_key_values
            
            prefill_time = time.time() - prefill_start
            prefill_time_list.append(prefill_time)
            print(f"    Total incremental prefill time: {prefill_time:.4f}s")
            
            print(f"  [3/3] Generating final response...")
            full_input_ids = torch.LongTensor([accumulated_ids])
            full_input_ids = move_to_target_device(full_input_ids, device)
            full_attention_mask = torch.ones_like(full_input_ids)
            
            generation_prefix = generation_token_ids.unsqueeze(0)
            generation_input_ids = torch.cat([full_input_ids, generation_prefix], axis=1)
            generation_attention_mask = torch.cat([full_attention_mask, torch.ones_like(generation_prefix)], axis=1)
            
            # REAL TTFT: from experiment start (simulated_experiment_time is when last tool arrived)
            # plus the time to generate first token
            time_before_generate = time.time()
            
            # KEY METRIC: For async methods, most of the prefill is already done!
            ttft_after_last_start = time.time()
            ttft_start = time.time()
            with torch.no_grad():
                outputs = model.generate(
                    input_ids=generation_input_ids,
                    attention_mask=generation_attention_mask,
                    use_cache=True,
                    generation_config=generation_cfg,
                    past_key_values=past_key_values,
                    tokenizer=tokenizer,
                )
            ttft = time.time() - ttft_start
            ttft_list.append(ttft)
            
            # REAL TTFT: total time from planning end (0.3s) to first token
            # simulated_experiment_time is the time when last tool arrived (relative to planning end)
            real_ttft = simulated_experiment_time + (time.time() - time_before_generate)
            real_ttft_list.append(real_ttft)
            
            # KEY METRIC: TTFT after LAST tool arrived!
            ttft_after_last = time.time() - ttft_after_last_start
            ttft_after_last_tool_list.append(ttft_after_last)
            
            generated_seq = tokenizer.decode(
                outputs[0, full_input_ids.size(1):].tolist(),
            )
            response = generated_seq.split("<|start_header_id|>assistant<|end_header_id|>")[-1].strip().split("<|eot_id|>")[0]
            
            print(f"  Response: {response[:100]}...")
            print(f"  Gold answer: {gold_answer[:100]}...")
            
            score = best_subspan_em(response, [gold_answer])
            correct_num = correct_num + int(score)
            
            res_list.append({
                "response": response,
                "gold_answer": gold_answer,
                "score": score,
                "prefill_time": prefill_time,
                "ttft": ttft,
                "num_tools": len(tool_returns)
            })
            
            sample_time = time.time() - sample_start_time
            total_time += sample_time
            
            print(f"  Score: {score}, Sample time: {sample_time:.4f}s")
            print(f"  Correct so far: {correct_num}/{sample_idx + 1}")
            
        except Exception as e:
            print(f"  Error processing sample {sample_idx}: {e}")
            import traceback
            traceback.print_exc()
        
        prog_bar.update(1)
    
    accuracy = correct_num / len(dataset)
    avg_time = total_time / len(dataset)
    avg_ttft = np.mean(ttft_list) if ttft_list else 0.0
    avg_real_ttft = np.mean(real_ttft_list) if real_ttft_list else 0.0
    avg_ttft_after_last = np.mean(ttft_after_last_tool_list) if ttft_after_last_tool_list else 0.0
    avg_prefill_time = np.mean(prefill_time_list) if prefill_time_list else 0.0
    
    print(f"\n{'='*80}")
    print(f"FINAL RESULTS (TEST 4: LAP ASYNC with {attn_type.upper()} ATTENTION)")
    print(f"{'='*80}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Average time per sample: {avg_time:.4f}s")
    print(f"Average TTFT (compute only): {avg_ttft:.4f}s")
    print(f"Average REAL TTFT (from start): {avg_real_ttft:.4f}s")
    print(f"=== KEY METRIC ===")
    print(f"Avg TTFT AFTER LAST TOOL ARRIVED: {avg_ttft_after_last:.4f}s")
    print(f"Average incremental prefill time: {avg_prefill_time:.4f}s")
    print(f"Total time (computation): {total_time:.4f}s")
    print(f"Total wait time: {total_wait_time:.4f}s")
    print(f"Total time (including waiting): {total_time + total_wait_time:.4f}s")
    print(f"{'='*80}")
    
    result_dir = args.output_dir
    os.makedirs(result_dir, exist_ok=True)
    
    file_name = os.path.join(result_dir, f"test4_lap_async_{attn_type}_accuracy_{accuracy:.4f}.jsonl")
    
    with open(file_name, "w", encoding="utf-8") as f:
        for entry in res_list:
            json_line = json.dumps(entry, ensure_ascii=False)
            f.write(json_line + "\n")
    
    print(f"\nDumped results at {file_name}")
    
    stats_file = os.path.join(result_dir, f"test4_lap_async_{attn_type}_stats.json")
    with open(stats_file, "w", encoding="utf-8") as f:
        json.dump({
            "accuracy": accuracy,
            "avg_time": avg_time,
            "avg_ttft": avg_ttft,
            "avg_real_ttft": avg_real_ttft,
            "avg_ttft_after_last_tool": avg_ttft_after_last,
            "avg_prefill_time": avg_prefill_time,
            "total_time_compute": total_time,
            "total_wait_time": total_wait_time,
            "total_time_total": total_time + total_wait_time,
            "num_samples": len(dataset),
            "attn_type": attn_type,
            "description": "Test 4: LAP Async - First come first prefill"
        }, f, indent=2, ensure_ascii=False)
    
    print(f"Dumped stats at {stats_file}")


if __name__ == "__main__":
    main()
