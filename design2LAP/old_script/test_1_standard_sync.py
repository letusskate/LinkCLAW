"""
Test 1: STANDARD SYNCHRONOUS - Wait for all modules before prefill.
The slowest traditional approach: waits for everything to arrive, then prefill all at once.

Usage:
python test_standard_sync.py \
    --dataset_path ../design3toolbenchdataset/async_toolbench_staggered.jsonl \
    --num_samples 2 \
    --attn_type blocked
"""
import argparse
import datetime
import json
import os
import string
import time
import sys
from typing import Dict, List, Any

# Add project root to path
sys.path.insert(0, "/home/dell/wyb/point1/model_trunk_sparse/LocalClaw")

import datasets
import numpy as np
import regex
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm as auto_tqdm
from transformers import AutoTokenizer, GenerationConfig, LlamaForCausalLM, AutoModelForCausalLM

from src.common import move_to_target_device
from src.data.titan_preprocessor import make_segment_mask
from jsonl_loader import load_jsonl_dataset


parser = argparse.ArgumentParser(description="Test 1: Standard Sync - Wait for all.")
parser.add_argument("--dataset_path", type=str, required=True, help="Path to async dataset (JSONL).")
parser.add_argument("--model_path", type=str, default=None, help="Optional path to model ckpt.")
parser.add_argument("--batch_size", type=int, default=1, help="Batch size.")
parser.add_argument("--num_samples", type=int, default=100, help="Number of samples.")
parser.add_argument("--hf", type=bool, default=False, help="Use HuggingFace checkpoints.")
parser.add_argument("--reencode_num", type=int, default=5, help="Number of link tokens per segment.")
parser.add_argument(
    "--attn_type",
    type=str,
    default="blocked",
    choices=["standard", "blocked"],
    help="Attention type."
)
parser.add_argument("--output_dir", type=str, required=True, help="Output directory for results.")

args = parser.parse_args()


def normalize_answer(s: str) -> str:
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
    for gt in ground_truths:
        normalized_gt = normalize_answer(gt)
        if normalized_gt.lower() in normalized_prediction.lower():
            return 1.0
    return 0.0


def preprocess_with_links(example: Dict[str, Any], tokenizer, reencode_num: int, mem_start: int, mem_end: int, special_token_start: int) -> Dict[str, Any]:
    """Build input with link tokens for standard LLM (sync, wait for all tools)."""
    user_req = example.get("user_request", "")
    final_answer = example.get("final_answer", "")
    tool_calls = example.get("tool_calls", [])
    
    prompt_parts = []
    segment_ids = []
    
    # System prefix
    system = (
        "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
        "You are an intelligent AI assistant. Answer based on the conversation history.\n"
        "<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
    )
    system_ids = tokenizer(system, add_special_tokens=False)["input_ids"]
    prompt_parts.extend(system_ids)
    segment_ids.extend([0]*len(system_ids))
    
    # Static user/planning prefix
    linearized = example.get("linearized_transcript", "")
    static_ids = tokenizer(linearized, add_special_tokens=False)["input_ids"]
    prompt_parts.extend(static_ids)
    segment_ids.extend([0]*len(static_ids))
    
    # Wait for ALL tools, then add each tool as a segment with link tokens!
    sorted_tools = sorted(tool_calls, key=lambda x: x.get("end_time", 0))
    for seg_id, tool in enumerate(sorted_tools, 1):
        tool_name = tool.get("tool_name", "")
        output = tool.get("output", "")
        tool_text = f"[Tool: {tool_name}]\n{json.dumps(output, ensure_ascii=False)}\n"
        tool_ids = tokenizer(tool_text, add_special_tokens=False)["input_ids"]
        prompt_parts.extend(tool_ids)
        segment_ids.extend([seg_id]*len(tool_ids))
        
        # Add link tokens for this segment!
        link_ids = [special_token_start + reencode_num*(seg_id-1) + sub_idx for sub_idx in range(reencode_num)]
        prompt_parts.extend(link_ids)
        segment_ids.extend([0]*len(link_ids))
    
    return {
        "input_ids": prompt_parts,
        "segment_ids": segment_ids,
        "gold_answer": [final_answer] if final_answer else [],
        "max_tool_delay": max([t.get("end_time", 0) for t in tool_calls]) - 0.3 if tool_calls else 0.0
    }


class SimpleListDataset(Dataset):
    """Simple Dataset wrapper for a list of preprocessed items."""
    def __init__(self, items):
        self.items = items
    
    def __len__(self):
        return len(self.items)
    
    def __getitem__(self, idx):
        return self.items[idx]


class DataCollatorForLinkGeneration:
    def __init__(self, pad_id: int):
        self.pad_id = pad_id
    def __call__(self, batch):
        input_ids_list = []
        segment_ids_list = []
        attention_mask_list = []
        gold_answers = []
        max_delays = []
        
        max_len = max(len(x["input_ids"]) for x in batch)
        
        for item in batch:
            seq_len = len(item["input_ids"])
            pad_len = max_len - seq_len
            
            input_ids_list.append(item["input_ids"] + [self.pad_id]*pad_len)
            segment_ids_list.append(item["segment_ids"] + [-1]*pad_len)
            attention_mask_list.append([1]*seq_len + [0]*pad_len)
            gold_answers.append(item["gold_answer"])
            max_delays.append(item["max_tool_delay"])
        
        return {
            "input_ids": torch.LongTensor(input_ids_list),
            "segment_ids": torch.LongTensor(segment_ids_list),
            "attention_mask": torch.LongTensor(attention_mask_list),
            "gold_answers": gold_answers,
            "max_delays": max_delays
        }


def main():
    dataset_path = args.dataset_path
    model_path = args.model_path
    batch_size = args.batch_size
    num_samples = args.num_samples
    hf = args.hf
    reencode_num = args.reencode_num
    attn_type = args.attn_type
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    mem_start = 128254
    mem_end = 128255
    special_token_start = 128011
    
    print("="*80)
    print(f"TEST 1: STANDARD SYNCHRONOUS with {attn_type.upper()} ATTENTION")
    print("(Wait for ALL modules, then prefill everything together)")
    print("="*80)
    
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
    
    # Preprocess dataset manually
    preprocessed_samples = []
    for example in dataset:
        processed = preprocess_with_links(
            example,
            tokenizer=tokenizer,
            reencode_num=reencode_num,
            mem_start=mem_start,
            mem_end=mem_end,
            special_token_start=special_token_start
        )
        preprocessed_samples.append(processed)
    
    collate_fn = DataCollatorForLinkGeneration(pad_id=tokenizer.pad_token_id)
    eval_dataloader = DataLoader(SimpleListDataset(preprocessed_samples), batch_size=batch_size, collate_fn=collate_fn)
    
    # Generation setup
    eot_id = tokenizer.convert_tokens_to_ids("<|eot_id|>")
    generation_cfg = GenerationConfig(
        do_sample=False,
        num_beams=1,
        max_new_tokens=200,
        stop_strings=["<|end_of_text|>", "<|eot_id|>"],
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=eot_id,
    )
    assistant_prompt = "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    assistant_ids = tokenizer(assistant_prompt, add_special_tokens=False)["input_ids"]
    assistant_ids = torch.LongTensor(assistant_ids)
    assistant_ids = move_to_target_device(assistant_ids, device)
    
    # Run evaluation
    correct_num = 0
    res_list = []
    total_compute_time = 0.0
    total_time_with_wait = 0.0
    ttft_list = []
    real_ttft_list = []  # TTFT from experiment start (planning end + 0.3s)
    ttft_after_last_tool_list = []  # KEY METRIC: TTFT after LAST tool arrival
    total_wait = 0.0
    prog_bar = auto_tqdm(range(len(eval_dataloader)))
    
    for batch_id, batch in enumerate(eval_dataloader):
        curr_bsz = batch["input_ids"].size(0)
        batch_gold = batch["gold_answers"]
        
        # START TIMING FROM BEGINNING (like test3 does)
        sample_start_time = time.time()
        
        # SIMULATE WAITING FOR ALL TOOLS (record time without actual sleeping)
        max_d = max(batch["max_delays"])
        if max_d > 0:
            print(f"  Simulated waiting {max_d:.2f}s for all tools to return...")
            total_wait += max_d
        
        compute_start_time = time.time()
        
        input_ids = batch["input_ids"]
        segment_ids = batch["segment_ids"]
        pad_mask = batch["attention_mask"]
        
        with torch.no_grad():
            input_ids = move_to_target_device(input_ids, device)
            segment_ids = move_to_target_device(segment_ids, device)
            pad_mask = move_to_target_device(pad_mask, device)
            
            if attn_type == "blocked":
                # Use BLOCKED attention with segment mask!
                attn_mask = make_segment_mask(
                    source_segments=segment_ids,
                    target_segments=segment_ids,
                    add_causal_lm_mask=True
                )
                attn_mask_4d = attn_mask.unsqueeze(1)
                attn_mask_4d = move_to_target_device(attn_mask_4d, device)
                if attn_mask_4d.dtype != model.dtype:
                    attn_mask_4d = attn_mask_4d.to(model.dtype)
                prefill_out = model(input_ids=input_ids, attention_mask=attn_mask_4d)
            else:
                # Use STANDARD full attention
                prefill_out = model(input_ids=input_ids, attention_mask=pad_mask)
            
            past_kv = prefill_out.past_key_values
            
            # Generate response
            gen_prefix = assistant_ids.repeat(curr_bsz, 1)
            gen_input = torch.cat([input_ids, gen_prefix], dim=1)
            gen_pad_mask = torch.cat([pad_mask, torch.ones_like(gen_prefix)], dim=1)
            
            # KEY METRIC: For test1, this is all the compute time after last tool arrived!
            ttft_after_last_start = time.time()
            ttft_start = time.time()
            outputs = model.generate(
                input_ids=gen_input,
                attention_mask=gen_pad_mask,
                use_cache=True,
                generation_config=generation_cfg,
                past_key_values=past_kv,
                tokenizer=tokenizer
            )
            ttft = time.time() - ttft_start
            ttft_list.append(ttft)
            
            # REAL TTFT: from experiment start (planning_end=0.3s) to first token
            real_ttft = max_d + (time.time() - compute_start_time)
            real_ttft_list.append(real_ttft)
            
            # TTFT after LAST tool arrived: this is the key metric!
            ttft_after_last = time.time() - ttft_after_last_start
            ttft_after_last_tool_list.append(ttft_after_last)
        
        compute_time = time.time() - compute_start_time
        total_compute_time += compute_time
        total_time_with_wait += (time.time() - sample_start_time)
        
        # Process responses
        generated_seqs = [
            tokenizer.decode(outputs[i, input_ids.size(1):].tolist())
            for i in range(curr_bsz)
        ]
        responses = [
            seq.split("<|start_header_id|>assistant<|end_header_id|>")[-1].strip().split("<|eot_id|>")[0]
            for seq in generated_seqs
        ]
        
        for idx, resp in enumerate(responses):
            print(f"Response: {resp}")
            print(f"Ground truth: {batch_gold[idx]}")
            print("---")
            score = best_subspan_em(resp, batch_gold[idx])
            correct_num += int(score)
            res_list.append({
                "response": resp,
                "gold_answer": batch_gold[idx],
                "score": score
            })
        
        print(f"Correct: {correct_num}/{(batch_id+1)*curr_bsz}")
        prog_bar.update(1)
    
    # Final stats
    accuracy = correct_num / len(dataset)
    avg_compute_time = total_compute_time / len(dataset)
    avg_total_time = total_time_with_wait / len(dataset)
    avg_ttft = np.mean(ttft_list) if ttft_list else 0.0
    avg_real_ttft = np.mean(real_ttft_list) if real_ttft_list else 0.0
    avg_ttft_after_last = np.mean(ttft_after_last_tool_list) if ttft_after_last_tool_list else 0.0
    
    print(f"\nFINAL RESULTS (TEST 1: STANDARD SYNC):")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Avg compute time/sample: {avg_compute_time:.4f}s")
    print(f"Avg total time/sample (incl wait): {avg_total_time:.4f}s")
    print(f"Avg TTFT (compute only): {avg_ttft:.4f}s")
    print(f"Avg REAL TTFT (from start): {avg_real_ttft:.4f}s")
    print(f"=== KEY METRIC ===")
    print(f"Avg TTFT AFTER LAST TOOL ARRIVED: {avg_ttft_after_last:.4f}s")
    print(f"Total wait time: {total_wait:.4f}s")
    print(f"Total compute time: {total_compute_time:.4f}s")
    print(f"Total time (including wait): {total_time_with_wait:.4f}s")
    
    result_dir = args.output_dir
    os.makedirs(result_dir, exist_ok=True)
    
    res_file = os.path.join(result_dir, f"test1_standard_sync_{attn_type}_accuracy_{accuracy:.4f}.jsonl")
    with open(res_file, "w", encoding="utf-8") as f:
        for entry in res_list:
            f.write(json.dumps(entry) + "\n")
    print(f"\nResults saved to {res_file}")
    
    stats_file = os.path.join(result_dir, f"test1_standard_sync_{attn_type}_stats.json")
    with open(stats_file, "w", encoding="utf-8") as f:
        json.dump({
            "accuracy": accuracy,
            "avg_time": avg_compute_time,
            "avg_ttft": avg_ttft,
            "avg_real_ttft": avg_real_ttft,
            "avg_ttft_after_last_tool": avg_ttft_after_last,
            "total_time_compute": total_compute_time,
            "total_wait_time": total_wait,
            "total_time_total": total_time_with_wait,
            "num_samples": len(dataset),
            "batch_size": batch_size,
            "attn_type": attn_type,
            "description": "Test 1: Standard Sync - Wait for all modules then prefill together"
        }, f, indent=2)
    print(f"Stats saved to {stats_file}")


if __name__ == "__main__":
    main()
