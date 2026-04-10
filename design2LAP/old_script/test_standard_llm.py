"""
Test standard LLM (synchronous execution) with both standard and blocked attention.
Based on common_sparse.py but using async ToolBench dataset.

Usage:
python test_standard_llm.py \
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
import sys
from typing import Dict, List, Any

# Add project root to path
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


parser = argparse.ArgumentParser(description="Test standard LLM (sync) with both attention types.")
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
    print(f"STANDARD LLM (SYNCHRONOUS) with {attn_type.upper()} ATTENTION")
    print("="*80)
    
    dataset = datasets.load_dataset("json", data_files=dataset_path, split="train")
    if num_samples > 0:
        dataset = dataset.select(np.arange(min(num_samples, len(dataset))))
    print(f"Dataset loaded: {len(dataset)} samples")
    
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
    
    # Preprocess dataset
    exist_cols = dataset.column_names
    dataset = dataset.map(
        preprocess_with_links,
        batched=False,
        num_proc=1,
        remove_columns=exist_cols,
        fn_kwargs=dict(tokenizer=tokenizer, reencode_num=reencode_num, mem_start=mem_start, mem_end=mem_end, special_token_start=special_token_start)
    )
    
    collate_fn = DataCollatorForLinkGeneration(pad_id=tokenizer.pad_token_id)
    eval_dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn)
    
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
    total_time = 0.0
    ttft_list = []
    total_wait = 0.0
    prog_bar = auto_tqdm(range(len(eval_dataloader)))
    
    for batch_id, batch in enumerate(eval_dataloader):
        curr_bsz = batch["input_ids"].size(0)
        batch_gold = batch["gold_answers"]
        
        # SIMULATE WAITING FOR ALL TOOLS
        max_d = max(batch["max_delays"])
        if max_d > 0:
            print(f"  Waiting {max_d:.2f}s for all tools to return...")
            time.sleep(max_d)
            total_wait += max_d
        
        start_time = time.time()
        
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
        
        batch_time = time.time() - start_time
        total_time += batch_time
        
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
    avg_time = total_time / len(dataset)
    avg_ttft = np.mean(ttft_list) if ttft_list else 0.0
    
    print(f"\nFINAL RESULTS:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Avg time/sample (compute): {avg_time:.4f}s")
    print(f"Avg TTFT: {avg_ttft:.4f}s")
    print(f"Total wait time: {total_wait:.4f}s")
    print(f"Total time (including wait): {total_time + total_wait:.4f}s")
    
    current_ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    os.makedirs("result", exist_ok=True)
    
    res_file = f"result/standard_llm_{attn_type}_accuracy_{accuracy:.4f}_{current_ts}.jsonl"
    with open(res_file, "w", encoding="utf-8") as f:
        for entry in res_list:
            f.write(json.dumps(entry) + "\n")
    print(f"\nResults saved to {res_file}")
    
    stats_file = f"result/standard_llm_{attn_type}_stats_{current_ts}.json"
    with open(stats_file, "w", encoding="utf-8") as f:
        json.dump({
            "accuracy": accuracy,
            "avg_time": avg_time,
            "avg_ttft": avg_ttft,
            "total_time_compute": total_time,
            "total_wait_time": total_wait,
            "total_time_total": total_time + total_wait,
            "num_samples": len(dataset),
            "batch_size": batch_size,
            "attn_type": attn_type
        }, f, indent=2)
    print(f"Stats saved to {stats_file}")


if __name__ == "__main__":
    main()
