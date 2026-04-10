"""
Evaluate the performance of LLMs fine-tuned via TorchTitan on NQ open dataset.

1. Convert the Titan checkpoint from DCP to torch:
```
python -m torch.distributed.checkpoint.format_utils dcp_to_torch \
    torchtitan/outputs/checkpoint/step-1000 checkpoint.pt
```

2. Run evaluation:
```
python scripts/evaluation/cona_chunk.py \
    --pos 0 \
    --batch_size 10 \
    --reencode_num 5 \
    --attn_type "blocked" \
    --doc_chunk_tokens 256
```
"""
import argparse
import datetime
import json
import os
import string
from typing import Dict, List

import datasets
import numpy as np
import regex
import torch
from safetensors import safe_open
from torch.utils.data import DataLoader
from torchtune.models.convert_weights import tune_to_hf
from tqdm.auto import tqdm as auto_tqdm
from transformers import AutoTokenizer, GenerationConfig, LlamaForCausalLM, AutoModelForCausalLM

from src.common import move_to_target_device
from src.data.titan_preprocessor import LLaMA32Tokenizer, make_segment_mask

parser = argparse.ArgumentParser(description="Run script with specified ckpt and pos.")
parser.add_argument(
    "--ckpt_path",
    type=str,
    default=None,
    help="The path to the `checkpoint.pt` file.",
)
parser.add_argument("--pos", type=int, required=True, help="Position value")
parser.add_argument("--batch_size", type=int, default=1, help="Batch size of the evaluation.")
parser.add_argument(
    "--attn_type",
    type=str,
    required=True,
    help="attention types.",
    choices=["standard", "blocked"],
)
parser.add_argument("--reencode_num", type=int, default=5, help="Number of the link tokens.")
parser.add_argument(
    "--doc_chunk_tokens",
    type=int,
    default=256,
    help="Token length for small document chunks (<=0 disables sub-chunking).",
)
parser.add_argument("--hf", type=bool, default=False, help="Use HuggingFace checkpoints.")

args = parser.parse_args()

def load_model_weights(ckpt_path: str):
    safe_tensor_file = os.path.join(ckpt_path, "model.safetensors")
    if os.path.exists(safe_tensor_file):
        state_dict = {}
        with safe_open(safe_tensor_file, framework="pt", device="cpu") as f:
            for key in f.keys():
                state_dict[key] = f.get_tensor(key)
        # state_dict["output.weight"] = state_dict["tok_embeddings.weight"]
        return state_dict

    state_dict = torch.load(ckpt_path, weights_only=False)

    state_dict = state_dict["model"]
    state_dict["output.weight"] = state_dict["tok_embeddings.weight"]

    converted_state_dict = tune_to_hf(
        state_dict=state_dict,
        num_heads=32,
        num_kv_heads=8,
        dim=2048,
    )
    return converted_state_dict

def normalize_answer(s: str) -> str:
    """Normalization from the SQuAD evaluation script.

    See https://worksheets.codalab.org/rest/bundles/0x6b567e1cf2e041ec80d7098f031c5c9e/contents/blob/
    """

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
        normalized_ground_truth = normalize_answer(ground_truth)
        if normalized_ground_truth.lower() in normalized_prediction.lower():
            return 1.0
    return 0.0

def preprocess_fn(example: Dict[str, str], tokenizer: LLaMA32Tokenizer, target_position: int, reencode_num: int, mem_start: int, mem_end: int, special_token_start:int):
    system = (
        "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou are a intelligent AI assistant. Please answer questions based on the user's instruction. "
        "Below are some reference documents that may help you in answering the user's question.<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
    )
    question = example["question"]
    memory_list = []

    doc_list = [example["ctxs"][x]["text"] for x in range(10)]
    title_list  = [example["ctxs"][x]["title"] for x in range(10)]
    # If target_position == 1, for example, then the code will read from the data source that
    # always put groud-truth at position 0.
    if target_position not in [0, 4, 9]:
        ground_truth_doc = doc_list.pop(0)
        ground_truth_title = title_list.pop(0)
        doc_list.insert(target_position, ground_truth_doc)
        title_list.insert(target_position, ground_truth_title)

    for j in range(0,10):
        title = title_list[j]
        text = doc_list[j]
        memory_list.append(f"Document [{j+1}](Title: {title}) {text}\n")

    qa_system_input_ids = tokenizer(system, add_special_tokens = False)["input_ids"]
    input_ids = qa_system_input_ids[:] + [mem_start]
    segment_ids = [0] * len(input_ids)

    for mem_id, st in enumerate(memory_list):
        tem_id = tokenizer(st, add_special_tokens = False)["input_ids"]
        segment_ids = segment_ids + [mem_id + 1] * len(tem_id) + [0] * reencode_num

        for sub_idx in range(reencode_num):
            tem_id = tem_id + [special_token_start + reencode_num * mem_id + sub_idx]
        input_ids = input_ids + tem_id

    new_prompt = question
    prompt_id = tokenizer(new_prompt, add_special_tokens = False)["input_ids"]
    input_ids = input_ids + [mem_end] + prompt_id
    segment_ids = segment_ids + [0] + [0] * len(prompt_id)
    return {
        "input_ids": input_ids,
        "segment_ids": segment_ids,
    }

def split_tokens(token_ids: List[int], chunk_size: int):
    if chunk_size is None or chunk_size <= 0:
        yield token_ids
        return
    for start in range(0, len(token_ids), chunk_size):
        yield token_ids[start : start + chunk_size]

def iter_chunked_prefill(
    example: Dict[str, str],
    tokenizer: LLaMA32Tokenizer,
    target_position: int,
    reencode_num: int,
    mem_start: int,
    mem_end: int,
    special_token_start: int,
    doc_chunk_tokens: int,
):
    system = (
        "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou are a intelligent AI assistant. Please answer questions based on the user's instruction. "
        "Below are some reference documents that may help you in answering the user's question.<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
    )
    question = example["question"]

    doc_list = [example["ctxs"][x]["text"] for x in range(10)]
    title_list = [example["ctxs"][x]["title"] for x in range(10)]
    if target_position not in [0, 4, 9]:
        ground_truth_doc = doc_list.pop(0)
        ground_truth_title = title_list.pop(0)
        doc_list.insert(target_position, ground_truth_doc)
        title_list.insert(target_position, ground_truth_title)

    system_ids = tokenizer(system, add_special_tokens=False)["input_ids"]
    yield system_ids + [mem_start], [0] * (len(system_ids) + 1)

    for mem_id in range(10):
        title = title_list[mem_id]
        text = doc_list[mem_id]
        doc_text = f"Document [{mem_id + 1}](Title: {title}) {text}\n"
        doc_ids = tokenizer(doc_text, add_special_tokens=False)["input_ids"]

        for chunk_ids in split_tokens(doc_ids, doc_chunk_tokens):
            yield chunk_ids, [mem_id + 1] * len(chunk_ids)

        special_ids = [special_token_start + reencode_num * mem_id + sub_idx for sub_idx in range(reencode_num)]
        if special_ids:
            yield special_ids, [0] * len(special_ids)

    prompt_ids = tokenizer(question, add_special_tokens=False)["input_ids"]
    tail_ids = [mem_end] + prompt_ids
    yield tail_ids, [0] * len(tail_ids)

class DataCollatorForGeneration():
    def __init__(self, pad_id: int):
        self.pad_id = pad_id
        pass
    def __call__(self, batch):
        input_ids = []
        segment_ids = []
        attention_mask = []
        length_list = [len(x['input_ids']) for x in batch]

        max_length = max(length_list)

        for item in batch:
            seq_length = len(item['input_ids'])

            residual = max_length - seq_length
            # padded_input_ids = [self.pad_id] * residual + item['input_ids']
            # curr_attention_mask = [0] * residual + [1] * seq_length
            padded_input_ids = item['input_ids'] + [self.pad_id] * residual
            curr_attention_mask = [1] * seq_length + [0] * residual
            input_ids.append(padded_input_ids)
            attention_mask.append(curr_attention_mask)
            segment_ids.append(item["segment_ids"] + [-1] * residual)

        return {
            "input_ids": torch.LongTensor(input_ids),
            "segment_ids": torch.LongTensor(segment_ids),
            "attention_mask": torch.LongTensor(attention_mask),
        }


def main():
    ckpt_path = args.ckpt_path
    pos = args.pos
    reencode_num: int  = args.reencode_num
    batch_size: int = args.batch_size
    doc_chunk_tokens: int = args.doc_chunk_tokens
    hf: bool = args.hf
    device = torch.device("cuda")

    mem_start = 128254
    mem_end = 128255
    special_token_start = 128011

    # Diff vs nq_eval.py: stream dataset and manually build batches to keep chunked prefill behavior.
    if pos in [0, 4, 9]:
        data_path = f"data/raw/nq/nq-open-10_{pos}.jsonl"
    else:
        data_path = "data/raw/nq/nq-open-10_0.jsonl"
    dataset = datasets.load_dataset("json", data_files=data_path, split="train", streaming=True)
    print("Dataset loaded in streaming mode")

    # tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")
    tokenizer = AutoTokenizer.from_pretrained("/home/dell/wyb/my_models/models/LLM-Research/Llama-3___2-1B-Instruct")
    tokenizer.pad_token_id = 128004
    tokenizer.pad_token = "<|finetune_right_pad_id|>"

    model = LlamaForCausalLM.from_pretrained(
        # "meta-llama/Llama-3.2-1B-Instruct",
        "/home/dell/wyb/my_models/models/LLM-Research/Llama-3___2-1B-Instruct",
        torch_dtype=torch.bfloat16,
        # torch_dtype=torch.float32,
    )
    # model.load_state_dict(state_dict, strict=True)

    if args.ckpt_path is None:
        print("Will NOT load fine-tuned models!")
    elif hf:
        model = AutoModelForCausalLM.from_pretrained(args.ckpt_path, torch_dtype=torch.bfloat16)
    else:
        state_dict = load_model_weights(ckpt_path)
        model.load_state_dict(state_dict, strict=False)
    model = model.to(device)
    model.eval()

    correct_num = 0
    res_list = []
    collate_fn = DataCollatorForGeneration(pad_id=tokenizer.pad_token_id)
    
    # Process dataset in chunks with streaming
    prog_bar = auto_tqdm(total=None, desc="Processing batches")

    eot_id = tokenizer.convert_tokens_to_ids("<|eot_id|>")
    print(eot_id)
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
    print(generation_token_ids)
    generation_token_ids = torch.LongTensor(generation_token_ids)
    generation_token_ids: torch.LongTensor = move_to_target_device(generation_token_ids, device)

    # Diff vs nq_eval.py: stream chunked documents and prefill incrementally per chunk.
    total_processed = 0

    for example in dataset:
        total_processed += 1
        input_ids_accum: List[int] = []
        segment_ids_accum: List[int] = []
        past_key_values = None

        for chunk_ids, chunk_segments in iter_chunked_prefill(
            example=example,
            tokenizer=tokenizer,
            target_position=pos,
            reencode_num=reencode_num,
            mem_start=mem_start,
            mem_end=mem_end,
            special_token_start=special_token_start,
            doc_chunk_tokens=doc_chunk_tokens,
        ):
            input_ids_accum.extend(chunk_ids)
            segment_ids_accum.extend(chunk_segments)

            input_ids = torch.LongTensor([chunk_ids])
            segment_ids = torch.LongTensor([segment_ids_accum])
            attention_mask_for_pad = torch.ones((1, len(segment_ids_accum)), dtype=torch.long)

            with torch.no_grad():
                input_ids = move_to_target_device(input_ids, device)
                segment_ids = move_to_target_device(segment_ids, device)
                attention_mask_for_pad = move_to_target_device(attention_mask_for_pad, device)

                if args.attn_type == "blocked":
                    attention_mask = make_segment_mask(
                        source_segments=segment_ids,
                        target_segments=segment_ids,
                        add_causal_lm_mask=True,
                    )
                    q_len = input_ids.size(1)
                    attention_mask = attention_mask[:, -q_len:, :]
                    attention_mask_4d = attention_mask.unsqueeze(1)
                    attention_mask_4d = move_to_target_device(attention_mask_4d, device)
                    if attention_mask_4d.dtype != model.dtype:
                        attention_mask_4d = attention_mask_4d.to(model.dtype)
                    prefilling_outputs = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask_4d,
                        past_key_values=past_key_values,
                        use_cache=True,
                    )
                elif args.attn_type == "standard":
                    prefilling_outputs = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask_for_pad,
                        past_key_values=past_key_values,
                        use_cache=True,
                    )
                else:
                    raise ValueError()
                past_key_values = prefilling_outputs.past_key_values

        input_ids_full = torch.LongTensor([input_ids_accum])
        attention_mask_for_pad = torch.ones_like(input_ids_full)
        input_ids_full = move_to_target_device(input_ids_full, device)
        attention_mask_for_pad = move_to_target_device(attention_mask_for_pad, device)

        generation_prefix = generation_token_ids.unsqueeze(0)
        generation_input_ids = torch.cat([input_ids_full, generation_prefix], axis=1)
        attention_mask_for_pad = torch.cat([attention_mask_for_pad, torch.ones_like(generation_prefix)], axis=1)

        with torch.no_grad():
            outputs = model.generate(
                input_ids=generation_input_ids,
                attention_mask=attention_mask_for_pad,
                use_cache=True,
                generation_config=generation_cfg,
                past_key_values=past_key_values,
                tokenizer=tokenizer,
            )

        generated_seq = tokenizer.decode(
            outputs[0, input_ids_full.size(1):].tolist(),
        )
        response = generated_seq.split("<|start_header_id|>assistant<|end_header_id|>")[-1].strip().split("<|eot_id|>")[0]
        print(response)
        print("Ground-truth: ", example["answers"])
        print("------\n")

        score = best_subspan_em(response, example["answers"])
        correct_num = correct_num + int(score)
        res_list.append(
            {
                "response": response,
                "gold_answer": example["answers"],
                "score": score,
            }
        )
        print("Correct progress", correct_num)
        prog_bar.update(1)

    accuracy = correct_num / total_processed
    print(f"Final Accuracy: {accuracy}")

    current_time = datetime.datetime.now()
    time_str = current_time.strftime("%Y%m%d-%H%M%S")

    file_name = f"result/nq_open_{accuracy}_{time_str}.jsonl"
    if not os.path.exists(os.path.dirname(file_name)):
        os.makedirs(os.path.dirname(file_name))

    with open(file_name, "w", encoding="utf-8") as f:
        for entry in res_list:
            json_line = json.dumps(entry)
            f.write(json_line + "\n")

    print(f"Dumped at {file_name}")

if __name__ == "__main__":
    main()

