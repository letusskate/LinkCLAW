"""Generate synthetic block_qa.jsonl data.

Usage:
    python scripts/data_process/wyb_generate_block_qa.py --num 1000 --out data/raw/block_qa/block_qa.jsonl --append
    python scripts/data_process/wyb_generate_block_qa.py --num 10000 --out data/raw/block_qa/block_qa.jsonl --append --model_path /home/dell/wyb/my_models/Qwen/Qwen/Qwen2___5-0___5B-Instruct --device cuda

This script appends generated JSONL records if `--append` is set, otherwise it overwrites the target file.
Each record contains: question, answers, generated, ctxs (empty), documents (10 items with title,text,score).

If `--model_path` is provided, the script will load the local model/tokenizer and use it to generate the
`generated` field for each record. Set `--device` to `cuda` or `cpu` as appropriate.
"""
import argparse
import json
import random
import os
from typing import List, Dict, Optional

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)


SAMPLE_QUESTIONS = [
    ("What is the capital of {}?", "{} City"),
    ("Who wrote {}?", "Author of {}"),
    ("What is the boiling point of {}?", "100°C"),
    ("Define {}.", "Definition of {}"),
    ("What causes {}?", "Because of {}"),
    ("When did {} happen?", "In year {}"),
    ("How many {} are there in {}?", "{}"),
    ("Explain {} in simple terms.", "Simple explanation of {}")
]

SAMPLE_ENTITIES = [
    "France", "water", "photosynthesis", "the seasons", "World War II", "the internet", "electricity",
    "gravity", "photosynthesis", "Pride and Prejudice", "Python programming", "quantum mechanics",
]

SAMPLE_DOC_TEMPLATES = [
    "{entity} is a topic with many aspects. {entity} has historical and scientific importance.",
    "Overview: {entity} — background, significance, and common facts.",
    "Detailed description of {entity}. Key points include causes, effects, and examples.",
    "Applications and examples related to {entity}. Common use cases and notes.",
    "Historical notes about {entity}, including important dates and people.",
    "Technical summary about {entity} with definitions and concise explanations.",
]


def make_document(entity: str, idx: int) -> Dict:
    title = f"{entity} - Doc {idx}"
    text = random.choice(SAMPLE_DOC_TEMPLATES).format(entity=entity)
    return {"title": title, "text": text, "score": 0.0}


def make_record(i: int) -> Dict:
    entity = random.choice(SAMPLE_ENTITIES)
    q_template, ans_template = random.choice(SAMPLE_QUESTIONS)
    # Determine how many positional placeholders '{}' are in the question template
    q_placeholders = q_template.count('{}')
    # Build args for formatting: if two placeholders, pick a second entity or a year where appropriate
    if q_placeholders == 1:
        q_args = (entity,)
    elif q_placeholders == 2:
        # choose a second distinct entity where meaningful
        other = random.choice([e for e in SAMPLE_ENTITIES if e != entity])
        q_args = (entity, other)
    else:
        q_args = (entity,)

    try:
        question = q_template.format(*q_args)
    except Exception:
        question = q_template
    # Simple deterministic generated and answer text for sanity
    # Format answer template with same args if it contains placeholders
    if ans_template.count('{}') == len(q_args):
        try:
            answer_text = ans_template.format(*q_args)
        except Exception:
            answer_text = ans_template
    elif ans_template.count('{}') > 0:
        # if answer expects one slot, provide the primary entity
        try:
            answer_text = ans_template.format(entity)
        except Exception:
            answer_text = ans_template
    else:
        answer_text = ans_template
    # For variety, sometimes set a numeric or short answer
    if 'boiling' in question.lower():
        answer = ["100°C"]
        generated = "Water boils at 100°C at sea level."
    else:
        answer = [answer_text]
        generated = answer_text

    documents = [make_document(entity, j + 1) for j in range(10)]

    return {
        "question": question,
        "answers": answer,
        "generated": generated,
        "ctxs": [],
        "documents": documents,
    }


def generate_with_model(rec: Dict, model, tokenizer, device: str = "cuda", max_new_tokens: int = 64) -> str:
    # Build a short prompt including the top documents and the question
    docs_text = "\n".join([f"Document [{j+1}]: {d['text']}" for j, d in enumerate(rec['documents'])])
    prompt = (
        "You are an intelligent assistant. Use the following documents to answer the question.\n\n"
        f"{docs_text}\n\nQuestion: {rec['question']}\nAnswer:"
    )

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
    input_ids = inputs.input_ids.to(device)
    attention_mask = inputs.attention_mask.to(device)

    gen_kwargs = dict(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        eos_token_id=tokenizer.eos_token_id if tokenizer.eos_token_id is not None else None,
    )

    with torch.no_grad():
        outputs = model.generate(**gen_kwargs)

    # take the generated tokens after the prompt
    generated_text = tokenizer.decode(outputs[0][input_ids.shape[-1]:], skip_special_tokens=True).strip()
    # fallback: if empty, return simple baseline
    return generated_text if generated_text else rec["generated"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num", type=int, default=1000, help="Number of samples to generate")
    parser.add_argument("--out", type=str, default="data/raw/block_qa/block_qa.jsonl", help="Output JSONL path")
    parser.add_argument("--append", action="store_true", help="Append to existing file instead of overwrite")
    parser.add_argument("--model_path", type=str, default=None, help="Local model path to use for generation")
    parser.add_argument("--device", type=str, default=None, help="Device to load model on: cuda or cpu")
    parser.add_argument("--max_new_tokens", type=int, default=64, help="Max new tokens to generate per sample")
    args = parser.parse_args()

    out_dir = os.path.dirname(args.out)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    mode = "a" if args.append else "w"
    start_idx = 0
    if args.append and os.path.exists(args.out):
        # try to count existing lines to continue numbering if desired
        with open(args.out, "r", encoding="utf-8") as f:
            start_idx = sum(1 for _ in f)

    # optionally load local model/tokenizer for generation
    lm_model = None
    lm_tokenizer = None
    device = args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    if args.model_path:
        print(f"Loading model/tokenizer from: {args.model_path} on device {device}")
        lm_tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
        try:
            lm_model = AutoModelForCausalLM.from_pretrained(
                args.model_path,
                torch_dtype=torch.bfloat16 if device.startswith("cuda") else None,
                low_cpu_mem_usage=True,
            )
            lm_model.to(device)
        except Exception:
            # fallback to cpu load
            lm_model = AutoModelForCausalLM.from_pretrained(args.model_path)
            lm_model.to(device)

    with open(args.out, mode, encoding="utf-8") as f:
        for i in range(args.num):
            rec = make_record(start_idx + i + 1)
            if lm_model is not None and lm_tokenizer is not None:
                try:
                    rec_generated = generate_with_model(rec, lm_model, lm_tokenizer, device=device, max_new_tokens=args.max_new_tokens)
                    rec["generated"] = rec_generated
                except Exception as e:
                    # if generation fails, keep deterministic generated text and continue
                    print(f"Generation failed for sample {i}: {e}")
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"Wrote {args.num} samples to {args.out} (append={args.append})")


if __name__ == "__main__":
    main()
