"""最简单的下载
"""

import argparse
import os
from datasets import load_dataset
from datasets import DownloadConfig



def parse_args():
    p = argparse.ArgumentParser(description="Download small FineWeb sample and save to disk")
    p.add_argument("--num_samples", type=int, default=1000, help="number of samples to download (small for local testing)")
    p.add_argument("--save_dir", type=str, default="./fineweb_sample", help="output folder to save dataset")
    p.add_argument("--dataset_name", type=str, default="nvidia/Daring-Anteater", help="datasets identifier")
    p.add_argument("--dataset_config", type=str, default="sample-10BT", help="dataset config / subset name")
    p.add_argument("--split", type=str, default="train", help="which split to use")
    p.add_argument("--shuffle_seed", type=int, default=42, help="seed for shuffling before selecting samples")
    p.add_argument("--save_tokenizer", action="store_true", help="also download and save tokenizer/model files")
    p.add_argument("--model_name", type=str, default="meta-llama/Llama-3.2-1B-Instruct", help="model id for tokenizer if --save_tokenizer")
    p.add_argument("--hf_token", type=str, default="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", help="HuggingFace token (optional); or set HF_TOKEN env var")
    return p.parse_args()


def main():
    args = parse_args()
    hf_token = args.hf_token or os.environ.get("HF_TOKEN")

    print(f"Loading dataset {args.dataset_name} name={args.dataset_config} split={args.split} ...")
    # ds = load_dataset(args.dataset_name, name=args.dataset_config, split=args.split, streaming=True,  use_auth_token=hf_token) # 最初，错，说没有use_auth_token参数
    # fw = load_dataset("HuggingFaceFW/fineweb", name="CC-MAIN-2024-10", split="train", streaming=True) # 官方命令
    # ds = load_dataset(args.dataset_name, name=args.dataset_config, split=args.split, streaming=True, token=hf_token) #AttributeError: 'IterableDataset' object has no attribute 'save_to_disk'
    # ds = load_dataset(args.dataset_name, split=args.split, streaming=False, token=hf_token,cache_dir=r"E:\\huggingfaceee",) # 不使用 streaming（完整下载与处理）
    ds = load_dataset(args.dataset_name, split=args.split, streaming=False, token=hf_token) # 不使用 streaming（完整下载与处理）

    try:
        total = len(ds)
    except Exception:
        total = None
    print("dataset length:", total)

    # out_dir = args.save_dir
    # os.makedirs(out_dir, exist_ok=True)
    # print(f"Saving sample dataset to {out_dir} ...")
    # ds.save_to_disk(out_dir)
    # print("saved dataset to", out_dir)

    print("Done.")


if __name__ == "__main__":
    main()
