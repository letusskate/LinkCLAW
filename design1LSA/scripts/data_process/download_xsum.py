"""最简单的下载
"""

import argparse
import os
from datasets import load_dataset
from datasets import DownloadConfig



def main():
    # ds = load_dataset(args.dataset_name, name=args.dataset_config, split=args.split, streaming=True,  use_auth_token=hf_token) # 最初，错，说没有use_auth_token参数
    # fw = load_dataset("HuggingFaceFW/fineweb", name="CC-MAIN-2024-10", split="train", streaming=True) # 官方命令
    # ds = load_dataset(args.dataset_name, name=args.dataset_config, split=args.split, streaming=True, token=hf_token) #AttributeError: 'IterableDataset' object has no attribute 'save_to_disk'
    # ds = load_dataset(args.dataset_name, split=args.split, streaming=False, token=hf_token,cache_dir=r"E:\\huggingfaceee",) # 不使用 streaming（完整下载与处理）
    ds = load_dataset("EdinburghNLP/xsum", split="train", streaming=False, token="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") # 不使用 streaming（完整下载与处理）
# TODO 无法下载，报错
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
