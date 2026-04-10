"""
download the fineweb pre-training corpus

```
python scripts/data_process/fineweb.py --num_samples=10000000 --min_length_for_memory=2048 --validation_size=3000
```
"""

from typing import Any, Dict, List

from absl import app, flags
from datasets import load_dataset, load_from_disk, DatasetDict
from transformers import AutoTokenizer
import os

FLAGS = flags.FLAGS

def set_args():
    flags.DEFINE_integer(
        "num_samples",
        default=10_000_000,
        help="number of samples to sample from the FineWeb.",
    )
    flags.DEFINE_integer(
        "min_length_for_memory",
        default=2048,
        help="minimum length for pre-training text subset with memory.",
    )
    flags.DEFINE_integer(
        "validation_size",
        default=3_000,
        help="number of samples for validation set.",
    )




def main(argv):
    # tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")
    llm_path = r"/home/dell/wyb/my_models/models/LLM-Research/Llama-3___2-1B-Instruct"
    print("files:", os.listdir(llm_path))
    try:
        tokenizer = AutoTokenizer.from_pretrained(llm_path, local_files_only=True)
    except Exception as e:
        print(f"Failed to load tokenizer from {llm_path}: {e}")
        print("Tokenizer directory listing:", os.listdir(llm_path))
        raise
    num_samples = FLAGS.num_samples
    # dataset = load_dataset(
    #     "HuggingFaceFW/fineweb",
    #     name="sample-10BT",
    #     split="train",
    # )
    print("load tokenizer ok, load dataset now......")
    # prefer offline usage; if network is unavailable this will fail fast
    cache_dir = "/home/dell/wyb/huggingface"
    try:
        # dataset = load_dataset(
        #     "HuggingFaceFW/fineweb",
        #     name="sample-10BT",
        #     split="train",
        #     cache_dir=r"/home/dell/wyb/huggingface",
        #     local_files_only=True,
        # )

        # dataset = load_dataset(
        #     "HuggingFaceFW/fineweb",
        #     name="sample-10BT",
        #     split="train",
        #     cache_dir=cache_dir,
        #     local_files_only=True,
        # )
        ### wyb：完全离线
        # dataset = load_from_disk("/home/dell/wyb/huggingface/datasets/HuggingFaceFW___fineweb/sample-10BT/0.0.0/9bb295ddab0e05d785b879661af7260fed5140fc")
        dataset = load_dataset("arrow", data_files="/home/dell/wyb/huggingface/datasets/HuggingFaceFW___fineweb/sample-10BT/0.0.0/9bb295ddab0e05d785b879661af7260fed5140fc/*.arrow")
    except Exception as e:
        print("Failed to load dataset 'HuggingFaceFW/fineweb' with local_files_only=True.")
        # print("Error:", e)
        # print("If you're offline, ensure the dataset is cached under the cache_dir path or set local_files_only=False to allow downloads.")
        raise

    # If load_dataset returned a DatasetDict (multiple splits), select the
    # `train` split if available, otherwise pick the first split.
    if isinstance(dataset, DatasetDict):
        if "train" in dataset:
            dataset = dataset["train"]
        else:
            # pick first available split
            dataset = list(dataset.values())[0]

    total_samples = len(dataset)
    print("total samples num: ", total_samples)

    # Extract the last 90,000 samples
    last_half_samples = dataset.select(range(0, total_samples))

    def tokenize_texts(examples: Dict[str, List[Any]]):
        token_counts = tokenizer(examples["text"], add_special_tokens= False)["input_ids"]
        examples["num_tokens"] = [len(x) for x in token_counts]
        return examples

    dataset_with_token_num = last_half_samples.map(tokenize_texts, batched=True, num_proc=192)

    # def filter_fn(example: Dict[str, Any]):
    #     return example["num_tokens"] > FLAGS.min_length_for_memory
    # filtered_dataset = dataset_with_token_num.filter(filter_fn)
    def filter_fn(examples: Dict[str, List[Any]]):
        token_counts = examples["num_tokens"]
        return [x > FLAGS.min_length_for_memory for x in token_counts]
    filtered_dataset = dataset_with_token_num.filter(filter_fn, batched=True, num_proc=192)
    filtered_dataset = filtered_dataset.remove_columns("num_tokens")

    text_mem = filtered_dataset.select(range(0, len(filtered_dataset) // 2))
    text_inst = filtered_dataset.select(range(len(filtered_dataset) // 2, len(filtered_dataset)))

    random_seed = 42
    text = dataset.shuffle(seed=random_seed).select(range(0, num_samples))


    text_mem = text_mem.train_test_split(test_size=FLAGS.validation_size)
    text_inst = text_inst.train_test_split(test_size=FLAGS.validation_size)
    text = text.train_test_split(test_size=FLAGS.validation_size)

    # print("text:", len(text), "textmem:", len(text_mem), "text:", len(text_inst),)
    print("text:", text, "textmem:", text_mem, "text inst:", text_inst,)
    shards = {'train': 128, 'test': 4}
    text.save_to_disk("dataset_cache/processed/fineweb/text", num_shards=shards, num_proc=128)
    text_mem.save_to_disk("dataset_cache/processed/fineweb/text_mem", num_shards=shards, num_proc=128)
    text_inst.save_to_disk("dataset_cache/processed/fineweb/text_inst", num_shards=shards, num_proc=128)

if __name__ == "__main__":
    set_args()
    app.run(main)

