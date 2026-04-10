from src.data.titan_datasets import load_data_and_process_fn
from src.data.titan_preprocessor import SumAttentionPreprocessor
from src.data.titan_tokenizer import LLaMA32Tokenizer
from src.training.titan_training_utils import DATASET_MAPPING

version = "wybnoqasum"
path = "/home/dell/wyb/my_models/models/LLM-Research/Llama-3___2-1B-Instruct/original/tokenizer.model"

paths = {
    "text": "./dataset_cache/processed/fineweb/text",
    "text_mem": "./dataset_cache/processed/fineweb/text_mem",
    "text_inst": "./dataset_cache/processed/fineweb/text_inst",
    "sft": "./dataset_cache/processed/daringanteater/sft",
    "sft_mem": "./dataset_cache/processed/daringanteater/sft_mem",
    "tulu": "./dataset_cache/processed/tulu/sft",
    "qa": "./dataset_cache/processed/block_qa/qa",
    "qa_mem": "./dataset_cache/processed/block_qa/qa_mem",
    "xsum": "./dataset_cache/processed/xsum/xsum",
}

tok = LLaMA32Tokenizer(path)
pre = SumAttentionPreprocessor(
    tokenizer=tok,
    max_len=128,
    special_token_start=128011,
    mem_start=128254,
    mem_end=128255,
    reencode_num=5,
    max_memory_num=40,
)

print("DATASET_VERSION=", version)
print("DATA_COMPONENTS=", [x.dataset_name for x in DATASET_MAPPING[version]])

for comp in DATASET_MAPPING[version]:
    name = comp.dataset_name
    ds, fn, cols = load_data_and_process_fn(name, pre, training=True)
    print("\n===", name, "===")
    print("path=", paths.get(name, "unknown"))
    print("remove_columns(filtered)=", cols)
    print("column_names=", ds.column_names)

    it = iter(ds)
    for i in range(5):
        raw = next(it)
        proc = fn(raw)
        x = proc["input_ids"]
        y = proc["labels"]
        valid = sum(1 for t in y if t != -100)
        print(f"sample#{i} lenx={len(x)} leny={len(y)} valid={valid} all_ignore={valid == 0}")
        print("  x[:12]=", x[:12])
        print("  y[:12]=", y[:12])
        print("  y[-12:]=", y[-12:])
