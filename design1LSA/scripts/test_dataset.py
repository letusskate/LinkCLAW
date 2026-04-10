from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CACHE_STREAM_PATH = PROJECT_ROOT / "dataset_cache" / "processed" / "fineweb" / "text" / "test" / "cache-5c48ad9551069d1c_00000_of_00032.arrow"

# # 方法1: 使用 pyarrow
# table = pa.ipc.RecordBatchFileReader(PROJECT_ROOT / 'dataset_cache' / 'processed' / 'block_qa' / 'qa' / 'train' / 'data-00000-of-00128.arrow').read_all()
# df = table.to_pandas()

# 方法2: 使用 pyarrow.feather (如果文件是 Feather 格式)
# 注意: Feather 是 Arrow 的一种序列化格式
# df = pd.read_feather(PROJECT_ROOT / 'dataset_cache' / 'processed' / 'block_qa' / 'qa' / 'train' / 'data-00000-of-00128.arrow')

# # 方法3: 读取为 Arrow Table
# with pa.ipc.open_file(PROJECT_ROOT / 'dataset_cache' / 'processed' / 'block_qa' / 'qa' / 'train' / 'data-00000-of-00128.arrow') as reader:
#     table = reader.read_all()
    
# 方法4: 如果是流格式
# /home/dell/wyb/huggingface/datasets/HuggingFaceFW___fineweb/sample-10BT/0.0.0/9bb295ddab0e05d785b879661af7260fed5140fc/fineweb-train-00000-of-00102.arrow
# PROJECT_ROOT / 'dataset_cache' / 'processed' / 'fineweb' / 'text' / 'test' / 'cache-5c48ad9551069d1c_00000_of_00032.arrow'
with pa.ipc.open_stream(CACHE_STREAM_PATH) as reader:
    for batch in reader:
        print(batch)
        # 处理每个批次
        pass