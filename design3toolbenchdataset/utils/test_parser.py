import json

with open('./data/data/toolllama_G123_dfs_train.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

print(f"Total samples: {len(data)}")
print(f"First sample keys: {data[0].keys()}")
print(f"First sample: {json.dumps(data[0], indent=2, ensure_ascii=False)[:500]}")
