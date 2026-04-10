import json

with open('./data/data/toolllama_G123_dfs_train.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

print(f"Total samples: {len(data)}")
print("\nFirst sample keys:", data[0].keys())
print("\nConversations count:", len(data[0].get('conversations', [])))
print("\nConversations:")
for i, conv in enumerate(data[0].get('conversations', [])):
    print(f"\n  Conversation {i}:")
    print(f"    from: {conv.get('from', '')}")
    print(f"    value (first 500 chars): {conv.get('value', '')[:500]}")
