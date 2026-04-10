"""Simple JSONL loader that doesn't use datasets library."""
import json
from typing import List, Dict, Any


def load_jsonl_dataset(file_path: str, num_samples: int = -1) -> List[Dict[str, Any]]:
    """Load a JSONL file directly into a list of dictionaries."""
    samples = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                sample = json.loads(line)
                samples.append(sample)
                if num_samples > 0 and len(samples) >= num_samples:
                    break
            except Exception as e:
                print(f"Skipping line {line_num}: {e}")
                continue
    return samples
