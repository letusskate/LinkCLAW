#!/usr/bin/env python3
"""
生成nq_eval.py中最终prompt的示例
"""

# 模拟tokenizer的行为
class MockTokenizer:
    def __call__(self, text, add_special_tokens=False):
        # 简单模拟分词，返回text长度作为token数量
        return {"input_ids": [1] * len(text.split())}
    
    def convert_tokens_to_ids(self, token):
        return 128009

# 模拟数据
example = {
    "question": "What is the capital of France?",
    "ctxs": [
        {"title": "France", "text": "France is a country in Western Europe. Its capital is Paris."},
        {"title": "Paris", "text": "Paris is the largest city in France."},
        {"title": "Europe", "text": "Europe is a continent."},
        {"title": "Geography", "text": "Geography is the study of places."},
        {"title": "Capital Cities", "text": "Capital cities are the seats of government."},
        {"title": "Western Europe", "text": "Western Europe includes countries like France, Germany, and Spain."},
        {"title": "Tourism in France", "text": "France is a popular tourist destination."},
        {"title": "French Culture", "text": "French culture is known for its cuisine and art."},
        {"title": "History of France", "text": "France has a rich history."},
        {"title": "Government of France", "text": "France is a republic."}
    ]
}

# 模拟函数，生成prompt示例
def generate_prompt_example(example):
    system = (
        "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou are a intelligent AI assistant. Please answer questions based on the user's instruction. "
        "Below are some reference documents that may help you in answering the user's question.<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
    )
    question = example["question"]
    memory_list = []

    # 提取文档和标题
    doc_list = [example["ctxs"][x]["text"] for x in range(10)]
    title_list = [example["ctxs"][x]["title"] for x in range(10)]
    
    # 调整文档顺序（假设target_position=2）
    target_position = 2
    if target_position not in [0, 4, 9]:
        ground_truth_doc = doc_list.pop(0)
        ground_truth_title = title_list.pop(0)
        doc_list.insert(target_position, ground_truth_doc)
        title_list.insert(target_position, ground_truth_title)

    # 构建文档列表
    for j in range(0, 10):
        title = title_list[j]
        text = doc_list[j]
        memory_list.append(f"Document [{j+1}](Title: {title}) {text}\n")

    # 构建完整prompt
    prompt = system
    prompt += "<|mem_start|>"  # 模拟内存开始标记
    
    # 添加文档
    for i, doc in enumerate(memory_list):
        prompt += doc
        # 添加特殊标记
        for _ in range(5):  # 假设reencode_num=5
            prompt += f"<|link_token_{i}|>"
    
    prompt += "<|mem_end|>"  # 模拟内存结束标记
    prompt += question

    return prompt

# 生成示例prompt
if __name__ == "__main__":
    prompt = generate_prompt_example(example)
    print("=== 最终Prompt示例 ===")
    print(prompt)
