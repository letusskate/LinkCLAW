import json
import random
import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
source_file = os.path.join(PROJECT_ROOT, "data", "raw", "nq", "nq-open-10_0.jsonl")
data_file = os.path.join(PROJECT_ROOT, "scripts", "evaluation", "dyx_tmp", "data.json")

def handle_data(first_title, process_data_num = -1):

    # 确保输出目录存在
    os.makedirs(os.path.dirname(data_file), exist_ok=True)

    # 检查文件是否存在
    if not os.path.exists(source_file):
        print(f"错误：源文件不存在 {source_file}")
        return

    # 读取并处理数据
    data_list = []
    with open(source_file, 'r', encoding='utf-8') as f:
        line_count = 0
        for line in f:
            line_count += 1
            try:
                if process_data_num > 0 and line_count > process_data_num:
                    break
                # 解析每一行JSON
                data = json.loads(line.strip())
                
                # 检查是否有documents字段
                if first_title in data:
                    print(f"处理第 {line_count} 条数据，包含 {len(data[first_title])} 个documents")
                    # 为每个document添加coming_time字段
                    for doc in data[first_title]:
                        # 生成1-5秒的随机值，保留2位小数
                        doc['coming_time'] = round(random.uniform(1, 5), 3)
                else:
                    print(f"第 {line_count} 条数据没有documents字段")
                
                # 添加到结果列表
                data_list.append(data)
            except Exception as e:
                print(f"处理第 {line_count} 条数据时出错: {e}")

    # 写入处理后的数据
    with open(data_file, 'w', encoding='utf-8') as f:
        json.dump(data_list, f, ensure_ascii=False, indent=2)

    print(f"\n数据处理完成，已写入到 {data_file}")
    print(f"共处理 {len(data_list)} 条数据")

def stream_model_inference_use_data():
    pass

if __name__ == "__main__":
    handle_data("ctxs", 5)