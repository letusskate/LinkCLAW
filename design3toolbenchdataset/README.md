# 异步 ToolBench 数据集生成工具

本项目用于生成带异步时间戳的 ToolBench 数据集，适用于研究设备端智能体的时序执行问题。

当前生成器的原则是：

- 保留 tool_calls 的 logical order。
- 只让无依赖的 root tools 在 planning 结束后同时发起。
- 依赖前序结果的 tools 会在依赖完成后再发起，而不是被错误地和 root tools 同时启动。
- 每个工具的返回时间独立采样，因此可以实现“同时调用，不同时返回”。
- 默认过滤空 final_answer 和并行度不足的样本，避免评测集本身失真。

当前支持三种依赖策略：

- `balanced_dag` - 默认启发式依赖抽取，兼顾语义保真和并发度。
- `llm_dag` - 使用本地小模型逐条判断工具依赖，并支持 JSONL 缓存和断点续跑。
- `all_parallel` - 忽略工具间依赖，将所有工具都视为并行 root calls。

## 快速开始

### 1. 下载 ToolBench 数据集

**重要说明**：根据 ToolBench 官方 GitHub，数据集可通过以下方式获取：

#### 方法一：手动下载（推荐）

由于 Google Drive 可能存在地区限制或需要登录，建议手动下载：

1. 访问官方链接：https://drive.google.com/file/d/1XFjDxVZdUY7TXYF2yvzx3pJlS2fy78jk/view
2. 点击下载按钮，保存为 `data.zip`
3. 将文件放到 `./data/` 目录
4. 运行解压：

```bash
cd toolbenchdataset
python download_toolbench.py --output_dir ./data --skip_download
```

#### 方法二：使用 gdown 工具（Python）

```bash
pip install gdown
cd toolbenchdataset
gdown https://drive.google.com/uc?id=1XFjDxVZdUY7TXYF2yvzx3pJlS2fy78jk -O ./data/data.zip
python download_toolbench.py --output_dir ./data --skip_download
```

#### 方法三：使用 wget（Linux/Mac）

```bash
cd toolbenchdataset
wget --no-check-certificate 'https://drive.google.com/uc?export=download&id=1XFjDxVZdUY7TXYF2yvzx3pJlS2fy78jk&confirm=yes' -O ./data/data.zip
python download_toolbench.py --output_dir ./data --skip_download
```

#### 方法四：尝试自动下载（不推荐，可能失败）

```bash
python download_toolbench.py --source manual
```

### 2. 生成异步数据集

下载并解压数据后，运行：

```bash
python generate_async_from_toolbench.py \
	--input ./data/data/toolllama_G123_dfs_train.json \
	--output ./async_agent_bench/async_agent_bench.jsonl \
	--dependency_policy balanced_dag \
	--num_samples 100 \
	--min_parallel_roots 2
```

生成全量 Balanced-DAG 数据集：

```bash
python generate_async_from_toolbench.py \
	--input ./data/data/toolllama_G123_dfs_train.json \
	--output ./async_agent_bench/async_agent_bench_full_balanced.jsonl \
	--dependency_policy balanced_dag \
	--latency_profile balanced \
	--num_samples 0 \
	--min_parallel_roots 2
```

生成全量 All-Parallel 数据集：

```bash
python generate_async_from_toolbench.py \
	--input ./data/data/toolllama_G123_dfs_train.json \
	--output ./async_agent_bench/async_agent_bench_full_all_parallel.jsonl \
	--dependency_policy all_parallel \
	--latency_profile balanced \
	--num_samples 0 \
	--min_parallel_roots 2
```

生成全量 LLM-DAG 数据集：

```bash
python generate_async_from_toolbench.py \
	--input ./data/data/toolllama_G123_dfs_train.json \
	--output ./async_agent_bench/async_agent_bench_full_llm_dag.jsonl \
	--dependency_policy llm_dag \
	--llm_model_path /home/dell/wyb/my_models/Qwen/Qwen/Qwen2___5-0___5B-Instruct \
	--llm_cache_path ./async_agent_bench/dependency_cache/toolbench_qwen05b_llm_dag.jsonl \
	--latency_profile balanced \
	--num_samples 0 \
	--min_parallel_roots 2 \
	--resume
```

`llm_dag` 默认使用保守提示词。如果模型输出不是合法 JSON，则回退到 `balanced_dag` 并把结果写入缓存，便于后续续跑。

## 文件说明

- `generate_async_from_toolbench.py` - 兼容入口，转发到新主脚本
- `generate_async_from_toolbench_new.py` - 支持 dependency policy variants 的主生成器
- `README.md` - 本文件

## 数据结构

生成的异步数据集样本包含：

- `user_request` - 用户请求
- `user_request_time` - 请求时间戳
- `planning_segments` - 规划阶段
- `tool_calls` - 异步工具调用（含开始/结束时间）
- `tool_calls[].logical_index` - 工具在原始计划中的逻辑顺序
- `tool_calls[].dependencies` - 当前工具依赖的前序 call_id 列表
- `final_answer` - 最终答案
- `final_answer_time` - 答案时间戳
- `event_list` - 机器可读事件列表
- `linearized_transcript` - 线性化转录文本

## 测试

使用模拟数据快速测试：

```bash
python generate_async_from_toolbench.py --num_samples 5 --min_parallel_roots 2
```

## 依赖

- Python 3.8+
- requests
- torch
- transformers
