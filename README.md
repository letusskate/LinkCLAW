# design1LSA, LAP, and AsyncBench

## 1. Repository Overview

This release contains three connected components for long-context and asynchronous-agent research:

- **design1LSA**: training and evaluation code for **LSA (link sparse attention)** style models, implemented on top of TorchTitan/TorchTune-style infrastructure and packaged under `design1LSA` in this workspace.
- **design2LAP**: the implementation of **LAP (link-guided asynchronous prefill)**, including synchronous and asynchronous scheduling baselines, overlap-aware timing analysis, and canonical replay evaluation.
- **design3toolbenchdataset**: **AsyncBench**, a timestamped asynchronous agent dataset derived from ToolBench, with `Balanced-DAG`, `LLM-DAG`, and `All-Parallel` dependency-policy variants.

The intended workflow is:

1. Train or load a design1LSA checkpoint.
2. Generate AsyncBench from ToolBench trajectories.
3. Evaluate standard sync, chunked async, Link-Last, and LinkCLAW schedules on AsyncBench.

The three folders are designed to be usable independently, but they are most useful together: design1LSA provides the sparse-attention model, design2LAP provides the asynchronous prefill runtime and benchmark, and design3toolbenchdataset provides the asynchronous agent traces used by the benchmark.

## 2. File Tree

The tree below lists every top-level file in the three released folders and summarizes the main nested source and artifact directories.

```text
point1/
├── README.md                                              # Top-level guide for the three released components
├── design1LSA/                                            # LSA / link sparse attention training and evaluation code
│       ├── .gitignore                                     # Git ignore rules for training artifacts and caches
│       ├── README.md                                      # Folder-specific training/evaluation notes
│       ├── pyproject.toml                                 # Packaging and formatter/linter configuration
│       ├── requirements.txt                               # Python package dependencies for design1LSA
│       ├── setup.py                                       # Editable install entry
│       ├── lsa.yml                                        # Environment/config snapshot for LSA runs
│       ├── lsa2.yml                                       # Alternate environment/config snapshot
│       ├── wyb20260225.yml                                # Additional environment/config snapshot
│       ├── titan_trainer.py                               # Main multi-GPU trainer entry
│       ├── titan_trainer_lsa.py                           # Trainer variant focused on LSA / link-attention setups
│       ├── titan_trainer_singleGPU.py                     # Single-GPU training entry
│       ├── titan_trainer_3gpu.py                          # Three-GPU launcher entry
│       ├── titan_trainer_4gpu.py                          # Four-GPU launcher entry
│       ├── titan_trainer_a40.py                           # A40-oriented training entry
│       ├── wyb_test.py                                    # Local test / debugging script
│       ├── wyb_train.sh                                   # Generic shell launcher
│       ├── wyb_train_dcn11.sh                             # Host-specific launcher for dcn11
│       ├── wyb_train_dcn12.sh                             # Host-specific launcher for dcn12
│       ├── wyb_train_2_dcn11_nonccl.sh                    # Two-node non-NCCL launcher for dcn11
│       ├── wyb_train_2_dcn12_nonccl.sh                    # Two-node non-NCCL launcher for dcn12
│       ├── wyb_train_3_dcn9_nonccl.sh                     # Three-node non-NCCL launcher for dcn9
│       ├── wyb_train_3_dcn11_nonccl.sh                    # Three-node non-NCCL launcher for dcn11
│       ├── wyb_train_3_dcn12_nonccl.sh                    # Three-node non-NCCL launcher for dcn12
│       ├── wyb_train_4_dcn9_nonccl.sh                     # Four-node non-NCCL launcher for dcn9
│       ├── wyb_train_4_dcn10_nonccl.sh                    # Four-node non-NCCL launcher for dcn10
│       ├── wyb_train_4_dcn11_nonccl.sh                    # Four-node non-NCCL launcher for dcn11
│       ├── wyb_train_4_dcn12_nonccl.sh                    # Four-node non-NCCL launcher for dcn12
│       ├── ztrain_bash/                                   # Additional cluster launch scripts grouped by machine/setup
│       ├── src/                                           # Core library code
│       │   ├── __init__.py                                # Package marker
│       │   ├── common.py                                  # Shared runtime helpers
│       │   ├── data/                                      # Data preprocessing, tokenizer download, masks, datasets
│       │   ├── model/                                     # Model definitions and sparse-attention modules
│       │   ├── torchtitan/                                # TorchTitan-related integrations
│       │   ├── torchtune_models/                          # TorchTune model wrappers
│       │   ├── training/                                  # Training configs and utilities
│       │   └── utils/                                     # Miscellaneous shared helpers
│       ├── scripts/                                       # Standalone data and evaluation utilities
│       │   ├── data_process/                              # Preprocessing scripts for QA and language data
│       │   └── evaluation/                                # Evaluation scripts for HQA, NQ, and related tasks
│       ├── data/                                          # Raw data placeholders and source dataset checkouts
│       │   └── raw/                                       # Raw datasets such as block_qa and nq
│       ├── dataset_cache/                                 # Processed dataset cache and upload helpers
│       ├── result/                                        # Evaluation outputs and experiment result JSONLs
│       ├── run_logs/                                      # Training checkpoints, TensorBoard logs, and experiment dumps
│       └── design1lsa.egg-info/                           # Packaging metadata generated by installation
├── tools/
│   ├── sparse/                                            # External sparse-attention dependencies used by design1LSA
│   └── torchtitan20260128/                                # TorchTitan experiment code and configs
├── design2LAP/                                            # LAP benchmark and asynchronous prefill schedulers
│   ├── lap_benchmark.py                                   # Core benchmark, schedulers, metrics, and evaluation loop
│   ├── jsonl_loader.py                                    # AsyncBench JSONL dataset loader
│   ├── run_all_tests.sh                                   # Convenience runner for the main benchmark variants
│   ├── test_1_standard_sync.py                            # Wait-for-all baseline: prefill only after every tool returns
│   ├── test_3_simple_async.py                             # Chunk baseline: prefill the longest available logical prefix
│   ├── test_4_lap_async.py                                # Block-then-links schedule (Link-Last style)
│   ├── test_5_advanced_lap.py                             # Incremental block-plus-link-chain schedule
│   ├── test_6_link_last_canonical.py                      # Link-Last with canonical block-link-block-link replay scoring
│   ├── test_7_linkclaw_canonical.py                       # LinkCLAW with canonical replay scoring
│   ├── doc/                                               # Notes, derivations, and benchmark write-ups
│   ├── result/                                            # Benchmark outputs and stats JSON files
│   ├── old_result/                                        # Archived older benchmark outputs
│   ├── old_script/                                        # Archived earlier evaluation scripts
│   └── __pycache__/                                       # Python bytecode cache
└── design3toolbenchdataset/                               # AsyncBench construction from ToolBench trajectories
    ├── README.md                                          # Folder-specific dataset-generation guide
    ├── requirements.txt                                   # Dataset-generation dependencies
    ├── generate_async_from_toolbench.py                   # Compatibility launcher for the current generator
    ├── generate_async_from_toolbench_new.py               # Main AsyncBench generator with dependency-policy variants
    ├── async_agent_bench/                                 # Generated AsyncBench JSONL files and variant splits
    ├── data/                                              # ToolBench source data and unpacked API trajectories
    ├── doc/                                               # Paper text, policy tables, and supplementary documentation
    ├── old_bench/                                         # Archived earlier dataset drafts and older JSONL versions
    ├── utils/                                             # Download, inspection, and parser test helpers
    └── __pycache__/                                       # Python bytecode cache
```

## 3. Quick Start

### 3.1 Environment Requirements

- Linux is the primary tested platform.
- Python 3.11 in a Conda environment is recommended.
- CUDA-capable GPUs are strongly recommended for design1LSA training and for `LLM-DAG` dataset generation.
- A 24 GB-class GPU is a practical minimum for the small local dependency-parser model used in `LLM-DAG`.
- `Balanced-DAG` and `All-Parallel` AsyncBench generation can run on CPU, but GPU is still recommended for speed.

### 3.2 Package Dependencies

Create and activate an environment:

```bash
conda create -n lsa2 python=3.11 -y
conda activate lsa2
pip install --upgrade pip
```

Install PyTorch and the core Python stack. Replace the CUDA index URL if your machine uses a different CUDA version.

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install torchtune
pip install -r design1LSA/requirements.txt
pip install -r design3toolbenchdataset/requirements.txt
pip install numpy regex safetensors tqdm sentencepiece
pip install -e design1LSA
```

If your TorchTitan branch requires a nightly PyTorch build, replace the first line with a nightly install command before installing the remaining packages.

### 3.3 Quick Start: Train design1LSA

Download the tokenizer and base model first if you do not already have them locally:

```bash
cd design1LSA

python src/data/titan_download_tokenizer.py \
  --repo_id meta-llama/Llama-3.2-1B-Instruct \
  --tokenizer_path original \
  --local_dir data/titan_tokenizer/ \
  --hf_token <HF_TOKEN>

tune download meta-llama/Llama-3.2-1B-Instruct \
  --output-dir model_cache/Llama-3.2-1B-Instruct \
  --ignore-patterns "original/consolidated.00.pth" \
  --hf-token <HF_TOKEN>
```

Run a training job:

```bash
cd design1LSA

LOG_RANK=${LOG_RANK:-0}
NGPU=${NGPU:-8}

PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True" \
torchrun --nproc_per_node=${NGPU} --rdzv_backend c10d --rdzv_endpoint="localhost:0" \
  --local-ranks-filter ${LOG_RANK} --role rank --tee 3 \
  titan_trainer.py --config_name datav6_step6k_bsz64_reencode_5_selective_ckpt
```

Run evaluation on a converted checkpoint:

```bash
cd design1LSA

python scripts/evaluation/hqa_eval.py \
  --ckpt_path <CHECKPOINT.pt> \
  --batch_size 10 \
  --reencode_num 5 \
  --attn_type blocked
```

### 3.4 Quick Start: Generate AsyncBench

Balanced-DAG generation:

```bash
cd design3toolbenchdataset

python generate_async_from_toolbench.py \
  --input ./data/data/toolllama_G123_dfs_train.json \
  --output ./async_agent_bench/async_agent_bench_full_balanced.jsonl \
  --dependency_policy balanced_dag \
  --latency_profile balanced \
  --num_samples 0 \
  --min_parallel_roots 2
```

LLM-DAG generation with a small local instruct model:

```bash
cd design3toolbenchdataset

python generate_async_from_toolbench.py \
  --input ./data/data/toolllama_G123_dfs_train.json \
  --output ./async_agent_bench/async_agent_bench_full_llm_dag.jsonl \
  --dependency_policy llm_dag \
  --llm_model_path <LOCAL_SMALL_INSTRUCT_MODEL_DIR> \
  --llm_cache_path ./async_agent_bench/dependency_cache/llm_dag_cache.jsonl \
  --latency_profile balanced \
  --num_samples 0 \
  --min_parallel_roots 2 \
  --resume
```

All-Parallel generation:

```bash
cd design3toolbenchdataset

python generate_async_from_toolbench.py \
  --input ./data/data/toolllama_G123_dfs_train.json \
  --output ./async_agent_bench/async_agent_bench_full_all_parallel.jsonl \
  --dependency_policy all_parallel \
  --latency_profile balanced \
  --num_samples 0 \
  --min_parallel_roots 2
```

### 3.5 Quick Start: Run LAP Benchmarks

Evaluate a LinkCLAW checkpoint on AsyncBench:

```bash
cd design2LAP

python test_7_linkclaw_canonical.py \
  --dataset_path ../design3toolbenchdataset/async_agent_bench/async_agent_bench_full_balanced_eval100_seed42.jsonl \
  --num_samples 100 \
  --attn_type blocked \
  --model_path ../design1LSA/run_logs/<RUN_NAME>/checkpoint.pt \
  --primary_metric decomposed_pass \
  --output_dir result/linkclaw_demo
```

Run all major scheduling baselines on the same dataset:

```bash
cd design2LAP

bash run_all_tests.sh ../design3toolbenchdataset/async_agent_bench/async_agent_bench_full_balanced_eval100_seed42.jsonl
```

### 3.6 Notes

- Install `design1LSA` in editable mode before running `design2LAP`, because the LAP benchmark imports its preprocessing and attention-mask code.
- `LLM-DAG` generation supports JSONL caching and resume mode; interrupted runs can be restarted safely.
- The `async_agent_bench/`, `result/`, `run_logs/`, and `dataset_cache/` directories are artifact-heavy by design. They are useful for reproducibility but are not required to understand the core implementations.