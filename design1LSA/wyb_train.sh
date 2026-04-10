# bash ./wyb_train.sh
LOG_RANK=${LOG_RANK:-0}
# NGPU=${NGPU:-"8"}
# Set number of processes (set NGPU=1 for single-GPU)
NGPU=${NGPU:-1}

# Optionally restrict visible CUDA device (default to GPU 0)
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True" \

# # ensure dataset alias: if `qa_mem` missing but `qamem` exists, create symlink
# BLOCK_QA_DIR="dataset_cache/processed/block_qa"
# if [ -d "$BLOCK_QA_DIR" ] && [ ! -e "$BLOCK_QA_DIR/qa_mem" ] && [ -d "$BLOCK_QA_DIR/qamem" ]; then
#     echo "Creating symlink $BLOCK_QA_DIR/qa_mem -> qamem"
#     ln -s qamem "$BLOCK_QA_DIR/qa_mem"
# fi
# torchrun --nproc_per_node=8 --rdzv_backend c10d --rdzv_endpoint="localhost:0" \
torchrun --nproc_per_node=${NGPU} --rdzv_backend c10d --rdzv_endpoint="localhost:0" \
    --local-ranks-filter ${LOG_RANK} --role rank --tee 3 \
    titan_trainer_singleGPU.py --config_name wybtest \
    --enable_cpu_offload