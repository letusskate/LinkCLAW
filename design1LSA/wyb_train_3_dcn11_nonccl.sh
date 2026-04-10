# 尝试lsa2跑起来
# MASTER_ADDR=10.10.60.27 MASTER_PORT=29400 \
export NCCL_IB_DISABLE=1
export PYTORCH_ALLOC_CONF="expandable_segments:True,max_split_size_mb:128,garbage_collection_threshold:0.6"
MASTER_ADDR=${MASTER_ADDR:-10.10.60.27}
MASTER_PORT=${MASTER_PORT:-29400}
torchrun --nnodes=3 --node_rank=0 --nproc_per_node=1 \
    --master_addr=${MASTER_ADDR} --master_port=${MASTER_PORT} \
    titan_trainer_3gpu.py --config_name=wybtest --backend gloo
