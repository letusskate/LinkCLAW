## 环境 lsa 能跑，最基础版本
export NCCL_IB_DISABLE=1
MASTER_ADDR=${MASTER_ADDR:-10.10.60.27}
MASTER_PORT=${MASTER_PORT:-29400}
torchrun --nnodes=2 --node_rank=0 --nproc_per_node=1 \
    --rdzv_backend=c10d --rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT} \
    titan_trainer.py --config_name=wybtest