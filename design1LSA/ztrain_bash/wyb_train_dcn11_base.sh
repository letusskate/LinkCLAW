# 尝试lsa2跑起来 成功
# MASTER_ADDR=10.10.60.27 MASTER_PORT=29400 \
# MASTER_ADDR=10.10.60.27 MASTER_PORT=29400
export NCCL_IB_DISABLE=1
MASTER_ADDR=${MASTER_ADDR:-10.10.60.27}
MASTER_PORT=${MASTER_PORT:-29400}
torchrun --nnodes=2 --node_rank=0 --nproc_per_node=1 \
    --master_addr=${MASTER_ADDR} --master_port=${MASTER_PORT} \
    titan_trainer.py --config_name=wybtest --backend gloo
