# 尝试走网卡传输 暂时失败
# Force IPv4 + TCP store; set explicit interface to avoid hostname resolution issues.
export MASTER_ADDR=${MASTER_ADDR:-10.10.60.27}
export MASTER_PORT=${MASTER_PORT:-29400}
export NCCL_IB_DISABLE=${NCCL_IB_DISABLE:-1}
# export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-ens7f0np0}
# export GLOO_SOCKET_IFNAME=${GLOO_SOCKET_IFNAME:-eno8303}
export NCCL_SOCKET_FAMILY=AF_INET

torchrun --nnodes=2 --node_rank=1 --nproc_per_node=1 \
    --rdzv_backend=c10d --rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT} \
    --master_addr=${MASTER_ADDR} --master_port=${MASTER_PORT} \
    titan_trainer.py --config_name=wybtest