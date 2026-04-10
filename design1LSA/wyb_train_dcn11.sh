# export NCCL_DEBUG=INFO
# export NCCL_IB_DISABLE=0
# export NCCL_SOCKET_IFNAME=ens7f0np0     # 例如: enp4s0 或 eth0，作为 TCP 回退与初始探测
# export NCCL_IB_HCA=mlx5_0,mlx5_1                 # 实际 HCA 名称，至少包含 mlx5_0
# export NCCL_IB_GID_INDEX=3                       # RoCE v2 常用；若 IB 原生链路可省略或设 0
# export NCCL_ASYNC_ERROR_HANDLING=1               # 异步错误更易诊断

export NCCL_MIN_NCHANNELS=1
export NCCL_MAX_NCHANNELS=1
export NCCL_IB_QPS_PER_CONNECTION=1
export NCCL_IB_TC=32

unset NCCL_SOCKET_IFNAME
unset NCCL_IB_HCA
export NCCL_IB_GID_INDEX=3
# export NCCL_IB_HCA=mlx5_0
export NCCL_IB_HCA=rocep202s0f0
unset HF_ENDPOINT

export MY_JOB_ID=1
export NCCL_IB_DISABLE=0

MASTER_ADDR=${MASTER_ADDR:-192.168.1.21}
# 192.168.1.21 192.168.1.31

MASTER_PORT=${MASTER_PORT:-29400}
# 12348 12346 12350

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,garbage_collection_threshold:0.6,max_split_size_mb:128
# export PYTORCH_NO_CUDA_MEMORY_CACHING=1

    # --rdzv_backend=c10d --rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT} \
torchrun --nnodes=2 --node_rank=0 --nproc_per_node=1 \
    --master_addr=${MASTER_ADDR} --master_port=${MASTER_PORT} \
    titan_trainer.py --config_name=wybtest \
    --enable_cpu_offload
