# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch.nn.attention.flex_attention import and_masks

from torchtitan.models.attention import create_attention_mask, get_causal_mask_mod
from torchtitan.models.llama3.infra.parallelize import parallelize_llama
from torchtitan.models.llama3.model.args import RoPEScalingArgs, TransformerModelArgs
from torchtitan.models.llama3.model.model import Transformer
from torchtitan.protocols.model import AttentionMasksType


kvlink_llama3_2_args = {
    "llama3_2_1b": TransformerModelArgs(
        dim=2048,
        n_layers=16,
        n_heads=32,
        n_kv_heads=8,
        vocab_size=128256,
        multiple_of=256,
        ffn_dim_multiplier=None,
        norm_eps=1e-5,
        rope_theta=500000,
        rope_scaling_args=RoPEScalingArgs(
            scaling_factor=32.0,
            low_freq_factor=1.0,
            high_freq_factor=4.0,
            original_max_position_embeddings=8192,
        ),
        attn_type="flex",
        attn_mask_type="block_causal",
    ),
}


class KvlinkTransformer(Transformer):
    """Transformer with KVLink segment-aware attention masks (flex attention)."""

    def get_attention_masks(
        self,
        input_batch: torch.Tensor,
        tokenizer,
        extra_inputs: dict[str, torch.Tensor] | None = None,
    ) -> AttentionMasksType:
        if not extra_inputs or "segment_ids" not in extra_inputs:
            return super().get_attention_masks(input_batch, tokenizer, extra_inputs)

        segment_ids = extra_inputs["segment_ids"].to(device=input_batch.device)
        batch_size, seq_len = input_batch.shape

        def segment_mask_mod(
            b: torch.Tensor,
            h: torch.Tensor,
            q_idx: torch.Tensor,
            kv_idx: torch.Tensor,
        ) -> torch.Tensor:
            seg_q = segment_ids[b, q_idx]
            seg_kv = segment_ids[b, kv_idx]
            valid = (seg_q != -1) & (seg_kv != -1)
            allow = (seg_q == 0) | (seg_q == seg_kv)
            return valid & allow

        mask_mods = [get_causal_mask_mod(), segment_mask_mod]
        return create_attention_mask(
            and_masks(*mask_mods), batch_size, None, seq_len, seq_len
        )


__all__ = [
    "kvlink_llama3_2_args",
    "KvlinkTransformer",
    "parallelize_llama",
]