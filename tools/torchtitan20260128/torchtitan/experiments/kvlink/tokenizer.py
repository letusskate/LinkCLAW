# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any

from transformers import AutoTokenizer, PreTrainedTokenizerBase

from torchtitan.components.tokenizer import BaseTokenizer
from torchtitan.config import JobConfig


class KVLinkTokenizer(BaseTokenizer):
    def __init__(self, tokenizer: PreTrainedTokenizerBase):
        super().__init__()
        self.tokenizer = tokenizer

        self.bos_id = tokenizer.bos_token_id
        self.eos_id = tokenizer.eos_token_id
        self.pad_id = tokenizer.pad_token_id

    def __call__(self, *args: Any, **kwargs: Any):
        return self.tokenizer(*args, **kwargs)

    def encode(self, *args: Any, **kwargs: Any) -> list[int]:
        return self.tokenizer.encode(*args, **kwargs)

    def decode(self, *args: Any, **kwargs: Any) -> str:
        return self.tokenizer.decode(*args, **kwargs)

    def get_vocab_size(self) -> int:
        return self.tokenizer.vocab_size


def build_kvlink_tokenizer(job_config: JobConfig, **kwargs):
    del kwargs
    tokenizer_path = job_config.model.hf_assets_path
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path,
        use_fast=True,
        local_files_only=True,
    )

    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    return KVLinkTokenizer(tokenizer)


__all__ = ["KVLinkTokenizer", "build_kvlink_tokenizer"]