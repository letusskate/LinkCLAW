# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass, field

from torchtitan.config import JobConfig


@dataclass
class KVLinkConfig:
    dataset_version: str = "original"
    enable_packing: bool = False

    # KVLink preprocessor parameters
    reencode_num: int = 5
    max_memory_num: int = 40
    special_token_start: int = 128011
    mem_start: int = 128254
    mem_end: int = 128255
    qa_document_num: int = 10
    min_memory_num: int = 5
    min_mem_length: int = 500
    max_mem_length: int = 1500


@dataclass
class JobConfigKVLink(JobConfig):
    kvlink: KVLinkConfig = field(default_factory=KVLinkConfig)


JobConfig = JobConfigKVLink