# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from torchtitan.components.loss import build_cross_entropy_loss
from torchtitan.components.lr_scheduler import build_lr_schedulers
from torchtitan.components.optimizer import build_optimizers
from torchtitan.components.validate import build_validator
from torchtitan.distributed.pipeline_parallel import pipeline_llm
from torchtitan.protocols.train_spec import TrainSpec
from torchtitan.models.llama3.model.state_dict_adapter import Llama3StateDictAdapter

from .data import build_kvlink_dataloader
from .model import KvlinkTransformer, kvlink_llama3_2_args, parallelize_llama
from .tokenizer import build_kvlink_tokenizer


def get_train_spec() -> TrainSpec:
    return TrainSpec(
        model_cls=KvlinkTransformer,
        model_args=kvlink_llama3_2_args,
        parallelize_fn=parallelize_llama,
        pipelining_fn=pipeline_llm,
        build_optimizers_fn=build_optimizers,
        build_lr_schedulers_fn=build_lr_schedulers,
        build_dataloader_fn=build_kvlink_dataloader,
        build_tokenizer_fn=build_kvlink_tokenizer,
        build_loss_fn=build_cross_entropy_loss,
        build_validator_fn=build_validator,
        state_dict_adapter=Llama3StateDictAdapter,
    )