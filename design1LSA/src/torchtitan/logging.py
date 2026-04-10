# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os

## wyb add to record in /design1LSA/run_logs/data_original_step6k_bsz64_link_5_selective_ckpt
def _add_file_handler_if_configured() -> None:
    log_dir = os.environ.get("TORCHTITAN_LOG_DIR")
    if not log_dir:
        return

    os.makedirs(log_dir, exist_ok=True)
    rank = os.environ.get("RANK") or os.environ.get("LOCAL_RANK") or "0"
    log_path = os.path.join(log_dir, f"rank_{rank}.log")

    for handler in logger.handlers:
        if isinstance(handler, logging.FileHandler) and getattr(
            handler, "baseFilename", None
        ) == os.path.abspath(log_path):
            return

    fh = logging.FileHandler(log_path)
    fh.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    fh.setFormatter(formatter)
    logger.addHandler(fh)

logger = logging.getLogger()


def init_logger():
    logger.setLevel(logging.INFO)
    has_stream = any(isinstance(h, logging.StreamHandler) for h in logger.handlers)
    if not has_stream:
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        ch.setFormatter(formatter)
        logger.addHandler(ch)

    _add_file_handler_if_configured()

    # suppress verbose torch.profiler logging
    os.environ["KINETO_LOG_LEVEL"] = "5"


################ before wyb change ###########

# import logging
# import os

# logger = logging.getLogger()


# def init_logger():
#     logger.setLevel(logging.INFO)
#     ch = logging.StreamHandler()
#     ch.setLevel(logging.INFO)
#     formatter = logging.Formatter(
#         "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
#     )
#     ch.setFormatter(formatter)
#     logger.addHandler(ch)

#     # suppress verbose torch.profiler logging
#     os.environ["KINETO_LOG_LEVEL"] = "5"