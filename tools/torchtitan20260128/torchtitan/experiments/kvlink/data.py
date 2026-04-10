# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pickle
import random
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Tuple

import datasets
import torch
from datasets.distributed import split_dataset_by_node
from torch.distributed.checkpoint.stateful import Stateful
from torch.utils.data import IterableDataset

from torchtitan.components.dataloader import ParallelAwareDataloader
from torchtitan.config import JobConfig
from torchtitan.tools.logging import logger

from .tokenizer import KVLinkTokenizer


@dataclass
class DataComponent:
    dataset_name: str
    weight: float


DATA_ORIGINAL: List[DataComponent] = [
    DataComponent(dataset_name="text", weight=20.0),
    DataComponent(dataset_name="tulu", weight=30.0),
    DataComponent(dataset_name="sft_mem", weight=25.0),
    DataComponent(dataset_name="qa", weight=10.0),
    DataComponent(dataset_name="qa_mem", weight=10.0),
    DataComponent(dataset_name="xsum", weight=5.0),
]

DATA_NOSUM: List[DataComponent] = [
    DataComponent(dataset_name="text", weight=21.1),
    DataComponent(dataset_name="tulu", weight=31.6),
    DataComponent(dataset_name="sft_mem", weight=26.3),
    DataComponent(dataset_name="qa", weight=10.5),
    DataComponent(dataset_name="qa_mem", weight=10.5),
]

DATA_NOSFTMEM: List[DataComponent] = [
    DataComponent(dataset_name="text", weight=26.7),
    DataComponent(dataset_name="tulu", weight=40.0),
    DataComponent(dataset_name="qa", weight=13.3),
    DataComponent(dataset_name="qa_mem", weight=13.3),
    DataComponent(dataset_name="xsum", weight=6.7),
]

DATA_QAONLY: List[DataComponent] = [
    DataComponent(dataset_name="qa_mem", weight=100.0),
]

DATASET_MAPPING: dict[str, List[DataComponent]] = {
    "original": DATA_ORIGINAL,
    "nosum": DATA_NOSUM,
    "nosftmem": DATA_NOSFTMEM,
    "qaonly": DATA_QAONLY,
}


def _get_dataset_root(dataset_path: str | None) -> str:
    return dataset_path or "./dataset_cache/processed"


def load_data_and_process_fn(
    *,
    dataset_root: str,
    data_component_name: str,
    preprocessor,
    training: bool = True,
) -> Tuple[datasets.IterableDataset, datasets.Dataset, List[str]]:
    remove_columns: List[str] = []
    if data_component_name in ["text", "text_mem", "text_inst"]:
        data_path = f"{dataset_root}/fineweb/{data_component_name}"
        if data_component_name == "text":
            preprocessor_fn = preprocessor.process_text
        elif data_component_name == "text_mem":
            preprocessor_fn = preprocessor.process_textmem
        elif data_component_name == "text_inst":
            preprocessor_fn = preprocessor.process_textinst
        else:
            raise NotImplementedError()
        remove_columns = [
            "text",
            "id",
            "dump",
            "url",
            "date",
            "file_path",
            "language",
            "language_score",
            "token_count",
        ]
        if data_component_name in ["text_mem", "text_inst"]:
            remove_columns.append("num_tokens")
        num_shards = 512
    elif data_component_name in ["sft", "sft_mem"]:
        data_path = f"{dataset_root}/daringanteater/{data_component_name}"
        if data_component_name == "sft":
            preprocessor_fn = preprocessor.process_sft
        elif data_component_name == "sft_mem":
            preprocessor_fn = preprocessor.process_sftmem
        else:
            raise NotImplementedError()
        remove_columns = ["system", "mask", "dataset", "conversations"]
        num_shards = 32
    elif data_component_name in ["tulu"]:
        data_path = f"{dataset_root}/tulu/sft"
        preprocessor_fn = preprocessor.process_tulu
        remove_columns = ["id", "messages", "source"]
        num_shards = 32
    elif data_component_name in ["qa", "qa_mem"]:
        data_path = f"{dataset_root}/block_qa/{data_component_name}"
        if data_component_name == "qa":
            preprocessor_fn = preprocessor.process_qa
        elif data_component_name == "qa_mem":
            preprocessor_fn = preprocessor.process_qamem
        else:
            raise NotImplementedError()
        remove_columns = [
            "prompt",
            "question",
            "answers",
            "generated",
            "inputs",
            "documents",
        ]
        num_shards = 32
    elif data_component_name in ["xsum"]:
        data_path = f"{dataset_root}/xsum/{data_component_name}"
        preprocessor_fn = preprocessor.process_xsum
        remove_columns = ["document", "summary", "id"]
        num_shards = 32
    else:
        raise ValueError(f"Unrecognized dataset name {data_component_name}.")

    data_component: datasets.DatasetDict = datasets.load_from_disk(data_path)
    if training:
        ds = data_component["train"].to_iterable_dataset(num_shards=num_shards)
    else:
        ds = data_component["test"]

    filtered_remove_columns = [col for col in remove_columns if col in ds.column_names]
    if len(remove_columns) != len(filtered_remove_columns):
        logger.warning(
            "For dataset %s, some columns to remove are not found. Requested: %s, found: %s.",
            data_component_name,
            remove_columns,
            ds.column_names,
        )

    return ds, preprocessor_fn, filtered_remove_columns


class HuggingFaceDataset(IterableDataset, Stateful):
    def __init__(
        self,
        *,
        dataset_name: str,
        tokenizer: KVLinkTokenizer,
        preprocessor,
        seq_len: int,
        world_size: int,
        rank: int,
        infinite: bool = False,
        packing_mode: str = "padding",
        dataset_root: str,
    ) -> None:
        self.packing_mode = packing_mode
        assert self.packing_mode in ["padding", "packing"]

        ds, preprocess_fn, columns_to_remove = load_data_and_process_fn(
            dataset_root=dataset_root,
            data_component_name=dataset_name,
            preprocessor=preprocessor,
            training=True,
        )

        self.dataset_name = dataset_name
        self._data = split_dataset_by_node(ds, rank, world_size)
        self._tokenizer = tokenizer
        self.seq_len = seq_len
        self.infinite = infinite
        self.preprocess_fn = preprocess_fn
        self.columns_to_remove = columns_to_remove

        # Variables for checkpointing
        self._sample_idx = 0
        self._all_tokens: List[int] = []
        self._all_labels: List[int] = []

    def _get_data_iter(self):
        if self._sample_idx == 0:
            return iter(self._data)

        if isinstance(self._data, datasets.Dataset) and self._sample_idx == len(
            self._data
        ):
            return iter([])

        return iter(self._data.skip(self._sample_idx))

    def __iter__(self):
        max_buffer_token_len = self.seq_len

        while True:
            for sample in self._get_data_iter():
                processed_example = self.preprocess_fn(sample)
                if self.packing_mode == "padding":
                    yield processed_example
                    self._sample_idx += 1
                else:
                    input_ids = processed_example["input_ids"]
                    labels = processed_example["labels"]
                    self._all_tokens.extend(input_ids)
                    self._all_labels.extend(labels)
                    self._sample_idx += 1

                    while len(self._all_tokens) >= max_buffer_token_len:
                        packed_tokens = self._all_tokens[:max_buffer_token_len]
                        packed_labels = self._all_labels[:max_buffer_token_len]
                        self._all_tokens = self._all_tokens[max_buffer_token_len:]
                        self._all_labels = self._all_labels[max_buffer_token_len:]
                        packed_example = {
                            "input_ids": packed_tokens,
                            "labels": packed_labels,
                            "biased_index": None,
                        }
                        yield packed_example

            if not self.infinite:
                logger.warning("Dataset %s has run out of data", self.dataset_name)
                break
            else:
                self._sample_idx = 0
                logger.warning("Dataset %s is being re-looped", self.dataset_name)

    def load_state_dict(self, state_dict):
        self._sample_idx = state_dict["sample_idx"]
        if self.packing_mode == "packing":
            self._all_tokens = state_dict["token_buffer"]
            self._all_labels = state_dict["label_buffer"]

    def state_dict(self):
        return {
            "token_buffer": self._all_tokens,
            "label_buffer": self._all_labels,
            "sample_idx": self._sample_idx,
        }


class WeightedAggregatorDataset(IterableDataset, Stateful):
    def __init__(
        self,
        datasets: List[IterableDataset],
        weights: List[float],
        seed: int = 42,
        infinite: bool = True,
    ):
        super().__init__()
        assert len(datasets) == len(weights), "datasets and weights must be the same length"
        self.datasets = datasets
        self.weights = weights
        self.infinite = infinite

        total = sum(weights)
        self.probs = [w / total for w in weights]
        self.iters = [None] * len(datasets)

        self._rng_state = None
        self._random_gen = random.Random(seed)

    def state_dict(self):
        state = {
            "rng_state": self._rng_state,
            "children": {},
        }
        for i, ds in enumerate(self.datasets):
            state["children"][i] = ds.state_dict()
        return state

    def load_state_dict(self, state_dict):
        if not state_dict:
            return

        self._rng_state = state_dict["rng_state"]
        if self._rng_state is not None:
            self._random_gen.setstate(self._rng_state)

        for i, ds in enumerate(self.datasets):
            ds.load_state_dict(state_dict["children"][i])

    def __iter__(self):
        self.iters = [iter(ds) for ds in self.datasets]

        while True:
            choice = self._random_gen.choices(
                range(len(self.datasets)), weights=self.probs, k=1
            )[0]

            try:
                sample = next(self.iters[choice])
                yield sample
            except StopIteration:
                if not self.infinite:
                    break
                else:
                    self.iters[choice] = iter(self.datasets[choice])
                    continue

    def __next__(self):
        return next(iter(self))

    def _save_rng_state(self):
        self._rng_state = self._random_gen.getstate()


class BlockAttnCollator:
    def __init__(self, pad_token_idx: int):
        self.pad_token_idx = pad_token_idx

    def __call__(self, features: List[Dict[str, Any]]):
        input_ids = []
        labels = []
        segment_ids = []
        length_list = [len(x["input_ids"]) for x in features]
        max_length = max(length_list)

        for idx in range(len(features)):
            curr_input = features[idx]["input_ids"]
            curr_labels = features[idx]["labels"]
            curr_segments = features[idx].get("segment_ids", [0] * len(curr_input))
            seq_length = len(curr_input)

            if len(curr_labels) < seq_length:
                curr_labels = curr_labels + [-100] * (seq_length - len(curr_labels))
            elif len(curr_labels) > seq_length:
                curr_labels = curr_labels[:seq_length]

            if len(curr_segments) < seq_length:
                curr_segments = curr_segments + [-1] * (seq_length - len(curr_segments))
            elif len(curr_segments) > seq_length:
                curr_segments = curr_segments[:seq_length]

            residual = max_length - seq_length
            input_ids.append(curr_input + [self.pad_token_idx] * residual)
            labels.append(curr_labels + [-100] * residual)
            segment_ids.append(curr_segments + [-1] * residual)

        return {
            "input": torch.LongTensor(input_ids),
            "segment_ids": torch.LongTensor(segment_ids),
        }, torch.LongTensor(labels)


def build_kvlink_dataloader(
    *,
    dp_world_size: int,
    dp_rank: int,
    tokenizer: KVLinkTokenizer,
    job_config: JobConfig,
    infinite: bool = True,
):
    kv_cfg = getattr(job_config, "kvlink", None)
    if kv_cfg is None:
        raise ValueError("KVLink config is missing. Set job.custom_config_module.")

    data_components = DATASET_MAPPING[kv_cfg.dataset_version]
    dataset_root = _get_dataset_root(job_config.training.dataset_path)

    preprocessor = SumAttentionPreprocessor(
        tokenizer=tokenizer,
        max_len=job_config.training.seq_len,
        special_token_start=kv_cfg.special_token_start,
        mem_start=kv_cfg.mem_start,
        mem_end=kv_cfg.mem_end,
        reencode_num=kv_cfg.reencode_num,
        min_memory_num=kv_cfg.min_memory_num,
        max_memory_num=kv_cfg.max_memory_num,
        qa_document_num=kv_cfg.qa_document_num,
        min_mem_length=kv_cfg.min_mem_length,
        max_mem_length=kv_cfg.max_mem_length,
    )

    pad_token_idx = tokenizer.pad_id if tokenizer.pad_id is not None else tokenizer.eos_id
    collator = BlockAttnCollator(pad_token_idx=pad_token_idx)

    all_datasets = []
    dataset_weights = []
    for data_component in data_components:
        dataset_name = data_component.dataset_name
        weight = data_component.weight
        if kv_cfg.enable_packing:
            packing_mode = (
                "packing" if dataset_name in ["sft", "tulu", "text"] else "padding"
            )
        else:
            packing_mode = "padding"

        hf_ds = HuggingFaceDataset(
            dataset_name=dataset_name,
            tokenizer=tokenizer,
            preprocessor=preprocessor,
            seq_len=job_config.training.seq_len,
            world_size=dp_world_size,
            rank=dp_rank,
            infinite=infinite,
            packing_mode=packing_mode,
            dataset_root=dataset_root,
        )
        all_datasets.append(hf_ds)
        dataset_weights.append(weight)

    seed = job_config.debug.seed if job_config.debug.seed is not None else 42
    combined_ds = WeightedAggregatorDataset(
        all_datasets,
        dataset_weights,
        seed=seed,
        infinite=infinite,
    )

    dataloader_kwargs = {
        **asdict(job_config.training.dataloader),
        "batch_size": job_config.training.local_batch_size,
    }

    return ParallelAwareDataloader(
        combined_ds,
        dp_rank=dp_rank,
        dp_world_size=dp_world_size,
        collate_fn=collator,
        **dataloader_kwargs,
    )


class SumAttentionPreprocessor:
    def __init__(
        self,
        tokenizer: KVLinkTokenizer,
        max_len: int,
        special_token_start: int,
        mem_start: int,
        mem_end: int,
        reencode_num: int,
        min_memory_num: int = 5,
        max_memory_num: int = 40,
        qa_document_num: int = 10,
        min_mem_length: int = 500,
        max_mem_length: int = 1500,
    ) -> None:
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.special_token_start = special_token_start
        self.mem_start = mem_start
        self.mem_end = mem_end
        self.reencode_num = reencode_num
        self.min_memory_num = min_memory_num
        self.max_memory_num = max_memory_num
        self.qa_document_num = qa_document_num
        self.min_memory_length = min_mem_length
        self.max_memory_length = max_mem_length

        self.bos_token_id = self.tokenizer.bos_id
        self.use_hf_tokenizer = True
        self.prepare_preprocessor()

    def prepare_preprocessor(self):
        self.sft_system = [
            "You are an AI assistant. Provide helpful, accurate, and clear answers. When uncertain, explain your reasoning or request clarification.",
            "You are an AI assistant. Focus on achieving the user's goal in each interaction. Use concise yet informative explanations.",
            "You are an AI assistant. Speak clearly and stay consistent with prior statements. If you need more information, politely ask for it.",
            "You are an AI assistant. Provide truthful, well-sourced information whenever possible. Acknowledge any limitations and avoid speculation if unsure.",
        ]
        self.sft_system_input_id_list = [
            self.tokenizer(
                "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
                + system_prompt
                + "<|eot_id|>",
                add_special_tokens=False,
            )["input_ids"]
            for system_prompt in self.sft_system
        ]

        self.qa_system = [
            "You are an AI assistant. Use the provided documents to answer the user’s question. If the information is insufficient, acknowledge the gap or request clarification.",
            "You are an AI assistant. Always ground your answers in the retrieved documents and do not add unsupported details. If the documents lack sufficient information, indicate that.",
            "You are an AI assistant. Rely solely on the given documents for evidence when answering questions. When necessary, cite or paraphrase the document content accurately.",
            "You are an AI assistant. Base your replies on the retrieved documents, ensuring completeness and correctness. Ask for more details if the documents do not cover the question fully.",
        ]
        self.qa_system_input_id_list = [
            self.tokenizer(
                "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
                + system_prompt
                + "<|eot_id|>",
                add_special_tokens=False,
            )["input_ids"]
            for system_prompt in self.qa_system
        ]

        self.summary_system = [
            "You are an AI assistant. Read the provided text and produce a concise summary. Capture the main points without unnecessary detail.",
            "You are an AI assistant. Summarize the essential ideas from the given text. Avoid minor details and focus on critical insights.",
            "You are an AI assistant. Provide a brief, high-level overview of the text. Ensure clarity and coherence, prioritizing key themes.",
            "You are an AI assistant. Summarize the text clearly and logically. Organize the main ideas in a coherent sequence.",
        ]
        self.summary_system_input_id_list = [
            self.tokenizer(
                "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
                + system_prompt
                + "<|eot_id|>",
                add_special_tokens=False,
            )["input_ids"]
            for system_prompt in self.summary_system
        ]

        text_inst_system = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou're an assistant who will complete the sentence after the text chunks given below<|eot_id|>"
        text_inst_sys_tokens = self.tokenizer(text_inst_system, add_special_tokens=False)[
            "input_ids"
        ]
        self.text_inst_sys_tokens = text_inst_sys_tokens + [self.mem_start]

        user_start_tokens = "<|start_header_id|>user<|end_header_id|>\n\n"
        self.user_start_token_ids = self.tokenizer(
            user_start_tokens, add_special_tokens=False
        )["input_ids"]
        self.eot_token_id = self.tokenizer("<|eot_id|>", add_special_tokens=False)[
            "input_ids"
        ][0]
        assistant_start_tokens = "<|start_header_id|>assistant<|end_header_id|>\n\n"
        self.assistant_start_token_ids = self.tokenizer(
            assistant_start_tokens, add_special_tokens=False
        )["input_ids"]

        self.all_memory_sum_tokens = [
            [
                self.special_token_start + idx * self.reencode_num + offset
                for offset in range(self.reencode_num)
            ]
            for idx in range(self.max_memory_num)
        ]

    def process_sftmem(self, example: Dict[str, str]):
        conversation = example["conversations"]
        input_texts = [conversation[i]["value"] for i in range(len(conversation))]
        all_conversation_texts_ids = self.tokenizer(
            input_texts,
            add_special_tokens=False,
            padding=False,
            truncation=False,
            return_attention_mask=False,
            return_token_type_ids=False,
            return_offsets_mapping=False,
            return_special_tokens_mask=False,
            return_length=False,
        )["input_ids"]

        sft_mem_system_input_ids = random.choice(self.sft_system_input_id_list)
        input_ids = sft_mem_system_input_ids + [self.mem_start]
        labels = [-100] * (len(sft_mem_system_input_ids) + 1)
        segment_ids = [0] * (len(sft_mem_system_input_ids) + 1)

        for idx in range(0, len(conversation) - 2, 2):
            if (
                conversation[idx]["from"] == "User"
                and conversation[idx + 1]["from"] == "Assistant"
            ):
                chat_input_ids = (
                    self.user_start_token_ids
                    + all_conversation_texts_ids[idx]
                    + [self.eot_token_id]
                    + self.assistant_start_token_ids
                    + all_conversation_texts_ids[idx + 1]
                    + [self.eot_token_id]
                )
                chat_input_ids = chat_input_ids + self.all_memory_sum_tokens[int(idx / 2)]
                input_ids = input_ids + chat_input_ids

                mem_len = len(chat_input_ids)

                chat_segment_ids = [
                    (idx + 2) // 2
                ] * (mem_len - self.reencode_num) + [0] * self.reencode_num
                segment_ids = segment_ids + chat_segment_ids

        last_q_input_ids = (
            [self.mem_end]
            + self.user_start_token_ids
            + all_conversation_texts_ids[-2]
            + [self.eot_token_id]
            + self.assistant_start_token_ids
            + all_conversation_texts_ids[-1]
            + [self.eot_token_id]
        )
        last_q_segment_ids = [0] * len(last_q_input_ids)

        input_ids = input_ids + last_q_input_ids
        segment_ids = segment_ids + last_q_segment_ids

        seq_len = len(input_ids)
        ans_len = len(all_conversation_texts_ids[-1]) + 1
        labels = (
            [-100] * (seq_len - ans_len)
            + all_conversation_texts_ids[-1]
            + [self.eot_token_id]
        )

        return {
            "input_ids": input_ids,
            "labels": labels,
            "segment_ids": segment_ids,
        }

    def process_sft(self, example: Dict[str, str]):
        conversation = example["conversations"]
        input_texts = [conversation[i]["value"] for i in range(len(conversation))]
        all_conversation_texts_ids = self.tokenizer(
            input_texts,
            add_special_tokens=False,
            padding=False,
            truncation=False,
            return_attention_mask=False,
            return_token_type_ids=False,
            return_offsets_mapping=False,
            return_special_tokens_mask=False,
            return_length=False,
        )["input_ids"]

        sft_system_input_ids = random.choice(self.sft_system_input_id_list)
        input_ids = sft_system_input_ids
        labels = [-100] * len(sft_system_input_ids)
        for i in range(len(conversation)):
            if conversation[i]["from"] == "User":
                user_msg_input_ids = (
                    self.user_start_token_ids
                    + all_conversation_texts_ids[i]
                    + [self.eot_token_id]
                )
                if len(labels) + len(user_msg_input_ids) >= self.max_len:
                    break

                labels.extend([-100] * len(user_msg_input_ids))
                input_ids += user_msg_input_ids
            elif conversation[i]["from"] == "Assistant":
                assist_msg_input_ids = (
                    self.assistant_start_token_ids + all_conversation_texts_ids[i]
                )
                if len(labels) + len(assist_msg_input_ids) > self.max_len - 1:
                    assist_msg_input_ids = assist_msg_input_ids[
                        : self.max_len - 1 - len(labels)
                    ]

                assist_msg_input_ids += [self.eot_token_id]
                labels.extend(assist_msg_input_ids)
                input_ids += assist_msg_input_ids

        segment_ids = [0] * len(input_ids)
        return {
            "input_ids": input_ids,
            "labels": labels,
            "segment_ids": segment_ids,
        }

    def process_textinst(self, example: Dict[str, str]):
        sys_len = len(self.text_inst_sys_tokens)

        user = "<|start_header_id|>user<|end_header_id|>\n\nPlease complete the sentence<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        user_tokens = self.tokenizer(user, add_special_tokens=False)["input_ids"]
        user_tokens = [self.mem_end] + user_tokens
        user_len = len(user_tokens)

        text = example["text"]
        input_ids = self.tokenizer(text, add_special_tokens=False)["input_ids"]
        input_ids = input_ids[: self.max_len - user_len - sys_len]

        mem_len = random.randint(self.min_memory_length, self.max_memory_length)
        mem_num = random.randint(self.min_memory_num, self.max_memory_num)

        breaks = sorted(random.sample(range(1, mem_len), mem_num - 1))
        breaks = [0] + breaks + [mem_len]
        each_mem_len = [breaks[i + 1] - breaks[i] for i in range(mem_num)]

        input_len = len(input_ids)
        input_ids = input_ids[: input_len - self.reencode_num * mem_num]

        memory_ids = input_ids[:mem_len]
        remaining_ids = input_ids[mem_len:]

        segment_ids = [0] * len(self.text_inst_sys_tokens)
        concat_ids = self.text_inst_sys_tokens[:]

        split_memory_ids = []
        index = 0
        for size in each_mem_len:
            split_memory_ids.append(memory_ids[index : index + size])
            index += size

        for i in range(mem_num):
            tem_mem_id = split_memory_ids[i] + self.all_memory_sum_tokens[i]
            concat_ids += tem_mem_id
            segment_ids = segment_ids + [i + 1] * len(split_memory_ids[i]) + [
                0
            ] * self.reencode_num

        concat_ids = concat_ids + user_tokens + remaining_ids
        segment_ids = segment_ids + [0] * (len(user_tokens) + len(remaining_ids))
        mem_len = mem_len + self.reencode_num * mem_num
        labels = [-100] * (sys_len + mem_len + user_len) + remaining_ids

        return {
            "input_ids": concat_ids,
            "labels": labels,
            "segment_ids": segment_ids,
        }

    def process_text(self, example: Dict[str, str]):
        text_tokens = self.tokenizer(example["text"], add_special_tokens=False)[
            "input_ids"
        ][: self.max_len]
        labels = text_tokens
        segment_ids = [0] * len(text_tokens)
        return {
            "input_ids": text_tokens,
            "labels": labels,
            "segment_ids": segment_ids,
        }

    def process_textmem(self, example: Dict[str, str]):
        sys_tokens = [self.bos_token_id, self.mem_start]
        sys_len = len(sys_tokens)

        user_tokens = [self.mem_end]
        user_len = len(user_tokens)

        text = example["text"]
        input_ids = self.tokenizer(text, add_special_tokens=False)["input_ids"]
        input_ids = input_ids[: self.max_len - user_len - sys_len]

        mem_len = random.randint(self.min_memory_length, self.max_memory_length)
        mem_num = random.randint(self.min_memory_num, self.max_memory_num)

        breaks = sorted(random.sample(range(1, mem_len), mem_num - 1))
        breaks = [0] + breaks + [mem_len]
        each_mem_len = [breaks[i + 1] - breaks[i] for i in range(mem_num)]

        input_len = len(input_ids)
        input_ids = input_ids[: input_len - self.reencode_num * mem_num]

        memory_ids = input_ids[:mem_len]
        remaining_ids = input_ids[mem_len:]
        concat_ids = sys_tokens

        split_memory_ids = []
        index = 0
        for size in each_mem_len:
            split_memory_ids.append(memory_ids[index : index + size])
            index += size

        segment_ids = [0] * sys_len
        for mem_id in range(mem_num):
            tem_mem_id = split_memory_ids[mem_id]
            for sub_idx in range(self.reencode_num):
                tem_mem_id = tem_mem_id + [
                    self.special_token_start + self.reencode_num * mem_id + sub_idx
                ]
            segment_ids = segment_ids + [mem_id + 1] * each_mem_len[mem_id] + [
                0
            ] * self.reencode_num
            concat_ids += tem_mem_id

        concat_ids = concat_ids + user_tokens + remaining_ids
        mem_len = mem_len + self.reencode_num * mem_num
        labels = [-100] * (sys_len + mem_len + user_len) + remaining_ids
        segment_ids = segment_ids + [0] * len(user_tokens) + [0] * len(remaining_ids)

        return {
            "input_ids": concat_ids,
            "labels": labels,
            "segment_ids": segment_ids,
        }

    def process_qamem(self, example: Dict[str, str]):
        qa_system_input_ids = random.choice(self.qa_system_input_id_list)
        input_ids = qa_system_input_ids[:] + [self.mem_start]
        segment_ids = [0] * len(input_ids)

        formated_input_text_list = [
            f"Document [{j+1}](Title: {example['documents'][j]['title']}) "
            f"{example['documents'][j]['text']}\n"
            for j in range(10)
        ]
        formated_input_ids = self.tokenizer(
            formated_input_text_list,
            add_special_tokens=False,
            padding=False,
            truncation=False,
            return_attention_mask=False,
            return_token_type_ids=False,
            return_offsets_mapping=False,
            return_special_tokens_mask=False,
            return_length=False,
        )["input_ids"]

        for idx in range(self.qa_document_num):
            input_ids = input_ids + formated_input_ids[idx] + self.all_memory_sum_tokens[idx]
            segment_ids = segment_ids + [idx + 1] * len(formated_input_ids[idx]) + [
                0
            ] * self.reencode_num

        user_input_ids = (
            [self.mem_end]
            + [self.eot_token_id]
            + self.user_start_token_ids
            + self.tokenizer(example["question"], add_special_tokens=False)["input_ids"]
            + [self.eot_token_id]
            + self.assistant_start_token_ids
        )
        input_ids = input_ids + user_input_ids
        segment_ids = segment_ids + [0] * len(user_input_ids)

        ans_id = self.tokenizer(
            example["generated"] + "<|eot_id|>", add_special_tokens=False
        )["input_ids"]
        input_ids += ans_id
        segment_ids = segment_ids + [0] * len(ans_id)

        ans_len = len(ans_id)
        input_len = len(input_ids)
        labels = [-100] * (input_len - ans_len) + ans_id

        return {
            "input_ids": input_ids,
            "labels": labels,
            "segment_ids": segment_ids,
        }

    def process_qa(self, example: Dict[str, str]):
        qa_system_input_ids = random.choice(self.qa_system_input_id_list)
        system_input_ids = qa_system_input_ids
        input_ids = system_input_ids

        for j in range(self.qa_document_num):
            title = example["documents"][j]["title"]
            text = example["documents"][j]["text"]
            tem_id = self.tokenizer(
                f"Document [{j+1}](Title: {title}) {text}\n",
                add_special_tokens=False,
            )["input_ids"]

            input_ids += tem_id

        user_input_ids = (
            [self.eot_token_id]
            + self.user_start_token_ids
            + self.tokenizer(example["question"], add_special_tokens=False)["input_ids"]
            + [self.eot_token_id]
            + self.assistant_start_token_ids
        )
        input_ids += user_input_ids

        ans_id = self.tokenizer(
            example["generated"] + "<|eot_id|>", add_special_tokens=False
        )["input_ids"]
        input_ids += ans_id

        ans_len = len(ans_id)
        input_len = len(input_ids)
        labels = [-100] * (input_len - ans_len) + ans_id

        segment_ids = [0] * len(input_ids)

        return {
            "input_ids": input_ids,
            "labels": labels,
            "segment_ids": segment_ids,
        }

    def process_tulu(self, example: Dict[str, str]):
        conversation = example["messages"]
        input_texts = [conversation[i]["content"] for i in range(len(conversation))]
        all_conversation_texts_ids = self.tokenizer(
            input_texts,
            add_special_tokens=False,
            padding=False,
            truncation=False,
            return_attention_mask=False,
            return_token_type_ids=False,
            return_offsets_mapping=False,
            return_special_tokens_mask=False,
            return_length=False,
        )["input_ids"]

        tulu_system_input_ids = random.choice(self.sft_system_input_id_list)
        input_ids = tulu_system_input_ids
        labels = [-100] * len(input_ids)

        for i in range(len(conversation)):
            if conversation[i]["role"] == "user":
                user_msg_input_ids = (
                    self.user_start_token_ids
                    + all_conversation_texts_ids[i]
                    + [self.eot_token_id]
                )
                if len(labels) + len(user_msg_input_ids) >= self.max_len:
                    break

                labels.extend([-100] * len(user_msg_input_ids))
                input_ids += user_msg_input_ids
            elif conversation[i]["role"] == "assistant":
                assist_msg_input_ids = (
                    self.assistant_start_token_ids + all_conversation_texts_ids[i]
                )
                if len(labels) + len(assist_msg_input_ids) > self.max_len - 1:
                    assist_msg_input_ids = input_ids[: self.max_len - 1 - len(labels)]

                assist_msg_input_ids += [self.eot_token_id]
                labels.extend(assist_msg_input_ids)
                input_ids += assist_msg_input_ids

        segment_ids = [0] * len(input_ids)
        return {
            "input_ids": input_ids,
            "labels": labels,
            "segment_ids": segment_ids,
        }

    def process_xsum(self, example: Dict[str, str]):
        text_ids = self.tokenizer(example["document"], add_special_tokens=False)[
            "input_ids"
        ]
        if len(text_ids) > 1900:
            chunk_size = 220
        else:
            chunk_size = 100
        chunks = [text_ids[i : i + chunk_size] for i in range(0, len(text_ids), chunk_size)]

        xsum_system_input_ids = random.choice(self.summary_system_input_id_list)
        input_ids = xsum_system_input_ids + [self.mem_start]
        segment_ids = [0] * len(input_ids)

        for j in range(len(chunks)):
            input_ids = input_ids + chunks[j] + self.all_memory_sum_tokens[j]
            segment_ids = segment_ids + [j + 1] * len(chunks[j]) + [
                0
            ] * self.reencode_num

        ans_id = self.tokenizer(
            example["summary"] + "<|eot_id|>", add_special_tokens=False
        )["input_ids"]
        assistant_input_ids = (
            [self.mem_end, self.eot_token_id] + self.assistant_start_token_ids + ans_id
        )

        input_ids = input_ids + assistant_input_ids
        labels = [-100] * (len(input_ids) - len(ans_id)) + ans_id
        segment_ids = segment_ids + [0] * len(assistant_input_ids)

        return {
            "input_ids": input_ids,
            "labels": labels,
            "segment_ids": segment_ids,
        }


__all__ = [
    "DataComponent",
    "DATASET_MAPPING",
    "build_kvlink_dataloader",
    "SumAttentionPreprocessor",
]