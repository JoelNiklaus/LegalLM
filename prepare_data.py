import copy
import logging
from dataclasses import dataclass
from typing import Dict, Sequence

import transformers
from torch.utils.data import Dataset

import torch

import utils

from sample_lawinstruct import generate_lawinstruct

IGNORE_INDEX = -100
PROMPT_DICT = {
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
    ),
    "prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:"
    ),
}


def _tokenize_fn(strings: Sequence[str], tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
        )
        for text in strings
    ]
    input_ids = labels = [tokenized.input_ids[0] for tokenized in tokenized_list]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item() for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )


def preprocess(
        sources: Sequence[str],
        targets: Sequence[str],
        tokenizer: transformers.PreTrainedTokenizer,
        supervised: bool = True,
) -> Dict:
    """Preprocess the data by tokenizing."""
    examples = [s + t for s, t in zip(sources, targets)]
    examples_tokenized, sources_tokenized = [_tokenize_fn(strings, tokenizer) for strings in (examples, sources)]
    input_ids = examples_tokenized["input_ids"]
    labels = copy.deepcopy(input_ids)
    if supervised:
        for label, source_len in zip(labels, sources_tokenized["input_ids_lens"]):
            label[:source_len] = IGNORE_INDEX  # ignore source, all tokens are set to -100

    # Filtering out examples that are too long
    # to make sure that we don't train on examples where we have no or partial targets
    for i in range(len(input_ids)):
        # filter if longer than tokenizer.model_max_length
        if len(input_ids[i]) > tokenizer.model_max_length:
            input_ids[i] = None
            labels[i] = None
    input_ids = [x for x in input_ids if x is not None]
    labels = [x for x in labels if x is not None]

    return dict(input_ids=input_ids, labels=labels)


class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: str, tokenizer: transformers.PreTrainedTokenizer, use_template=False):
        super(SupervisedDataset, self).__init__()
        if "max-seq-len:" in data_path and "samples:" in data_path:
            max_seq_len, num_samples = [int(x.split(":")[1]) for x in data_path.split("_")]
            logging.warning(f"Generating data with max_seq_len={max_seq_len} and num_samples={num_samples} ...")
            list_data_dict = generate_lawinstruct(max_seq_len=max_seq_len, num_samples=num_samples, debug=False)
        else:  # it is a real data path
            logging.warning(f"Loading data from {data_path} ...")
            list_data_dict = utils.jload(data_path)

        logging.warning("Formatting inputs...")

        if use_template:
            sources = [
                PROMPT_DICT["prompt_input"].format_map(example)
                if example.get("input", "") != "" else
                PROMPT_DICT["prompt_no_input"].format_map(example)
                for example in list_data_dict if 'instruction' in example
            ]
        else:
            sources = [f"{example['instruction']}\n\n{example['input']}\n\n"
                       if example.get("input", "") != "" else
                       f"{example['instruction']}\n\n"
                       for example in list_data_dict
                       if 'instruction' in example and example.get("instruction", "") != ""]
        targets = [f"{example['output']}\n\n{tokenizer.eos_token}" for example in list_data_dict]

        logging.warning("Tokenizing inputs... This may take some time...")
        data_dict = preprocess(sources, targets, tokenizer, supervised=True)

        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(input_ids=self.input_ids[i], labels=self.labels[i])


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )


def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer, data_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = SupervisedDataset(tokenizer=tokenizer, data_path=data_args.data_path)
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator)
