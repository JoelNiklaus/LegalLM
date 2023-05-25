#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.
import os
from dataclasses import dataclass, field
from typing import Optional, Dict

import transformers
from transformers import Trainer, TrainerCallback, TrainerState, TrainerControl

from peft import LoraConfig, get_peft_model
import torch
import torch.nn as nn
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
from prepare_data import make_supervised_data_module

import bitsandbytes as bnb

DEFAULT_PAD_TOKEN = "[PAD]"

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")


@dataclass
class DataArguments:
    data_path: str = field(default=None, metadata={"help": "Path to the training data."})


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    train_with_peft: bool = field(default=False)
    gradient_checkpointing: bool = field(default=False)
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=512,
        metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )


class SavePeftModelCallback(TrainerCallback):
    """https://github.com/huggingface/peft/issues/96#issuecomment-1460080427"""

    def on_save(
            self,
            args: TrainingArguments,
            state: TrainerState,
            control: TrainerControl,
            **kwargs,
    ):
        checkpoint_folder = os.path.join(
            args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}"
        )

        peft_model_path = os.path.join(checkpoint_folder, "adapter_model")
        kwargs["model"].save_pretrained(peft_model_path)

        pytorch_model_path = os.path.join(checkpoint_folder, "pytorch_model.bin")
        if os.path.exists(pytorch_model_path):
            os.remove(pytorch_model_path)
        return control


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """Collects the state dict and dump to disk."""
    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


def smart_tokenizer_and_embedding_resize(
        special_tokens_dict: Dict,
        tokenizer: transformers.PreTrainedTokenizer,
        model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


def train():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        trust_remote_code=True,
        # loading in 8 bit might lead to problems:
        # but so far we don't know if 8 bit is the issue or training a fp16 trained model in bf16 or both
        # since I did not see huge gains in memory usage when loading in 8 bit,
        # fp16 pythia 6.7b with 8 bit: 110s/it; without 8 bit: 30s/it, memory usage the same ==> disable it for now
        # however, in the lora colab it is enabled: https://colab.research.google.com/drive/1jCkpikz0J2o20FBQmYmAGdiKmJGOMo-o?usp=sharing#scrollTo=AQ_HCYruWIHU
        # load_in_8bit=True if training_args.train_with_peft else False,
        device_map='auto',
    )
    print(model.config)
    print(model)

    if 'mosaicml/mpt' in model_args.model_name_or_path:
        model.config.attn_config['attn_impl'] = 'triton'

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
    )
    if tokenizer.pad_token is None:
        print(f"Adding {DEFAULT_PAD_TOKEN} to the tokenizer.")
        smart_tokenizer_and_embedding_resize(
            special_tokens_dict=dict(pad_token=DEFAULT_PAD_TOKEN),
            tokenizer=tokenizer,
            model=model,
        )

    if training_args.train_with_peft:
        for param in model.parameters():
            param.requires_grad = False  # freeze the model - train adapters later
            if param.ndim == 1:
                # cast the small parameters (e.g. layernorm) to fp32 for stability
                param.data = param.data.to(torch.float32)

        if training_args.gradient_checkpointing:
            model.gradient_checkpointing_enable()  # reduce number of stored activations ==> greatly reduces memory usage
        model.enable_input_require_grads()

        class CastOutputToFloat(nn.Sequential):
            def forward(self, x): return super().forward(x).to(torch.float32)

        if isinstance(model.config, transformers.GPTNeoXConfig):
            model.embed_out = CastOutputToFloat(model.embed_out)
            target_modules = ["query_key_value", "dense"]
        elif isinstance(model.config, transformers.OPTConfig):
            model.lm_head = CastOutputToFloat(model.lm_head)
            target_modules = ["q_proj", "v_proj"]
        elif isinstance(model.config, transformers.GPT2Config):
            model.lm_head = CastOutputToFloat(model.lm_head)
            target_modules = ["c_attn", "c_proj"]
        elif isinstance(model.config, transformers.LlamaConfig):
            model.lm_head = CastOutputToFloat(model.lm_head)
            target_modules = ["q_proj", "v_proj"]
        else:
            raise ValueError(f"Unknown model: {model_args.model_name_or_path}")

        config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=target_modules,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM"
        )

        model = get_peft_model(model, config)
        model.print_trainable_parameters()

        model.config.use_cache = False  # silence the warnings. Please re-enable for inference!

    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)

    trainer = Trainer(model=model, tokenizer=tokenizer,
                      callbacks=[SavePeftModelCallback],
                      args=training_args,
                      **data_module)
    trainer.train()

    hf_name = f"lawinstruct/LegalLM-{model_args.model_name_or_path.split('/')[1]}"
    if training_args.train_with_peft:
        model.save_pretrained(training_args.output_dir)
        hf_name += "-lora"
    else:
        trainer.save_state()
        safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)

    model.push_to_hub(hf_name, use_auth_token=True, private=True)


if __name__ == "__main__":
    train()
