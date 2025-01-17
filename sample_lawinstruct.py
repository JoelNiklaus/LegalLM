import random
import json

from datasets import get_dataset_config_names
from datasets import load_dataset

import pandas as pd


def should_be_sampled(example, max_num_whitespace_tokens):
    example = {k: ("" if v is None else v) for k, v in example.items()}
    text = example["instruction"] + " " + example["prompt"] + " " + example["answer"]
    return text and len(text.split()) < max_num_whitespace_tokens


def get_example_dict(example):
    return {
        "instruction": example["instruction"],
        "input": example["prompt"],
        "output": example["answer"]
    }


def calc_max_num_whitespace_tokens(max_seq_len, use_template=False):
    token_deduction = int(max_seq_len / 3)  # each whitespace word expands to approx. 1.5 tokens
    if use_template:
        template_deduction = 50  # 50 tokens for the instruction template in the LegalLM code
    else:
        template_deduction = 0
    max_num_whitespace_tokens = max_seq_len - token_deduction - template_deduction
    # with template_deduction: 512 ==> 292, 1024 ==> 633, 2048 ==> 1316
    return max_num_whitespace_tokens


def generate_instruction_data(dataset_name,
                              configs,
                              max_seq_len=512,
                              num_samples=500,
                              use_fast_way=True,
                              do_shuffle=True,
                              only_english_tasks=False):
    """
    We only use zero-shot examples for now because our examples are very long.
    """

    instruction_data = []
    filename = f"law_instruction_data_len:{max_seq_len}_samples:{num_samples}.json"
    stats = {"config": [], "num_examples": [], "jurisdiction": [], "task_type": [],
             "instruction_language": [], "prompt_language": [], "answer_language": []}
    for config in configs:
        print(f"Loading {dataset_name}:{config}...")
        dataset = load_dataset(dataset_name,
                               config,
                               split="train",
                               use_auth_token=True,
                               streaming=True)

        max_num_whitespace_tokens = calc_max_num_whitespace_tokens(max_seq_len)
        print(
            f"Filtering out examples with more than {max_seq_len} tokens "
            f"({max_num_whitespace_tokens} whitespace separated words) and sampling {num_samples} examples..."
        )
        if only_english_tasks:
            example = next(iter(dataset))
            if example["prompt_language"] != "en":
                print(f"The first example's language of the dataset {config} was {example['prompt_language']}. "
                      f"Skipping...")
                continue

        if use_fast_way:
            num_samples_taken = 0
            for example in dataset:
                if should_be_sampled(example, max_num_whitespace_tokens):
                    instruction_data.append(get_example_dict(example))
                    num_samples_taken += 1
                if num_samples_taken >= num_samples:
                    break
        else:  # this is a cleaner way which probably takes longer
            # this slows it down considerably for large datasets,
            # but could be more easily parallelized when using non-streaming datasets
            dataset = dataset.filter(lambda example: should_be_sampled(example, max_num_whitespace_tokens))
            dataset = dataset.shuffle(seed=42)
            examples_to_add = [get_example_dict(example) for example in dataset.take(num_samples)]
            instruction_data.extend(examples_to_add)
            num_samples_taken = len(examples_to_add)

        stats["config"].append(config)
        stats["num_examples"].append(num_samples_taken)

        stats["jurisdiction"].append(example["jurisdiction"])
        stats["task_type"].append(example["task_type"])
        stats["instruction_language"].append(example["instruction_language"])
        stats["prompt_language"].append(example["prompt_language"])
        stats["answer_language"].append(example["answer_language"])

    if do_shuffle:
        print(f"Shuffling {len(instruction_data)} examples...")
        random.shuffle(instruction_data)

    # convert stats to pandas dataframe and save to csv
    stats_df = pd.DataFrame(stats)
    stats_df.to_csv(f"stats_{filename}.csv", index=False)

    print(f"Writing {len(instruction_data)} examples to {filename}...")
    with open(filename, "w") as file:
        json.dump(instruction_data, file, indent=4)
    return instruction_data


def generate_lawinstruct(max_seq_len=512, num_samples=10000, debug=False,
                         instruction_language="english", only_english_tasks=True,
                         tasks="all"):
    dataset_name = f"lawinstruct/lawinstruct_{instruction_language}"
    configs = get_dataset_config_names(dataset_name)
    if debug:
        configs = configs[:1]
    print(f"Available configs: {configs}")

    non_legal_configs = ['NaturalInstructionsOther', 'XP3MT']
    faulty_configs = []
    configs = [config for config in configs
               if config not in non_legal_configs and config not in faulty_configs and config != 'all']
    if tasks == 'without-mc':
        mc_configs = [
            'BrCAD5-brcad5_mc',
            'CAIL2022-cail_2022_mc',
            'ProfessionalLaw-professional_law_examples',
            'ProfessionalLaw-professional_law_zero_shot',
            'ReClor-reclor',
            'Sara-sara_entailment',
            'Sara-sara_tax_liability',
            'SwissJudgmentPrediction-swiss_judgment_multiple_choice',
            'TsccAlqac-tscc_alqac_question_answering',
            'TurkishConstitutionalCourt-MainSubset',
            'TurkishConstitutionalCourt-turkish_constitutional_multiple_choice',
            'ValidWills-MainSubset',
        ]
        configs = [config for config in configs if config not in mc_configs]  # filter out multiple choice datasets
    elif tasks == 'without-nli':
        nli_configs = [
            'LawngNli-lawng_nli_entailment',
            'ValidWills-MainSubset',
            'Sara-sara_entailment',
            'ContractNLI-contract_nli',
            'COLIEE-task3_generate_entailed_question',
            'COLIEE-task3_passage_entailment'
        ]
        configs = [config for config in configs if config not in nli_configs]  # exclude the entailment sets
    elif tasks == 'only-lexglue':
        configs = [config for config in configs if "LexGLUE" in config]  # only take lex glue tasks
    elif tasks == 'only-casehold':
        configs = ['LexGLUE-case_hold']  # only take casehold
    else:
        pass
    print(f"Using configs: {configs}")
    return generate_instruction_data(dataset_name, configs, max_seq_len=max_seq_len, num_samples=num_samples,
                                     only_english_tasks=only_english_tasks)


if __name__ == '__main__':
    # TODO check why some datasets only contain very few examples even when using 2048 tokens
    for max_seq_len in [512, 1024, 2048]:
        for num_samples in [10, 100, 1000, 10000]:
            generate_lawinstruct(max_seq_len=max_seq_len, num_samples=num_samples)
