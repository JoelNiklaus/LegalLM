import json

from datasets import get_dataset_config_names
from datasets import load_dataset


def generate_instruction_data(dataset_name,
                              configs,
                              max_seq_len=512,
                              num_samples=500,
                              use_fast_way=True):
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

    instruction_data = []
    filename = f"law_instruction_data_len:{max_seq_len}_samples:{num_samples}.json"
    for config in configs:
        print(f"Loading {dataset_name}:{config}...")
        dataset = load_dataset(dataset_name,
                               config,
                               split="train",
                               use_auth_token=True,
                               streaming=True)

        token_deduction = int(max_seq_len / 3)  # each whitespace word expands to approx. 1.5 tokens
        template_deduction = 50  # 50 tokens for the instruction template in the LegalLM code
        max_num_whitespace_tokens = max_seq_len - token_deduction - template_deduction
        # 512 ==> 292, 1024 ==> 633, 2048 ==> 1316
        print(
            f"Filtering out examples with more than {max_seq_len} tokens "
            f"({max_num_whitespace_tokens} whitespace separated words) and sampling {num_samples} examples..."
        )
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
    print(f"Writing {len(instruction_data)} examples to {filename}...")
    with open(filename, "w") as file:
        json.dump(instruction_data, file, indent=4)
    return instruction_data


def generate_lawinstruct(max_seq_len=512, num_samples=10000, debug=False):
    dataset_name = "lawinstruct/lawinstruct"
    configs = get_dataset_config_names(dataset_name)
    if debug:
        configs = configs[:2]
    print(configs)

    non_legal_configs = ['NaturalInstructionsOther', 'XP3MT']
    faulty_configs = []
    configs = [config for config in configs
               if config not in non_legal_configs and config not in faulty_configs and config != 'all']
    return generate_instruction_data(dataset_name, configs, max_seq_len=max_seq_len, num_samples=num_samples)


if __name__ == '__main__':
    for max_seq_len in [512, 1024, 2048]:
        for num_samples in [100, 1000, 10000]:
            generate_lawinstruct(max_seq_len=max_seq_len, num_samples=num_samples)
