import json
import re
import os
from datasets import load_dataset
import random
from pathlib import Path

# Dataset splits
list_split = ['train.jsonl', 'dev.jsonl', 'test.jsonl']

# Load and save DialogSum dataset
dataset = load_dataset("knkarthick/dialogsum")
print(dataset)
dataset["train"].to_json("dialogsum_split/train.jsonl", orient="records", lines=True)
dataset["validation"].to_json("dialogsum_split/dev.jsonl", orient="records", lines=True)
dataset["test"].to_json("dialogsum_split/test.jsonl", orient="records", lines=True)

# Load and save Alpaca dataset
dataset = load_dataset("tatsu-lab/alpaca")
print(dataset)
dataset["train"].to_json("alpaca_raw/train.jsonl", orient="records", lines=True)

# Load and save Self-Instruct dataset
dataset = load_dataset("yizhongw/self_instruct", "super_natural_instructions")
print(dataset)
dataset["train"].to_json("self_instruct_raw/train.jsonl", orient="records", lines=True)
dataset["test"].to_json("self_instruct_raw/test.jsonl", orient="records", lines=True)

# Process Alpaca dataset
def process_jsonl_file_alpaca(input_file, output_file):
    transformed_data = []
    with open(input_file, 'r') as file:
        for line in file:
            example = json.loads(line.strip())
            if 'prompt' in example:
                del example['prompt']
            if 'text' in example:
                del example['text']
            transformed_data.append(example)

    with open(output_file, 'w') as file:
        for data in transformed_data:
            file.write(json.dumps(data) + '\n')

    print(f"Transformed examples saved to '{output_file}'")

files = ['alpaca_raw/train.jsonl']
for file in files:
    process_jsonl_file_alpaca(file, file)

# Process DialogSum dataset
def process_dialogsum_dataset(input_file, output_file):
    transformed_data = []
    with open(input_file, 'r') as file:
        for line in file:
            example = json.loads(line.strip())
            if 'dialogue' in example and 'summary' in example:
                transformed_data.append({
                    "instruction": "Generate a summary of the dialogue, focusing specifically on the provided topic: " + example['topic'].strip() + '.',
                    # "instruction": "Generate a summary of the given dialogue.",
                    "input": example['dialogue'].strip(),
                    "output": example['summary'].strip()
                })
    with open(output_file, 'w') as file:
        for data in transformed_data:
            file.write(json.dumps(data) + '\n')

    print(f"Transformed DialogSum dataset saved to '{output_file}'")

files = ['dialogsum_split/train.jsonl', 'dialogsum_split/dev.jsonl', 'dialogsum_split/test.jsonl']
for file in files:
    process_dialogsum_dataset(file, file)

# Process Self-Instruct dataset
def process_jsonl_file_self_instruct(input_file, output_file):
    transformed_data = []
    with open(input_file, 'r') as file:
        for line in file:
            example = json.loads(line.strip())
            if 'prompt' in example and 'completion' in example:
                match = re.search(r"Input:\s*(.*?)\s*Output:", example["prompt"], re.DOTALL)
                if match:
                    input_text = match.group(1).strip()
                    transformed_data.append({
                        "instruction": example["prompt"].split("\n\n")[0].strip(),
                        "input": input_text,
                        "output": example["completion"].strip()
                    })
            elif 'instruction' in example and 'input' in example and 'output' in example:
                transformed_data.append({
                    "instruction": example["instruction"].strip(),
                    "input": example["input"].strip(),
                    "output": example["output"].strip()
                })

    with open(output_file, 'w') as file:
        for data in transformed_data:
            file.write(json.dumps(data) + '\n')

    print(f"Transformed examples saved to '{output_file}'")

files = ['self_instruct_raw/train.jsonl', 'self_instruct_raw/test.jsonl']
for file in files:
    process_jsonl_file_self_instruct(file, file)

def read_jsonl(file_path, min_output_length=11):
    parsed_data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            data = json.loads(line.strip())
            if len(data['input'].strip()) != 0 and len(data['output'].strip().split()) >= min_output_length:
                parsed_data.append({
                    'instruction': data['instruction'],
                    'input': data['input'],
                    'output': data['output'],
                })
    return parsed_data

def read_jsonl_test_self_instruct(file_path, min_output_length=11, max_prompt_length=350):
    parsed_data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            data = json.loads(line.strip())
            if len(data['input'].strip()) != 0 and len(data['output'].strip().split()) >= min_output_length and (len(data['instruction'].strip().split())+len(data['instruction'].strip().split())) <= max_prompt_length:
                parsed_data.append({
                    'instruction': data['instruction'],
                    'input': data['input'],
                    'output': data['output'],
                })
    return parsed_data


def save_jsonl(data, file_path):
    with open(file_path, 'w', encoding='utf-8') as file:
        for item in data:
            file.write(json.dumps(item) + '\n')

def split_and_save(dataset_path, output_dir, dataset_name, split_sizes, min_output_length=11, seed=42):
    data = read_jsonl(dataset_path, min_output_length)
    random.seed(seed)
    random.shuffle(data)

    train_size, dev_size, test_size = split_sizes
    train_size = len(data) - dev_size - test_size
    train_data = data[:train_size]
    dev_data = data[train_size:train_size + dev_size]
    test_data = data[train_size + dev_size:train_size + dev_size + test_size]

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    save_jsonl(train_data, os.path.join(output_dir, f"train.jsonl"))
    save_jsonl(dev_data, os.path.join(output_dir, f"dev.jsonl"))
    save_jsonl(test_data, os.path.join(output_dir, f"test.jsonl"))
    print(f"{dataset_name} dataset split and saved to {output_dir}")

def split_and_save_one(dataset_path, output_dir, dataset_name, name, min_output_length=11, max_prompt_length = 350, seed=42):
    test_data = read_jsonl_test_self_instruct(dataset_path, min_output_length, max_prompt_length)
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    save_jsonl(test_data, os.path.join(output_dir, f"{name}.jsonl"))
    print(f"{dataset_name} dataset split and saved to {output_dir}")

datasets = {
    "alpaca": {
        "path": "alpaca_raw/train.jsonl",
        "output_dir": "alpaca_split",
        "split_sizes": (10396, 500, 500),
        "min_output_length": 11
    },

    "self_instruct": {
        "path": "self_instruct_raw/train.jsonl",
        "output_dir": "self_instruct_split",
        "split_sizes": (10414, 500, 0),  
        "min_output_length": 6
    },
}

# Split and save each dataset
for dataset_name, config in datasets.items():
    split_and_save(
        dataset_path=config["path"],
        output_dir=config["output_dir"],
        dataset_name=dataset_name,
        split_sizes=config["split_sizes"],
        min_output_length=config["min_output_length"],
        seed=42
    )

split_and_save_one(
    dataset_path="self_instruct_raw/test.jsonl",
    output_dir="self_instruct_split",
    dataset_name='self_instruct',
    name='test',  
    min_output_length= 11,
    seed=42
)
import shutil
splits = ['train', 'train_rationale_augment', 'dev', 'test']
data_dirs = ['dolly', 'alpaca_split', 'dialogsum_split', 'self_instruct_split']

for data_dir in data_dirs:
    for split in splits:
        if data_dir == 'dolly' and split == 'dev':
            continue
        if data_dir == 'dolly' and split == 'valid':
            continue
        if data_dir == 'dolly' and split == 'test':
            continue
        if data_dir == 'dolly' and split == 'train':
            continue
        # Always read from 'train.jsonl' for 'train_rationale_v1'
        input_file = os.path.join(
            data_dir,
            'train.jsonl' if split in ['train_rationale_v1', 'train_rationale_augment'] else f"{split}.jsonl"
        )        
        output_dir = data_dir.replace("_split", "")
        os.makedirs(output_dir, exist_ok=True)  # Ensure the output directory exists
        
        # Determine the correct output file name
        output_file = os.path.join(output_dir, f"{split}.jsonl")
        
        with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
            for line in infile:
                line = json.loads(line.strip())
                # if split == 'train_rationale_v1':
                    
                #     if "input" not in line or len(line["input"]) == 0:
                #         template = (
                #             "Below is an instruction that describes a task. "
                #             "Write a response that appropriately completes the request.\n\n"
                #             "Please give a short explanation why for the response.\n\n"
                #             "### Instruction:\n{instruction}\n\n### Response:\n"
                #         )
                #         prompt_str = template.format(instruction=line["instruction"])
                #     else:
                #         template = (
                #             "Below is an instruction that describes a task, paired with an input that provides further context. "
                #             "Write a response that appropriately completes the request.\n\n"
                #             "Please give a short explanation why for the response.\n\n"
                #             "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n\n ### Explain:\n"
                #         )
                #         prompt_str = template.format(instruction=line["instruction"], input=line["input"])
                #     line['prompt'] = prompt_str
                    # Add rationale prompt for 'train_rationale_v1'
                    # if split == 'train_rationale_v1':
                    #     line["prompt"] = prompt_str + "\nPlease give a short explanation why.\n"
                    # else:
                    #     line['prompt'] = prompt_str
                if split == 'train_rationale_augment':
                    if "input" not in line or len(line["input"]) == 0:
                        template = (
                            "Below is an instruction that describes a task. "
                            "Write a response that appropriately completes the request.\n\n"
                            "Please give a short explanation why for the response.\n\n"
                            "### Instruction:\n{instruction}\n\n### Response:\n"
                        )
                        prompt_str = template.format(instruction=line["instruction"])
                    else:    
                        # template = (
                        #     "Below is an instruction that describes a task. "
                        #     "Write a response that appropriately completes the request.\n\n"
                        #     "Please give a short explanation why for the response.\n\n"
                        #     "### Instruction:\n{instruction}\n\n### Response:\n"
                        #     "Let's think step by step.\n"
                        # )
                        template = (
                            "Below is an instruction that describes a task. "
                            "Write a response with reasoning step by step that appropriately completes the request.\n\n"
                            "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n"
                            "Let's think step by step.\n"
                        )
                        prompt_str = template.format(instruction=line["instruction"], input=line["input"])
                    line['prompt'] = prompt_str
            
                else:
                    if "input" not in line or len(line["input"]) == 0:
                        template = (
                            "Below is an instruction that describes a task. "
                            "Write a response that appropriately completes the request.\n\n"
                            "### Instruction:\n{instruction}\n\n### Response:\n"
                        )
                        prompt_str = template.format(instruction=line["instruction"])
                    else:
                        template = (
                            "Below is an instruction that describes a task, paired with an input that provides further context. "
                            "Write a response that appropriately completes the request.\n\n"
                            "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n"
                        )
                        prompt_str = template.format(instruction=line["instruction"], input=line["input"])
                    line['prompt'] = prompt_str
                    
                outfile.write(json.dumps(line) + '\n')
# "Please give a short explanation why for the response.\n\n""

# Cleanup
shutil.rmtree('alpaca_raw')
shutil.rmtree('self_instruct_raw')
shutil.rmtree('alpaca_split')
shutil.rmtree('self_instruct_split')
shutil.rmtree('dialogsum_split')

# def rename_and_move_files(folder_pairs):
#     for source_folder, dest_folder in folder_pairs:
#         old_path = os.path.join(source_folder, 'train.jsonl')
#         new_path = os.path.join(source_folder, 'train_rationale.jsonl')
#         if os.path.exists(old_path):
#             os.rename(old_path, new_path)
        
#         split_path = os.path.join(f"{source_folder}_split", 'train.jsonl')
#         dest_path = os.path.join(dest_folder, 'train.jsonl')
#         if os.path.exists(split_path):
#             shutil.move(split_path, dest_path)

# # Define the folder pairs
# folders = [
#     ('self_instruct', 'self_instruct'),
#     ('alpaca', 'alpaca'),
#     ('dialogsum', 'dialogsum')
# ]

# rename_and_move_files(folders)