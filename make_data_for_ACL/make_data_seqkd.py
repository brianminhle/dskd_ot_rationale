import json

def process_files(input_file, output_file):
    with open(input_file, 'r') as input_f, open(output_file, 'r') as output_f:
        processed_lines = []
        
        for input_line, output_line in zip(input_f, output_f):
            try:
                input_data = json.loads(input_line.strip())
                output_data = json.loads(output_line.strip())
                
                input_data['output'] = output_data['text']
                
                processed_lines.append(input_data)
            except json.JSONDecodeError as e:
                print(f"Error processing line: {e}")
        
        return processed_lines

# Paths
input_file = '/home/aac/DSKD/data/dolly/train.jsonl'
output_file = '/home/aac/DSKD/outputs/qwen257B_Instruct/Qwen2.5-7B-Instruct/sft/raw/dolly/criterion=cross_entropy__lora-rank=256-alpha=8-dropout=0.1-bf16__epoch=10__bsz=8x1x4=32__lr=0.001/answers.jsonl'

processed_data = process_files(input_file, output_file)

processed_output_file = '/home/aac/DSKD/data/dolly/train_seqkd.jsonl'
with open(processed_output_file, 'w') as f:
    for item in processed_data:
        f.write(json.dumps(item) + '\n')

print(f"Processed {len(processed_data)} lines.")

import os
# Rename the files
train_raw_file = '/home/aac/DSKD/data/dolly/train_raw.jsonl'
os.rename(input_file, train_raw_file)  # Rename train.jsonl to train_raw.jsonl

new_train_file = '/home/aac/DSKD/data/dolly/train.jsonl'
os.rename(processed_output_file, new_train_file)  # Rename train_seqkd.jsonl to train.jsonl

print("File renaming complete:")
print(f"{input_file} -> {train_raw_file}")
print(f"{processed_output_file} -> {new_train_file}")