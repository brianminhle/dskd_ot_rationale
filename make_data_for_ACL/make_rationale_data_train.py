import json
import os 
# Base paths
PATH_TO_DATASET = '/mnt/bn/magellan-product-llm-data/tu.vu/matrix_one/dskd_ot_rationale/data/dialogsum'
TEACHER_PATH = '/mnt/bn/magellan-product-llm-data/tu.vu/matrix_one/dskd_ot_rationale/outputs/qwen/Qwen2.5-7B-Instruct/sft/raw/dialogsum/criterion=cross_entropy__lora-rank=256-alpha=8-dropout=0.1-bf16__epoch=10__bsz=16x1x4=64__lr=0.001'
# biến model_type cũng cần phải thay đổi để có train_rationale_mistral
MODEL_TYPE="qwen"    # gpt2, qwen, mistral, llama2

# Input file paths
train_cot_path = os.path.join(PATH_TO_DATASET, 'train_rationale_augment.jsonl')
# train_old_path = os.path.join(PATH_TO_DATASET, 'train.jsonl')
train_path = os.path.join(PATH_TO_DATASET, 'train.jsonl')
answer_path = os.path.join(TEACHER_PATH, 'answers.jsonl')
output_path = os.path.join(PATH_TO_DATASET,  f'train_rationale_{MODEL_TYPE}.jsonl')

# if os.path.exists(train_path):
#     print(f"Warning: {train_path} already exists")
# else:
#     os.rename(train_old_path, train_path)
#     print(f"Successfully renamed {train_old_path} to {train_path}")
# answer_path = '/home/aac/DSKD/outputs/mistral/mistral-7b-v0.1/sft/criterion=cross_entropylora-rank=256-alpha=8-dropout=0.1-bf16epoch=10bsz=8x1x4=32lr=0.001/answers.jsonl'

# Read train_cot.jsonl
with open(train_cot_path, 'r', encoding = 'utf-8') as f:
    train_cot_data = [json.loads(line) for line in f]

# Read answer.jsonl
with open(answer_path, 'r', encoding = 'utf-8') as f:
    answer_data = [json.loads(line) for line in f]

with open(train_path, 'r', encoding = 'utf-8') as f:
    raw_data = [json.loads(line) for line in f]
    
    
# Ensure both files have the same length
assert len(train_cot_data) == len(answer_data), "Mismatch in number of records between the files."
assert len(answer_data) == len(raw_data)


# Combine the files
combined_data = []

# train_cot_data 
# instruction
# prompt 
# input 
# output 

# answer
# only text 

# raw_data
# instruction
# prompt 
# input
# output 

# combined record 
# instruction
# prompt 
# input
# output 
# raw_prompt 
# raw_output 

for train_record, answer_record, raw_record in zip(train_cot_data, answer_data, raw_data):
    if raw_record['input'].strip() == '':
        new_record = {} 
        new_record['instruction'] = raw_record['instruction']
        new_record['prompt'] = raw_record['prompt']
        new_record['input'] = raw_record['input']
        new_record['output'] = raw_record['output']
        new_record['prompt_raw'] = ''
        new_record ['output_raw'] = '' 
        combined_data.append(new_record)
    else:

        # neu input khac rong thi phai vut 
        new_record = {}
        new_record['instruction'] = raw_record['instruction']
        new_record['prompt'] = raw_record['prompt']
        new_record['input'] = raw_record['input']
        new_record['output'] = raw_record['output']
        # new_record['prompt_raw'] = raw_record['prompt']
        # new_record ['output_raw'] = raw_record['output']
        new_record['prompt_raw'] = ''
        new_record ['output_raw'] = '' 
        combined_data.append(new_record)
        
        new_record = {}
        new_record['instruction'] = raw_record['instruction']
        new_record['prompt'] = train_record['prompt']
        new_record['input'] = raw_record['input']
        new_record['output'] = answer_record['text'] + '\nTherefore, the answer is: ' + raw_record['output']
        new_record['prompt_raw'] = raw_record['prompt']
        new_record ['output_raw'] = raw_record['output']

        combined_data.append(new_record)

                
        # new_record
        # create a new then append  to raw_record 
        
        # raw_record['input'] = 
    # train_record['cot_output'] = answer_record['text']  # Add the 'cot_output' attribute
    # train_record['cot_prompt'] = train_record['prompt']
    # train_record['prompt'] = raw_record['prompt']

    # combined_data.append(train_record)

# Write the combined data to a new file
with open(output_path, 'w') as f:
    for record in combined_data:
        f.write(json.dumps(record) + '\n')

print(f"Combined data written to {output_path}")


# instruction
# prompt
# input : check xem input co empty hay ko
# output :
# cot_prompt :
# cot_output :    
    
# data moi 
# instruction
# prompt : prompt la ca cau binh thuong lan cau kia 
# input : check xem input co empty hay ko
# output :
# prompt_raw :
# output_raw : 