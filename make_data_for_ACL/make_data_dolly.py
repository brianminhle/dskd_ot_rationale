# import json

# # Open and read the JSONL file
# x = 0
# y = 0

# with open('train.jsonl', 'r', encoding='utf-8') as file:
#     for line in file:
#         try:
#             d = json.loads(line.strip())
#             if d['input'].strip() != '':
#                 x += 1
#             else:
#                 y += 1
#         except json.JSONDecodeError as e:
#             print(f"Error decoding JSON line: {e}")

# print(f"Lines with non-empty input: {x}")
# print(f"Lines with empty input: {y}")


import json

path_to_dolly_data_folder = '/path/to/dolly/data/folder/'
# Input and output file paths
input_file = path_to_dolly_data_folder + 'train.jsonl'
output_file = path_to_dolly_data_folder + 'train_rationale_v1.jsonl'

# Open the input file for reading and the output file for writing
with open(input_file, 'r', encoding='utf-8') as infile, open(output_file, 'w', encoding='utf-8') as outfile:
    for line in infile:
        d = json.loads(line.strip())
        d['prompt'] = d['prompt'].strip() + "\nPlease give a short explanation why.\n"

        outfile.write(json.dumps(d) + '\n')

print(f"Updated file saved to {output_file}")
    