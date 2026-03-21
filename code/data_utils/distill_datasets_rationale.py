import torch
import os
import json
import numpy as np
from torch.utils.data import Dataset
import torch.distributed as dist
from tqdm import tqdm

from utils import log_rank
from typing import Dict, Optional
from transformers import AutoTokenizer


class DistillDatasetRationale(Dataset):
    def __init__(
        self, 
        args, 
        split: str,
        student_tokenizer: Dict[str, AutoTokenizer], 
        teacher_tokenizers: Optional[Dict[str, AutoTokenizer]] = {},
    ):
        self.args = args
        self.split = split
        self.student_tokenizer = student_tokenizer
        self.teacher_tokenizers = teacher_tokenizers
        self.max_length = args.max_length
        self.max_prompt_length = args.max_prompt_length
        self.dataset = self._load_and_process_data()
        # log_rank(f"Num of data instances: {len(self.dataset)}")

    def __len__(self):
        return len(self.dataset)
   
    def __getitem__(self, index):
        return self.dataset[index]
    
    def _load_and_process_data(self):
        dataset = []
        # if self.args.dataset = 
        # path = os.path.join(self.args.data_dir, f"{self.split}.jsonl")
        # self.split == 'train_rationale'
        if self.split == 'train_rationale':
            train_file_name = 'train_rationale_' + self.args.teacher_model_type
            path = os.path.join(self.args.data_dir, f"{train_file_name}.jsonl")
        else:
            path = os.path.join(self.args.data_dir, f"{self.split}.jsonl")

        # self.split la train dev test 
        # if self.args.dataset == 'rationale':
        #     train_file_name = 'train_rationale_' + self.args.teacher_model_type
        #     path = os.path.join(self.args.data_dir, f"{train_file_name}.jsonl")

        # else:
        #     train_file_name = 'train'
        #     path = os.path.join(self.args.data_dir, f"{train_file_name}.jsonl")
            
        if self.split == 'train_rationale':
            if os.path.exists(path):
                with open(path) as f:
                    raw_data = [json.loads(l) for l in f.readlines()]
                    self.answers = [x["output"] if isinstance(x["output"], list) else [x["output"]] for x in raw_data]
                
                log_rank("Processing dataset for student model (and all teacher models)...")  
                seg = np.iinfo(np.int32).max * 2 + 1        
                for data in tqdm(raw_data, disable=(dist.get_rank() != 0)):
                    # normal input
                    student_prompt_ids = self.student_tokenizer.encode(
                        data["prompt"], add_special_tokens=False
                    )
                    student_prompt_ids = student_prompt_ids[:self.max_prompt_length]
                    # cot input 
                    student_raw_prompt_ids = self.student_tokenizer.encode(
                        data["prompt_raw"], add_special_tokens=False
                    )
                    student_raw_prompt_ids = student_raw_prompt_ids[:self.max_prompt_length]
                    # normal output 
                    student_response_ids = self.student_tokenizer.encode(
                        data["output"], add_special_tokens=False
                    )
                    student_response_ids = student_response_ids \
                                     + [self.student_tokenizer.eos_token_id]
                    # cot output 
                    student_raw_response_ids = self.student_tokenizer.encode(
                        data["output_raw"], add_special_tokens=False
                    )
                    student_raw_response_ids = student_raw_response_ids \
                                        + [self.student_tokenizer.eos_token_id]

                    if data["input"].strip() == '' or data["prompt_raw"].strip() == '':
                        tokenized_data = {
                            "student_input_ids": student_prompt_ids + [seg] + student_response_ids,
                            "student_raw_input_ids": [-10000] + [seg]
                        }
                    else:
                        tokenized_data = {
                            "student_input_ids": student_prompt_ids + [seg] + student_response_ids,
                            "student_raw_input_ids": student_raw_prompt_ids + [seg] + student_raw_response_ids 
                        }
                    
            
                    for model_type in self.teacher_tokenizers:
                        if self.teacher_tokenizers[model_type] is None: continue
                        
                        # normal input
                        teacher_prompt_ids = self.teacher_tokenizers[model_type].encode(
                            data["prompt"], add_special_tokens=False
                        )
                        teacher_prompt_ids = teacher_prompt_ids[:self.max_prompt_length]
                        # cot_input 
                        teacher_raw_prompt_ids = self.teacher_tokenizers[model_type].encode(
                            data["prompt_raw"], add_special_tokens=False
                        )
                        teacher_raw_prompt_ids = teacher_raw_prompt_ids[:self.max_prompt_length]
                        # normal output
                        teacher_response_ids = self.teacher_tokenizers[model_type].encode(
                            data["output"], add_special_tokens=False
                        )
                        teacher_response_ids = teacher_response_ids \
                                                + [self.teacher_tokenizers[model_type].eos_token_id]
                        # cot output 
                        teacher_raw_response_ids = self.teacher_tokenizers[model_type].encode(
                            data["output_raw"], add_special_tokens=False
                        )
                        teacher_raw_response_ids = teacher_raw_response_ids \
                                                + [self.teacher_tokenizers[model_type].eos_token_id]
                         
                        if data["input"].strip() == '' or data["prompt_raw"].strip() == '':
                            tokenized_data[f"teacher_{model_type}_input_ids"] = \
                            teacher_prompt_ids + [seg] + teacher_response_ids
                        
                            tokenized_data[f"teacher_raw_{model_type}_input_ids"] = [-10000] + [seg]
                        else:  
                            # student_prompt_ids = [-10000]  # Mark empty input with special value
                            tokenized_data[f"teacher_{model_type}_input_ids"] = \
                                teacher_prompt_ids + [seg] + teacher_response_ids
                            
                            tokenized_data[f"teacher_raw_{model_type}_input_ids"] = \
                                teacher_raw_prompt_ids + [seg] + teacher_raw_response_ids 

                    dataset.append(tokenized_data)
                return dataset
            else:
                raise FileNotFoundError(f"No such file named {path}")
        else:
            if os.path.exists(path):
                with open(path) as f:
                    raw_data = [json.loads(l) for l in f.readlines()]
                    self.answers = [x["output"] if isinstance(x["output"], list) else [x["output"]] for x in raw_data]
                
                log_rank("Processing dataset for student model (and all teacher models)...")  
                seg = np.iinfo(np.int32).max * 2 + 1        
                for data in tqdm(raw_data, disable=(dist.get_rank() != 0)):
                    student_prompt_ids = self.student_tokenizer.encode(
                        data["prompt"], add_special_tokens=False
                    )
                    student_prompt_ids = student_prompt_ids[:self.max_prompt_length]
                    student_response_ids = self.student_tokenizer.encode(
                        data["output"], add_special_tokens=False
                    )
                    student_response_ids = student_response_ids \
                                        + [self.student_tokenizer.eos_token_id]
                    tokenized_data = {
                        "student_input_ids": student_prompt_ids + [seg] + student_response_ids,
                    }
            
                    for model_type in self.teacher_tokenizers:
                        if self.teacher_tokenizers[model_type] is None: continue
                            
                        teacher_prompt_ids = self.teacher_tokenizers[model_type].encode(
                            data["prompt"], add_special_tokens=False
                        )
                        teacher_prompt_ids = teacher_prompt_ids[:self.max_prompt_length]
                        teacher_response_ids = self.teacher_tokenizers[model_type].encode(
                            data["output"], add_special_tokens=False
                        )
                        teacher_response_ids = teacher_response_ids \
                                                + [self.teacher_tokenizers[model_type].eos_token_id]
                        tokenized_data[f"teacher_{model_type}_input_ids"] = \
                            teacher_prompt_ids + [seg] + teacher_response_ids

                    dataset.append(tokenized_data)
                return dataset
            else:
                raise FileNotFoundError(f"No such file named {path}")    

    def _process_lm(
        self, i, samp, model_data, no_model_data, gen_data, 
        teacher_model_data, teacher_no_model_data
    ):
        if self.split == 'train_rationale':
            seg = np.iinfo(np.int32).max * 2 + 1
            # normal input
            input_ids = np.array(samp["student_input_ids"])
            source_len = np.where(input_ids == seg)[0][0]
            prompt = input_ids[:source_len]
            input_ids = np.concatenate(
                [input_ids[:source_len], input_ids[source_len+1:]], axis=0
            )
            input_ids = input_ids[:self.max_length]
            input_len = len(input_ids)
            model_data["input_ids"][i][:input_len-1] = torch.tensor(input_ids[:-1], dtype=torch.long)
            model_data["attention_mask"][i][:input_len-1] = 1.0
            if self.args.model_type in ["gpt2"]:
                model_data["position_ids"][i][:input_len-1] = torch.arange(0, input_len-1, dtype=torch.long)
            no_model_data["label"][i][:input_len-1] = torch.tensor(input_ids[1:], dtype=torch.long)
            no_model_data["label"][i][:source_len-1] = -100
            no_model_data["loss_mask"][i][:input_len-1] = 1.0
            no_model_data["loss_mask"][i][:source_len-1] = 0
            
            gen_data["input_ids"][i][-len(prompt):] = torch.tensor(prompt, dtype=torch.long)
            gen_data["attention_mask"][i][-len(prompt):] = 1.0
            
            # cot input 
            input_cot_ids = np.array(samp["student_raw_input_ids"])

            source_cot_len = np.where(input_cot_ids == seg)[0][0]
            prompt_cot = input_cot_ids[:source_cot_len]
            input_cot_ids = np.concatenate(
                [input_cot_ids[:source_cot_len], input_cot_ids[source_cot_len+1:]]
            )
            input_cot_ids = input_cot_ids[:self.max_length]
            input_cot_len = len(input_cot_ids)
            model_data["input_raw_ids"][i][:input_cot_len-1] = torch.tensor(input_cot_ids[:-1], dtype=torch.long) 
            model_data["attention_raw_mask"][i][:input_cot_len-1] = 1.0 
            if self.args.model_type in ["gpt2"]:
                model_data["position_raw_ids"][i][:input_cot_len-1] = torch.arange(0, input_cot_len-1, dtype=torch.long)
            no_model_data["label_raw"][i][:input_cot_len-1] = torch.tensor(input_cot_ids[1:], dtype=torch.long)
            no_model_data["label_raw"][i][:source_cot_len-1] = -100
            no_model_data["loss_mask_raw"][i][:input_cot_len-1] = 1.0
            no_model_data["loss_mask_raw"][i][:source_cot_len-1] = 0   
            
            gen_data["input_raw_ids"][i][-len(prompt_cot):] = torch.tensor(prompt_cot, dtype=torch.long)
            gen_data["attention_raw_mask"][i][-len(prompt_cot):] = 1.0
            
            
            for model_type in self.teacher_tokenizers:
                t_input_ids = np.array(samp[f"teacher_{model_type}_input_ids"])
                t_source_len = np.where(t_input_ids == seg)[0][0]
                t_input_ids = np.concatenate(
                    [t_input_ids[:t_source_len], t_input_ids[t_source_len+1:]], axis=0
                )
                t_input_ids = t_input_ids[:self.max_length]
                t_input_len = len(t_input_ids)
                teacher_model_data[model_type]["input_ids"][i][:t_input_len-1] = \
                    torch.tensor(t_input_ids[:-1], dtype=torch.long)
                teacher_model_data[model_type]["attention_mask"][i][:t_input_len-1] = 1.0
                if model_type in ["gpt2"]:
                    teacher_model_data[model_type]["position_ids"][i][:t_input_len-1] = \
                        torch.arange(0, t_input_len-1, dtype=torch.long)
                teacher_no_model_data[model_type]["label"][i][:t_input_len-1] = \
                    torch.tensor(t_input_ids[1:], dtype=torch.long)
                teacher_no_model_data[model_type]["label"][i][:t_source_len-1] = -100
                teacher_no_model_data[model_type]["loss_mask"][i][:t_input_len-1] = 1.0
                teacher_no_model_data[model_type]["loss_mask"][i][:t_source_len-1] = 0
                
                
                t_cot_input_ids = np.array(samp[f"teacher_raw_{model_type}_input_ids"])
                t_cot_source_len = np.where(t_cot_input_ids == seg)[0][0]
                t_cot_input_ids = np.concatenate(
                    [t_cot_input_ids[:t_cot_source_len], t_cot_input_ids[t_cot_source_len+1:]], axis=0
                )
                t_cot_input_ids = t_cot_input_ids[:self.max_length]
                t_cot_input_len = len(t_cot_input_ids)
                teacher_model_data[model_type]["input_raw_ids"][i][:t_cot_input_len-1] = \
                    torch.tensor(t_cot_input_ids[:-1], dtype=torch.long)
                teacher_model_data[model_type]["attention_raw_mask"][i][:t_cot_input_len-1] = 1.0
                if model_type in ["gpt2"]:
                    teacher_model_data[model_type]["position_raw_ids"][i][:t_cot_input_len-1] = \
                        torch.arange(0, t_cot_input_len-1, dtype=torch.long)
                teacher_no_model_data[model_type]["label_raw"][i][:t_cot_input_len-1] = \
                    torch.tensor(t_cot_input_ids[1:], dtype=torch.long)
                teacher_no_model_data[model_type]["label_raw"][i][:t_cot_input_len-1] = -100
                teacher_no_model_data[model_type]["loss_mask_raw"][i][:t_cot_input_len-1] = 1.0
                teacher_no_model_data[model_type]["loss_mask_raw"][i][:t_cot_input_len-1] = 0
        else:
            seg = np.iinfo(np.int32).max * 2 + 1
            input_ids = np.array(samp["student_input_ids"])
            source_len = np.where(input_ids == seg)[0][0]
            prompt = input_ids[:source_len]
            input_ids = np.concatenate(
                [input_ids[:source_len], input_ids[source_len+1:]], axis=0
            )
            input_ids = input_ids[:self.max_length]
            input_len = len(input_ids)
            model_data["input_ids"][i][:input_len-1] = torch.tensor(input_ids[:-1], dtype=torch.long)
            model_data["attention_mask"][i][:input_len-1] = 1.0
            if self.args.model_type in ["gpt2"]:
                model_data["position_ids"][i][:input_len-1] = torch.arange(0, input_len-1, dtype=torch.long)
            no_model_data["label"][i][:input_len-1] = torch.tensor(input_ids[1:], dtype=torch.long)
            no_model_data["label"][i][:source_len-1] = -100
            no_model_data["loss_mask"][i][:input_len-1] = 1.0
            no_model_data["loss_mask"][i][:source_len-1] = 0
            
            gen_data["input_ids"][i][-len(prompt):] = torch.tensor(prompt, dtype=torch.long)
            gen_data["attention_mask"][i][-len(prompt):] = 1.0

            for model_type in self.teacher_tokenizers:
                t_input_ids = np.array(samp[f"teacher_{model_type}_input_ids"])
                t_source_len = np.where(t_input_ids == seg)[0][0]
                t_input_ids = np.concatenate(
                    [t_input_ids[:t_source_len], t_input_ids[t_source_len+1:]], axis=0
                )
                t_input_ids = t_input_ids[:self.max_length]
                t_input_len = len(t_input_ids)
                teacher_model_data[model_type]["input_ids"][i][:t_input_len-1] = \
                    torch.tensor(t_input_ids[:-1], dtype=torch.long)
                teacher_model_data[model_type]["attention_mask"][i][:t_input_len-1] = 1.0
                if model_type in ["gpt2"]:
                    teacher_model_data[model_type]["position_ids"][i][:t_input_len-1] = \
                        torch.arange(0, t_input_len-1, dtype=torch.long)
                teacher_no_model_data[model_type]["label"][i][:t_input_len-1] = \
                    torch.tensor(t_input_ids[1:], dtype=torch.long)
                teacher_no_model_data[model_type]["label"][i][:t_source_len-1] = -100
                teacher_no_model_data[model_type]["loss_mask"][i][:t_input_len-1] = 1.0
                teacher_no_model_data[model_type]["loss_mask"][i][:t_source_len-1] = 0

    def move_to_device(self, datazip, device):
        for data in datazip:
            for k in data:
                if isinstance(data[k], torch.Tensor):
                    data[k] = data[k].to(device)
                elif isinstance(data[k], dict):
                    for kk in data[k]:
                        data[k][kk] = data[k][kk].to(device)

    def collate(self, samples):
        if self.split == 'train_rationale':
            bs = len(samples)
            max_length = self.max_length

            model_data = {
                "input_ids": torch.ones(bs, max_length, dtype=torch.long) \
                            * self.student_tokenizer.eos_token_id,
                "attention_mask": torch.zeros(bs, max_length),
                "input_raw_ids": torch.ones(bs, max_length, dtype=torch.long) \
                            * self.student_tokenizer.eos_token_id,
                "attention_raw_mask": torch.zeros(bs, max_length),
            }
            
            if self.args.model_type in ["gpt2"]:
                model_data["position_ids"] = torch.zeros(bs, max_length, dtype=torch.long)
                model_data["position_raw_ids"] = torch.zeros(bs, max_length, dtype=torch.long)

                
            no_model_data = {
                "label": torch.ones(bs, max_length, dtype=torch.long) * -100,
                "loss_mask": torch.zeros(bs, max_length),
                "label_raw": torch.ones(bs, max_length, dtype=torch.long) * -100,
                "loss_mask_raw": torch.zeros(bs, max_length),        }
            
            gen_data = {
                "input_ids": torch.ones(bs, self.max_prompt_length, dtype=torch.long) \
                            * self.student_tokenizer.eos_token_id,
                "attention_mask": torch.zeros(bs, self.max_prompt_length, dtype=torch.long),
                "input_raw_ids": torch.ones(bs, self.max_prompt_length, dtype=torch.long) \
                            * self.student_tokenizer.eos_token_id,
                "attention_raw_mask": torch.zeros(bs, self.max_prompt_length, dtype=torch.long),
            }

            teacher_model_data = {
                model_type: {
                    "input_ids": torch.ones(bs, max_length, dtype=torch.long) \
                                * self.teacher_tokenizers[model_type].eos_token_id,
                    "attention_mask": torch.zeros(bs, max_length),
                    
                    "input_raw_ids": torch.ones(bs, max_length, dtype=torch.long) \
                                * self.teacher_tokenizers[model_type].eos_token_id,
                    "attention_raw_mask": torch.zeros(bs, max_length),
                } for model_type in self.teacher_tokenizers
            }

            for model_type in self.teacher_tokenizers:
                if model_type in ["gpt2"]:
                    teacher_model_data[model_type]["position_ids"] = torch.zeros(
                        bs, max_length, dtype=torch.long
                    )
                    teacher_model_data[model_type]["position_raw_ids"] = torch.zeros(
                        bs, max_length, dtype=torch.long
                    )

            teacher_no_model_data = {
                model_type: {
                    "label": torch.ones(bs, max_length, dtype=torch.long) * -100,
                    "loss_mask": torch.zeros(bs, max_length),
                    "label_raw": torch.ones(bs, max_length, dtype=torch.long) * -100,
                    "loss_mask_raw": torch.zeros(bs, max_length),
                } for model_type in self.teacher_tokenizers
            }

            for i, samp in enumerate(samples):
                self._process_lm(
                    i, samp, model_data, no_model_data, gen_data, 
                    teacher_model_data, teacher_no_model_data
                )

            for model_type in teacher_model_data:
                prefix = f"teacher_{model_type}_"
                for key in teacher_model_data[model_type]:
                    model_data[f"{prefix}{key}"] = teacher_model_data[model_type][key]
                    
                for key in teacher_no_model_data[model_type]:
                    no_model_data[f"{prefix}{key}"] = teacher_no_model_data[model_type][key]
            
            return model_data, no_model_data, gen_data
        else:
            bs = len(samples)
            max_length = self.max_length

            model_data = {
                "input_ids": torch.ones(bs, max_length, dtype=torch.long) \
                            * self.student_tokenizer.eos_token_id,
                "attention_mask": torch.zeros(bs, max_length),
            }
            
            if self.args.model_type in ["gpt2"]:
                model_data["position_ids"] = torch.zeros(bs, max_length, dtype=torch.long)
                
            no_model_data = {
                "label": torch.ones(bs, max_length, dtype=torch.long) * -100,
                "loss_mask": torch.zeros(bs, max_length)
            }
            
            gen_data = {
                "input_ids": torch.ones(bs, self.max_prompt_length, dtype=torch.long) \
                            * self.student_tokenizer.eos_token_id,
                "attention_mask": torch.zeros(bs, self.max_prompt_length, dtype=torch.long),
            }

            teacher_model_data = {
                model_type: {
                    "input_ids": torch.ones(bs, max_length, dtype=torch.long) \
                                * self.teacher_tokenizers[model_type].eos_token_id,
                    "attention_mask": torch.zeros(bs, max_length),
                } for model_type in self.teacher_tokenizers
            }

            for model_type in self.teacher_tokenizers:
                if model_type in ["gpt2"]:
                    teacher_model_data[model_type]["position_ids"] = torch.zeros(
                        bs, max_length, dtype=torch.long
                    )

            teacher_no_model_data = {
                model_type: {
                    "label": torch.ones(bs, max_length, dtype=torch.long) * -100,
                    "loss_mask": torch.zeros(bs, max_length),
                } for model_type in self.teacher_tokenizers
            }

            for i, samp in enumerate(samples):
                self._process_lm(
                    i, samp, model_data, no_model_data, gen_data, 
                    teacher_model_data, teacher_no_model_data
                )

            for model_type in teacher_model_data:
                prefix = f"teacher_{model_type}_"
                for key in teacher_model_data[model_type]:
                    model_data[f"{prefix}{key}"] = teacher_model_data[model_type][key]
                    
                for key in teacher_no_model_data[model_type]:
                    no_model_data[f"{prefix}{key}"] = teacher_no_model_data[model_type][key]
            
            return model_data, no_model_data, gen_data

# input len lon hon source len 
# input la ca output con source len la chi minh input 


# cái gen_data ko bao giờ dùng 
# train , dev , valid là tập test 

# model_data: Input data for the student model.
# no_model_data: Ground truth labels and loss masks for training the student model.
# gen_data: Prompt data for sequence generation tasks.
# teacher_model_data: Input data for teacher models.
# teacher_no_model_data: Labels and loss masks for teacher models.