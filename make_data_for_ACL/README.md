step1:
+) run: final_make_data.py
=> 3 folders 
- alpaca
- dialogsum
- self_instruct

step2:
- DSKD/code/evaluate_main.py line 28 
def prepare_dataset_main(args, tokenizer): 
     data = {} 
     data["test"] = PromptDataset( 
         args,  
         tokenizer,  
         "valid",  
         data_path=args.data_dir,  
         num=args.dev_num 
     ) 
  
     return data 

change to 

def prepare_dataset_main(args, tokenizer): 
     data = {} 
     data["test"] = PromptDataset( 
         args,  
         tokenizer,  
         "train",  
         data_path=args.data_dir,  
         num=args.dev_num 
     ) 
  
     return data 

- run DSKD/scripts/eval/eval_main_lora_gen_rationale.sh

After running the script, you will find a file named answers.jsonl in the directory of the teacher checkpoint, which records the responses of the teacher corresponding to the training data. You can replace the "output" of the training data train.jsonl with the"text" in answers.jsonl to get the new training data

step 3:

- run make_rationale_data_train.py
 change name dataset in PATH_TO_DATASET to make dataset 

step 4:
- change DSKD/code/evaluate_main.py line 28 to valid again 
