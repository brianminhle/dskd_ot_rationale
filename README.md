+ EVAL 
run_eval_lora chỉnh tham số DATASET


+ TRAININIG 
chỉnh tham số 
dskd_cma_tinyllama.sh
- CRITERION # xem list criterion trong code/criterions/__init__.py
- KD_RATE=0.5 chỉnh hệ số trọng số KD , KD rate càng cao thì càng nghiêng về KD hơn 
- DATASET là co dung rationale hay ko , ví dụ raw, rationale
- DATASET_NAME là ten folder dataset , ví dụ dolly  
- SAVE_PATH="${BASE_PATH}/outputs/${CKPT_TYPE}/${CKPT_NAME}/${TASK}/${DATASET}/${DATASET_NAME}/${SETTING}" 
- >> ${SAVE_PATH}/abc.log 2>&1 & với abc.log là tên file log 

+ mới chỉnh folder scripts/tinyllama