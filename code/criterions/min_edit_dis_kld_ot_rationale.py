import logging
import torch
import torch.distributed
import torch.nn.functional as F
import numpy as np
import transformers
import editdistance
from typing import Dict, List
from .various_divergence import VariousDivergence
import math

import torch.nn as nn

TOKENIZER_TO_SPECIAL_TOKEN = {
    transformers.LlamaTokenizer: "▁",
    transformers.LlamaTokenizerFast: "▁",
    transformers.GPTNeoXTokenizerFast: "Ġ",
    transformers.GPT2Tokenizer: "Ġ",
    transformers.GPT2TokenizerFast: "Ġ",
    transformers.Qwen2Tokenizer: "Ġ",
    transformers.Qwen2TokenizerFast: "Ġ",
}

def pairwise_euclidean_distance(x, y):
    return torch.cdist(x, y, p=2)  # Computes pairwise Euclidean distance
def pairwise_cosin_distance(a, b, eps=1e-8):
    """
    added eps for numerical stability
    """
    # a = a.float()
    # b = b.float()
    a_n, b_n = a.norm(dim=1)[:, None], b.norm(dim=1)[:, None]
    a_norm = a / torch.max(a_n, eps * torch.ones_like(a_n, dtype=torch.bfloat16))
    b_norm = b / torch.max(b_n, eps * torch.ones_like(b_n, dtype=torch.bfloat16))
    sim_mt = torch.mm(a_norm, b_norm.transpose(0, 1))
    
    sim_mt = 1 - sim_mt
    return sim_mt

def pairwise_attention_distance(x, y, eps=1e-8):
    # x = x.float()
    # y = y.float()
    
    d = x.shape[1]
   
    sim_mt = torch.mm(x, y.transpose(0, 1)) / math.sqrt(d)
    attention_weights = torch.softmax(sim_mt, dim=1)

    dist_mt = 1.0 - attention_weights
    return dist_mt

class ETP(nn.Module):
    def __init__(self, sinkhorn_alpha=0.1, stopThr=1e-9, OT_max_iter=100, epsilon=1e-9, ot_dist_type='attention'):
        super(ETP, self).__init__()
        self.sinkhorn_alpha = sinkhorn_alpha
        self.stopThr = stopThr
        self.OT_max_iter = OT_max_iter
        self.epsilon = epsilon
        self.ot_dist_type = ot_dist_type

    def forward(self, x, y):

        if self.ot_dist_type == 'euclidean':
            M = pairwise_euclidean_distance(x, y)
        elif self.ot_dist_type == 'cosine':
            M = pairwise_cosin_distance(x, y)
        else:
            M = pairwise_attention_distance(x, y)
        
        device = M.device
        # Sinkhorn's algorithm

        # Initialize a and b, also in bf16
        a = (torch.ones(M.shape[0]) / M.shape[0]).unsqueeze(1).to(device).to(dtype=torch.bfloat16)
        b = (torch.ones(M.shape[1]) / M.shape[1]).unsqueeze(1).to(device).to(dtype=torch.bfloat16)

        u = (torch.ones_like(a) / a.size()[0]).to(device)

        # K matrix
        K = torch.exp(-M * self.sinkhorn_alpha)
        err = 1
        cpt = 0
        while err > self.stopThr and cpt < self.OT_max_iter:
            v = torch.div(b, torch.matmul(K.t(), u) + self.epsilon)
            u = torch.div(a, torch.matmul(K, v) + self.epsilon)
            cpt += 1
            if cpt % 50 == 1:
                bb = torch.mul(v, torch.matmul(K.t(), u))
                err = torch.norm(torch.sum(torch.abs(bb - b), dim=0), p=float('inf'))

        transp = u * (K * v.T)

        loss_ETP = torch.sum(transp * M)

        return loss_ETP, transp
    


class MinEditDisForwardKLD_OT_Rationale(VariousDivergence):
    def __init__(self, args, padding_id=-100) -> None:
        super(MinEditDisForwardKLD_OT_Rationale, self).__init__(args, padding_id=padding_id)
        self.kd_rate = args.kd_rate
        self.kd_temp = args.kd_temperature
        self.etp = ETP()

    
    def forward(
        self, 
        distiller, 
        input_data, 
        output_data, 
        logging_output, 
        batch_denom,
    ):
        model = distiller.student_model
        teacher_model = distiller.teacher_model
        self.distiller = distiller

        with torch.no_grad():
            teacher_model.eval()
            teacher_outputs = teacher_model(
                input_ids=input_data[f"teacher_{distiller.teacher_model_type}_input_ids"],
                attention_mask=input_data[f"teacher_{distiller.teacher_model_type}_attention_mask"],
                position_ids=input_data.get(f"teacher_{distiller.teacher_model_type}_position_ids", None),
                output_hidden_states=True
            )

        outputs = model(
            input_ids=input_data["input_ids"],
            attention_mask=input_data["attention_mask"],
            position_ids=input_data.get("position_ids", None),
            output_hidden_states=True
        )
        log = {}
        output_align_output_cot_first = []
        output_align_output_cot_last = []
        output_align_output_cot_teacher_first = []
        output_align_output_cot_teacher_last = []

        input_cot_data = []
        attention_cot_mask = []
        position_cot_ids = []

        input_cot_data_teacher = []
        attention_cot_mask_teacher = []
        position_cot_ids_teacher = []

        pad_mask_raw_cot_list = []
        teacher_pad_mask_raw_cot_list = []
        pad_mask_cot_list = []
        teacher_pad_mask_cot_list = []

        # mask cot 
        target = output_data["label"]
        teacher_target = output_data[f"teacher_{distiller.teacher_model_type}_label"]
         
        # start here
        
        target_cot = output_data["label_raw"]
        teacher_target_cot = output_data[f"teacher_{distiller.teacher_model_type}_label_raw"]
        
        for bs in range(len(input_data["input_raw_ids"])):
            if input_data["input_raw_ids"][bs][0] != -10000:                # Student model alignment data
                output_align_output_cot_first.append(outputs.hidden_states[0][bs]) 
                output_align_output_cot_last.append(outputs.hidden_states[-1][bs]) 

                input_cot_data.append(input_data["input_raw_ids"][bs])
                attention_cot_mask.append(input_data["attention_raw_mask"][bs])
                if "position_raw_ids" in input_data:
                    position_cot_ids.append(input_data["position_raw_ids"][bs])
                else:
                    position_cot_ids.append(None)
                    
                # Teacher model alignment data
                output_align_output_cot_teacher_first.append(teacher_outputs.hidden_states[0][bs])
                output_align_output_cot_teacher_last.append(teacher_outputs.hidden_states[-1][bs])

                input_cot_data_teacher.append(input_data[f"teacher_{distiller.teacher_model_type}_input_raw_ids"][bs])
                attention_cot_mask_teacher.append(input_data[f"teacher_{distiller.teacher_model_type}_attention_raw_mask"][bs])
                if f"teacher_{distiller.teacher_model_type}_position_raw_ids" in input_data:
                            position_cot_ids_teacher.append(input_data[f"teacher_{distiller.teacher_model_type}_position_raw_ids"][bs])
                else:
                    position_cot_ids_teacher.append(None)
                    
                # Create the 4 masks
                pad_mask_raw_cot = target[bs].ne(self.padding_id)
                teacher_pad_mask_raw_cot = teacher_target[bs].ne(self.padding_id)
                pad_mask_cot = target_cot[bs].ne(self.padding_id)
                teacher_pad_mask_cot = teacher_target_cot[bs].ne(self.padding_id)

                # Append masks to the respective lists
                pad_mask_raw_cot_list.append(pad_mask_raw_cot)
                teacher_pad_mask_raw_cot_list.append(teacher_pad_mask_raw_cot)
                pad_mask_cot_list.append(pad_mask_cot)
                teacher_pad_mask_cot_list.append(teacher_pad_mask_cot)
        # Convert collected data to tensors
        output_align_output_cot_first = torch.stack(output_align_output_cot_first) if output_align_output_cot_first else None
        output_align_output_cot_last = torch.stack(output_align_output_cot_last) if output_align_output_cot_last else None

        output_align_output_cot_teacher_first = torch.stack(output_align_output_cot_teacher_first) if output_align_output_cot_teacher_first else None
        output_align_output_cot_teacher_last = torch.stack(output_align_output_cot_teacher_last) if output_align_output_cot_teacher_last else None

        input_cot_data = torch.stack(input_cot_data) if input_cot_data else None
        attention_cot_mask = torch.stack(attention_cot_mask) if attention_cot_mask else None
        position_cot_ids = (
            torch.stack(position_cot_ids) if position_cot_ids and all(x is not None for x in position_cot_ids) else None
        )
        
        input_cot_data_teacher = torch.stack(input_cot_data_teacher) if input_cot_data_teacher else None
        attention_cot_mask_teacher = torch.stack(attention_cot_mask_teacher) if attention_cot_mask_teacher else None
        position_cot_ids_teacher = (
            torch.stack(position_cot_ids_teacher) if position_cot_ids_teacher and all(x is not None for x in position_cot_ids_teacher) else None
        )
        
        # Stack the masks
        pad_mask_raw_cot = torch.stack(pad_mask_raw_cot_list) if pad_mask_raw_cot_list else None
        teacher_pad_mask_raw_cot = torch.stack(teacher_pad_mask_raw_cot_list) if teacher_pad_mask_raw_cot_list else None
        pad_mask_cot = torch.stack(pad_mask_cot_list) if pad_mask_cot_list else None
        teacher_pad_mask_cot = torch.stack(teacher_pad_mask_cot_list) if teacher_pad_mask_cot_list else None
        
        if input_cot_data is not None:
            outputs_cot = model(
                input_ids=input_cot_data,
                attention_mask=attention_cot_mask,
                position_ids=position_cot_ids,
                output_hidden_states=True
            )
        else:
            outputs_cot = None

        if input_cot_data_teacher is not None:
            with torch.no_grad():
                teacher_outputs_cot = teacher_model(
                    input_ids=input_cot_data_teacher,
                    attention_mask=attention_cot_mask_teacher,
                    position_ids=position_cot_ids_teacher,
                    output_hidden_states=True
                )
        else:
            teacher_outputs_cot = None

        logits = outputs.logits
        loss_ce = self.compute_cross_entropy_loss(
            logits,
            output_data["label"],
            log=log
        )[0]

        hidden_state_student = outputs.hidden_states[-1]  # (batch_size, seq_len_student, hidden_dim_student)
        hidden_state_student_first = outputs.hidden_states[0]
        hidden_state_student_cot = outputs_cot.hidden_states[-1]
        hidden_state_student_first_cot = outputs_cot.hidden_states[0]
        
        hidden_state_teacher = teacher_outputs.hidden_states[-1]  # (batch_size, seq_len_teacher, hidden_dim_teacher)
        hidden_state_teacher_first = teacher_outputs.hidden_states[0]
        hidden_state_teacher_cot = teacher_outputs_cot.hidden_states[-1]
        hidden_state_teacher_first_cot  = teacher_outputs_cot.hidden_states[0]
        
        ot_loss = 0.0
        # normal ot 
        target = output_data["label"]
        teacher_target = output_data[f"teacher_{distiller.teacher_model_type}_label"]
        
        pad_mask = target.ne(self.padding_id)
        teacher_pad_mask = teacher_target.ne(self.padding_id)
        
        ot_loss_last, log = self.compute_etp_loss(distiller, hidden_state_student, hidden_state_teacher, pad_mask, teacher_pad_mask, log)
        ot_loss_first, log = self.compute_etp_loss(distiller, hidden_state_student_first, hidden_state_teacher_first, pad_mask, teacher_pad_mask, log)
        
        ot_loss += (ot_loss_first + ot_loss_last)
        # ot raw output and cot output 
        
        # y1 cot, y2 cot 
        # y1 cot, y2 raw 
        # y1 raw, y2 cot 
        
        # pad_mask_raw_cot   y1 raw
        # teacher_pad_mask_raw_cot y2 raw 
        # pad_mask_cot      y1 cot
        # teacher_pad_mask_cot y2 cot  
        
        if output_align_output_cot_first is not None:
            # y1 raw, y2 cot 
            ot_loss_last_cot, log = self.compute_etp_loss(distiller, hidden_state_student_cot, output_align_output_cot_teacher_last, pad_mask_cot, teacher_pad_mask_raw_cot, log, logits = True)
            ot_loss_first_cot, log = self.compute_etp_loss(distiller, hidden_state_student_first_cot, output_align_output_cot_teacher_first, pad_mask_cot, teacher_pad_mask_raw_cot, log, logits = True)
            
            ot_loss += (ot_loss_last_cot + ot_loss_first_cot)
            
            # y1 cot, y2 raw 
            ot_loss_last_cot, log = self.compute_etp_loss(distiller, output_align_output_cot_last, hidden_state_teacher_cot, pad_mask_raw_cot, teacher_pad_mask_cot, log, logits = True)
            ot_loss_first_cot, log = self.compute_etp_loss(distiller, output_align_output_cot_first, hidden_state_teacher_first_cot, pad_mask_raw_cot, teacher_pad_mask_cot, log, logits = True)
            
            ot_loss += (ot_loss_last_cot + ot_loss_first_cot)
            
            
            
        teacher_logits = self.get_aligned_teacher_logits(
            logits, 
            teacher_outputs.logits, 
            input_data,
            output_data,
            distiller
        )
        target = output_data["label"]
        teacher_target = output_data[f"teacher_{distiller.teacher_model_type}_label"]
        
        kd_loss = self.compute_forward_kl_divergence(
            logits, 
            teacher_logits, 
            output_data["label"],
            log=log
        )

        loss = (1.0 - self.kd_rate) * (loss_ce) + self.kd_rate * (kd_loss + ot_loss*batch_denom)
        log["loss"] = loss

        accuracy = self.compute_token_accuracy(
            logits, 
            output_data["label"], 
        )
        log["accuracy"] = accuracy

        logging_output = self.record_logging_output(
            logging_output, 
            batch_denom,
            log
        )
        return loss / batch_denom, logging_output

    def compute_etp_loss(
            self, distiller ,student_outputs, teacher_outputs, attention_mask_student, attention_mask_teacher, log, logits = False
        ):
            """
            Compute OT loss between teacher and student outputs
            
            Args:
                teacher_outputs: tensor of shape (batch_size, seq_len1, input_dim)
                student_outputs: tensor of shape (batch_size, seq_len2, output_dim)
                
            Returns:
                loss: scalar tensor
            """
        
            if logits:
                teacher_outputs = distiller.projectors["ot"](teacher_outputs)
            else:
                teacher_outputs = distiller.projectors["ot"](teacher_outputs)

            batch_size = teacher_outputs.size(0)
            total_loss = 0
            for b in range(batch_size):
                # Get sequences for current batch
                teacher_seq = teacher_outputs[b]
                student_seq = student_outputs[b]

                teacher_mask = attention_mask_teacher[b]  # (seq_len1)
                student_mask = attention_mask_student[b]  # (seq_len2)
                
                # Prune sequences based on the mask
                teacher_seq = teacher_seq[teacher_mask.bool()]  # Shape: (valid_seq_len1, hidden_dim)
                student_seq = student_seq[student_mask.bool()]  # Shape: (valid_seq_len2, hidden_dim)
                
                # Compute ETP loss
                etp_loss, transp = self.etp(student_seq, teacher_seq)
                total_loss += etp_loss

            loss =  total_loss / (batch_size*4)
            return loss, log
    
    
    def get_aligned_teacher_logits(
        self, logits, teacher_logits, input_data, output_data, distiller,
    ):
        target = output_data["label"]
        pad_mask = target.ne(self.padding_id)
        teacher_target = output_data[f"teacher_{distiller.teacher_model_type}_label"]
        target_ids = torch.where(
            pad_mask, 
            target, 
            torch.ones_like(target) * distiller.student_tokenizer.eos_token_id
        )
        stu_tokenizer = distiller.student_tokenizer
        tea_tokenizer = distiller.teacher_tokenizers[distiller.teacher_model_type]

        bsz = target.shape[0]
        aligned_tea_logits = []
        for i in range(bsz):
            stu_content_idx = torch.nonzero(target[i].ne(self.padding_id)).view(-1)
            stu_input_ids = input_data["input_ids"][i, stu_content_idx]
            stu_target_ids = target_ids[i, stu_content_idx]

            tea_content_idx = torch.nonzero(teacher_target[i].ne(self.padding_id)).view(-1)
            tea_input_ids = input_data[f"teacher_{distiller.teacher_model_type}_input_ids"][i, tea_content_idx]

            stu_per_step_logits = logits[i, stu_content_idx, :].float()
            tea_per_step_logits = teacher_logits[i, tea_content_idx, :].float()   # [slen, vocab]

            aligned_tea_content_per_step_logits = self.transform_step_logits_fast(
                stu_tokenizer,
                tea_tokenizer,
                stu_input_ids,
                stu_per_step_logits,
                stu_target_ids,
                tea_input_ids,
                tea_per_step_logits,
                blending_to_base_mapping=distiller.tea2stu_id_mapping,
                base_to_blending_mapping_blending_ids=distiller.stu2tea_id_mapping_tea,
                base_to_blending_mapping_base_ids=distiller.stu2tea_id_mapping_stu
            )

            aligned_tea_per_step_logits = logits[i].float().detach()
            aligned_tea_per_step_logits[stu_content_idx] = aligned_tea_content_per_step_logits
            aligned_tea_logits.append(aligned_tea_per_step_logits)
        
        aligned_tea_logits = torch.stack(aligned_tea_logits, 0)
        return aligned_tea_logits

    def transform_step_logits_fast(
        self,
        base_model_tokenizer: transformers.tokenization_utils_base.PreTrainedTokenizerBase,
        blending_model_tokenizer: transformers.tokenization_utils_base.PreTrainedTokenizerBase,
        base_model_input_ids: torch.LongTensor,
        base_model_per_step_logits: torch.FloatTensor,
        base_model_target_ids: torch.LongTensor,
        blending_model_input_ids: torch.LongTensor,
        blending_model_per_step_logits: torch.FloatTensor,
        blending_to_base_mapping: torch.LongTensor = None,
        base_to_blending_mapping_blending_ids: torch.LongTensor = None,
        base_to_blending_mapping_base_ids: torch.LongTensor = None,
        device: str = None,
    ):
        """faster implementation to align logits"""
        base_model_tokens = base_model_tokenizer.convert_ids_to_tokens(base_model_input_ids)
        blending_model_tokens = blending_model_tokenizer.convert_ids_to_tokens(
            blending_model_input_ids
        )
        base_model_special_token = TOKENIZER_TO_SPECIAL_TOKEN[
            base_model_tokenizer.__class__
        ]
        blending_model_special_token = TOKENIZER_TO_SPECIAL_TOKEN[
            blending_model_tokenizer.__class__
        ]

        def dist_fn(a, b):
            """Calculate editdistance between two tokens, a is from blending model, b is from base model."""
            aa = a.replace(blending_model_special_token, "")
            bb = b.replace(base_model_special_token, "")
            dist = editdistance.eval(aa, bb)
            return dist

        # obtain sequence token alignment (each stu token to which tea token)
        _, _, _, base_to_blending, _ = self.dtw(
            blending_model_tokens, base_model_tokens, norm_func=dist_fn
        ) 
        unalign_mask = [1 if len(a) == 1 else 0 for a in base_to_blending]
        unalign_mask = torch.tensor(unalign_mask).to(base_model_input_ids.device)

        # for one-to-one mapping, align their logits; for one-to-many mapping, use ground-truth one-hot target
        base_to_blending = [a[0] if len(a) == 1 else 0 for a in base_to_blending]
        base_to_blending = torch.LongTensor(base_to_blending).to(base_model_input_ids.device)
        # for one-to-one mapping, ensure they are really similar
        unalign_mask = unalign_mask & base_model_input_ids.eq(blending_to_base_mapping[blending_model_input_ids[base_to_blending]])
        # get the logits of mapped tea tokens
        blending_model_per_step_logits = blending_model_per_step_logits[base_to_blending]
        blending_model_per_step_logits = blending_model_per_step_logits[
            :, base_to_blending_mapping_blending_ids.view(-1)
        ]
        blending_model_per_step_logits = blending_model_per_step_logits.view(
            -1, 
            base_to_blending_mapping_blending_ids.shape[0], 
            base_to_blending_mapping_blending_ids.shape[1]
        ).max(-1)[0]
        # transform teacher logits to student logits
        blending_to_base_logits = torch.ones_like(base_model_per_step_logits) * (-100000)
        blending_to_base_logits[:, base_to_blending_mapping_base_ids] = blending_model_per_step_logits
        
        unalign_mask = unalign_mask \
                     & blending_to_base_logits.max(-1)[0].ne(-100000)
        # mask unaligned position, use ground-truth target (one-hot)
        one_hot_logits = F.one_hot(base_model_target_ids, num_classes=base_model_per_step_logits.shape[-1])
        one_hot_logits = (1 - one_hot_logits) * (-100000) + (one_hot_logits) * 100
        
        unalign_mask = unalign_mask.unsqueeze(-1)
        blending_to_base_logits = torch.where(
            unalign_mask.repeat(1, base_model_per_step_logits.shape[-1]).eq(1),
            blending_to_base_logits,
            one_hot_logits
        )

        return blending_to_base_logits


    def transform_step_logits(
        self,
        base_model_tokenizer: transformers.tokenization_utils_base.PreTrainedTokenizerBase,
        blending_model_tokenizer: transformers.tokenization_utils_base.PreTrainedTokenizerBase,
        base_model_vocab: Dict[str, int],
        base_model_input_ids: List[int],
        blending_model_input_ids: List[int],
        blending_model_per_step_logits: List[List[float]],
        blending_model_per_step_indices: List[List[int]],
        vocab_align_type: str = "hard",
        blending_to_base_mapping: Dict[str, str] = None,
    ):
        """Align blending model per step logits & indices with base model. (original implementation in FuseLLM)"""
        base_model_tokens = base_model_tokenizer.convert_ids_to_tokens(base_model_input_ids)
        blending_model_tokens = blending_model_tokenizer.convert_ids_to_tokens(
            blending_model_input_ids
        )
        base_model_special_token = TOKENIZER_TO_SPECIAL_TOKEN[
            base_model_tokenizer.__class__
        ]
        blending_model_special_token = TOKENIZER_TO_SPECIAL_TOKEN[
            blending_model_tokenizer.__class__
        ]

        def dist_fn(a, b):
            """Calculate editdistance between two tokens, a is from blending model, b is from base model."""
            aa = a.replace(blending_model_special_token, "")
            bb = b.replace(base_model_special_token, "")
            dist = editdistance.eval(aa, bb)
            return dist

        _, _, _, base_to_blending, _ = self.dtw(
            blending_model_tokens, base_model_tokens, norm_func=dist_fn
        )
        aligned_blending_model_per_step_logits, aligned_blending_model_per_step_indices = (
            [],
            [],
        )
        for i, blending_idx in enumerate(base_to_blending):
            aligned_blending_model_per_step_logit = []
            aligned_blending_model_per_step_index = []
            if len(blending_idx) == 1:  # one base token map to one blending token
                j = blending_idx[0]
                base_token = base_model_tokens[i]
                blending_token = blending_model_tokens[j].replace(
                    blending_model_special_token, base_model_special_token
                )
                if (
                    (
                        blending_model_tokenizer.__class__
                        == transformers.GPTNeoXTokenizerFast
                        or blending_model_tokenizer.__class__
                        == transformers.GPT2TokenizerFast
                    )
                    and i == 0
                    and base_token.startswith(base_model_special_token)
                    and not blending_token.startswith(base_model_special_token)
                ):
                    blending_token = (
                        base_model_special_token + blending_token
                    )  # special case for mpt
                if vocab_align_type == "hard":
                    if (
                        base_token == blending_token
                    ):  # find the aligned mapping, use the corresponding logits
                        # the logits and indices at this step
                        for blending_logit, blending_index in zip(
                            blending_model_per_step_logits[j],
                            blending_model_per_step_indices[j],
                        ):
                            # the token corresponds to the logit and indices
                            blending_t = blending_model_tokenizer.convert_ids_to_tokens(
                                [blending_index]
                            )[0].replace(
                                blending_model_special_token, base_model_special_token
                            )
                            if blending_t in base_model_vocab:
                                aligned_index = base_model_vocab[
                                    blending_t
                                ]  # the index of the token in base model vocab
                                if (
                                    aligned_index
                                    not in aligned_blending_model_per_step_index
                                ):
                                    aligned_blending_model_per_step_index.append(
                                        aligned_index
                                    )
                                    aligned_blending_model_per_step_logit.append(
                                        blending_logit
                                    )
                    else:  # find error aligned mapping, use the one-hot logits
                        aligned_blending_model_per_step_index.append(
                            base_model_vocab[base_token]
                        )
                        aligned_blending_model_per_step_logit.append(1.0)
                elif vocab_align_type == "soft":
                    if (base_token == blending_token) or (
                        blending_token in blending_to_base_mapping
                        and base_token == blending_to_base_mapping[blending_token]
                    ):  # find the aligned mapping, use the corresponding logits
                        # the logits and indices at this step
                        for blending_logit, blending_index in zip(
                            blending_model_per_step_logits[j],
                            blending_model_per_step_indices[j],
                        ):  
                            # the token corresponds to the logit and indices
                            blending_t = blending_model_tokenizer.convert_ids_to_tokens(
                                [blending_index]
                            )[0].replace(
                                blending_model_special_token, base_model_special_token
                            )
                            blending_t = blending_to_base_mapping[blending_t]
                            if blending_t in base_model_vocab:
                                aligned_index = base_model_vocab[
                                    blending_t
                                ]  # the index of the token in base model vocab
                                if (
                                    aligned_index
                                    not in aligned_blending_model_per_step_index
                                ):
                                    aligned_blending_model_per_step_index.append(
                                        aligned_index
                                    )
                                    aligned_blending_model_per_step_logit.append(
                                        blending_logit
                                    )
                            else:
                                logging.warning(
                                    f"blending_t: {blending_t} not in base_model_vocab!"
                                )
                    else:  # find error aligned mapping, use the one-hot logits
                        aligned_blending_model_per_step_index.append(
                            base_model_vocab[base_token]
                        )
                        aligned_blending_model_per_step_logit.append(1.0)
                else:
                    logging.warning(
                        f"The vocab_align_type: '{vocab_align_type}' is not support!"
                    )
                    raise NotImplementedError
            else:  # one base token map to multiple blending token, in this case only fit base token. use the one-hot logits
                base_token = base_model_tokens[i]
                aligned_blending_model_per_step_index.append(base_model_vocab[base_token])
                aligned_blending_model_per_step_logit.append(1.0)
            aligned_blending_model_per_step_indices.append(
                aligned_blending_model_per_step_index
            )
            aligned_blending_model_per_step_logits.append(
                aligned_blending_model_per_step_logit
            )
        return (
            aligned_blending_model_per_step_logits,
            aligned_blending_model_per_step_indices,
        )
    
    def dtw(self, series_1, series_2, norm_func=np.linalg.norm):
        """Use dynamic time wrapping to align to tokenizers, modified from:
        https://github.com/talcs/simpledtw/blob/master/simpledtw.py"""
        matrix = np.zeros((len(series_1) + 1, len(series_2) + 1))
        matrix[0, :] = np.inf
        matrix[:, 0] = np.inf
        matrix[0, 0] = 0
        for i, vec1 in enumerate(series_1):
            for j, vec2 in enumerate(series_2):
                cost = norm_func(vec1, vec2)
                matrix[i + 1, j + 1] = cost + min(
                    matrix[i, j + 1], matrix[i + 1, j], matrix[i, j]
                )
        matrix = matrix[1:, 1:]
        i = matrix.shape[0] - 1
        j = matrix.shape[1] - 1
        matches = []
        mappings_series_1 = [list() for v in range(matrix.shape[0])]
        mappings_series_2 = [list() for v in range(matrix.shape[1])]
        while i > 0 or j > 0:
            matches.append((i, j))
            mappings_series_1[i].append(j)
            mappings_series_2[j].append(i)
            option_diag = matrix[i - 1, j - 1] if i > 0 and j > 0 else np.inf
            option_up = matrix[i - 1, j] if i > 0 else np.inf
            option_left = matrix[i, j - 1] if j > 0 else np.inf
            move = np.argmin([option_diag, option_up, option_left])
            if move == 0:
                i -= 1
                j -= 1
            elif move == 1:
                i -= 1
            else:
                j -= 1
        matches.append((0, 0))
        mappings_series_1[0].append(0)
        mappings_series_2[0].append(0)
        matches.reverse()
        for mp in mappings_series_1:
            mp.reverse()
        for mp in mappings_series_2:
            mp.reverse()

        return matches, matrix[-1, -1], mappings_series_1, mappings_series_2, matrix
