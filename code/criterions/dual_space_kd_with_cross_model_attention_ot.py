import torch
import torch.nn as nn
from .various_divergence import VariousDivergence
import math
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
# class ETP(nn.Module):
#     def __init__(self, sinkhorn_alpha=0.1, stopThr=1e-9, OT_max_iter=1000, epsilon=1e-9, ot_dist_type='attention'):
#         super(ETP, self).__init__()
#         self.sinkhorn_alpha = sinkhorn_alpha
#         self.stopThr = stopThr
#         self.OT_max_iter = OT_max_iter
#         self.epsilon = epsilon
#         self.ot_dist_type = ot_dist_type
#     def forward(self, x, y):
#         if self.ot_dist_type == 'euclidean':
#             M = pairwise_euclidean_distance(x, y)
#         elif self.ot_dist_type == 'cosine':
#             M = pairwise_cosin_distance(x,y)
#         else:
#             M = pairwise_attention_distance(x,y)
           
#         device = M.device
#         # Sinkhorn's algorithm
#         a = (torch.ones(M.shape[0]) / M.shape[0]).unsqueeze(1).to(device)
#         b = (torch.ones(M.shape[1]) / M.shape[1]).unsqueeze(1).to(device)

#         u = (torch.ones_like(a) / a.size()[0]).to(device)  # Kx1

#         K = torch.exp(-M * self.sinkhorn_alpha)
#         err = 1
#         cpt = 0
#         while err > self.stopThr and cpt < self.OT_max_iter:
#             v = torch.div(b, torch.matmul(K.t(), u) + self.epsilon)
#             u = torch.div(a, torch.matmul(K, v) + self.epsilon)
#             cpt += 1
#             if cpt % 50 == 1:
#                 bb = torch.mul(v, torch.matmul(K.t(), u))
#                 err = torch.norm(torch.sum(torch.abs(bb - b), dim=0), p=float('inf'))

#         transp = u * (K * v.T)
       
#         loss_ETP = torch.sum(transp * M)

#         return loss_ETP, transp

class DualSpaceKDWithCMA_OT(VariousDivergence):
    def __init__(self, args, padding_id=-100) -> None:
        super().__init__(args, padding_id=padding_id)
        self.args = args
        self.kd_rate = args.kd_rate
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
        # output_align_output_cot_first = []
        # output_align_output_cot_last = []
        # output_align_output_cot_teacher_first = []
        # output_align_output_cot_teacher_last = []

        # input_cot_data = []
        # attention_cot_mask = []
        # position_cot_ids = []

        # input_cot_data_teacher = []
        # attention_cot_mask_teacher = []
        # position_cot_ids_teacher = []

        # pad_mask_raw_cot_list = []
        # teacher_pad_mask_raw_cot_list = []
        # pad_mask_cot_list = []
        # teacher_pad_mask_cot_list = []

        # mask cot 
        target = output_data["label"]
        teacher_target = output_data[f"teacher_{distiller.teacher_model_type}_label"]
        
        # pad_mask_raw_cot = target.ne(self.padding_id)
        # teacher_pad_mask_raw_cot = teacher_target.ne(self.padding_id)
        
        # start here
        
        # target_cot = output_data["label_raw"]
        # teacher_target_cot = output_data[f"teacher_{distiller.teacher_model_type}_label_raw"]
        
        # pad_mask_cot = target_cot.ne(self.padding_id)
        # teacher_pad_mask_cot = teacher_target_cot.ne(self.padding_id)
        
        
        # for bs in range(len(input_data["input_raw_ids"])):
        #     if input_data["input_raw_ids"][bs][0] != -10000:                # Student model alignment data
        #         output_align_output_cot_first.append(outputs.hidden_states[0][bs]) 
        #         output_align_output_cot_last.append(outputs.hidden_states[-1][bs]) 

        #         input_cot_data.append(input_data["input_raw_ids"][bs])
        #         attention_cot_mask.append(input_data["attention_raw_mask"][bs])
        #         if "position_raw_ids" in input_data:
        #             position_cot_ids.append(input_data["position_raw_ids"][bs])
        #         else:
        #             position_cot_ids.append(None)
                    
        #         # Teacher model alignment data
        #         output_align_output_cot_teacher_first.append(teacher_outputs.hidden_states[0][bs])
        #         output_align_output_cot_teacher_last.append(teacher_outputs.hidden_states[-1][bs])

        #         input_cot_data_teacher.append(input_data[f"teacher_{distiller.teacher_model_type}_input_raw_ids"][bs])
        #         attention_cot_mask_teacher.append(input_data[f"teacher_{distiller.teacher_model_type}_attention_raw_mask"][bs])
        #         if f"teacher_{distiller.teacher_model_type}_position_raw_ids" in input_data:
        #                     position_cot_ids_teacher.append(input_data[f"teacher_{distiller.teacher_model_type}_position_raw_ids"][bs])
        #         else:
        #             position_cot_ids_teacher.append(None)
                    
        #         # Create the 4 masks
        #         pad_mask_raw_cot = target[bs].ne(self.padding_id)
        #         teacher_pad_mask_raw_cot = teacher_target[bs].ne(self.padding_id)
        #         pad_mask_cot = target_cot[bs].ne(self.padding_id)
        #         teacher_pad_mask_cot = teacher_target_cot[bs].ne(self.padding_id)

        #         # Append masks to the respective lists
        #         pad_mask_raw_cot_list.append(pad_mask_raw_cot)
        #         teacher_pad_mask_raw_cot_list.append(teacher_pad_mask_raw_cot)
        #         pad_mask_cot_list.append(pad_mask_cot)
        #         teacher_pad_mask_cot_list.append(teacher_pad_mask_cot)
        # # Convert collected data to tensors
        # output_align_output_cot_first = torch.stack(output_align_output_cot_first) if output_align_output_cot_first else None
        # output_align_output_cot_last = torch.stack(output_align_output_cot_last) if output_align_output_cot_last else None

        # output_align_output_cot_teacher_first = torch.stack(output_align_output_cot_teacher_first) if output_align_output_cot_teacher_first else None
        # output_align_output_cot_teacher_last = torch.stack(output_align_output_cot_teacher_last) if output_align_output_cot_teacher_last else None

        # input_cot_data = torch.stack(input_cot_data) if input_cot_data else None
        # attention_cot_mask = torch.stack(attention_cot_mask) if attention_cot_mask else None
        # position_cot_ids = (
        #     torch.stack(position_cot_ids) if position_cot_ids and all(x is not None for x in position_cot_ids) else None
        # )
        
        # input_cot_data_teacher = torch.stack(input_cot_data_teacher) if input_cot_data_teacher else None
        # attention_cot_mask_teacher = torch.stack(attention_cot_mask_teacher) if attention_cot_mask_teacher else None
        # position_cot_ids_teacher = (
        #     torch.stack(position_cot_ids_teacher) if position_cot_ids_teacher and all(x is not None for x in position_cot_ids_teacher) else None
        # )
        
        # # Stack the masks
        # pad_mask_raw_cot = torch.stack(pad_mask_raw_cot_list) if pad_mask_raw_cot_list else None
        # teacher_pad_mask_raw_cot = torch.stack(teacher_pad_mask_raw_cot_list) if teacher_pad_mask_raw_cot_list else None
        # pad_mask_cot = torch.stack(pad_mask_cot_list) if pad_mask_cot_list else None
        # teacher_pad_mask_cot = torch.stack(teacher_pad_mask_cot_list) if teacher_pad_mask_cot_list else None
        
        # if input_cot_data is not None:
        #     outputs_cot = model(
        #         input_ids=input_cot_data,
        #         attention_mask=attention_cot_mask,
        #         position_ids=position_cot_ids,
        #         output_hidden_states=True
        #     )
        # else:
        #     outputs_cot = None

        # if input_cot_data_teacher is not None:
        #     with torch.no_grad():
        #         teacher_outputs_cot = teacher_model(
        #             input_ids=input_cot_data_teacher,
        #             attention_mask=attention_cot_mask_teacher,
        #             position_ids=position_cot_ids_teacher,
        #             output_hidden_states=True
        #         )
        # else:
        #     teacher_outputs_cot = None

        # Compute cross-entropy loss for the main outputs
        logits = outputs.logits
        loss_ce = self.compute_cross_entropy_loss(
            logits,
            output_data["label"],
            log=log
        )[0]

        hidden_state_student = outputs.hidden_states[-1]  # (batch_size, seq_len_student, hidden_dim_student)
        hidden_state_student_first = outputs.hidden_states[0]
        # hidden_state_student_cot = outputs_cot.hidden_states[-1]
        # hidden_state_student_first_cot = outputs_cot.hidden_states[0]
        
        hidden_state_teacher = teacher_outputs.hidden_states[-1]  # (batch_size, seq_len_teacher, hidden_dim_teacher)
        hidden_state_teacher_first = teacher_outputs.hidden_states[0]
        # hidden_state_teacher_cot = teacher_outputs_cot.hidden_states[-1]
        # hidden_state_teacher_first_cot  = teacher_outputs_cot.hidden_states[0]
        
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
        
        # if output_align_output_cot_first is not None:
        #     # y1 raw, y2 cot 
        #     ot_loss_last_cot, log = self.compute_etp_loss(distiller, hidden_state_student_cot, output_align_output_cot_teacher_last, pad_mask_cot, teacher_pad_mask_raw_cot, log, logits = True)
        #     ot_loss_first_cot, log = self.compute_etp_loss(distiller, hidden_state_student_first_cot, output_align_output_cot_teacher_first, pad_mask_cot, teacher_pad_mask_raw_cot, log, logits = True)
            
        #     ot_loss += (ot_loss_last_cot + ot_loss_first_cot)
            
        #     # y1 cot, y2 raw 
        #     ot_loss_last_cot, log = self.compute_etp_loss(distiller, output_align_output_cot_last, hidden_state_teacher_cot, pad_mask_raw_cot, teacher_pad_mask_cot, log, logits = True)
        #     ot_loss_first_cot, log = self.compute_etp_loss(distiller, output_align_output_cot_first, hidden_state_teacher_first_cot, pad_mask_raw_cot, teacher_pad_mask_cot, log, logits = True)
            
        #     ot_loss += (ot_loss_last_cot + ot_loss_first_cot)
        
        kd_loss, log = self.compute_dual_space_kd_loss_with_cma(
            outputs, teacher_outputs, input_data, output_data, distiller, log
        )

        loss = (1.0 - self.kd_rate) * (loss_ce) + self.kd_rate * (kd_loss + ot_loss * batch_denom)
        # loss = 0.001*loss_ce + 0.3*ot_loss + 0.9*kd_loss
        # loss = ot_loss + kd_loss
    
        # loss = 2*(ot_loss/3) + kd_loss
        log["loss"] = loss 
        log["ot_loss"] = ot_loss

        accuracy = self.compute_token_accuracy(
            logits, output_data["label"], 
        )
        log["accuracy"] = accuracy


        logging_output = self.record_logging_output(
            logging_output, batch_denom, log
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
        # loss =  total_loss 

        
        return loss, log
    
    def compute_dual_space_kd_loss_with_cma(
        self, outputs, teacher_outputs, input_data, output_data, distiller, log
    ):
        target = output_data["label"]
        teacher_target = output_data[f"teacher_{distiller.teacher_model_type}_label"]
        
        pad_mask = target.ne(self.padding_id)
        teacher_pad_mask = teacher_target.ne(self.padding_id)

        hiddens = outputs.hidden_states[-1]
        teacher_hiddens = teacher_outputs.hidden_states[-1]

        if hasattr(distiller.student_model, "model") \
            and hasattr(distiller.student_model.model, "embed_tokens"):
            stu_embed_tokens = distiller.student_model.model.embed_tokens
        elif hasattr(distiller.student_model, "model") \
            and hasattr(distiller.student_model.model, "model") \
            and hasattr(distiller.student_model.model.model, "embed_tokens"):
            stu_embed_tokens = distiller.student_model.model.model.embed_tokens
        elif hasattr(distiller.student_model, "transformer") \
            and hasattr(distiller.student_model.transformer, "wte"):
            stu_embed_tokens = distiller.student_model.transformer.wte
        else:
            raise NotImplementedError

        if hasattr(distiller.teacher_model, "model") \
            and hasattr(distiller.teacher_model.model, "embed_tokens"):
            tea_embed_tokens = distiller.teacher_model.model.embed_tokens
        elif hasattr(distiller.teacher_model, "model") \
            and hasattr(distiller.teacher_model.model, "model") \
            and hasattr(distiller.teacher_model.model.model, "embed_tokens"):
            tea_embed_tokens = distiller.teacher_model.model.model.embed_tokens
        elif hasattr(distiller.teacher_model, "transformer") \
            and hasattr(distiller.teacher_model.model, "wte"):
            tea_embed_tokens = distiller.teacher_model.transformer.wte
        else:
            raise NotImplementedError

        formal_target = torch.where(pad_mask, target, torch.zeros_like(target))
        formal_input = torch.where(pad_mask, input_data["input_ids"], torch.zeros_like(target))
        stu_input_embeds = stu_embed_tokens(formal_input).detach()
        stu_target_embeds = stu_embed_tokens(formal_target).detach()

        formal_teacher_target = torch.where(teacher_pad_mask, teacher_target, torch.zeros_like(teacher_target))
        formal_teacher_input = torch.where(teacher_pad_mask, input_data[f"teacher_{distiller.teacher_model_type}_input_ids"], torch.zeros_like(teacher_target))
        tea_input_embeds = tea_embed_tokens(formal_teacher_input).detach()
        tea_target_embeds = tea_embed_tokens(formal_teacher_target).detach()

        stu_index_embeds = torch.cat([stu_input_embeds, stu_target_embeds], -1)
        tea_index_embeds = torch.cat([tea_input_embeds, tea_target_embeds], -1)

        norm_tea_index_embeds = tea_index_embeds / tea_index_embeds.std()
        norm_tea_target_embeds = tea_target_embeds / tea_target_embeds.std()
        norm_teacher_hiddens = teacher_hiddens / teacher_hiddens.std()

        stu_q_hiddens = distiller.projectors["query"](stu_index_embeds).float()
        tea_k_hiddens = norm_tea_index_embeds.float()

        stu_v_hiddens = distiller.projectors["s2t"](hiddens).float()
        tea_v_hiddens = distiller.projectors["t2s"](
            norm_teacher_hiddens + norm_tea_target_embeds
        ).float()
        
        align = stu_q_hiddens.matmul(tea_k_hiddens.transpose(-1, -2))
        align = align / math.sqrt(2 * teacher_hiddens.shape[-1])
        align_mask = pad_mask.float().unsqueeze(-1) * teacher_pad_mask.float().unsqueeze(1)
        align = align + (1.0 - align_mask) * (-100000)

        t2s_weight = torch.softmax(align, -1)        
        t2s_hiddens = t2s_weight.matmul(tea_v_hiddens).to(hiddens)
        t2s_logits = t2s_hiddens.matmul(
            distiller.student_model.lm_head.weight.detach().transpose(-1, -2)
        )
        t2s_ce_loss = self.compute_cross_entropy_loss(t2s_logits, target)[0]
        t2s_acc_mask = t2s_logits.argmax(-1).eq(target)
        t2s_acc = (t2s_acc_mask * pad_mask).sum()
        max_probs = (t2s_logits.softmax(-1).max(-1)[0] * pad_mask).sum()
        log["t2s_ce_loss"] = t2s_ce_loss
        log["t2s_acc"] = t2s_acc
        log["max_t2s_prob"] = max_probs
        
        if not self.args.only_save_projector:  # skip if only train projectors (pre-train projectors)
            t2s_kd_loss = self.dist_func(
                outputs.logits, t2s_logits.detach(), target, reduction="none", use_tea_temp=True
            )
            t2s_kd_loss = (t2s_kd_loss * pad_mask * t2s_acc_mask).sum()

            s2t_weight = torch.softmax(align.transpose(-1, -2), -1)
            s2t_hiddens = s2t_weight.matmul(stu_v_hiddens).to(hiddens)
            s2t_logits = s2t_hiddens.matmul(
            distiller.teacher_model.lm_head.weight.detach().transpose(-1, -2)
            )

            s2t_kd_loss = self.compute_forward_kl_divergence(
                s2t_logits, teacher_outputs.logits, teacher_target, reduction="none"
            )
            s2t_kd_loss = (s2t_kd_loss * teacher_pad_mask).sum()
            s2t_acc = (s2t_logits.argmax(-1).eq(teacher_target) * teacher_pad_mask).sum() * pad_mask.sum() / teacher_pad_mask.sum()

            kd_loss = t2s_ce_loss + t2s_kd_loss + s2t_kd_loss
            log["t2s_kd_loss"] = t2s_kd_loss
            log["s2t_kd_loss"] = s2t_kd_loss
            log["s2t_acc"] = s2t_acc
        else:
            kd_loss = t2s_ce_loss

        log["kd_loss"] = kd_loss
        return kd_loss, log

