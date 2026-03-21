import torch
from .cross_entropy_loss import CrossEntropyLoss
import math
from torch import nn

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

class VariousDivergence(CrossEntropyLoss):
    def __init__(self, args, padding_id=-100) -> None:
        super(VariousDivergence, self).__init__(args, padding_id=padding_id)
        self.kd_rate = args.kd_rate
        self.kd_temp = args.kd_temperature
        self.tea_temp = args.teacher_temperature
        self.kd_objective = args.kd_objective
        self.args = args
        self.etp = ETP()


        if self.kd_objective == "forward_kl":
            self.dist_func = self.compute_forward_kl_divergence
        elif self.kd_objective == "reverse_kl":
            self.dist_func = self.compute_reverse_kl_divergence
        elif self.kd_objective == "adaptive_kl":
            self.dist_func = self.compute_adaptive_kl_divergence
        elif self.kd_objective == "skewed_forward_kl":
            self.dist_func = self.compute_skewed_forward_kl_divergence
        elif self.kd_objective == "skewed_reverse_kl":
            self.dist_func = self.compute_skewed_reverse_kl_divergence
        elif self.kd_objective == "js_divergence":
            self.dist_func = self.compute_js_divergence
        else:
            raise NameError(f"Unsupported kd_objective for `{self.kd_objective}'")
    
    def forward(
        self, 
        distiller, 
        input_data, 
        output_data, 
        logging_output, 
        batch_denom, 
    ):
        self.distiller = distiller
        model = distiller.student_model
        teacher_model = distiller.teacher_model
        outputs = model(
            input_data["input_ids"],
            attention_mask=input_data["attention_mask"],
            position_ids=input_data.get("position_ids", None), 
            output_hidden_states=True
        )
        logits = outputs.logits
        log = {}
        loss_ce = self.compute_cross_entropy_loss(
            outputs.logits, output_data["label"], log=log
        )[0]

        with torch.no_grad():
            teacher_model.eval()
            teacher_outputs = teacher_model(
                input_data[f"teacher_{distiller.teacher_model_type}_input_ids"],
                attention_mask=input_data[f"teacher_{distiller.teacher_model_type}_attention_mask"],
                position_ids=input_data.get(f"teacher_{distiller.teacher_model_type}_position_ids", None), 
                output_hidden_states=True)
            teacher_logits = teacher_outputs.logits
        
        # Qwen has different vocab_size for models in different sizes (see https://github.com/QwenLM/Qwen/issues/419)
        if self.args.model_type == "qwen":
            logits = logits[..., :151851]
            teacher_logits = teacher_logits[..., :151851]
        


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
        
        # pad_mask_raw_cot = target.ne(self.padding_id)
        # teacher_pad_mask_raw_cot = teacher_target.ne(self.padding_id)
        
        # start here
        
        target_cot = output_data["label_raw"]
        teacher_target_cot = output_data[f"teacher_{distiller.teacher_model_type}_label_raw"]
        
        # pad_mask_cot = target_cot.ne(self.padding_id)
        # teacher_pad_mask_cot = teacher_target_cot.ne(self.padding_id)
        
        
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

        # Compute cross-entropy loss for the main outputs
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






        kd_loss = self.dist_func(logits, teacher_logits, output_data["label"])
        log["kd_loss"] = kd_loss

        loss = (1.0 - self.kd_rate) * loss_ce + self.kd_rate * (kd_loss + ot_loss*batch_denom)
        log["loss"] = loss

        accuracy = self.compute_token_accuracy(logits, output_data["label"])
        log["accuracy"] = accuracy

        if self.args.report_logits:
            self.record_logits(
                logits, 
                output_data["label"], 
                log, 
                teacher_logits=teacher_logits, 
                teacher_target=output_data[f"teacher_{distiller.teacher_model_type}_label"]
            )

        logging_output = self.record_logging_output(logging_output, batch_denom, log)
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

    def compute_forward_kl_divergence(
        self, 
        logits, 
        teacher_logits, 
        target, 
        reduction="sum", 
        log=None, 
        use_tea_temp=False
    ):
        logits = logits / self.kd_temp
        teacher_logits = teacher_logits / self.kd_temp
        teacher_logits = teacher_logits / self.tea_temp if use_tea_temp else teacher_logits

        lprobs = torch.log_softmax(logits, -1, dtype=torch.float32)
        teacher_probs = torch.softmax(teacher_logits, -1, dtype=torch.float32)
        teacher_lprobs = torch.log_softmax(teacher_logits, -1, dtype=torch.float32)
        kld = (teacher_probs * (teacher_lprobs - lprobs))
        inf_mask = logits.isinf()
        kld = kld.masked_fill_(inf_mask, 0.0).sum(-1)
        
        if reduction == "sum":
            pad_mask = target.eq(self.padding_id)
            kld = kld.masked_fill_(pad_mask, 0.0)
            kld = kld.sum()

            if log is not None:
                log["forward_kl"] = kld

        return kld
    
    def compute_reverse_kl_divergence(
        self, 
        logits, 
        teacher_logits, 
        target, 
        reduction="sum", 
        log=None, 
        use_tea_temp=False
    ):
        logits = logits / self.kd_temp
        teacher_logits = teacher_logits / self.kd_temp
        teacher_logits = teacher_logits / self.tea_temp if use_tea_temp else teacher_logits

        probs = torch.softmax(logits, -1, dtype=torch.float32)
        lprobs = torch.log_softmax(logits, -1, dtype=torch.float32)
        teacher_lprobs = torch.log_softmax(teacher_logits, -1, dtype=torch.float32)
        kld = (probs * (lprobs - teacher_lprobs))
        inf_mask = logits.isinf() | teacher_logits.isinf()
        kld = kld.masked_fill_(inf_mask, 0.0).sum(-1)

        if reduction == "sum":
            pad_mask = target.eq(self.padding_id)
            kld = kld.masked_fill_(pad_mask, 0.0)
            kld = kld.sum()

            if log is not None:
                log["reverse_kl"] = kld

        return kld
    
    def compute_adaptive_kl_divergence(
        self, 
        logits, 
        teacher_logits, 
        target, 
        reduction="sum", 
        log=None, 
        use_tea_temp=False
    ):
        alpha = self.args.adaptive_kl_alpha
        probs = torch.softmax(
            logits / self.kd_temp, dim=-1, dtype=torch.float32
        )
        if use_tea_temp:
            teacher_probs = torch.softmax(
                teacher_logits / self.tea_temp / self.kd_temp, dim=-1, dtype=torch.float32
            )
        else:
            teacher_probs = torch.softmax(
                teacher_logits / self.kd_temp, dim=-1, dtype=torch.float32
            )
        sorted_teacher_probs, sorted_idx = teacher_probs.sort(-1)
        sorted_probs = probs.gather(-1, sorted_idx)
        gap = (sorted_teacher_probs - sorted_probs).abs()
        cum_teacher_probs = torch.cumsum(sorted_teacher_probs, -1)
        tail_mask = cum_teacher_probs.lt(alpha).float()
        g_head = (gap * (1 - tail_mask)).sum(-1).detach()
        g_tail = (gap * tail_mask).sum(-1).detach()

        fkl = self.compute_forward_kl_divergence(logits, teacher_logits, target, reduction="none", use_tea_temp=use_tea_temp)
        rkl = self.compute_reverse_kl_divergence(logits, teacher_logits, target, reduction="none", use_tea_temp=use_tea_temp)

        akl = (g_head / (g_head + g_tail)) * fkl + (g_tail / (g_head + g_tail)) * rkl
        
        if reduction == "sum":
            pad_mask = target.eq(self.padding_id)
            akl = akl.masked_fill_(pad_mask, 0.0)
            akl = akl.sum()

            if log is not None:
                log["adaptive_kl"] = akl

        return akl
    
    def compute_skewed_forward_kl_divergence(
        self, 
        logits, 
        teacher_logits, 
        target, 
        reduction="sum", 
        log=None, 
        use_tea_temp=False
    ):
        logits = logits / self.kd_temp
        teacher_logits = teacher_logits / self.kd_temp
        teacher_logits = teacher_logits / self.tea_temp if use_tea_temp else teacher_logits

        student_probs = torch.softmax(logits, -1, dtype=torch.float32)
        teacher_probs = torch.softmax(teacher_logits, -1, dtype=torch.float32)
        mixed_probs = self.args.skew_lambda * teacher_probs + (1 - self.args.skew_lambda) * student_probs
        mixed_lprobs = torch.log(mixed_probs)
        teacher_lprobs = torch.log_softmax(teacher_logits, -1, dtype=torch.float32)
        kld = (teacher_probs * (teacher_lprobs - mixed_lprobs))
        inf_mask = logits.isinf() | teacher_logits.isinf()
        kld = kld.masked_fill_(inf_mask, 0.0).sum(-1)
        
        if reduction == "sum":
            pad_mask = target.eq(self.padding_id)
            kld = kld.masked_fill_(pad_mask, 0.0)
            kld = kld.sum()

            if log is not None:
                log["skewed_forward_kl"] = kld

        return kld
    
    def compute_skewed_reverse_kl_divergence(
        self, 
        logits, 
        teacher_logits, 
        target, 
        reduction="sum", 
        log=None, 
        use_tea_temp=False
    ):
        logits = logits / self.kd_temp
        teacher_logits = teacher_logits / self.kd_temp
        teacher_logits = teacher_logits / self.tea_temp if use_tea_temp else teacher_logits

        student_probs = torch.softmax(logits, -1, dtype=torch.float32)
        teacher_probs = torch.softmax(teacher_logits, -1, dtype=torch.float32)
        mixed_probs = (1 - self.args.skew_lambda) * teacher_probs + self.args.skew_lambda * student_probs
        mixed_lprobs = torch.log(mixed_probs)
        student_lprobs = torch.log_softmax(logits, -1, dtype=torch.float32)
        # teacher_lprobs = torch.log_softmax(teacher_logits / self.tea_temp / self.kd_temp, -1, dtype=torch.float32)
        kld = (student_probs * (student_lprobs - mixed_lprobs))
        inf_mask = logits.isinf() | teacher_logits.isinf()
        kld = kld.masked_fill_(inf_mask, 0.0).sum(-1)
        
        if reduction == "sum":
            pad_mask = target.eq(self.padding_id)
            kld = kld.masked_fill_(pad_mask, 0.0)
            kld = kld.sum()

            if log is not None:
                log["skewed_reverse_kl"] = kld

        return kld

    def compute_js_divergence(
        self, 
        logits, 
        teacher_logits, 
        target, 
        reduction="sum", 
        log=None, 
        use_tea_temp=False
    ):
        # temperature scaling
        logits = logits / self.kd_temp
        teacher_logits = teacher_logits / self.kd_temp
        teacher_logits = teacher_logits / self.tea_temp if use_tea_temp else teacher_logits

        probs = torch.softmax(logits, -1, dtype=torch.float32)
        teacher_probs = torch.softmax(teacher_logits, -1, dtype=torch.float32)
        m_probs = (probs + teacher_probs) / 2
        
        lprobs = torch.log(probs + 1e-9)
        teacher_lprobs = torch.log(teacher_probs + 1e-9)
        m_lprobs = torch.log(m_probs + 1e-9)

        kld1 = teacher_probs * (teacher_lprobs - m_lprobs)
        kld2 = probs * (lprobs - m_lprobs)
        kld = (kld1 + kld2) / 2
        inf_mask = logits.isinf() | teacher_logits.isinf()
        kld = kld.masked_fill_(inf_mask, 0.0).sum(-1)
        
        if reduction == "sum":
            pad_mask = target.eq(self.padding_id)
            kld = kld.masked_fill_(pad_mask, 0.0)
            kld = kld.sum()

            if log is not None:
                log["js_div"] = kld

        return kld
