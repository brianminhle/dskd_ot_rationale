from .cross_entropy_loss import CrossEntropyLoss
from .various_divergence import VariousDivergence
from .dual_space_kd import DualSpaceKD
from .dual_space_kd_with_cross_model_attention import DualSpaceKDWithCMA
from .universal_logit_distillation import UniversalLogitDistillation
from .min_edit_dis_kld import MinEditDisForwardKLD


from .dual_space_kd_with_cross_model_attention_ot_rationale import DualSpaceKDWithCMA_OT_Rationale
from .universal_logit_distillation_ot_rationale import UniversalLogitDistillation_OT_Rationale
from .min_edit_dis_kld_ot_rationale import MinEditDisForwardKLD_OT_Rationale


from .dual_space_kd_with_cross_model_attention_ot import DualSpaceKDWithCMA_OT

criterion_list = {
    "cross_entropy": CrossEntropyLoss,
    "various_divergence": VariousDivergence,
    "dual_space_kd": DualSpaceKD,
    "dual_space_kd_with_cma": DualSpaceKDWithCMA,
    "universal_logit_distillation": UniversalLogitDistillation,
    "min_edit_dis_kld": MinEditDisForwardKLD,
    "dual_space_kd_with_cma_ot_rationale": DualSpaceKDWithCMA_OT_Rationale,
    "universal_logit_distillation_ot_rationale": UniversalLogitDistillation_OT_Rationale,
    "min_edit_dis_kld_ot_rationale": MinEditDisForwardKLD_OT_Rationale,
    "dual_space_kd_with_cma_ot": DualSpaceKDWithCMA_OT,
}

def build_criterion(args):
    if criterion_list.get(args.criterion, None) is not None:
        return criterion_list[args.criterion](args)
    else:
        raise NameError(f"Undefined criterion for {args.criterion}!")