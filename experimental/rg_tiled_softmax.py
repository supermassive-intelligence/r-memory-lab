"""Inference-only Qwen eager attention with row-tiled FP32 softmax.

QK and PV GEMMs, scale, mask, dtype and reduction length stay unchanged.
Only independent query rows of the softmax are evaluated in smaller batches.
This is an explicit instrument, never a default numerical correction.
Revision 2 reuses the fresh QK buffer for scale and same-dtype additive mask;
the corresponding out-of-place BF16 temporaries exceeded the 12 GiB cap.
"""
import torch
from torch import nn


def tiled_eager(module,query,key,value,attention_mask,scaling,dropout=0.,**kwargs):
    from transformers.models.qwen2.modeling_qwen2 import repeat_kv
    if torch.is_grad_enabled() or module.training or dropout:
        raise ValueError('row-tiled eager is declared for eval inference only')
    key_states=repeat_kv(key,module.num_key_value_groups)
    value_states=repeat_kv(value,module.num_key_value_groups)
    weights=torch.matmul(query,key_states.transpose(2,3))
    weights.mul_(scaling)
    if attention_mask is not None:
        if torch.promote_types(weights.dtype,attention_mask.dtype)==weights.dtype:
            weights.add_(attention_mask)
        else:
            weights=weights+attention_mask
    for start in range(0,query.shape[-2],128):
        block=weights[...,start:start+128,:]
        block.copy_(nn.functional.softmax(block,dim=-1,dtype=torch.float32).to(query.dtype))
    weights=nn.functional.dropout(weights,p=dropout,training=module.training)
    output=torch.matmul(weights,value_states).transpose(1,2).contiguous()
    return output,weights


def install():
    from transformers.models.qwen2 import modeling_qwen2 as qwen
    qwen.eager_attention_forward=tiled_eager
