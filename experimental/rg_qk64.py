"""RG-QK64-1 single candidate; original default arithmetic remains untouched."""
import torch
from experimental.rg_equivalent import wide_state as original_wide_state

STUDY='RG-QK64-1'


def qk64_state(q,k,v,mask=None,scale=None):
    if q.ndim!=4 or k.ndim!=4 or k.shape!=v.shape:
        raise ValueError('expected B,H,T,D tensors and matching K/V')
    if q.shape[0]!=k.shape[0] or q.shape[-1]!=k.shape[-1] or q.shape[1]%k.shape[1]:
        raise ValueError('incompatible attention dimensions')
    if q.device!=k.device or k.device!=v.device or q.dtype!=k.dtype or k.dtype!=v.dtype:
        raise ValueError('one partition requires matching dtype/device')
    if q.dtype==torch.float64 or k.shape[-2]==0:
        return original_wide_state(q,k,v,mask,scale)
    groups=q.shape[1]//k.shape[1]
    keys=k.double().repeat_interleave(groups,dim=1)
    values=v.float().repeat_interleave(groups,dim=1)
    scores=((q.double()@keys.transpose(-1,-2))*(q.shape[-1]**-.5 if scale is None else scale)).float()
    if mask is not None:
        if mask.dtype==torch.bool:scores=scores.masked_fill(~mask,-torch.inf)
        else:
            excluded=torch.isneginf(mask)|(mask<=torch.finfo(mask.dtype).min/2)
            scores=(scores+mask.float()).masked_fill(excluded,-torch.inf)
    lse=torch.logsumexp(scores,dim=-1);empty=torch.isneginf(lse)
    weights=torch.softmax(torch.where(empty[...,None],torch.zeros_like(scores),scores),dim=-1).masked_fill(empty[...,None],0)
    return weights@values,lse


def install():
    # Called only inside the separately labelled child process, before imports
    # of model/diagnostic harnesses. FP64 reference dispatch stays original.
    from experimental import rg_equivalent
    rg_equivalent.wide_state=qk64_state
