"""RG-FP64-ACC1: FP64 partition attention/merge, one FP32 output boundary.

BF16 KV storage is unchanged. This opt-in correctness candidate is not a speed
claim, native bitwise-equivalence claim, or change to the reference converter.
"""
import torch
from experimental.rg_equivalent import wide_state, stable_merge

STUDY='RG-FP64-ACC1'


def fp64_parts(q,kr,vr,kg,vg,mr=None,mg=None,scale=None):
    g=wide_state(q.double(),kg.double(),vg.double(),mg,scale)
    if kr.shape[-2]:
        r=wide_state(q.to(kr.device).double(),kr.double(),vr.double(),
                     None if mr is None else mr.to(kr.device),scale)
        merged=stable_merge(g,tuple(x.to(q.device) for x in r))
    else:merged=g
    # Identical declared conversion for matched GPU and CPU/GPU branches.
    # Reference FP64 calls remain FP64; production BF16 calls return FP32.
    return merged if q.dtype==torch.float64 else tuple(x.float() for x in merged)


def install():
    from experimental import rg_equivalent
    rg_equivalent.wide_parts=fp64_parts
