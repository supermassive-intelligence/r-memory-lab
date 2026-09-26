"""Independent binary64 -> BF16 nearest/even reference conversion.

Reference instrumentation only. Does not change model or attention arithmetic.
Search exact BF16 endpoints in binary64 instead of rounding through float32.
"""
import torch
import os

_GRIDS={}


def rounded_reference(reference, dtype):
    """Explicit new instrument only; default legacy behavior is unchanged."""
    instrument=os.environ.get('RG_REFERENCE_INSTRUMENT','legacy')
    if instrument not in ('legacy','RG-RNE-VALID1'):
        raise ValueError('unknown reference instrument')
    if instrument=='RG-RNE-VALID1':
        if dtype!=torch.bfloat16:raise ValueError('RNE validation requires BF16')
        return direct_bf16(reference)
    return reference.to(dtype)


def direct_bf16(reference):
    if reference.dtype!=torch.float64:raise ValueError('binary64 reference required')
    device=reference.device
    if device not in _GRIDS:
        _GRIDS[device]=torch.arange(0,0x7f80,dtype=torch.int16,device=device).view(torch.bfloat16).double()
    grid=_GRIDS[device]
    absolute=reference.abs().contiguous()
    high=torch.searchsorted(grid,absolute).clamp(max=len(grid)-1)
    low=(high-1).clamp(min=0)
    lo,hi=grid[low],grid[high]
    dl,dh=(absolute-lo).abs(),(hi-absolute).abs()
    choose_high=(dh<dl)|((dh==dl)&((high&1)==0))
    codes=torch.where(choose_high,high,low)
    # Max finite BF16 + half its ULP; ties overflow to the even infinity code.
    codes=torch.where(absolute>=float.fromhex('0x1.ffp127'),0x7f80,codes)
    codes=torch.where(torch.isnan(reference),0x7fc0,codes)
    codes=codes.to(torch.int32)|(torch.signbit(reference).to(torch.int32)<<15)
    return codes.to(torch.int16).view(torch.bfloat16)
