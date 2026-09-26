"""Inference-only attention-state prototype; no storage, retrieval or scheduling.

Inputs are already position-encoded and carry the caller's original mask.
Natural-log LSE throughout. CPU placement changes execution, not exposure.
"""
from __future__ import annotations

import torch


def attention_state(q, k, v, mask=None, scale=None):
    """Return normalized output and log normalizer, B,Hq,Tq,D / B,Hq,Tq."""
    if q.ndim != 4 or k.ndim != 4 or k.shape != v.shape:
        raise ValueError('expected B,H,T,D tensors and matching K/V')
    if q.shape[0] != k.shape[0] or q.shape[-1] != k.shape[-1]:
        raise ValueError('batch/head dimension mismatch')
    if q.shape[1] % k.shape[1]:
        raise ValueError('query heads must be divisible by KV heads')
    if q.device != k.device or k.device != v.device or q.dtype != k.dtype or k.dtype != v.dtype:
        raise ValueError('one partition must use one dtype and device')
    acc = torch.float64 if q.dtype == torch.float64 else torch.float32
    if k.shape[-2] == 0:
        return torch.zeros_like(q), torch.full(q.shape[:-1], -torch.inf, dtype=acc, device=q.device)
    nrep = q.shape[1] // k.shape[1]
    # Same GQA ordering as Qwen2 repeat_kv. This eager prototype may materialize.
    k = k.repeat_interleave(nrep, dim=1)
    v = v.repeat_interleave(nrep, dim=1)
    scores = (q @ k.transpose(-1, -2)) * (q.shape[-1] ** -.5 if scale is None else scale)
    if mask is not None:
        if mask.dtype == torch.bool:
            scores = scores.masked_fill(~mask, -torch.inf)
        else:
            scores = scores + mask
    scores = scores.to(acc)
    lse = torch.logsumexp(scores, dim=-1)
    empty = torch.isneginf(lse)
    # Avoid NaN for entirely masked partitions (including both partitions empty).
    safe = torch.where(empty[..., None], torch.zeros_like(scores), scores)
    weights = torch.softmax(safe, dim=-1).masked_fill(empty[..., None], 0).to(q.dtype)
    return weights @ v, lse


def merge_states(a, b):
    oa, la = a
    ob, lb = b
    if oa.shape != ob.shape or la.shape != lb.shape or oa.shape[:-1] != la.shape:
        raise ValueError('attention state shape mismatch')
    if oa.device != ob.device or la.device != oa.device or lb.device != oa.device:
        raise ValueError('merge states must share a device')
    lse = torch.logaddexp(la, lb)
    safe = torch.where(torch.isneginf(lse), torch.zeros_like(lse), lse)
    wa, wb = torch.exp(la - safe)[..., None], torch.exp(lb - safe)[..., None]
    out = (wa * oa + wb * ob).to(oa.dtype)
    # Empty-side identity is an exact contract, not a rounded multiply-by-one.
    out = torch.where(torch.isneginf(la)[..., None], ob, out)
    out = torch.where(torch.isneginf(lb)[..., None], oa, out)
    return out, lse


def attend_parts(q, kr, vr, kg, vg, mr=None, mg=None, scale=None):
    """R KV is resident where kr lives. Transfer Q out, then O/LSE back.

Synchronous diagnostic: no claim that CPU/GPU work overlaps yet.
"""
    g = attention_state(q, kg, vg, mg, scale)
    if kr.shape[-2] == 0:
        return g
    r = attention_state(q.to(kr.device), kr, vr,
                        None if mr is None else mr.to(kr.device), scale)
    return merge_states(g, tuple(x.to(q.device) for x in r))
