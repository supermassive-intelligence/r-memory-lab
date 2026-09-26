"""RG-NE-v1: wide accumulation, distinct from the historical bitwise track.

Model KV storage remains BF16 and unmodified. QK, softmax, PV and merging
use FP32 (FP64 for an explicitly FP64 reference); cast only at the boundary.
"""
import torch

CONTRACT = {
    'id': 'RG-NE-v1',
    'precast_relative_l2_max': 1e-5,
    'precast_relative_linf_max': 1e-5,
    'bf16_relative_l2_max': 1e-3,
    'bf16_relative_linf_max': 1 / 128,
    'normalization_floor': 1e-12,
    'quality_loss_margin': 0.01,
    'evidence_gain_target': 0.05,
    'legacy_bitwise_admission_unchanged': True,
}

# Additional matched-placement check, fixed before the corrected GPU run.
# PyTorch 2.9 assert_close FP32 defaults; not a universal attention standard.
PLACEMENT_CONTRACT = {'id': 'RG-placement-v1', 'precast_atol': 1e-5,
                      'precast_rtol': 1.3e-6, 'all_elements_required': True}


def stable_merge(a, b):
    """Normalize relative weights before forming O; avoid exp(LSE-logaddexp).

    At LSE around 1,000, rounding logaddexp before subtracting produces a
    common multiplicative error. The original exact-identity path is untouched.
    """
    oa, la = a
    ob, lb = b
    if oa.shape != ob.shape or la.shape != lb.shape or oa.shape[:-1] != la.shape:
        raise ValueError('attention state shape mismatch')
    maximum = torch.maximum(la, lb)
    safe = torch.where(torch.isneginf(maximum), torch.zeros_like(maximum), maximum)
    wa, wb = torch.exp(la - safe), torch.exp(lb - safe)
    total = wa + wb
    denominator = torch.where(total == 0, torch.ones_like(total), total)
    out = (wa / denominator)[..., None] * oa + (wb / denominator)[..., None] * ob
    out = torch.where(torch.isneginf(la)[..., None], ob, out)
    out = torch.where(torch.isneginf(lb)[..., None], oa, out)
    return out, maximum + torch.log(total)


def wide_state(q, k, v, mask=None, scale=None):
    if q.ndim != 4 or k.ndim != 4 or k.shape != v.shape:
        raise ValueError('expected B,H,T,D tensors and matching K/V')
    if q.shape[0] != k.shape[0] or q.shape[-1] != k.shape[-1] or q.shape[1] % k.shape[1]:
        raise ValueError('incompatible attention dimensions')
    if q.device != k.device or k.device != v.device or q.dtype != k.dtype or k.dtype != v.dtype:
        raise ValueError('one partition requires matching dtype/device')
    acc = torch.float64 if q.dtype == torch.float64 else torch.float32
    if k.shape[-2] == 0:
        return (torch.zeros(q.shape, dtype=acc, device=q.device),
                torch.full(q.shape[:-1], -torch.inf, dtype=acc, device=q.device))
    groups = q.shape[1] // k.shape[1]
    keys = k.to(acc).repeat_interleave(groups, dim=1)
    values = v.to(acc).repeat_interleave(groups, dim=1)
    scores = (q.to(acc) @ keys.transpose(-1, -2)) * (q.shape[-1] ** -.5 if scale is None else scale)
    if mask is not None:
        if mask.dtype == torch.bool:
            scores = scores.masked_fill(~mask, -torch.inf)
        else:
            # HF causal masks use finite dtype-min sentinels. Preserve exclusion
            # semantics when a partition is wholly masked, rather than uniform
            # softmax over its sentinel values. Ordinary additive biases remain.
            excluded = torch.isneginf(mask) | (mask <= torch.finfo(mask.dtype).min / 2)
            scores = (scores + mask.to(acc)).masked_fill(excluded, -torch.inf)
    lse = torch.logsumexp(scores, dim=-1)
    empty = torch.isneginf(lse)
    safe = torch.where(empty[..., None], torch.zeros_like(scores), scores)
    weights = torch.softmax(safe, dim=-1).masked_fill(empty[..., None], 0)
    return weights @ values, lse


def wide_parts(q, kr, vr, kg, vg, mr=None, mg=None, scale=None):
    g = wide_state(q, kg, vg, mg, scale)
    if kr.shape[-2] == 0:
        return g
    r = wide_state(q.to(kr.device), kr, vr,
                   None if mr is None else mr.to(kr.device), scale)
    return stable_merge(g, tuple(x.to(q.device) for x in r))


def relative_errors(actual, reference):
    if actual.shape != reference.shape:
        raise ValueError('comparison shape mismatch')
    a, r = actual.double(), reference.to(actual.device).double()
    delta = (a - r).abs()
    floor = CONTRACT['normalization_floor']
    return dict(max_absolute=float(delta.max()), mean_absolute=float(delta.mean()),
        relative_l1=float(delta.sum() / r.abs().sum().clamp_min(floor)),
        relative_l2=float(torch.linalg.vector_norm(delta) / torch.linalg.vector_norm(r).clamp_min(floor)),
        relative_linf=float(delta.max() / r.abs().max().clamp_min(floor)),
        finite=bool(torch.isfinite(a).all() and torch.isfinite(r).all()))


def numerical_pass(precast, rounded):
    return bool(precast['finite'] and rounded['finite']
        and precast['relative_l2'] <= CONTRACT['precast_relative_l2_max']
        and precast['relative_linf'] <= CONTRACT['precast_relative_linf_max']
        and rounded['relative_l2'] <= CONTRACT['bf16_relative_l2_max']
        and rounded['relative_linf'] <= CONTRACT['bf16_relative_linf_max'])


def placement_errors(actual, reference):
    result = relative_errors(actual, reference)
    delta = (actual.double() - reference.double()).abs()
    allowed = PLACEMENT_CONTRACT['precast_atol'] + PLACEMENT_CONTRACT['precast_rtol'] * reference.double().abs()
    result.update(elements=delta.numel(), violating_elements=int((delta > allowed).sum()),
                  worst_tolerance_ratio=float((delta / allowed).max()))
    result['pass'] = result['finite'] and result['violating_elements'] == 0
    return result
