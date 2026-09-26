"""Controlled arithmetic variants, diagnostics only; not production repairs."""
import torch

from experimental.rg_attention import merge_states


def bitwise_equal(actual, reference):
    """Exact logical tensor bytes, including signed zero; never cast dtype."""
    if actual.shape != reference.shape or actual.dtype != reference.dtype:
        return False
    a = actual.detach().contiguous().reshape(-1).view(torch.uint8)
    b = reference.detach().to(actual.device).contiguous().reshape(-1).view(torch.uint8)
    return torch.equal(a, b)


def score_state(scores, values, wide=False):
    """Use already-computed masked scores to remove QK-kernel confounding."""
    acc = torch.float64 if scores.dtype == torch.float64 else torch.float32
    if scores.shape[-1] == 0:
        shape = (*scores.shape[:-1], values.shape[-1])
        dtype = acc if wide else values.dtype
        return (torch.zeros(shape, device=scores.device, dtype=dtype),
                torch.full(scores.shape[:-1], -torch.inf, device=scores.device, dtype=acc))
    scores = scores.to(acc)
    lse = torch.logsumexp(scores, dim=-1)
    empty = torch.isneginf(lse)
    safe = torch.where(empty[..., None], torch.zeros_like(scores), scores)
    weights = torch.softmax(safe, -1).masked_fill(empty[..., None], 0)
    dtype = acc if wide else values.dtype
    return weights.to(dtype) @ values.to(dtype), lse


def fixed_score_split(scores, values, cut, wide=False):
    if not 0 <= cut <= scores.shape[-1]:
        raise ValueError('partition outside key range')
    a = score_state(scores[..., :cut], values[..., :cut, :], wide)
    b = score_state(scores[..., cut:], values[..., cut:, :], wide)
    return merge_states(a, b)[0].to(values.dtype)


def global_weight_split(scores, values, cut):
    """Round GLOBAL probabilities once, then split PV; isolates PV reduction."""
    acc = torch.float64 if scores.dtype == torch.float64 else torch.float32
    weights = torch.softmax(scores.to(acc), -1).to(values.dtype)
    left = weights[..., :cut] @ values[..., :cut, :]
    right = weights[..., cut:] @ values[..., cut:, :]
    return (left.to(acc) + right.to(acc)).to(values.dtype)


def pv_rounding_diagnostic(scores, values, cut):
    """Isolate PV rounding with native GLOBAL probabilities held fixed.

    Not an R protocol: global probabilities require global normalization.
    FP64 is a diagnostic, not a replacement reference or admission tolerance.
    Return whole and split outputs separately to expose endpoint failures too.
    """
    if not 0 <= cut <= scores.shape[-1]:
        raise ValueError('partition outside key range')
    acc = torch.float64 if scores.dtype == torch.float64 else torch.float32
    weights = torch.softmax(scores.to(acc), -1).to(values.dtype)
    result = {'native-global-pv': weights @ values}
    for name, dtype in (('bf16', values.dtype), ('fp32', torch.float32),
                        ('fp64', torch.float64)):
        w, v = weights.to(dtype), values.to(dtype)
        whole = w @ v
        left = w[..., :cut] @ v[..., :cut, :]
        right = w[..., cut:] @ v[..., cut:, :]
        # FP32/FP64 variants do not round either partial to BF16 before merging.
        merge_dtype = torch.float64 if dtype == torch.float64 else torch.float32
        merged = left.to(merge_dtype) + right.to(merge_dtype)
        result[name + '-whole'] = whole.to(values.dtype)
        result[name + '-split'] = merged.to(values.dtype)
    return result


def difference(actual, reference):
    if actual.shape != reference.shape:
        raise ValueError('comparison shape mismatch')
    delta = (actual.double() - reference.double()).abs()
    return dict(exact=bitwise_equal(actual, reference),
                max_abs_error=float(delta.max()), mean_abs_error=float(delta.mean()),
                unequal_elements=int((actual != reference).sum()),
                elements=actual.numel(), finite=bool(torch.isfinite(actual).all()))
