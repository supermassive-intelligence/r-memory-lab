import pytest

torch = pytest.importorskip('torch')
from experimental.rg_equivalent import (wide_state, wide_parts, relative_errors,
    numerical_pass, stable_merge, placement_errors)
from experimental.rg_numerics import bitwise_equal


@pytest.mark.parametrize('cut', [0, 3, 7])
@pytest.mark.parametrize('dtype', [torch.float32, torch.float64, torch.bfloat16])
def test_wide_algebra_and_endpoints(cut, dtype):
    gen = torch.Generator().manual_seed(7321)
    q = torch.randn(1, 4, 3, 8, generator=gen).to(dtype)
    k = torch.randn(1, 2, 7, 8, generator=gen).to(dtype)
    v = torch.randn(1, 2, 7, 8, generator=gen).to(dtype)
    reference = wide_state(q.double(), k.double(), v.double())[0]
    actual = wide_parts(q, k[..., :cut, :], v[..., :cut, :], k[..., cut:, :], v[..., cut:, :])[0]
    pre = relative_errors(actual, reference)
    rounded = relative_errors(actual.to(torch.bfloat16), reference.to(torch.bfloat16))
    assert numerical_pass(pre, rounded)
    assert bitwise_equal(actual, wide_parts(q, k[..., :cut, :], v[..., :cut, :], k[..., cut:, :], v[..., cut:, :])[0])
    if cut in (0, 7):
        assert bitwise_equal(actual, wide_state(q, k, v)[0])


@pytest.mark.parametrize('mask_type', ['bool', 'finite', 'infinite'])
def test_masked_partition_and_all_masked(mask_type):
    q = torch.ones(1, 2, 2, 4, dtype=torch.bfloat16)
    k = torch.ones(1, 1, 4, 4, dtype=torch.bfloat16)
    v = torch.arange(16, dtype=torch.bfloat16).reshape(1, 1, 4, 4)
    mask = torch.tensor([[True, True, False, False], [False, False, False, False]])
    if mask_type != 'bool':
        value = torch.finfo(q.dtype).min if mask_type == 'finite' else -torch.inf
        mask = torch.zeros(2, 4, dtype=q.dtype).masked_fill(~mask, value)
    actual = wide_parts(q, k[..., :2, :], v[..., :2, :], k[..., 2:, :], v[..., 2:, :], mask[..., :2], mask[..., 2:])[0]
    assert bitwise_equal(actual, wide_state(q, k, v, mask)[0])
    assert (actual[..., 1, :] == 0).all()
    assert torch.isfinite(actual).all()


def test_fixed_budget_cannot_accept_nonfinite_or_large_errors():
    ref = torch.ones(2)
    good = relative_errors(ref, ref)
    assert numerical_pass(good, good)
    assert not numerical_pass(relative_errors(ref + 1, ref), good)
    assert not numerical_pass(relative_errors(ref * torch.nan, ref), good)
    with pytest.raises(ValueError):
        relative_errors(ref, ref[:1])


def test_stable_merge_large_common_log_normalizer():
    # Identical normalized outputs must remain identical under a common score
    # shift. Rounded logaddexp then subtraction loses that property at LSE1259.
    output = torch.ones(1, 2, 3, 4)
    lse = torch.full(output.shape[:-1], 1259.0)
    actual, combined_lse = stable_merge((output, lse), (output, lse))
    assert bitwise_equal(actual, output)
    assert torch.isfinite(combined_lse).all()


def test_direct_placement_gate_checks_every_element_and_near_zero():
    ref = torch.tensor([0., 1., 100.])
    assert placement_errors(ref + 1e-6, ref)['pass']
    wrong = ref.clone()
    wrong[0] = 2e-5
    result = placement_errors(wrong, ref)
    assert not result['pass'] and result['violating_elements'] == 1
    assert not placement_errors(ref * torch.nan, ref)['pass']
