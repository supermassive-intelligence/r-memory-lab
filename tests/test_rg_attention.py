"""Prototype tensor identities. Model-domain guards run separately in the job."""
import pytest

torch = pytest.importorskip('torch')
from experimental.rg_attention import attention_state, attend_parts, merge_states


@pytest.mark.parametrize('length', [1, 17, 128])
@pytest.mark.parametrize('heads', [1, 4])
def test_split_algebra(length, heads):
    gen = torch.Generator().manual_seed(length)
    q = torch.randn(1, heads, 3, 8, generator=gen, dtype=torch.float64)
    k = torch.randn(1, 1, length, 8, generator=gen, dtype=torch.float64)
    v = torch.randn(1, 1, length, 8, generator=gen, dtype=torch.float64)
    ref = attention_state(q, k, v)
    cut = length // 2
    actual = attend_parts(q, k[..., :cut, :], v[..., :cut, :], k[..., cut:, :], v[..., cut:, :])
    # Algebra unit-test tolerance only; NOT a model admission threshold.
    torch.testing.assert_close(actual[0], ref[0], atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(actual[1], ref[1], atol=1e-12, rtol=1e-12)


def test_empty_zero_is_bit_identical():
    q = torch.randn(1, 4, 2, 8)
    k = torch.randn(1, 2, 7, 8)
    v = torch.randn_like(k)
    ref = attention_state(q, k, v)
    empty = attention_state(q, k[..., :0, :], v[..., :0, :])
    for merged in (merge_states(ref, empty), merge_states(empty, ref)):
        assert torch.equal(merged[0], ref[0])
        assert torch.equal(merged[1], ref[1])
    both = merge_states(empty, empty)
    assert torch.equal(both[0], torch.zeros_like(q))
    assert torch.isneginf(both[1]).all()


def test_causal_and_all_masked_partition():
    q = torch.randn(1, 2, 4, 8, dtype=torch.float64)
    k = torch.randn(1, 1, 4, 8, dtype=torch.float64)
    v = torch.randn_like(k)
    mask = torch.ones(4, 4, dtype=torch.bool).tril()
    ref = attention_state(q, k, v, mask)
    actual = attend_parts(q, k[..., 2:, :], v[..., 2:, :], k[..., :2, :], v[..., :2, :], mask[..., 2:], mask[..., :2])
    torch.testing.assert_close(actual[0], ref[0], atol=1e-12, rtol=1e-12)
    assert torch.isfinite(actual[0]).all()


def test_invalid_inputs_fail():
    q = torch.zeros(1, 3, 1, 4)
    k = torch.zeros(1, 2, 2, 4)
    with pytest.raises(ValueError, match='divisible'):
        attention_state(q, k, k)
    with pytest.raises(ValueError, match='shape'):
        merge_states((q, q[..., 0]), (k, k[..., 0]))
