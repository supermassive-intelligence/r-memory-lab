import pytest

torch = pytest.importorskip('torch')
from experimental.rg_numerics import (bitwise_equal, difference, fixed_score_split,
                                    global_weight_split, score_state,
                                    pv_rounding_diagnostic)


@pytest.mark.parametrize('cut', [0, 1, 3, 7])
@pytest.mark.parametrize('wide', [False, True])
def test_fixed_score_algebra_and_endpoints(cut, wide):
    gen = torch.Generator().manual_seed(43)
    scores = torch.randn(1, 2, 3, 7, dtype=torch.float64, generator=gen)
    values = torch.randn(1, 2, 7, 4, dtype=torch.float64, generator=gen)
    control = torch.softmax(scores, -1) @ values
    actual = fixed_score_split(scores, values, cut, wide)
    # Unit algebra tolerance only, never a model admission threshold.
    torch.testing.assert_close(actual, control, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(global_weight_split(scores, values, cut), control, rtol=1e-12, atol=1e-12)
    if cut in (0, 7):
        assert torch.equal(actual, control)


def test_masked_and_empty_partition():
    scores = torch.zeros(1, 2, 3, 4, dtype=torch.float64)
    scores[..., :2] = -torch.inf
    values = torch.ones(1, 2, 4, 5, dtype=torch.float64)
    actual = fixed_score_split(scores, values, 2)
    assert torch.equal(actual, torch.ones_like(actual))
    scores.fill_(-torch.inf)
    actual = fixed_score_split(scores, values, 2, True)
    assert torch.equal(actual, torch.zeros_like(actual))
    state = score_state(scores[..., :0], values[..., :0, :], True)
    assert torch.isneginf(state[1]).all()


def test_difference_and_invalid_inputs():
    x = torch.tensor([1., 2.])
    assert difference(x, x)['exact']
    assert difference(x + 1, x)['max_abs_error'] == 1
    with pytest.raises(ValueError, match='shape'):
        difference(x, x[:1])
    with pytest.raises(ValueError, match='partition'):
        fixed_score_split(torch.zeros(1, 1, 1, 2), torch.zeros(1, 1, 2, 2), 3)


@pytest.mark.parametrize('cut', [0, 1, 4])
def test_pv_rounding_diagnostic_exact_control_and_replay(cut):
    scores = torch.tensor([[[[0., 1., -1., 2.]]]], dtype=torch.bfloat16)
    values = torch.arange(12, dtype=torch.bfloat16).reshape(1, 1, 4, 3)
    reference = torch.softmax(scores.float(), -1).to(values.dtype) @ values
    actual = pv_rounding_diagnostic(scores, values, cut)
    repeated = pv_rounding_diagnostic(scores, values, cut)
    assert torch.equal(actual['native-global-pv'], reference)
    assert torch.equal(actual['bf16-whole'], reference)
    for name, value in actual.items():
        assert value.dtype == values.dtype
        assert torch.equal(value, repeated[name])
    if cut in (0, 4):
        for precision in ('bf16', 'fp32', 'fp64'):
            assert torch.equal(actual[precision+'-split'], actual[precision+'-whole'])
    with pytest.raises(ValueError, match='partition'):
        pv_rounding_diagnostic(scores, values, 5)


def test_bitwise_identity_rejects_signed_zero_and_dtype_change():
    positive = torch.tensor([0.], dtype=torch.float32)
    negative = torch.tensor([-0.], dtype=torch.float32)
    assert torch.equal(positive, negative)  # Numerical equality is weaker.
    assert not bitwise_equal(positive, negative)
    assert not difference(positive, negative)['exact']
    assert not bitwise_equal(positive, positive.double())
    assert not bitwise_equal(positive, positive.reshape(1, 1))
    assert bitwise_equal(positive, positive.clone())
    assert bitwise_equal(positive[0], positive[0].clone())


def test_bitwise_identity_handles_layout_and_empty_tensors():
    x = torch.arange(6, dtype=torch.bfloat16).reshape(2, 3).transpose(0, 1)
    assert bitwise_equal(x, x.contiguous())
    assert bitwise_equal(x[:0], x[:0].clone())
