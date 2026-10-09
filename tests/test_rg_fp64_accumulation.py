import pytest
import torch
from experimental.rg_equivalent import wide_state,stable_merge
from experimental.rg_fp64_accumulation import fp64_parts


@pytest.mark.parametrize('cut',[0,3,9])
def test_declared_arithmetic_and_no_input_mutation(cut):
    gen=torch.Generator().manual_seed(27)
    q=torch.randn(1,4,2,8,generator=gen).bfloat16()
    k=torch.randn(1,2,9,8,generator=gen).bfloat16();v=torch.randn(k.shape,generator=gen).bfloat16()
    original=[x.clone() for x in (q,k,v)]
    g=wide_state(q.double(),k[...,cut:,:].double(),v[...,cut:,:].double())
    r=wide_state(q.double(),k[...,:cut,:].double(),v[...,:cut,:].double())
    expected=stable_merge(g,r) if cut else g
    actual=fp64_parts(q,k[...,:cut,:],v[...,:cut,:],k[...,cut:,:],v[...,cut:,:])
    assert all(x.dtype==torch.float32 and torch.equal(x,y.float()) for x,y in zip(actual,expected))
    assert all(torch.equal(a,b) for a,b in zip((q,k,v),original))


def test_fully_masked_output_zero_and_fp64_reference_preserved():
    q=torch.ones(1,2,1,8,dtype=torch.float64);k=torch.ones(1,1,2,8,dtype=torch.float64)
    mask=torch.zeros(1,1,1,1,dtype=torch.bool)
    out,lse=fp64_parts(q,k[...,:1,:],k[...,:1,:],k[...,1:,:],k[...,1:,:],mask,mask)
    assert out.dtype==torch.float64 and torch.equal(out,torch.zeros_like(out))
    assert torch.isneginf(lse).all()
