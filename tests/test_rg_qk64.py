import pytest
torch=pytest.importorskip('torch')
from experimental.rg_qk64 import qk64_state
from experimental.rg_equivalent import wide_state,stable_merge,relative_errors,numerical_pass
from experimental.rg_numerics import bitwise_equal


@pytest.mark.parametrize('cut',[0,3,7])
@pytest.mark.parametrize('dtype',[torch.bfloat16,torch.float32,torch.float64])
def test_qk64_partition_endpoints_and_repeat(cut,dtype):
    gen=torch.Generator().manual_seed(7321)
    q=torch.randn(1,4,3,8,generator=gen).to(dtype)
    k=torch.randn(1,2,7,8,generator=gen).to(dtype);v=torch.randn(1,2,7,8,generator=gen).to(dtype)
    a=qk64_state(q,k[...,:cut,:],v[...,:cut,:]);b=qk64_state(q,k[...,cut:,:],v[...,cut:,:])
    actual=stable_merge(a,b)[0];reference=wide_state(q.double(),k.double(),v.double())[0]
    assert numerical_pass(relative_errors(actual,reference),relative_errors(actual.bfloat16(),reference.bfloat16()))
    assert bitwise_equal(qk64_state(q,k,v)[0],qk64_state(q,k,v)[0])
    if cut in (0,7):assert bitwise_equal(actual,qk64_state(q,k,v)[0])
    if dtype==torch.float64:assert bitwise_equal(qk64_state(q,k,v)[0],wide_state(q,k,v)[0])


@pytest.mark.parametrize('kind',['bool','finite','infinite'])
def test_qk64_full_mask_and_partial_mask(kind):
    q=torch.ones(1,2,2,4,dtype=torch.bfloat16);k=torch.ones(1,1,4,4,dtype=q.dtype);v=k.clone()
    mask=torch.tensor([[True,True,False,False],[False,False,False,False]])
    if kind!='bool':mask=torch.zeros(2,4,dtype=q.dtype).masked_fill(~mask,torch.finfo(q.dtype).min if kind=='finite' else -torch.inf)
    out,lse=qk64_state(q,k,v,mask)
    assert torch.isfinite(out).all() and (out[...,1,:]==0).all()
    assert torch.isneginf(lse[...,1]).all()
