from types import SimpleNamespace
import pytest
import torch
from experimental.rg_tiled_softmax import tiled_eager


@pytest.mark.parametrize('length',[1,127,128,129,257])
def test_native_exact_rows_and_masks(length):
    from transformers.models.qwen2.modeling_qwen2 import eager_attention_forward
    g=torch.Generator().manual_seed(8096)
    q=torch.randn(1,4,length,16,generator=g).bfloat16()
    k=torch.randn(1,2,length+7,16,generator=g).bfloat16()
    v=torch.randn(k.shape,generator=g).bfloat16()
    mask=torch.zeros(1,1,length,length+7,dtype=q.dtype)
    mask[..., -3:]=torch.finfo(q.dtype).min
    module=SimpleNamespace(num_key_value_groups=2,training=False)
    with torch.inference_mode():
        a=eager_attention_forward(module,q,k,v,mask,.25)
        b=tiled_eager(module,q,k,v,mask,.25)
        assert all(torch.equal(x,y) for x,y in zip(a,b))


def test_training_and_grad_are_not_silently_supported():
    q=torch.zeros(1,1,1,4)
    with pytest.raises(ValueError):tiled_eager(SimpleNamespace(training=False),q,q,q,None,1.)
