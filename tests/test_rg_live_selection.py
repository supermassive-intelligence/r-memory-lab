import pytest
import torch
from experimental.rg_blend_prefill import prefill
from experimental.rg_live_selection import select


def fixture():
    from transformers import Qwen2Config,Qwen2ForCausalLM
    torch.manual_seed(28)
    config=Qwen2Config(vocab_size=64,hidden_size=32,intermediate_size=64,num_hidden_layers=3,
        num_attention_heads=4,num_key_value_heads=2,max_position_embeddings=64)
    config._attn_implementation='eager'
    return Qwen2ForCausalLM(config).eval(),torch.tensor([[1,2,3,4,5,6]]),[
        (torch.randn(1,2,3,8),torch.randn(1,2,3,8)) for _ in range(3)]


@pytest.mark.parametrize('ratio',[0.,.16,.67,1.])
@pytest.mark.parametrize('dtype',[torch.float32,torch.bfloat16])
def test_live_selection_matches_independent_native_trace(ratio,dtype):
    model,ids,raw=fixture()
    model=model.to(dtype);raw=[(k.to(dtype),v.to(dtype)) for k,v in raw]
    with torch.inference_mode():
        _,_,trace=prefill(model,ids,raw,[1,2,3],ratio=ratio)
        selected,receipt=select(model,ids,raw,[1,2,3],ratio)
        assert selected==trace[1]['selected_positions']
        again,_=select(model,ids,raw,[1,2,3],ratio)
        assert again==selected and receipt['ratio']==ratio
        assert torch.equal(receipt['stored_value'],raw[1][1])
        assert receipt['scores'].shape==(3,) and all(i in selected for i in [0,4,5])


def test_live_empty_and_reject_invalid_inputs():
    model,ids,raw=fixture()
    with torch.inference_mode():
        selected,r=select(model,ids,[],[])
        assert selected==list(range(6)) and r['mode']=='empty'
        for positions in ([1,1],[1,5],[-1]):
            with pytest.raises(ValueError):select(model,ids,raw,positions)
        with pytest.raises(ValueError):select(model,ids,raw,[])
    with pytest.raises(ValueError):select(model,ids,raw,[1,2,3])
