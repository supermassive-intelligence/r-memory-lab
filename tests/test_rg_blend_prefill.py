import pytest
import torch
from experimental.rg_blend_prefill import selected_rows, absolute_mask, prefill


def test_selection_always_recomputes_uncached_rows():
    pos=torch.tensor([1,3,5,7]);old=torch.zeros(1,2,4,2,dtype=torch.bfloat16)
    fresh=old.clone();fresh[:,:,2]=5
    assert selected_rows(fresh,old,pos,10,.25).tolist()==[0,2,4,5,6,8,9]
    assert selected_rows(fresh,old,pos,10,1).tolist()==list(range(10))
    assert selected_rows(fresh,old,pos,10,0).tolist()==[0,2,4,6,8,9]


def test_noncontiguous_queries_never_see_future():
    mask=absolute_mask(torch.tensor([1,4,9,15]),16,torch.float32)[0,0]
    for i,position in enumerate([1,4,9,15]):
        assert torch.equal((mask[i]==0).nonzero().flatten(),torch.arange(position+1))
    values=torch.arange(16,dtype=torch.float32);changed=values.clone();changed[2:]=10000
    probs=mask.softmax(-1)
    assert (probs[0]@values).item()==(probs[0]@changed).item()==.5


@pytest.mark.parametrize('mode,ratio',[('full',.16),('blend',1.)])
def test_full_endpoint_matches_native_model_and_decode(mode,ratio):
    from transformers import Qwen2Config,Qwen2ForCausalLM
    torch.manual_seed(2)
    config=Qwen2Config(vocab_size=64,hidden_size=32,intermediate_size=64,num_hidden_layers=3,
        num_attention_heads=4,num_key_value_heads=2,max_position_embeddings=64,attention_dropout=0.)
    config._attn_implementation='eager'
    model=Qwen2ForCausalLM(config).eval()
    ids=torch.tensor([[1,2,3,4,5,6]])
    raw=[(torch.randn(1,2,3,8),torch.randn(1,2,3,8)) for _ in range(3)]
    with torch.inference_mode():
        native=model(input_ids=ids,use_cache=True)
        logits,cache,trace=prefill(model,ids,raw,[1,2,3],mode,ratio)
        assert torch.equal(logits,native.logits[:,-1])
        assert all(torch.equal(a,b) for pair,control in zip(cache,native.past_key_values) for a,b in zip(pair[:2],control[:2]))
        token=logits.argmax(-1)[:,None]
        actual=model(input_ids=token,past_key_values=cache,use_cache=True)
        reference=model(input_ids=token,past_key_values=native.past_key_values,use_cache=True)
        assert torch.equal(actual.logits,reference.logits)
        assert all(row['projected_rows']==6 for row in trace)


def test_reject_invalid_document_positions():
    with pytest.raises(ValueError):absolute_mask(torch.tensor([9]),5,torch.float32)
    values=torch.zeros(1,2,2,2)
    with pytest.raises(ValueError):selected_rows(values,values,torch.tensor([1,1]),4,.16)


def test_partial_replay_and_empty_cache_native_zero():
    from transformers import Qwen2Config,Qwen2ForCausalLM
    torch.manual_seed(3)
    config=Qwen2Config(vocab_size=64,hidden_size=32,intermediate_size=64,num_hidden_layers=3,
        num_attention_heads=4,num_key_value_heads=2,max_position_embeddings=64)
    config._attn_implementation='eager';model=Qwen2ForCausalLM(config).eval()
    ids=torch.tensor([[1,2,3,4,5,6]])
    raw=[(torch.randn(1,2,3,8),torch.randn(1,2,3,8)) for _ in range(3)]
    with torch.inference_mode():
        native=model(input_ids=ids,use_cache=True)
        for mode in ('blend','reuse'):
            a,_,trace=prefill(model,ids,raw,[1,2,3],mode)
            b,_,_=prefill(model,ids,raw,[1,2,3],mode)
            assert torch.equal(a,b) and torch.isfinite(a).all()
            assert trace[-1]['projected_rows'] < 6
            zero,_,_=prefill(model,ids,[],[],mode)
            assert torch.equal(zero,native.logits[:,-1])
