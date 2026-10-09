import pytest
import torch
from experimental.rg_blend_split import SplitBlend,wide_native,causal


def fixture():
    from transformers import Qwen2Config,Qwen2ForCausalLM
    torch.manual_seed(28)
    cfg=Qwen2Config(vocab_size=64,hidden_size=32,intermediate_size=64,num_hidden_layers=3,
        num_attention_heads=4,num_key_value_heads=2,max_position_embeddings=64)
    cfg._attn_implementation='eager';model=Qwen2ForCausalLM(cfg).eval()
    ids=torch.tensor([[1,2,3,4,5,6]])
    raw=[(torch.randn(1,2,3,8),torch.randn(1,2,3,8)) for _ in range(3)]
    return model,ids,raw


@pytest.mark.parametrize('full',[False,True])
def test_full_and_zero_endpoints_exact_to_named_wide_native(full):
    model,ids,raw=fixture()
    with torch.inference_mode(),wide_native():
        reference=model(input_ids=ids,use_cache=True)
        split=SplitBlend(model,ids,raw if full else [],[1,2,3] if full else [],[0,2,4,5],full=full)
        assert torch.equal(split.logits,reference.logits[:,-1])
        token=int(split.logits.argmax())
        native=model(input_ids=torch.tensor([[token]]),past_key_values=reference.past_key_values,use_cache=True)
        assert torch.equal(split.decode(token,1),native.logits[:,-1])


def test_partial_partition_repeat_and_noncontiguous_causality():
    model,ids,raw=fixture()
    with torch.inference_mode():
        a=SplitBlend(model,ids,raw,[1,2,3],[0,2,4,5],cpu=True)
        b=SplitBlend(model,ids,raw,[1,2,3],[0,2,4,5],cpu=False)
        assert torch.equal(a.logits,b.logits)
        for step in range(1,4):
            token=int(a.logits.argmax());assert torch.equal(a.decode(token,step),b.decode(token,step))
        assert a.states[2]['rp'].tolist()==[1,3]
        assert a.states[2]['gp'].tolist()==[0,2,4,5,6,7,8]
        mask=causal(torch.tensor([0,2,5]),torch.tensor([1,3]))[0,0]
        assert mask.tolist()==[[False,False],[True,False],[True,True]]


def test_dropped_question_rows_rejected():
    model,ids,raw=fixture()
    with torch.inference_mode(),pytest.raises(ValueError):SplitBlend(model,ids,raw,[1,2,3],[0,2,5])
