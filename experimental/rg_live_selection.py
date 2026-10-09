"""G-side request-time CB4 selector; one native-BF16 first-layer scout pass.

The selected map is shared by the matched GPU and R/G engines. No historical
trace or answer enters selection. This extra work is not a speed optimization.
"""
import torch
from experimental.rg_acceptance import require
from experimental.rg_blend_prefill import selected_rows, absolute_mask

STUDY='CB5-LIVE-SELECT1'
RATIO=.16


def select(model,input_ids,raw,document_positions,ratio=RATIO):
    from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb,eager_attention_forward
    require(not model.training and not torch.is_grad_enabled(),'inference only')
    require(model.config.model_type=='qwen2' and not getattr(model.config,'use_sliding_window',False),'dense Qwen2 only')
    require(input_ids.ndim==2 and input_ids.shape[0]==1,'batch1 only')
    length=input_ids.shape[1];device=input_ids.device
    require(0<=ratio<=1 and length<=model.config.max_position_embeddings,'invalid policy/context')
    positions=torch.tensor(document_positions,dtype=torch.long,device=device)
    require(torch.equal(positions,positions.unique(sorted=True)) and
            bool((positions>=0).all()) and bool((positions<length-1).all()),'invalid document positions')
    if not len(positions):
        require(not raw,'unexpected raw memory')
        return list(range(length)),dict(mode='empty',ratio=ratio,layer=1,selected_positions=list(range(length)))
    require(len(raw)==len(model.model.layers) and len(raw)>=2,'missing layers')
    old=raw[1][1]
    require(old.device.type=='cpu' and old.dtype==model.dtype,'CPU-owned matching document V required')
    hidden=model.model.embed_tokens(input_ids);all_pos=torch.arange(length,device=device)
    cos,sin=model.model.rotary_emb(hidden,all_pos[None])
    layer=model.model.layers[0];attn=layer.self_attn
    normalized=layer.input_layernorm(hidden);shape=(*normalized.shape[:-1],-1,attn.head_dim)
    q=attn.q_proj(normalized).view(shape).transpose(1,2)
    k=attn.k_proj(normalized).view(shape).transpose(1,2)
    v=attn.v_proj(normalized).view(shape).transpose(1,2)
    q,k=apply_rotary_pos_emb(q,k,cos,sin)
    output,_=eager_attention_forward(attn,q,k,v,absolute_mask(all_pos,length,q.dtype),attn.scaling,dropout=0.)
    hidden=hidden+attn.o_proj(output.reshape(1,length,-1).contiguous())
    hidden=hidden+layer.mlp(layer.post_attention_layernorm(hidden))
    layer=model.model.layers[1];attn=layer.self_attn
    fresh=attn.v_proj(layer.input_layernorm(hidden)).view(1,length,-1,attn.head_dim).transpose(1,2)[:,:,positions]
    old_device=old.to(device)
    chosen=selected_rows(fresh,old_device,positions,length,ratio)
    scores=((fresh-old_device)**2).sum(dim=(0,1,3))
    require(bool(torch.isfinite(scores).all()),'nonfinite selection scores')
    return chosen.tolist(),dict(mode='native-BF16-G-scout',layer=1,ratio=ratio,
        selected_positions=chosen.tolist(),document_positions=positions.tolist(),
        input_ids=input_ids.cpu(),fresh_value=fresh.cpu(),stored_value=old.cpu(),scores=scores.cpu())
