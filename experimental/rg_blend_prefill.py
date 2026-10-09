"""GPU/HF reference for required CB3; not a serving or R-attention backend.

Adapted from the admitted causal-corrected CacheBlend worker's layer-1,
V-difference policy. Only document rows are eligible for reuse: instructions,
separators outside documents and question rows are always fresh. This declared
document-only workload is not the original question-cache artifact.
"""
import torch
from experimental.rg_acceptance import require


def selected_rows(fresh_value, old_value, document_positions, length, ratio):
    require(0 <= ratio <= 1 and type(length) is int and length > 0, 'invalid selection policy')
    require(document_positions.ndim == 1 and document_positions.dtype == torch.long, 'invalid document positions')
    require(document_positions.numel() > 0 and bool((document_positions >= 0).all()) and
            bool((document_positions < length).all()), 'document position outside request')
    require(torch.equal(document_positions,document_positions.unique(sorted=True)), 'duplicate/unordered positions')
    require(fresh_value.shape == old_value.shape and fresh_value.ndim == 4 and
            fresh_value.shape[2] == document_positions.numel(), 'selection value shape mismatch')
    # Preserve BF16 difference arithmetic instead of silently upcasting scores.
    scores=((fresh_value-old_value)**2).sum(dim=(0,1,3))
    count=int(document_positions.numel()*ratio)
    chosen=document_positions[torch.topk(scores,k=count).indices]
    fresh=torch.ones(length,dtype=torch.bool,device=document_positions.device)
    fresh[document_positions]=False
    fresh[chosen]=True
    return fresh.nonzero().flatten()


def absolute_mask(selected, length, dtype):
    require(selected.ndim == 1 and selected.dtype == torch.long and selected.numel() > 0, 'invalid query positions')
    require(bool((selected >= 0).all()) and bool((selected < length).all()), 'query position outside request')
    keys=torch.arange(length,device=selected.device)
    return torch.where(keys[None,:] <= selected[:,None],
        torch.tensor(0.,device=selected.device,dtype=dtype),
        torch.tensor(torch.finfo(dtype).min,device=selected.device,dtype=dtype))[None,None]


def prefill(model, input_ids, raw_document_pairs, document_positions, mode='blend', ratio=.16):
    """Return final prompt logits, native decode cache and projection-row trace.

    Native/full endpoint still executes these layer operations. It is not a
    shortcut calling the uninstrumented model. Empty evidence bypass is an
    explicitly named native control; partial results need their own admission.
    """
    from transformers import DynamicCache
    from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb, eager_attention_forward
    require(mode in ('full','blend','reuse'), 'unknown composition mode')
    require(not model.training and model.config.model_type == 'qwen2', 'only inference Qwen2 supported')
    require(input_ids.ndim == 2 and input_ids.shape[0] == 1, 'single request only')
    require(not getattr(model.config,'use_sliding_window',False), 'sliding window unsupported')
    length=input_ids.shape[1];device=input_ids.device
    require(length <= model.config.max_position_embeddings, 'context overflow')
    positions=torch.as_tensor(document_positions,dtype=torch.long,device=device)
    require(positions.ndim == 1 and torch.equal(positions,positions.unique(sorted=True)), 'invalid document positions')
    require(bool((positions >= 0).all()) and bool((positions < length-1).all()), 'final question token must be fresh')
    if positions.numel() == 0:
        require(not raw_document_pairs, 'unexpected KV without documents')
        output=model(input_ids=input_ids,use_cache=True)
        return output.logits[:,-1],output.past_key_values,[]
    require(len(raw_document_pairs) == len(model.model.layers), 'missing document layers')
    require(len(model.model.layers) >= 2, 'selection layer1 missing')
    for k,v in raw_document_pairs:
        require(k.shape == v.shape == (1,model.config.num_key_value_heads,len(positions),
            model.config.hidden_size//model.config.num_attention_heads), 'invalid document KV shape')
        require(k.dtype == v.dtype == model.dtype, 'document dtype mismatch')
    selected=torch.arange(length,device=device)
    if mode == 'reuse':
        keep=torch.ones(length,dtype=torch.bool,device=device);keep[positions]=False
        selected=keep.nonzero().flatten()
    hidden=model.model.embed_tokens(input_ids[:,selected])
    cache=DynamicCache();trace=[]
    # The native model computes shared cos/sin once outside the decoder layers.
    full_cos,full_sin=model.model.rotary_emb(hidden,torch.arange(length,device=device)[None])
    for index,layer in enumerate(model.model.layers):
        residual=hidden;normalized=layer.input_layernorm(hidden);attn=layer.self_attn
        shape=(*normalized.shape[:-1],-1,attn.head_dim)
        q=attn.q_proj(normalized).view(shape).transpose(1,2)
        k=attn.k_proj(normalized).view(shape).transpose(1,2)
        v=attn.v_proj(normalized).view(shape).transpose(1,2)
        projected_rows=len(selected)
        cos,sin=full_cos[:,selected],full_sin[:,selected]
        q,k=apply_rotary_pos_emb(q,k,cos,sin)
        full_fresh=mode == 'full' or (mode == 'blend' and index <= 1)
        if full_fresh:
            full_k,full_v=k,v
        else:
            old_k,old_v=(x.to(device) for x in raw_document_pairs[index])
            old_k=apply_rotary_pos_emb(old_k,old_k,full_cos[:,positions],full_sin[:,positions])[1]
            full_k=torch.zeros(1,attn.config.num_key_value_heads,length,attn.head_dim,dtype=k.dtype,device=device)
            full_v=torch.zeros_like(full_k)
            full_k[:,:,positions]=old_k;full_v[:,:,positions]=old_v
            full_k[:,:,selected]=k;full_v[:,:,selected]=v
        if mode == 'blend' and index == 1:
            selected=selected_rows(v[:,:,positions],raw_document_pairs[index][1].to(device),positions,length,ratio)
            q=q[:,:,selected];residual=residual[:,selected]
        mask=absolute_mask(selected,length,q.dtype)
        output,_=eager_attention_forward(attn,q,full_k,full_v,mask,attn.scaling,dropout=0.)
        output=output.reshape(1,len(selected),-1).contiguous()
        hidden=residual+attn.o_proj(output)
        residual=hidden;hidden=layer.post_attention_layernorm(hidden)
        hidden=residual+layer.mlp(hidden)
        cache.update(full_k,full_v,index)
        trace.append(dict(layer=index,projected_rows=projected_rows,selected_positions=selected.tolist(),
            full_fresh_kv=full_fresh,mask='absolute-causal',document_rows=len(positions)))
    hidden=model.model.norm(hidden)
    require(int(selected[-1]) == length-1,'final sampling row missing')
    return model.lm_head(hidden)[:,-1],cache,trace
