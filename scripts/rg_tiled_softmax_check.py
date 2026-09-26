"""Exact GPU comparison of native and row-tiled eager attention."""
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import torch
from scripts.artifacts import write
from experimental.rg_acceptance import require
from experimental.rg_tiled_softmax import tiled_eager

LENGTHS=(1,127,128,129,512,1024,2048)


def run(out):
    from transformers.models.qwen2.modeling_qwen2 import eager_attention_forward
    require(eager_attention_forward is not tiled_eager,'native comparison was replaced')
    generator=torch.Generator(device='cuda').manual_seed(20260926)
    rows=[];module=SimpleNamespace(num_key_value_groups=8,training=False)
    with torch.inference_mode():
        for n in LENGTHS:
            q=torch.randn(1,16,n,128,generator=generator,device='cuda').bfloat16()
            k=torch.randn(1,2,n+7,128,generator=generator,device='cuda').bfloat16()
            v=torch.randn(k.shape,generator=generator,device='cuda').bfloat16()
            for kind in ('causal','key-mask'):
                blocked=(torch.arange(n+7,device='cuda')[None,:]>torch.arange(n,device='cuda')[:,None]+7)
                if kind=='key-mask':blocked[:]=False;blocked[:,-3:]=True
                mask=torch.zeros(1,1,n,n+7,dtype=q.dtype,device='cuda').masked_fill(blocked,torch.finfo(q.dtype).min)
                native=eager_attention_forward(module,q,k,v,mask,128**-.5)
                actual=tiled_eager(module,q,k,v,mask,128**-.5)
                replay=tiled_eager(module,q,k,v,mask,128**-.5)
                row=dict(query_tokens=n,key_tokens=n+7,mask=kind,
                    output_exact=torch.equal(native[0],actual[0]),probabilities_exact=torch.equal(native[1],actual[1]),
                    repeat_exact=all(torch.equal(a,b) for a,b in zip(actual,replay)))
                rows.append(row);write(out/'tiled-softmax-check.json',dict(status='running',checks=rows))
                require(all(row[key] is True for key in ('output_exact','probabilities_exact','repeat_exact')),'tiled softmax differs from native')
                del native,actual,replay,mask
            del q,k,v
    source=Path(__file__).resolve().parents[1]/'experimental/rg_tiled_softmax.py'
    result=dict(status='PASS',checks=rows,source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                scope='exact tensor/mask/replay check, no quality or latency claim')
    write(out/'tiled-softmax-check.json',result);return result
