import copy
import pytest
import torch
from experimental.rg_blend_cache import POLICY, pack_document, validate_document, compose_documents


def example(n=3, offset=0):
    encoding=dict(model_snapshot='model',config_sha256='config',tokenizer_sha256='tokenizer',capture_policy=POLICY)
    pairs=[(torch.arange(n*4,dtype=torch.bfloat16).reshape(1,2,n,2)+offset,
            torch.ones(1,2,n,2,dtype=torch.bfloat16)*i) for i in range(2)]
    entry,chunks=pack_document(encoding,list(range(offset,offset+n)),pairs)
    return encoding,pairs,entry,chunks


def test_roundtrip_and_stable_identity():
    encoding,pairs,entry,chunks=example()
    assert entry==pack_document(encoding,entry['tokens'],pairs)[0]
    for actual,expected in zip(validate_document(entry,chunks,encoding),pairs):
        assert all(torch.equal(a,b) for a,b in zip(actual,expected))


@pytest.mark.parametrize('mutation',['encoding','tokens','payload','padding','missing','metadata'])
def test_invalid_document_fails(mutation):
    encoding,pairs,entry,chunks=example();entry=copy.deepcopy(entry)
    if mutation=='encoding':entry['encoding']['model_snapshot']='other'
    if mutation=='tokens':entry['tokens'][0]+=1
    if mutation=='payload':chunks[0][0,0,0,0]+=1
    if mutation=='padding':chunks[0][0,0,-1,0]=1
    if mutation=='missing':chunks=[]
    if mutation=='metadata':entry['metadata']['tokens']=2
    with pytest.raises(ValueError):validate_document(entry,chunks,encoding)


def test_reorder_repeat_positions_and_singleton_limit():
    encoding,pairs,a,ac=example();_,bp,b,bc=example(2,10)
    seen=[]
    def rotate(k,pos):
        seen.append(pos);return k+torch.tensor(pos,dtype=k.dtype)[None,None,:,None]
    out,trace=compose_documents([(b,bc),(a,ac),(b,bc)],encoding,7,rotate)
    assert [(t['start'],t['end']) for t in trace]==[(7,9),(9,12),(12,14)]
    assert torch.equal(out[0][0][:,:,:2],bp[0][0]+torch.tensor([7,8],dtype=torch.bfloat16)[None,None,:,None])
    single,_=compose_documents([(a,ac)],encoding,0,lambda k,p:k.clone())
    assert all(torch.equal(x,y) for a,b in zip(single,pairs) for x,y in zip(a,b))
    assert torch.equal(validate_document(b,bc,encoding)[0][0],bp[0][0])


def test_reject_post_rope_and_empty_composition():
    encoding,pairs,entry,chunks=example();bad=dict(encoding,capture_policy='post-rope-prefix')
    with pytest.raises(ValueError):pack_document(bad,entry['tokens'],pairs)
    with pytest.raises(ValueError):compose_documents([],encoding,0,lambda k,p:k)
    with pytest.raises(ValueError):compose_documents([(entry,chunks)],encoding,-1,lambda k,p:k)


def test_rotary_layout_guard():
    encoding,pairs,entry,chunks=example()
    with pytest.raises(ValueError):compose_documents([(entry,chunks)],encoding,0,lambda k,p:k.float())
