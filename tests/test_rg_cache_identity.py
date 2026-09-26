import copy
import pytest
from experimental.rg_cache_identity import identity,validate_entry


def fixture():
    encoding=dict(model_snapshot='snapshot',config_sha256='a'*64,tokenizer_sha256='b'*64,prefix_policy='policy')
    meta=dict(layers=36,heads=2,tokens=2,head_dim=128,dtype='torch.bfloat16',chunk_size=256,chunks=1,layout='2,L,T,H*D')
    entry=dict(encoding=encoding,tokens=[1,2],metadata=meta,chunk_hashes=['c'*64])
    entry['identity']=identity(encoding,entry['tokens'],meta)
    return entry


def test_stable_identity_is_independent_of_request_question():
    a=fixture();b=fixture();b.update(question='another question',request_id='another request')
    assert validate_entry(a,a['encoding'])==validate_entry(b,b['encoding'])


@pytest.mark.parametrize('field',['model_snapshot','config_sha256','tokenizer_sha256','prefix_policy'])
def test_encoding_change_cannot_hit_old_key(field):
    entry=fixture();changed=copy.deepcopy(entry['encoding']);changed[field]+='x'
    with pytest.raises(ValueError):validate_entry(entry,changed)
    assert identity(changed,entry['tokens'],entry['metadata'])!=entry['identity']


@pytest.mark.parametrize('damage',['tokens','valid-length','layers','hash-count','identity','missing-encoding'])
def test_corrupt_entry_rejected(damage):
    e=fixture()
    if damage=='tokens':e['tokens'][0]=9
    elif damage=='valid-length':e['metadata']['tokens']=3
    elif damage=='layers':e['metadata']['layers']=35
    elif damage=='hash-count':e['chunk_hashes']=[]
    elif damage=='identity':e['identity']='bad'
    else:del e['encoding']['config_sha256']
    with pytest.raises(ValueError):validate_entry(e,e['encoding'])
