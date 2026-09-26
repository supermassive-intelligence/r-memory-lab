import hashlib
import json
import pickle
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from scripts.rg_lmcache_prefix import PrefixStore


def fixture(tmp_path,monkeypatch,mode):
    import scripts.rg_lmcache_prefix as module
    clock=[0.]
    monkeypatch.setattr(module.time,'monotonic',lambda:clock[0])
    monkeypatch.setattr(module.time,'sleep',lambda seconds:clock.__setitem__(0,clock[0]+seconds))
    store=object.__new__(PrefixStore);store.disk=tmp_path
    store.command=['--l2-adapter',json.dumps(dict(type='fs',base_path=str(tmp_path)))]
    store.alive=Mock();store.key=lambda tokens,identity,index:SimpleNamespace(index=index,request_id=str(index))
    store.client=SimpleNamespace(prepare_store=Mock(),commit_store=Mock(),end_session=Mock())
    attempts={}
    def commit(key,instance,data):
        attempts[key.index]=attempts.get(key.index,0)+1
        if mode=='never' or (mode=='retry' and attempts[key.index]==1):
            return Mock(result=lambda **kwargs:True)  # false-success regression
        chunk=pickle.loads(data)[0]
        payload=chunk.contiguous().view(torch.uint8).numpy().tobytes()
        if mode=='corrupt':payload=b'X'*len(payload)
        (tmp_path/f'{key.index}.data').write_bytes(payload)
        if mode=='extra':(tmp_path/'extra.data').write_bytes(payload)
        return Mock(result=lambda **kwargs:True)
    store.client.commit_store.side_effect=commit
    return store,attempts


def test_false_success_retried_and_all_payloads_verified(tmp_path,monkeypatch):
    store,attempts=fixture(tmp_path,monkeypatch,'retry')
    chunks=[torch.arange(8,dtype=torch.int16),torch.ones(8,dtype=torch.int16)]
    result=store.store_persisted_fs([1],'case',chunks)
    assert attempts=={0:2,1:2} and result['retries']==2
    assert store.client.end_session.call_count==4
    for item in result['chunks']:
        assert item['sha256']==hashlib.sha256((tmp_path/item['file']).read_bytes()).hexdigest()


@pytest.mark.parametrize('mode,error',[('never',TimeoutError),('corrupt',ValueError),('extra',ValueError)])
def test_unpersisted_or_corrupt_store_is_not_admitted(tmp_path,monkeypatch,mode,error):
    store,_=fixture(tmp_path,monkeypatch,mode)
    with pytest.raises(error):store.store_persisted_fs([1],'case',[torch.zeros(8)],timeout=2)


def test_raw_backend_is_not_silently_treated_as_filesystem(tmp_path,monkeypatch):
    store,_=fixture(tmp_path,monkeypatch,'ok')
    store.command=['--l2-adapter','{"type":"raw_block"}']
    with pytest.raises(ValueError):store.store_persisted_fs([1],'case',[torch.zeros(8)])
