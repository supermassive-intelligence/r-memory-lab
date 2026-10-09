import pytest
from experimental.rg_blend_route import route,build_manifest
from experimental.rg_live_route import encoding_id
from experimental.rg_retrieval import digest


def inputs(monkeypatch):
    candidates=[dict(chunk_id=str(i),text='document '+str(i),content_sha256=digest('document '+str(i))) for i in range(3)]
    monkeypatch.setattr('experimental.rg_blend_route.search',lambda db,q:candidates if q else [])
    encoding={'version':'fixture'}
    manifest={c['chunk_id']:dict(cache_key='kv-'+c['chunk_id'],content_sha256=c['content_sha256'],
        encoding_id=encoding_id(encoding)) for c in candidates}
    return encoding,manifest


def test_top_two_and_empty(monkeypatch):
    encoding,manifest=inputs(monkeypatch)
    row=route(None,'question',manifest,encoding)
    assert row['document_keys']==['kv-0','kv-1'] and row['selected_chunk_ids']==['0','1']
    assert route(None,'',manifest,encoding)['status']=='empty-retrieval'


def test_partial_miss_does_not_substitute_lower_rank(monkeypatch):
    encoding,manifest=inputs(monkeypatch);del manifest['1']
    row=route(None,'question',manifest,encoding)
    assert row['status']=='cache-miss' and row['document_keys']==[]
    assert row['selected_chunk_ids']==['0','1'] and row['resolutions'][0]['cache_key']=='kv-0'


def test_corrupt_is_not_miss(monkeypatch):
    encoding,manifest=inputs(monkeypatch);manifest['0']['content_sha256']='wrong'
    with pytest.raises(ValueError):route(None,'question',manifest,encoding)


def test_manifest_binds_full_content_and_cache_tokens():
    import copy
    import torch
    from experimental.rg_blend_cache import pack_document,POLICY
    encoding=dict(model_snapshot='test',config_sha256='config',tokenizer_sha256='tokens',capture_policy=POLICY)
    pair=torch.zeros(1,2,3,4,dtype=torch.bfloat16)
    entry,_=pack_document(encoding,[1,2,3],[(pair,pair),(pair,pair)])
    case=dict(documents=[dict(chunk_id='c',content_sha256=digest('full text'),tokens=[1,2,3])],document_keys=[entry['key']])
    state=dict(status='complete',online_document_encodes=0,encoding=encoding,catalog={entry['key']:entry},cases=[case])
    assert build_manifest(state)['c']['cache_key']==entry['key']
    for mutation in ('tokens','identity','online'):
        changed=copy.deepcopy(state)
        if mutation=='tokens':changed['cases'][0]['documents'][0]['tokens']=[1]
        if mutation=='identity':changed['catalog'][entry['key']]['encoding']['model_snapshot']='other'
        if mutation=='online':changed['online_document_encodes']=1
        with pytest.raises(ValueError):build_manifest(changed)


class Tokenizer:
    def encode(self,text,add_special_tokens=False):return list(text.encode())


def test_fresh_composition_and_missing_fallback(monkeypatch):
    from experimental.rg_blend_route import compose_request,resolve_candidates
    encoding,manifest=inputs(monkeypatch);row=route(None,'new question',manifest,encoding)
    tokenizer=Tokenizer()
    catalog={key:dict(tokens=tokenizer.encode(c['text']+'\n\n')) for key,c in zip(row['document_keys'],row['candidates'])}
    composed=compose_request(row,catalog,tokenizer)
    assert composed['question']=='new question' and len(composed['documents'])==2
    missing=resolve_candidates('new question',row['candidates'],{},encoding)
    absent=compose_request(missing,catalog,tokenizer)
    assert absent['documents']==[] and absent['document_keys']==[]
    assert absent['suffix_tokens']==composed['suffix_tokens']
    assert len(composed['full_tokens'])-len(absent['full_tokens'])==sum(len(d['tokens']) for d in composed['documents'])
    catalog[row['document_keys'][0]]['tokens']=[999]
    with pytest.raises(ValueError,match='token mismatch'):compose_request(row,catalog,tokenizer)


def test_composition_rejects_invalid_route_shape(monkeypatch):
    from experimental.rg_blend_route import compose_request
    encoding,manifest=inputs(monkeypatch);row=route(None,'q',manifest,encoding)
    row['document_keys']=[]
    with pytest.raises(ValueError,match='cardinality'):compose_request(row,{},Tokenizer())
    row['status']='unknown'
    with pytest.raises(ValueError,match='unknown route'):compose_request(row,{},Tokenizer())
