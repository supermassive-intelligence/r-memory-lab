import sqlite3
import pytest
from experimental.rg_retrieval import init_index,set_meta,get_meta,page_chunks,add_chunks,search,resolve_cache


class Tokens:
    def encode(self,text,add_special_tokens=False):return list(text.encode())
    def decode(self,ids,skip_special_tokens=False):return bytes(ids).decode()


@pytest.fixture
def index():
    c=sqlite3.connect(':memory:');init_index(c)
    for id,text in [('2','Michael is a carpenter.'),('1','Michael is a carpenter.'),('3','Venus is a planet.')]:
        add_chunks(c,page_chunks(dict(wikipedia_id=id,wikipedia_title='Article',text=[text]),Tokens()))
    set_meta(c,'status','complete');yield c;c.close()


def test_rank_ties_and_repeat(index):
    a=search(index,'Michael occupation?')
    assert a==search(index,'Michael occupation?')
    assert [r['doc_id'] for r in a]==['1','2']
    assert search(index,'Michael',limit=1)==a[:1]
    assert get_meta(index,'absent',42)==42


def test_empty_and_query_injection(index):
    assert search(index,'!!!')==[]
    assert search(index,'nonexistentword')==[]
    assert search(index,'"Michael" : **')
    with pytest.raises(TypeError):search(index,dict(question='Michael',answers=['carpenter']))
    with pytest.raises(ValueError):search(index,'Michael',limit=6)
    set_meta(index,'status','building')
    with pytest.raises(ValueError):search(index,'Michael')


def test_cache_resolution_miss_and_integrity(index):
    selected=search(index,'Michael')[0]
    assert resolve_cache(None,{},'v1')['status']=='empty-retrieval'
    assert resolve_cache(selected,{},'v1')['status']=='cache-miss'
    entry=dict(content_sha256=selected['content_sha256'],encoding_id='v1',cache_key='key')
    manifest={selected['chunk_id']:entry}
    assert resolve_cache(selected,manifest,'v1')==dict(status='manifest-hit',cache_key='key')
    with pytest.raises(ValueError):resolve_cache(selected,manifest,'v2')
    with pytest.raises(ValueError):resolve_cache(dict(selected,text='wrong'),manifest,'v1')
    with pytest.raises(ValueError):add_chunks(index,[dict(selected,text='corrupt')])


def test_chunking_complete_and_stable():
    text='x'*600
    page=dict(wikipedia_id='7',wikipedia_title='Title',text=[text,'','last'])
    chunks=list(page_chunks(page,Tokens()))
    assert [c['token_count'] for c in chunks]==[256,256,88,4]
    assert ''.join(c['text'] for c in chunks[:3])==text
    assert [c['chunk_id'] for c in chunks]==['7:0:0','7:0:256','7:0:512','7:2:0']
    assert chunks==list(page_chunks(page,Tokens()))
    with pytest.raises(ValueError):list(page_chunks(dict(page,text=[None]),Tokens()))


def test_answers_and_gold_evidence_cannot_change_selection(index):
    from scripts.rg_retrieval_manifest import select_case
    record=dict(id='case',question='Michael occupation?',accepted_answers=['carpenter'],evidence='oracle one')
    a=select_case(index,record)
    b=select_case(index,dict(record,accepted_answers=['astronaut'],evidence='oracle two',resolved_provenance=['3']))
    assert a['candidates']==b['candidates'] and a['evidence']==b['evidence']
    assert a['answers']!=b['answers']
    assert a['cache_status']=='not-looked-up'
    empty=select_case(index,dict(record,question='unfindableword'))
    assert empty['evidence']=='' and empty['retrieval_fallback']=='empty-retrieval'


def test_generation_loader_requires_exact_selected_manifest(index,tmp_path):
    import hashlib,json
    from scripts.rg_retrieval_manifest import select_case
    from scripts.rg_lmcache_prefix import load_selected_cases
    rows=[select_case(index,dict(id='one',question='Michael',accepted_answers=['carpenter']))]
    path=tmp_path/'selection.json'
    def save():
        path.write_text(json.dumps(dict(cases=rows,selection_mode='retrieved',selection_sha256=
            hashlib.sha256(json.dumps(rows,sort_keys=True,ensure_ascii=False).encode()).hexdigest())))
    save();assert load_selected_cases(path,1)==rows
    with pytest.raises(ValueError):load_selected_cases(path,2)
    rows[0]['evidence']='wrong';save()
    with pytest.raises(ValueError):load_selected_cases(path,1)


@pytest.mark.parametrize('damage',['duplicate-query','duplicate-chunk','bad-rank','wrong-order','truthy-replay','wrong-fallback'])
def test_selection_injection_rejected(index,tmp_path,damage):
    import copy,hashlib,json
    from scripts.rg_retrieval_manifest import select_case
    from scripts.rg_lmcache_prefix import load_selected_cases
    row=select_case(index,dict(id='one',question='Michael',accepted_answers=['carpenter']))
    rows=[row]
    if damage=='duplicate-query':rows.append(copy.deepcopy(row))
    elif damage=='duplicate-chunk':row['candidates'].append(copy.deepcopy(row['candidates'][0]))
    elif damage=='bad-rank':row['candidates'][0]['bm25_rank']=float('nan')
    elif damage=='wrong-order':row['candidates'].reverse()
    elif damage=='truthy-replay':row['retrieval_repeat_exact']='PASS'
    elif damage=='wrong-fallback':row['retrieval_fallback']='support-miss'
    p=tmp_path/'cases.json'
    p.write_text(json.dumps(dict(cases=rows,selection_mode='retrieved',selection_sha256=
        hashlib.sha256(json.dumps(rows,sort_keys=True,ensure_ascii=False).encode()).hexdigest())))
    with pytest.raises(ValueError):load_selected_cases(p,1)
