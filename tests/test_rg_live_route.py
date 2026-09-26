import copy
import sqlite3
import pytest
from experimental.rg_cache_identity import identity
from experimental.rg_retrieval import init_index, set_meta, add_chunks, digest
from experimental.rg_live_route import build_manifest, route


@pytest.fixture
def fixture():
    db = sqlite3.connect(':memory:'); init_index(db); set_meta(db, 'status', 'complete')
    candidate = dict(chunk_id='1:0:0', doc_id='1', paragraph=0, token_start=0,
                     title='Article', text='Michael is a carpenter.', token_count=5,
                     content_sha256=digest('Michael is a carpenter.'))
    add_chunks(db, [candidate])
    encoding = dict(model_snapshot='revision', config_sha256='a'*64,
                    tokenizer_sha256='b'*64, prefix_policy='frozen-prefix')
    meta = dict(layers=36, heads=2, tokens=2, head_dim=128, dtype='torch.bfloat16',
                chunk_size=256, chunks=1, layout='2,L,T,H*D')
    key = identity(encoding, [1, 2], meta)
    entry = dict(identity=key, encoding=encoding, tokens=[1, 2], metadata=meta, chunk_hashes=['c'*64])
    row = dict(id='q1', prefix_token_ids=[1, 2], stable_identity=key,
               evidence=candidate['text'], evidence_sha256=candidate['content_sha256'],
               selected_chunk_id=candidate['chunk_id'], candidates=[candidate])
    state = dict(status='complete', source_evidence_mode='retrieved', evidence_encode_calls=0,
                 expected_cases=1, catalog={key:entry}, encoding=encoding, records={'stable-L':[row]})
    yield db, state
    db.close()


def test_live_question_routes_existing_key(fixture):
    db, state = fixture; manifest = build_manifest(state)
    answer = route(db, 'Michael occupation?', manifest, state['encoding'])
    assert answer['status'] == 'manifest-hit'
    assert answer['cache_key'] in state['catalog']
    assert answer == route(db, 'Michael occupation?', manifest, state['encoding'])
    assert 'answers' not in answer


def test_miss_empty_and_answer_blind_interface(fixture):
    db, state = fixture
    assert route(db, 'Michael', {}, state['encoding'])['status'] == 'cache-miss'
    assert route(db, '!!!', build_manifest(state), state['encoding'])['status'] == 'empty-retrieval'
    with pytest.raises(TypeError):
        route(db, {'question':'Michael', 'answers':['carpenter']}, {}, state['encoding'])


def test_incompatible_encoding_fails_not_fallback(fixture):
    db, state = fixture; encoding = dict(state['encoding'], model_snapshot='other')
    with pytest.raises(ValueError):route(db, 'Michael', build_manifest(state), encoding)


@pytest.mark.parametrize('damage', ['incomplete', 'mode', 'encoding', 'tokens', 'content',
                                   'selection', 'count', 'key', 'duplicate', 'reencode'])
def test_bad_catalog_rejected(fixture, damage):
    _, original = fixture; state = copy.deepcopy(original)
    row = state['records']['stable-L'][0]
    if damage == 'incomplete':state['status'] = 'generating'
    elif damage == 'mode':state['source_evidence_mode'] = 'supplied'
    elif damage == 'encoding':state['encoding']['model_snapshot'] = 'wrong'
    elif damage == 'tokens':row['prefix_token_ids'] = [9]
    elif damage == 'content':row['evidence'] += ' changed'
    elif damage == 'selection':row['selected_chunk_id'] = 'wrong'
    elif damage == 'count':state['expected_cases'] = 2
    elif damage == 'key':state['catalog'] = {'wrong':next(iter(state['catalog'].values()))}
    elif damage == 'duplicate':state['records']['stable-L'].append(row);state['expected_cases'] = 2
    else:state['evidence_encode_calls'] = 1
    with pytest.raises((ValueError, KeyError)):build_manifest(state)


def test_same_chunk_cannot_have_conflicting_prefix(fixture):
    _, state = fixture; row = copy.deepcopy(state['records']['stable-L'][0])
    entry = copy.deepcopy(state['catalog'][row['stable_identity']])
    entry['tokens'] = [3, 4]; entry['identity'] = identity(entry['encoding'], entry['tokens'], entry['metadata'])
    state['catalog'][entry['identity']] = entry
    row.update(id='q2', stable_identity=entry['identity'], prefix_token_ids=[3, 4])
    state['records']['stable-L'].append(row);state['expected_cases'] = 2
    with pytest.raises(ValueError):build_manifest(state)


def test_question_ids_do_not_change_cache_key(fixture):
    _, state = fixture; before = build_manifest(state)
    state['records']['stable-L'][0].update(id='another-question', answers=['wrong reference'])
    assert build_manifest(state) == before
