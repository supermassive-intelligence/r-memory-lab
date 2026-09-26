import copy
import hashlib
import pytest
from experimental.rg_acceptance import BASE_ARMS,RETRIEVAL_ARMS,STORAGE_GUARDS,validate_prefix_records


def fixture(retrieved=False):
    parent=dict(status='complete',equivalence_numerical_pass=True,
        guards={n:'PASS' for n in ('inherited','coverage','mp-gpu')})
    cases=[]
    for i in range(10):
        text='document' if retrieved or i else ''
        cases.append(dict(id=str(i),question='question',answers=['answer'],evidence=text,
            evidence_sha256=hashlib.sha256(text.encode()).hexdigest(),prefix_token_ids=[1],
            suffix_token_ids=[2],kv_identity='identity'))
    arms=BASE_ARMS+(RETRIEVAL_ARMS if retrieved else ())
    prefix=dict(status='complete',storage_numerical_pass=True,expected_cases=10,
        evidence_mode='retrieved' if retrieved else 'supplied',
        guards={n:'PASS' for n in STORAGE_GUARDS},records={})
    for arm in arms:
        prefix['records'][arm]=[dict(c,arm=arm,output_text='answer',output_token_ids=[3],
            step_logits_sha256=['a'*64],all_logits_finite=True) for c in cases]
    if retrieved:prefix['guards'].update({n+'_identity':'PASS' for n in RETRIEVAL_ARMS})
    return parent,prefix,dict(cases=cases)


@pytest.mark.parametrize('retrieved',[False,True])
def test_complete_evidence_passes_only_structural_scope(retrieved):
    result=validate_prefix_records(*fixture(retrieved))
    assert result['status']=='PASS' and 'not held-out quality' in result['scope']


@pytest.mark.parametrize('guard',STORAGE_GUARDS)
def test_missing_required_guard_cannot_be_hidden_by_true_summary(guard):
    p,s,m=fixture();del s['guards'][guard]
    with pytest.raises(ValueError):validate_prefix_records(p,s,m)


@pytest.mark.parametrize('change',['truthy-string','empty-guards','missing-arm','truncated-A','order',
    'logit-mismatch','no-logits','wrong-question','duplicate-id','unknown-guard','extra-arm','missing-R'])
def test_negative_guard_injection(change):
    p,s,m=fixture()
    if change=='truthy-string':s['storage_numerical_pass']='FAIL'
    elif change=='empty-guards':s['guards']={}
    elif change=='missing-arm':del s['records']['L']
    elif change=='truncated-A':s['records']['A'].pop()
    elif change=='order':s['records']['L'].reverse()
    elif change=='logit-mismatch':s['records']['L'][0]['step_logits_sha256']=['b'*64]
    elif change=='no-logits':s['records']['F'][0]['step_logits_sha256']=[]
    elif change=='wrong-question':s['records']['L'][0]['question']='changed'
    elif change=='duplicate-id':m['cases'][1]['id']='0'
    elif change=='unknown-guard':s['guards']['new-unresolved-guard']='UNKNOWN'
    elif change=='extra-arm':s['records']['unvalidated-new-path']=copy.deepcopy(s['records']['A'])
    with pytest.raises(ValueError):validate_prefix_records(p,s,m,require_r=change=='missing-R')


def test_retrieval_requires_exercised_forced_endpoints():
    p,s,m=fixture(True)
    for arm in RETRIEVAL_ARMS:
        changed=copy.deepcopy(s);del changed['records'][arm]
        with pytest.raises(ValueError):validate_prefix_records(p,changed,m)
        changed=copy.deepcopy(s);changed['records'][arm][0]['step_logits_sha256']=['b'*64]
        with pytest.raises(ValueError):validate_prefix_records(p,changed,m)


def test_supplied_absence_guard_cannot_pass_vacuously():
    p,s,m=fixture(True);s['evidence_mode']='supplied'
    for arm in RETRIEVAL_ARMS:del s['records'][arm]
    with pytest.raises(ValueError,match='vacuous'):validate_prefix_records(p,s,m)
