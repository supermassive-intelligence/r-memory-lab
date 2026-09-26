"""Fail-closed evidence checks, independent of producer summary flags.

No acceptance thresholds live here: this enforces already declared exact
integrity, coverage, cardinality and endpoint contracts. Research success and
held-out statistical quality remain separate, unimplemented admission gates.
"""
import hashlib

BASE_ARMS=('A','A-repeat','F','F-repeat','L-zero','L','L-repeat')
RETRIEVAL_ARMS=('retrieval-zero','cache-miss','selection-limit')
STORAGE_GUARDS=('missing_key_before','missing_key_after','kv_roundtrip_bytes',
    'restart_L2_delivery','lock_lifecycle','ten_generations_each','A_exact_replay',
    'A_absent_evidence_identity','F_all_step_identity_to_F','F-repeat_all_step_identity_to_F',
    'L-zero_all_step_identity_to_F','L_all_step_identity_to_F','L-repeat_all_step_identity_to_F')


def require(condition,message):
    if not condition:raise ValueError(message)


def require_statuses(values,names):
    require(isinstance(values,dict),'missing guard map')
    for name in names:require(values.get(name)=='PASS','missing/failed/unresolved guard: '+name)
    require(all(v=='PASS' for v in values.values()),'an additional guard is not PASS')


def identical(a,b):
    return a['output_token_ids']==b['output_token_ids'] and a['step_logits_sha256']==b['step_logits_sha256']


def validate_prefix_records(parent,prefix,manifest,require_r=False):
    require(parent.get('status')=='complete','parent execution incomplete')
    require(parent.get('equivalence_numerical_pass') is True,'numerical prerequisite missing/failed')
    require_statuses(parent.get('guards'),('inherited','coverage','mp-gpu'))
    require(prefix.get('status')=='complete','prefix execution incomplete')
    require(prefix.get('storage_numerical_pass') is True,'storage flag missing/failed')
    require_statuses(prefix.get('guards'),STORAGE_GUARDS)
    count=prefix.get('expected_cases')
    require(type(count) is int and count in (10,100),'invalid expected denominator')
    cases=manifest.get('cases',[])
    require(len(cases)==count,'manifest cardinality differs from declared workload')
    ids=[case['id'] for case in cases]
    require(len(set(ids))==count,'duplicate query IDs in workload')
    retrieved=prefix.get('evidence_mode')=='retrieved'
    arms=BASE_ARMS+(RETRIEVAL_ARMS if retrieved else ())+(('W','R','R-repeat') if require_r else ())
    records=prefix.get('records',{})
    require(set(records)==set(arms),'unexpected or missing arm; declare its controls before admission')
    for arm in arms:
        rows=records.get(arm,[])
        require(len(rows)==count,'missing/truncated arm: '+arm)
        for row,case in zip(rows,cases):
            require(row.get('arm')==arm and row.get('id')==case['id'],'arm/case order mismatch')
            for key in ('question','answers','evidence_sha256','prefix_token_ids','suffix_token_ids','kv_identity'):
                require(key in case and row.get(key)==case[key],'matched input drift: '+key)
            require(hashlib.sha256(row['evidence'].encode()).hexdigest()==row['evidence_sha256'],'evidence hash drift')
            require(isinstance(row.get('output_text'),str),'missing full generation text')
            tokens=row.get('output_token_ids');hashes=row.get('step_logits_sha256')
            require(isinstance(tokens,list) and 1<=len(tokens)<=32 and all(type(x) is int for x in tokens),'missing/invalid token trace')
            require(isinstance(hashes,list) and len(hashes)==len(tokens),'missing generated-step logits')
            require(all(isinstance(x,str) and len(x)==64 and all(c in '0123456789abcdef' for c in x) for x in hashes),'invalid logit digest')
            require(row.get('all_logits_finite') is True,'nonfinite/unverified logits')
    for arm in ('F-repeat','L-zero','L','L-repeat'):
        require(all(identical(a,b) for a,b in zip(records[arm],records['F'])),'full/zero storage endpoint mismatch: '+arm)
    require(all(identical(a,b) for a,b in zip(records['A'],records['A-repeat'])),'A replay mismatch')
    empty=[i for i,c in enumerate(cases) if not c['evidence']]
    for i in empty:require(identical(records['A'][i],records['F'][i]),'absent-evidence mismatch')
    if retrieved:
        for arm,control in [('retrieval-zero','A'),('cache-miss','A'),('selection-limit','F')]:
            require(all(identical(a,b) for a,b in zip(records[arm],records[control])),'forced retrieval endpoint mismatch: '+arm)
            require(prefix.get('guards',{}).get(arm+'_identity')=='PASS','forced endpoint guard missing')
    else:require(bool(empty),'absent-evidence guard was vacuous: no applicable cases')
    if require_r:
        require(prefix.get('r_prefix_numerical_pass') is True,'R numerical summary missing/failed')
        guards=prefix.get('r_prefix_guards',{})
        for name in ('complete_samples','exact_replay','same_input_numerical','absent_evidence_identity'):
            require(guards.get(name) is True,'R guard missing/failed: '+name)
        require(all(identical(a,b) for a,b in zip(records['R'],records['R-repeat'])),'R replay mismatch')
        for arm in ('W','R','R-repeat'):
            require(all(row.get('same_input_numerical_pass') is True for row in records[arm]),'R layer guard missing/failed')
    return dict(status='PASS',expected_cases=count,arms=list(arms),natural_absent_cases=len(empty),
                scope='development structural/endpoint integrity, not held-out quality acceptance')
