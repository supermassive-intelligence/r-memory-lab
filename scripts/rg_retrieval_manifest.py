"""Freeze answer-blind retrieval on the existing development allocation.

No confirmation records, answer-conditioned filtering, or retriever tuning.
This produces a selection manifest, not a quality result or KV-cache hit claim.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sqlite3
import time
from experimental.rg_retrieval import POLICY,get_meta,search,digest
from scripts.artifacts import write


def select_case(connection,record):
    # Only the question crosses the selection interface. References are attached
    # afterwards for the scorer, never used for selection/fallback decisions.
    candidates=search(connection,record['question'])
    repeated=search(connection,record['question'])
    if candidates!=repeated:raise ValueError('retrieval replay mismatch')
    chosen=candidates[0] if candidates else None
    evidence='' if chosen is None else chosen['text']
    return dict(id=record['id'],question=record['question'],answers=record['accepted_answers'],
        evidence=evidence,evidence_sha256=digest(evidence),selection_mode='retrieved',
        candidates=candidates,selected_chunk_id=None if chosen is None else chosen['chunk_id'],
        retrieval_fallback='empty-retrieval' if chosen is None else None,
        cache_status='not-looked-up',retrieval_repeat_exact=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--index',type=Path,required=True)
    p.add_argument('--development',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--index-audit',type=Path,required=True)
    p.add_argument('--count',type=int,choices=(10,100),default=100);a=p.parse_args()
    a.out.mkdir(parents=True,exist_ok=False)
    state=dict(status='selecting',quality_admissible=False,scope='development retrieval manifest only',
        policy=POLICY,expected_cases=a.count,started_unix=time.time())
    write(a.out/'state.json',state)
    try:
        db=sqlite3.connect(a.index.resolve().as_uri()+'?mode=ro',uri=True)
        db.execute('PRAGMA mmap_size=0')
        db.execute('PRAGMA cache_size=-32768')
        audit=json.loads(a.index_audit.read_text())
        from experimental.rg_acceptance import require,require_statuses
        require(audit.get('status')=='complete','independent index audit incomplete')
        require_statuses(audit.get('checks'),('metadata','source_sha256','sqlite_quick_check','chunk_count','unique_chunk_ids','source_pages'))
        stat=a.index.stat()
        require(audit.get('index')==str(a.index.resolve()) and audit.get('index_stat')==dict(size=stat.st_size,mtime_ns=stat.st_mtime_ns),'index changed after audit')
        if get_meta(db,'status')!='complete' or not get_meta(db,'source_verified'):
            raise ValueError('full-corpus index identity not verified')
        identity=get_meta(db,'identity')
        if identity['policy']!=POLICY or identity.get('smoke_pages'):
            raise ValueError('index/retriever policy mismatch')
        state.update(index_identity=identity,index_progress=get_meta(db,'progress'),sqlite_version=sqlite3.sqlite_version,
            index_audit_sha256=hashlib.sha256(a.index_audit.read_bytes()).hexdigest(),index_sha256=audit['index_sha256'],
            development_sha256=hashlib.sha256(a.development.read_bytes()).hexdigest(),
            selection='first existing historical development records, no correctness filtering')
        records=json.loads(a.development.read_text())['materialized_records'][:a.count]
        if len(records)!=a.count:raise ValueError('incomplete development allocation')
        if len({record['id'] for record in records})!=a.count:raise ValueError('duplicate development IDs')
        cases=[]
        for i,record in enumerate(records):
            case=select_case(db,record);cases.append(case)
            write(a.out/'cases.partial.json',dict(cases=cases,quality_admissible=False))
            state['completed_cases']=i+1;write(a.out/'state.json',state)
            print('COMPLETE_RETRIEVAL_CASE',json.dumps(case),flush=True)
        selection_bytes=json.dumps(cases,sort_keys=True,ensure_ascii=False).encode()
        write(a.out/'cases.json',dict(cases=cases,selection_sha256=hashlib.sha256(selection_bytes).hexdigest(),
            index_identity=identity,selection_mode='retrieved',quality_admissible=False,
            index_sha256=audit['index_sha256'],index_audit_sha256=state['index_audit_sha256'],
            note='selected evidence is not oracle evidence; KV not yet populated or looked up'))
        state.update(status='complete',selection_sha256=hashlib.sha256(selection_bytes).hexdigest(),
            next_action='prewarm selected document KV offline; same-arm forced-selection identity then 10-case A/F/L smoke')
        db.close()
    except BaseException as error:
        state.update(status='failed',error=repr(error));raise
    finally:write(a.out/'state.json',state)


if __name__=='__main__':main()
