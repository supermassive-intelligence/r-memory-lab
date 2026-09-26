"""Extracted LMCache prefix transport; historical generation harness is not bundled."""
import hashlib
import json
import os
from pathlib import Path
import pickle
import re
import socket
import subprocess
import sys
import time
import urllib.request
import torch
from scripts.artifacts import write

def ensure_ports_available(ports):
    for port in ports:
        with socket.socket() as probe:
            # Match the server's restart behavior: TIME_WAIT is harmless,
            # whereas an actual live listener still rejects the bind.
            probe.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1)
            probe.bind(('127.0.0.1',port))


def validate_l2_reads(log,expected_chunk_counts):
    reads=[(int(a),int(b)) for a,b in re.findall(r'\((\d+) L1, (\d+) L2\)',log)]
    reads=[(a,b) for a,b in reads if a+b]
    return reads==[(0,n) for n in expected_chunk_counts]


def service_ports(base=18580):
    if type(base) is not int or not 1024<=base<65535:
        raise ValueError('unprivileged loopback port pair required')
    return base,base+1


class PrefixStore:
    """Owned loopback MP service; restart clears L1, not the OS page cache."""
    def __init__(self,root,model_name,layers,hidden,namespace=None,reuse_existing=False,port_base=18580):
        from lmcache.v1.multiprocess.custom_types import RegisterEngineDrivenContextPayload
        self.root=root
        self.disk=root/'l2'
        self.disk.mkdir(exist_ok=reuse_existing)
        self.namespace=root.parent.name if namespace is None else namespace
        self.rpc_port,self.http_port=service_ports(port_base)
        self.model_name=model_name
        self.server=None
        self.client=None
        self.sequence=0
        self.events=[]
        self.metadata=RegisterEngineDrivenContextPayload(instance_id=1,model_name=model_name,
            world_size=1,block_size=16,num_layers=layers,hidden_dim_size=hidden,
            dtype_str='bfloat16',use_mla=False,num_physical_slots=256)
        self.command=[sys.executable,'-u','-m','lmcache.v1.multiprocess.http_server',
            '--host','127.0.0.1','--port',str(self.rpc_port),'--http-host','127.0.0.1','--http-port',str(self.http_port),
            '--l1-size-gb','.5','--eviction-policy','LRU','--chunk-size','256','--max-workers','2',
            '--supported-transfer-mode','engine_driven','--shm-name','',
            '--l2-adapter',json.dumps(dict(type='fs',base_path=str(self.disk)))]

    def start(self,label):
        from lmcache.v1.multiprocess.transport.factory import RequestClientFactory
        ensure_ports_available((self.rpc_port,self.http_port))
        with (self.root/f'{label}-server.log').open('x') as log:
            self.server=subprocess.Popen(self.command,stdout=log,stderr=subprocess.STDOUT)
        self.events.append(dict(event='start',label=label,pid=self.server.pid,command=self.command))
        write(self.root/'storage-events.json',self.events)
        deadline=time.monotonic()+120
        while True:
            self.alive()
            try:
                with urllib.request.urlopen(f'http://127.0.0.1:{self.http_port}/healthcheck',timeout=2) as response:
                    if response.status==200:
                        break
            except OSError:
                pass
            if time.monotonic()>deadline:
                raise TimeoutError('cache server healthcheck')
            time.sleep(.2)
        self.client=RequestClientFactory.create(f'tcp://127.0.0.1:{self.rpc_port}')
        self.client.register_kv_cache_engine_driven_context(self.metadata).result(timeout=60)

    def alive(self):
        if self.server is None or self.server.poll() is not None:
            raise RuntimeError('owned LMCache server exited')

    def stop(self):
        if self.client is not None:
            self.client.close()
            self.client=None
        if self.server is not None and self.server.poll() is None:
            self.server.terminate()
            try:
                self.server.wait(timeout=30)
            except subprocess.TimeoutExpired:
                self.server.kill()
                self.server.wait(timeout=10)
        self.server=None

    def key(self,tokens,identity,index,end_index=None):
        from lmcache.v1.multiprocess.custom_types import IPCCacheServerKey
        self.sequence+=1
        padded=tokens+[0]*((-len(tokens))%256)
        return IPCCacheServerKey.from_token_ids(self.model_name,1,0,padded,
            start=index*256,end=(index+1 if end_index is None else end_index)*256,request_id=f'prefix-{self.sequence}',
            cache_salt=self.namespace+'-'+identity)

    def store(self,tokens,identity,chunks):
        for index,chunk in enumerate(chunks):
            key=self.key(tokens,identity,index)
            self.client.prepare_store(key,1).result(timeout=60)
            if not self.client.commit_store(key,1,pickle.dumps([chunk])).result(timeout=60):
                raise ValueError('store rejected')

    def retrieve(self,tokens,identity,count):
        # Lookup hashes the whole prefix, irrespective of the retrieve range.
        # Fetch all its chunks once, not one full-prefix prefetch per chunk.
        key=self.key(tokens,identity,0,end_index=count)
        self.client.lookup(key.no_worker_id_version(),1).result(timeout=60)
        deadline=time.monotonic()+120
        while True:
            hits=self.client.query_prefetch_status(key.request_id).result(timeout=60)
            if hits is not None:
                break
            self.alive()
            if time.monotonic()>deadline:
                raise TimeoutError('LMCache prefetch')
            time.sleep(.01)
        if hits==0:
            self.client.end_session(key.request_id).result(timeout=60)
            return None
        response=self.client.prepare_retrieve(key,1).result(timeout=60)
        if not response.success:
            raise ValueError('lookup hit but retrieve failed')
        # Only bytes produced by this owned, loopback-only trusted service.
        chunks=pickle.loads(response.data)
        if not self.client.commit_retrieve(key,1).result(timeout=60):
            raise ValueError('retrieve finalization failed')
        self.client.end_session(key.request_id).result(timeout=60)
        if len(chunks)!=count:
            raise ValueError('prefix transport chunk count mismatch')
        return chunks

    def store_persisted_fs(self,tokens,identity,chunks,timeout=120):
        """Offline filesystem population with bounded backpressure and raw proof.

        Pickle COMMIT_STORE can acknowledge an empty reservation after L1 OOM.
        Do not trust that acknowledgement: admit each new file only after its
        complete raw payload matches. Retry a missing write after allowing the
        eviction controller to run. One 120-second deadline covers the case.
        Timed retrieval, L1 capacity and eviction settings are not changed.
        Requires this service's exclusively owned filesystem output directory.
        """
        adapter=json.loads(self.command[self.command.index('--l2-adapter')+1])
        if adapter.get('type')!='fs' or Path(adapter['base_path'])!=self.disk:
            raise ValueError('durable population requires owned filesystem adapter')
        deadline=time.monotonic()+timeout
        known=set(self.disk.glob('*.data'));receipts=[]
        for index,chunk in enumerate(chunks):
            payload=chunk.contiguous().view(torch.uint8).numpy().tobytes()
            expected=hashlib.sha256(payload).hexdigest();size=len(payload);del payload
            key=self.key(tokens,identity,index);attempts=0;next_try=0
            while True:
                self.alive()
                added=set(self.disk.glob('*.data'))-known
                if len(added)>1:
                    raise ValueError('unexpected extra persistence objects')
                if added:
                    path=next(iter(added))
                    observed_size=path.stat().st_size
                    if observed_size>size:
                        raise ValueError('persisted payload is oversized')
                    if observed_size==size:
                        if hashlib.sha256(path.read_bytes()).hexdigest()!=expected:
                            raise ValueError('persisted payload checksum mismatch')
                        receipts.append(dict(index=index,file=path.name,bytes=size,sha256=expected,attempts=attempts))
                        known.add(path);break
                now=time.monotonic()
                if now>=deadline:
                    raise TimeoutError(f'persisted store incomplete: {identity} chunk {index}; attempts={attempts}')
                if now>=next_try and not added:
                    self.client.prepare_store(key,1).result(timeout=min(60,deadline-now))
                    self.client.commit_store(key,1,pickle.dumps([chunk])).result(timeout=max(.001,min(60,deadline-time.monotonic())))
                    self.client.end_session(key.request_id).result(timeout=max(.001,min(60,deadline-time.monotonic())))
                    attempts+=1;next_try=time.monotonic()+1
                time.sleep(.05)
        return dict(identity=identity,chunks=receipts,retries=sum(max(0,r['attempts']-1) for r in receipts))

    def flush(self,expected_bytes):
        deadline=time.monotonic()+120
        while sum(p.stat().st_size for p in self.disk.glob('*.data'))!=expected_bytes:
            self.alive()
            if time.monotonic()>deadline:
                raise TimeoutError('L2 persistence byte count')
            time.sleep(.1)


def load_selected_cases(path,count):
    """Validate frozen retrieved inputs; never select or filter by references."""
    payload=json.loads(Path(path).read_text())
    cases=payload['cases']
    encoded=json.dumps(cases,sort_keys=True,ensure_ascii=False).encode()
    if hashlib.sha256(encoded).hexdigest()!=payload.get('selection_sha256'):
        raise ValueError('retrieval selection manifest hash mismatch')
    if payload.get('selection_mode')!='retrieved' or len(cases)<count:
        raise ValueError('retrieval selection mode or count mismatch')
    if len({row['id'] for row in cases})!=len(cases):
        raise ValueError('duplicate selected query IDs')
    for row in cases:
        if row.get('selection_mode')!='retrieved' or row.get('retrieval_repeat_exact') is not True:
            raise ValueError('unverified retrieval case')
        if hashlib.sha256(row['evidence'].encode()).hexdigest()!=row['evidence_sha256']:
            raise ValueError('retrieved evidence hash mismatch')
        candidates=row['candidates']
        import math
        if not isinstance(candidates,list) or len(candidates)>5:
            raise ValueError('invalid candidate count')
        if len({c['chunk_id'] for c in candidates})!=len(candidates):
            raise ValueError('duplicate retrieved chunk ID')
        for candidate in candidates:
            if not math.isfinite(candidate['bm25_rank']) or hashlib.sha256(candidate['text'].encode()).hexdigest()!=candidate['content_sha256']:
                raise ValueError('invalid candidate score or content hash')
        if candidates!=sorted(candidates,key=lambda c:(c['bm25_rank'],c['chunk_id'])):
            raise ValueError('candidate ranking order mismatch')
        selected=candidates[0] if candidates else None
        if selected is None:
            if row['evidence'] or row['selected_chunk_id'] is not None or row.get('retrieval_fallback')!='empty-retrieval':
                raise ValueError('empty retrieval must be exact no-evidence fallback')
        elif (row['evidence']!=selected['text'] or row['selected_chunk_id']!=selected['chunk_id']
              or row['evidence_sha256']!=selected['content_sha256'] or row.get('retrieval_fallback') is not None):
            raise ValueError('selected top-one evidence differs from ranked manifest')
    return cases[:count]
