"""Stable prefix identity; request/question IDs never enter the cache key."""
import hashlib
import json
from experimental.rg_acceptance import require


def identity(encoding,tokens,metadata):
    require(set(encoding)=={'model_snapshot','config_sha256','tokenizer_sha256','prefix_policy'},'incomplete encoding identity')
    require(all(isinstance(x,str) and x for x in encoding.values()),'empty encoding field')
    require(isinstance(tokens,list) and tokens and all(type(x) is int and x>=0 for x in tokens),'invalid prefix tokens')
    require(metadata.get('tokens')==len(tokens),'prefix valid length mismatch')
    require(set(metadata)=={'layers','heads','tokens','head_dim','dtype','chunk_size','chunks','layout'},'incomplete layout identity')
    for key in ('layers','heads','head_dim','chunk_size','chunks'):
        require(type(metadata[key]) is int and metadata[key]>0,'invalid layout')
    require(metadata['chunks']==(len(tokens)+metadata['chunk_size']-1)//metadata['chunk_size'],'chunk count mismatch')
    require(metadata['dtype']=='torch.bfloat16' and metadata['layout']=='2,L,T,H*D','unsupported layout')
    return hashlib.sha256(json.dumps(dict(encoding=encoding,tokens=tokens,metadata=metadata,
        positions=list(range(len(tokens))),mask='causal'),sort_keys=True).encode()).hexdigest()


def validate_entry(entry,encoding):
    require(entry['encoding']==encoding,'incompatible encoding; not a cache miss')
    require(entry['identity']==identity(encoding,entry['tokens'],entry['metadata']),'corrupt cache identity')
    hashes=entry['chunk_hashes']
    require(len(hashes)==entry['metadata']['chunks'] and all(isinstance(h,str) and len(h)==64 for h in hashes),'invalid payload manifest')
    return entry['identity']
