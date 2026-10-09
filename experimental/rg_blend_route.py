"""Fresh question -> top-two document cache keys; no model or answer input."""
from experimental.rg_acceptance import require
from experimental.rg_blend_cache import cache_identity
from experimental.rg_live_route import encoding_id
from experimental.rg_retrieval import search,resolve_cache


def build_manifest(state):
    require(state.get('status')=='complete','storage not complete')
    require(state.get('online_document_encodes')==0,'source encoded evidence online')
    catalog=state['catalog'];encoding=state['encoding'];manifest={}
    for key,entry in catalog.items():
        require(entry['encoding']==encoding and entry['key']==key==
                cache_identity(encoding,entry['tokens'],entry['metadata']),'invalid catalog identity')
    for case in state['cases']:
        require(len(case['documents'])==len(case['document_keys'])<=2,'composition cardinality drift')
        for doc,key in zip(case['documents'],case['document_keys'],strict=True):
            require(doc['tokens']==catalog[key]['tokens'],'document token binding changed')
            value=dict(cache_key=key,content_sha256=doc['content_sha256'],encoding_id=encoding_id(encoding))
            require(doc['chunk_id'] not in manifest or manifest[doc['chunk_id']]==value,'conflicting chunk mapping')
            manifest[doc['chunk_id']]=value
    return manifest


def route(connection,question,manifest,encoding):
    candidates=search(connection,question)
    return resolve_candidates(question,candidates,manifest,encoding)


def resolve_candidates(question,candidates,manifest,encoding):
    """Resolve an already retrieved ranking; never rerank for cache availability."""
    selected=candidates[:2]
    resolved=[resolve_cache(c,manifest,encoding_id(encoding)) for c in selected]
    if not selected:status='empty-retrieval'
    elif any(r['status']=='cache-miss' for r in resolved):status='cache-miss'
    else:status='manifest-hit'
    # Explicit all-or-nothing fallback. Never replace a missing top-ranked
    # document with a lower-ranked cached one or filter using a gold answer.
    return dict(question=question,candidates=candidates,
        selected_chunk_ids=[c['chunk_id'] for c in selected],status=status,
        document_keys=[r['cache_key'] for r in resolved] if status=='manifest-hit' else [],
        resolutions=resolved,miss_policy='whole-request no-evidence; no reencoding')


def compose_request(row,catalog,tokenizer):
    """Construct model inputs from question/ranking, without an evaluation case."""
    common=tokenizer.encode('Answer the question using the passages. Give only the short answer.\n\nPassages:\n',add_special_tokens=False)
    suffix=tokenizer.encode('\nQuestion: '+row['question']+'\nAnswer:',add_special_tokens=False)
    full=list(common);documents=[]
    require(row['status'] in ('manifest-hit','cache-miss','empty-retrieval'),'unknown route status')
    if row['status']=='manifest-hit':
        selected=row['candidates'][:2]
        require(len(selected)==len(row['document_keys'])>0,'route cardinality mismatch')
        for candidate,key in zip(selected,row['document_keys'],strict=True):
            tokens=tokenizer.encode(candidate['text']+'\n\n',add_special_tokens=False)[:256]
            require(tokens==catalog[key]['tokens'],'fresh text/cache token mismatch')
            full+=tokens
            documents.append(dict(chunk_id=candidate['chunk_id'],content_sha256=candidate['content_sha256'],tokens=tokens))
    else:require(not row['document_keys'],'fallback retained cache keys')
    return dict(question=row['question'],documents=documents,document_keys=row['document_keys'],
                full_tokens=full,suffix_tokens=suffix)
