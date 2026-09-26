"""Answer-blind, deterministic SQLite FTS5/BM25 retrieval for M1r.

This selects evidence IDs, never LMCache objects by semantic similarity.
Gold answers and provenance are deliberately absent from the query interface.
"""
import hashlib
import json
import re
import sqlite3

POLICY=dict(id='rg-bm25-v1',backend='sqlite-fts5-bm25',chunk_tokens=256,
            overlap_tokens=0,paragraph_boundaries=True,query_terms='unicode-word-OR',
            title_weight=1.0,text_weight=1.0,top_candidates=5,consumed_chunks=1,
            ties='chunk_id ascending',empty_query='no evidence')


def digest(text): return hashlib.sha256(text.encode('utf8')).hexdigest()


def init_index(connection):
    connection.execute('CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY,value TEXT NOT NULL)')
    connection.execute('CREATE VIRTUAL TABLE IF NOT EXISTS chunks USING fts5('
        'chunk_id UNINDEXED,doc_id UNINDEXED,paragraph UNINDEXED,token_start UNINDEXED,'
        'title,text,content_sha256 UNINDEXED,token_count UNINDEXED,tokenize="unicode61")')


def set_meta(connection,key,value):
    connection.execute('INSERT OR REPLACE INTO metadata VALUES (?,?)',(key,json.dumps(value,sort_keys=True)))


def get_meta(connection,key,default=None):
    row=connection.execute('SELECT value FROM metadata WHERE key=?',(key,)).fetchone()
    return default if row is None else json.loads(row[0])


def page_chunks(page,tokenizer):
    doc_id=str(page['wikipedia_id']); title=page['wikipedia_title']
    if not isinstance(title,str) or not isinstance(page['text'],list):
        raise ValueError('invalid KILT document')
    for paragraph,text in enumerate(page['text']):
        if not isinstance(text,str): raise ValueError('invalid KILT paragraph')
        tokens=tokenizer.encode(text,add_special_tokens=False)
        for start in range(0,len(tokens),POLICY['chunk_tokens']):
            part=tokens[start:start+POLICY['chunk_tokens']]
            chunk=tokenizer.decode(part,skip_special_tokens=False)
            if not chunk.strip():continue
            yield dict(chunk_id=f'{doc_id}:{paragraph}:{start}',doc_id=doc_id,
                paragraph=paragraph,token_start=start,title=title,text=chunk,
                content_sha256=digest(chunk),token_count=len(part))


def add_chunks(connection,records):
    for row in records:
        if digest(row['text'])!=row['content_sha256']:
            raise ValueError('chunk content hash mismatch')
        connection.execute('INSERT INTO chunks VALUES (?,?,?,?,?,?,?,?)',tuple(row[k] for k in
            ('chunk_id','doc_id','paragraph','token_start','title','text','content_sha256','token_count')))


def search(connection,question,*,limit=5):
    if not isinstance(question,str):raise TypeError('query must be text, not an evaluation record')
    if not isinstance(limit,int) or not 1<=limit<=5:raise ValueError('fixed maximum of five candidates')
    if get_meta(connection,'status')!='complete':raise ValueError('incomplete index cannot serve evaluation')
    terms=sorted(set(re.findall(r'\w+',question.casefold(),flags=re.UNICODE)))
    if not terms:return []
    query=' OR '.join('"'+term.replace('"','""')+'"' for term in terms)
    rows=connection.execute('SELECT chunk_id,doc_id,paragraph,token_start,title,text,content_sha256,token_count,'
        'bm25(chunks) AS rank FROM chunks WHERE chunks MATCH ? ORDER BY rank,chunk_id LIMIT ?',
        (query,limit)).fetchall()
    results=[]
    for row in rows:
        item=dict(zip(('chunk_id','doc_id','paragraph','token_start','title','text','content_sha256','token_count','bm25_rank'),row))
        if digest(item['text'])!=item['content_sha256']:raise ValueError('retrieved content hash mismatch')
        results.append(item)
    return results


def resolve_cache(selected,manifest,encoding_id):
    """Manifest lookup only: wrong-but-compatible evidence is not gold-filtered."""
    if selected is None:return dict(status='empty-retrieval',cache_key=None)
    if digest(selected['text'])!=selected['content_sha256']:raise ValueError('selected text is corrupt')
    entry=manifest.get(selected['chunk_id'])
    if entry is None:return dict(status='cache-miss',cache_key=None)
    if (entry.get('content_sha256')!=selected['content_sha256'] or
        entry.get('encoding_id')!=encoding_id or not entry.get('cache_key')):
        raise ValueError('incompatible/corrupt KV manifest; not a cache miss')
    return dict(status='manifest-hit',cache_key=entry['cache_key'])
