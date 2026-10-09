"""Same BM25 ranking with native FTS5 rank ordering and exact boundary ties.

Candidate only: this does not replace the admitted retriever. No stopwords,
weight/ranking changes, approximate ties, document filtering, or new index.
"""
import bisect
import math
import re
from experimental.rg_retrieval import get_meta,digest

FIELDS=('chunk_id','doc_id','paragraph','token_start','title','text','content_sha256','token_count','bm25_rank')


def search_ranked(connection,question,*,limit=5):
    if not isinstance(question,str):raise TypeError('query must be text, not an evaluation record')
    if not isinstance(limit,int) or not 1<=limit<=5:raise ValueError('fixed maximum of five candidates')
    if get_meta(connection,'status')!='complete':raise ValueError('incomplete index cannot serve evaluation')
    terms=sorted(set(re.findall(r'\w+',question.casefold(),flags=re.UNICODE)))
    if not terms:return []
    query=' OR '.join('"'+term.replace('"','""')+'"' for term in terms)
    # Explicit query-local bm25() mapping prevents a persistent rank override
    # from changing semantics. Read beyond the kth rank through ALL exact ties;
    # keep only the lexicographically smallest IDs within a bounded top-five.
    cursor=connection.execute('SELECT chunk_id,doc_id,paragraph,token_start,title,text,content_sha256,token_count,rank '
        "FROM chunks WHERE chunks MATCH ? AND rank MATCH 'bm25()' ORDER BY rank",(query,))
    best=[];previous=None
    try:
        for values in cursor:
            row=dict(zip(FIELDS,values));rank=row['bm25_rank']
            if not isinstance(rank,(int,float)) or not math.isfinite(rank):raise ValueError('nonfinite rank')
            if previous is not None and rank<previous:raise ValueError('native rank order violated')
            previous=rank
            if len(best)==limit and rank>best[-1][0][0]:break
            if digest(row['text'])!=row['content_sha256']:raise ValueError('retrieved content hash mismatch')
            key=(rank,row['chunk_id'])
            position=bisect.bisect_left([entry[0] for entry in best],key)
            best.insert(position,(key,row))
            if len(best)>limit:best.pop()
    finally:cursor.close()
    return [entry[1] for entry in best]
