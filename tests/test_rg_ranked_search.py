import random
import sqlite3
import pytest
from experimental.rg_retrieval import init_index,set_meta,add_chunks,digest,search
from experimental.rg_ranked_search import search_ranked


def fixture():
    db=sqlite3.connect(':memory:');init_index(db);rng=random.Random(20261005)
    records=[]
    for i in range(120):
        # Many exact BM25 ties; insert in reverse ID order so stopping at five
        # native-rank rows cannot accidentally satisfy the required tie policy.
        text=('repeated identical passage' if i<40 else ' '.join(rng.choices(['red','blue','green','gold','café'],k=10)))
        records.append(dict(chunk_id=f'{119-i}:0:0',doc_id=str(i),paragraph=0,token_start=0,
                            title='title',text=text,content_sha256=digest(text),token_count=len(text.split())))
    add_chunks(db,records);set_meta(db,'status','complete');db.commit();return db


@pytest.mark.parametrize('question',['repeated','red blue','café GOLD','title','!!!','absent','" OR * blue','red red red'])
@pytest.mark.parametrize('limit',[1,3,5])
def test_exact_ranking_including_all_boundary_ties(question,limit):
    db=fixture()
    try:assert search_ranked(db,question,limit=limit)==search(db,question,limit=limit)
    finally:db.close()


def test_query_local_mapping_ignores_persistent_rank_override():
    db=fixture()
    try:
        db.execute("INSERT INTO chunks(chunks,rank) VALUES('rank','bm25(0,0,0,0,10,1)')")
        assert search_ranked(db,'red title')==search(db,'red title')
    finally:db.close()


def test_invalid_inputs_and_unfinished_index():
    db=fixture()
    try:
        for query,limit in (({},5),('q',0),('q',6)):
            with pytest.raises((TypeError,ValueError)):search_ranked(db,query,limit=limit)
        set_meta(db,'status','building')
        with pytest.raises(ValueError):search_ranked(db,'q')
    finally:db.close()
