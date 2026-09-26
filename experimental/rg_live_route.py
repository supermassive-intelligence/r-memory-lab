"""Question-only retrieval to a validated stable KV catalog; never encode on miss."""
import hashlib
import json
from experimental.rg_acceptance import require
from experimental.rg_cache_identity import validate_entry
from experimental.rg_retrieval import digest, search, resolve_cache


def encoding_id(encoding):
    return hashlib.sha256(json.dumps(encoding, sort_keys=True).encode()).hexdigest()


def build_manifest(state):
    require(state.get('status') == 'complete', 'cache producer incomplete')
    require(state.get('source_evidence_mode') == 'retrieved', 'not a retrieved cache')
    require(state.get('evidence_encode_calls') == 0, 'unexpected evidence encoding')
    catalog = state['catalog']; encoding = state['encoding']; manifest = {}
    rows = state['records']['stable-L']
    require(len(rows) == state['expected_cases'], 'incomplete route catalog')
    require(len({r['id'] for r in rows}) == len(rows), 'duplicate source query')
    for key, entry in catalog.items():
        require(validate_entry(entry, encoding) == key, 'catalog key mismatch')
    for row in rows:
        key = row['stable_identity']; entry = catalog[key]
        require(entry['tokens'] == row['prefix_token_ids'], 'prefix binding mismatch')
        require(digest(row['evidence']) == row['evidence_sha256'], 'evidence corrupt')
        chunk = row['selected_chunk_id']
        if chunk is None:
            require(row['evidence'] == '' and not row['candidates'], 'invalid empty selection')
            continue
        require(row['candidates'] and row['candidates'][0]['chunk_id'] == chunk,
                'not the frozen top-ranked chunk')
        candidate = row['candidates'][0]
        require(candidate['text'] == row['evidence'] and
                candidate['content_sha256'] == row['evidence_sha256'], 'selected content changed')
        value = dict(cache_key=key, content_sha256=row['evidence_sha256'],
                     encoding_id=encoding_id(encoding))
        require(chunk not in manifest or manifest[chunk] == value, 'conflicting chunk identity')
        manifest[chunk] = value
    return manifest


def route(connection, question, manifest, encoding):
    # Only text enters search; references, correctness labels and source IDs do not.
    candidates = search(connection, question)
    selected = candidates[0] if candidates else None
    result = resolve_cache(selected, manifest, encoding_id(encoding))
    return dict(question=question, candidates=candidates,
                selected_chunk_id=None if selected is None else selected['chunk_id'],
                **result)
