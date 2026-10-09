"""Independent pre-RoPE document KV identity and request-position composition.

This is storage/placement infrastructure, NOT selective recomputation.
"""
import hashlib
import json
import torch
from experimental.rg_acceptance import require
from experimental.rg_prefix import pack_prefix, unpack_prefix

POLICY = 'cb-mvp1-independent-document-pre-rope-bf16-v1'


def tensor_hash(tensor):
    return hashlib.sha256(tensor.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def cache_identity(encoding, tokens, metadata):
    require(set(encoding) == {'model_snapshot', 'config_sha256', 'tokenizer_sha256', 'capture_policy'},
            'incomplete encoding identity')
    require(encoding['capture_policy'] == POLICY, 'not independent pre-RoPE document KV')
    require(all(isinstance(x, str) and x for x in encoding.values()), 'empty encoding identity')
    require(isinstance(tokens, list) and tokens and all(type(t) is int and t >= 0 for t in tokens), 'invalid tokens')
    require(set(metadata) == {'layers','heads','tokens','head_dim','dtype','chunk_size','chunks','layout'}, 'invalid metadata')
    require(all(type(metadata[k]) is int and metadata[k] > 0 for k in
                ('layers','heads','tokens','head_dim','chunk_size','chunks')), 'invalid dimensions')
    require(metadata['tokens'] == len(tokens) and metadata['chunks'] ==
            (len(tokens)+metadata['chunk_size']-1)//metadata['chunk_size'], 'invalid valid length')
    require(metadata['dtype'] == 'torch.bfloat16' and metadata['layout'] == '2,L,T,H*D', 'unsupported layout')
    body = dict(encoding=encoding, tokens=tokens, metadata=metadata,
                original_positions=list(range(len(tokens))), mask='independent-document-causal', key_state='pre-rope')
    return hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()


def pack_document(encoding, tokens, pairs):
    chunks, metadata = pack_prefix(pairs)
    entry = dict(encoding=dict(encoding), tokens=list(tokens), metadata=metadata,
                 key=cache_identity(encoding, tokens, metadata), chunk_hashes=[tensor_hash(t) for t in chunks])
    return entry, chunks


def validate_document(entry, chunks, encoding):
    require(entry['encoding'] == encoding, 'incompatible encoding; not a cache miss')
    require(entry['key'] == cache_identity(encoding, entry['tokens'], entry['metadata']), 'identity mismatch')
    require(len(chunks) == entry['metadata']['chunks'], 'chunk cardinality mismatch')
    pairs = unpack_prefix(chunks, entry['metadata'])
    require([tensor_hash(t) for t in chunks] == entry['chunk_hashes'], 'payload integrity failure')
    tail = entry['metadata']['tokens'] % entry['metadata']['chunk_size']
    require(not tail or bool((chunks[-1][:,:,tail:,:] == 0).all()), 'nonzero padding')
    return pairs


def compose_documents(documents, encoding, start, rotate):
    """rotate(K, absolute_positions) is the model's native RoPE on its device.

    Output remains CPU-owned. Repeated documents are legal and keep distinct
    request positions. No re-encoding or question input is accepted here.
    """
    require(type(start) is int and start >= 0, 'invalid start position')
    require(bool(documents), 'empty composition needs explicit no-evidence fallback')
    layers = None; trace = []; cursor = start
    for entry, chunks in documents:
        pairs = validate_document(entry, chunks, encoding)
        if layers is None: layers = [[] for _ in pairs]
        require(len(layers) == len(pairs), 'mixed layer count')
        positions = list(range(cursor, cursor+len(entry['tokens'])))
        for i, (key, value) in enumerate(pairs):
            rotated = rotate(key, positions)
            require(rotated.shape == key.shape and rotated.dtype == key.dtype and rotated.device.type == 'cpu',
                    'rotary callback changed layout')
            layers[i].append((rotated, value))
        trace.append(dict(key=entry['key'], start=cursor, end=cursor+len(positions), positions=positions))
        cursor += len(positions)
    return [(torch.cat([p[0] for p in rows],dim=2), torch.cat([p[1] for p in rows],dim=2))
            for rows in layers], trace
