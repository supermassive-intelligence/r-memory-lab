# R Memory Lab

Research on **external KV memory for frozen language models**: persist reusable
document-prefix KV with LMCache, retrieve compatible evidence, and compare
GPU-resident attention with an emulated R-side attention path.

This is an experimental component release, **not a production R machine or a
completed end-to-end demo**. Numerical, retrieval-quality, and latency gates are
reported separately. A failed gate is not hidden by a passing smoke test.

## What are R and G?

| Component | Role | What is available today |
|---|---|---|
| G | Runs the frozen model and local attention on GPU | GPU-based research harnesses |
| R | Owns external KV; may perform external-memory attention | CPU/storage emulation, not physical R hardware |
| LMCache | Cache identity, storage, lookup and KV delivery | Experimental engine-driven filesystem adapter |
| Evidence retriever | Selects documents from question text | Frozen SQLite FTS5/BM25 prototype |

```text
Offline document encoding -> per-layer KV -> LMCache-backed storage
                                                    |
Question -> evidence retrieval -> compatible KV key -+
                                                    |
                         +--------------------------+-----------------+
                         |                                            |
                    Reload KV to G                     Emulated R attention
                         |                             (partial output + LSE)
                         +--------------------------+-----------------+
                                                    |
                              Compare generation and numerical controls
```

Retrieval and cache lookup are different operations. A cache hit does not mean
the selected evidence answers the question. Stored KV remains model-, tokenizer-,
position-, prompt-policy-, and layout-specific. This is not context-window extension.

## Current checkpoint

- Retrieved-document stable-cache reuse passed a 100-case development evaluation
  and independent repeat, preserving native-control tokens and per-step logits.
- Retrieved R attention passed ten cases but failed its larger numerical audit.
  No admitted 100-case retrieved R-quality result is claimed.
- Live question-to-cache routing is in progress; it is not yet a live generation endpoint.
- The native prefill sweep has accepted diagnostic points through 4,096 tokens;
  no practical cache/recompute crossover or physical NAND performance claim is established.

See [status and provenance](docs/STATUS.md), [acceptance contracts](docs/ACCEPTANCE.md),
and the [MVP roadmap](docs/ROADMAP.md). Status is a dated snapshot, not a live monitor.

## Layout

| Path | Contents |
|---|---|
| `experimental/` | Split attention, unchanged numerical contracts, KV packing, cache identity, retrieval and routing |
| `scripts/rg_lmcache_prefix.py` | Extracted real LMCache transport, persistence verification and manifest validation |
| `scripts/rg_retrieval_manifest.py` | Answer-blind deterministic selection-manifest tool |
| `tests/` | Unmodified selected component/regression tests from the research workspace |
| `docs/` | Scope, reproduction requirements, gates, provenance and next milestones |

## Run component tests

Use a dedicated environment with PyTorch and Transformers installed. The verified
environment versions and test receipt are in [VALIDATION.md](docs/VALIDATION.md).
We do not silently install or replace your GPU framework.

```bash
python -m pip install -r requirements-dev.txt
python -m pip install --no-deps -e .
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 CUDA_VISIBLE_DEVICES="" python -m pytest -q
```

Missing framework dependencies may cause some existing tests to skip; such a run
is **not equivalent** to the recorded complete component validation. The published
validation explicitly reports skips and failures. There is no GPU model download
or corpus download in these component tests.

For LMCache integration requirements and experiment boundaries, read
[REPRODUCTION.md](docs/REPRODUCTION.md). The historical fleet/model-generation
launchers are not included in this first release. These tests do not reproduce
the full historical quality or performance measurements.

## Related projects

- [LMCache](https://github.com/LMCache/LMCache): KV cache storage and reuse.
- [CacheBlend](https://arxiv.org/abs/2405.16444): related work motivating reuse
  beyond exact prefixes. This release does not claim to reproduce its paper numbers.

No model weights, corpus passages, user prompts, credentials, or raw experiment
tensors are distributed here. No repository-wide license grant is declared yet;
upstream dependencies retain their respective licenses.
