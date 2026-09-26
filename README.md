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

## MVP checklist

Last checked: **2026-09-26 18:33 UTC**. Checked means the stated subtask has
supporting evidence, not that its whole milestone has passed. Historical research
results are separate from the code included in this component release.
Run IDs and claim boundaries are in [STATUS.md](docs/STATUS.md).

- [x] **M1 — Reference harness:** freeze evidence, model, positions and named
  native/zero/miss controls; audit complete generations and raw logits.
- [x] **M2 — Storage fidelity:** retrieved-document LMCache persistence/reload
  passes 100-case development and an independent 100-case repeat, with exact
  native-control tokens and per-step logits.
- [x] **M1r — Retrieval preparation:** build/audit the frozen corpus index;
  freeze 100 answer-blind BM25 selections and compatible document-KV identities.
- [x] **M1r — Development evaluation:** evaluate all 100 queries, including wrong
  evidence and misses. Completed with a negative quality result; not an MVP pass.
- [ ] **M1r — Live routing:** finish 100 fresh question-to-LMCache delivery cases
  and their raw audit (49/100 complete; first ten-case audit passed).
- [ ] **M0/M2 — Retrieved R numerical admission:** diagnose and repair the
  100-case numerical failure, then rerun unchanged gates and independent repeat.
  Captured-failure replay is complete; a passing repair is not.
- [ ] **M2/M4 — Live generation:** connect fresh routing to GPU generation with
  native/zero/miss controls, then to validated R-side attention. Routing alone
  is not a live answer endpoint.
- [ ] **M3 — Quality confirmation:** freeze untouched confirmation data; meet
  the evidence-benefit and R non-inferiority gates; independently reproduce.
- [x] **M4 — Recorded viewer prototype:** historical case replay/export exists
  in the research workspace; it is not yet part of this public component release.
- [ ] **M4 — Interactive MVP:** ship the live/replay viewer with explicit evidence,
  cache-hit/fallback and guard status; verify browser access and case export.
- [ ] **M5 — Delayed-consumption extension:** run all K=0…36 on the validated
  R/LMCache path, then confirm at most two frozen selections. Historical
  delayed-injection sweeps do not complete this new-path milestone.
- [ ] **Supporting performance work:** resolve the 8,192-token allocator failure,
  retain exact native overlap, and measure matched end-to-end cache/recompute
  curves. No practical crossover is established.
- [x] **Public component release:** publish the extracted components and passing
  147-test receipt, with explicit reproduction limitations.
- [ ] **Final handoff:** publish the portable full-model harness, accepted MVP
  report, reproduction commands and durable evidence receipts.

At milestone boundaries, update this checklist, [status](docs/STATUS.md) and
[roadmap](docs/ROADMAP.md) together. Keep failed gates unchecked; a completed
negative experiment may be checked only when explicitly labeled as such.

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
- [CacheBlend](https://arxiv.org/abs/2405.16444): motivates reuse beyond exact
  prefixes. We attempted reproduction and completed both an adapted 350-case
  quality study and a corrected original-layout Blackwell port, with independent
  repeats. Exact paper-number reproduction, including latency and throughput,
  remains unestablished. See [experiments, results and limitations](docs/CACHEBLEND.md).

No model weights, corpus passages, user prompts, credentials, or raw experiment
tensors are distributed here. No repository-wide license grant is declared yet;
upstream dependencies retain their respective licenses.
