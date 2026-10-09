# R Memory Lab

Research on **external KV memory for frozen language models**: persist reusable
document-prefix KV with LMCache, retrieve compatible evidence, and compare
GPU-resident attention with an emulated R-side attention path.

The target MVP also **requires CacheBlend-style multi-document KV composition
and selective recomputation**. The current strict-prefix path is a baseline,
not the final demo. See the [required integration plan](docs/CACHEBLEND_MVP.md).

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

**Paused at the user's request, October 9, 2026.** No experiments or demo
services are running for this project; publication does not resume them.

- Retrieved-document stable-cache reuse passed a 100-case development evaluation
  and independent repeat, preserving native-control tokens and per-step logits.
- The FP64-accumulation retrieved-prefix R path passed 100 cases and a fresh
  repeat under unchanged numerical gates. Evidence quality did not improve;
  historical failed arithmetic paths remain failed.
- Fresh question-to-cache routing and generation passed 100 development cases
  and an independent 100-case repeat, including actual LMCache reload.
- Combined R/CacheBlend fixed-row 100-case development and fresh repeat passed
  correctness gates. Request-time G-side row selection also passed100 cases and
  a fresh repeat. Matched FP64 GPU/R tokens and every-step logits agree on100/100.
  Development EM15% versus11% without evidence is inconclusive (+4pp;95%[-1,+10]).
- Bounded live HTTP transport passed ten questions and a fresh ten-question
  repeat. The separate manual demo was stopped before its first request.
- The separate1GiB L1/24GiB allocator sweep passed through20,480tokens. Cache
  first beat staged recompute at the measured16K point; this is not a production
  serving or physical NAND claim. [Curve and measured data](docs/SWEEP.md).

See [status and provenance](docs/STATUS.md), [acceptance contracts](docs/ACCEPTANCE.md),
and the [MVP roadmap](docs/ROADMAP.md). Status is a dated snapshot, not a live monitor.

## MVP checklist

Last checked: **2026-10-09**. Checked means the stated subtask has
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
- [x] **M1r — Live routing:** 100 fresh question-to-LMCache delivery cases
  completed with all ten ten-case raw audits passing; not live generation.
- [x] **M0/M2 — Retrieved-prefix R numerical admission:** FP64 accumulation
  passed 100 cases and a fresh repeat under unchanged gates. This is not native
  bitwise identity, combined CacheBlend admission, or a quality-benefit pass.
- [x] **M2/M4 — Development live generation:** fresh routing through LMCache and
  GPU/R generation passed10/100/fresh100 in `rg-ranked-online-20261006-a`.
- [x] **M4 — Bounded HTTP transport:** ten actual HTTP-triggered questions and
  fresh ten-case repeat passed in `rg-http-20261007-a`. Fixed development
  questions, not arbitrary-prompt serving; the service is now stopped.
- [x] **M2b/CB0 — CacheBlend reference:** recover the completed corrected
  Blackwell port, repeats and failed original-mask diagnosis.
- [x] **M2b/CB1–CB2 — Ten-case document persistence/composition:** independently capture
  pre-RoPE document KV; persist/restart/reload through LMCache; validate new
  request positions and exact agreement with the resident-composition control.
  Successor `rg-blend-storage-20260926-b` passed its ten-case/70-generation raw
  audit; durable raw archive is checksum-verified.
- [x] **M2b/CB3 — First GPU selective-recompute checkpoint:**
  `rg-blend-prefill-20260926-b` passed ten cases/100 complete generations,
  native full-limit, cache-disabled zero, resident/reloaded and replay checks.
  Durable raw archive verified; no R or quality-benefit claim.
- [x] **M2b/CB4 — All-query CacheBlend development experiment:** fresh matched no-evidence/miss
  smoke, 100-case document storage, 100-case selective/reuse/full comparison and
  independent fresh-engine repeat. `rg-blend-development-20260926-a` completed;
  raw guards/repeat and durable archive passed. EM: absent11%, full12%, blend15%,
  independent reuse17%. Experiment complete; quality-benefit gate **not passed**.
- [x] **M2b/CB5 — Fixed-row R CacheBlend correctness:** matched 100-case
  development and fresh repeat passed numerical, ownership, limit/zero and
  raw-output guards in `rg-blend-r-20260928-b`. Exact agreement is with the
  matched FP64 GPU control, not native BF16. Evidence-benefit gate still unpassed.
- [x] **M2b/CB5 — Request-time selection:** compute rows from current request
  and loaded document KV, then share the map with W/R. `rg-blend-live-20261005-a`
  passed10-case smoke,100-case development and independent100repeat, with
  263,952 numerical checks per100-case run and verified raw archival. Not a live endpoint.
- [ ] **M3 — Quality confirmation:** freeze untouched confirmation data; meet
  the evidence-benefit and R non-inferiority gates; independently reproduce.
- [x] **M4 — Recorded viewer prototype:** historical case replay/export exists
  in the research workspace; it is not yet part of this public component release.
- [ ] **M4 — Interactive MVP:** ship the live/replay viewer with explicit evidence,
  cache-hit/fallback, cached/recomputed rows and guard status; verify browser
  access and case export. Required CacheBlend integration must be admitted.
- [ ] **M5 — Delayed-consumption extension:** run all K=0…36 on the validated
  R/LMCache path, then confirm at most two frozen selections. Historical
  delayed-injection sweeps do not complete this new-path milestone.
- [x] **Supporting measured sweep:** separately declared24GiB allocator/1GiB L1
  series passed8K/10K/12K/16K/20K raw audits. Shared-resource timing diagnostic
  only; production latency remains unpassed. Older failures are preserved.
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
