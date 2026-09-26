# Verified checkpoint — 2026-09-26 18:33 UTC

## CacheBlend integration update — 2026-09-26 19:52 UTC

- `rg-blend-storage-20260926-a` failed before generation on Transformers cache
  iterator metadata. Failed raw archive checksum verified; no result admitted.
- `rg-blend-storage-20260926-b` (`bc490fd`) completed all ten cases and 70
  complete generations. Fresh inherited/coverage/GPU transport gates and raw
  storage/position/output audit PASS. Twenty independent pre-RoPE documents;
  exact resident-versus-LMCache-reloaded tokens and every-step logits. Terminal
  archival is in progress. This is not selective recomputation or R quality.
- `rg-blend-prefill-20260926-b` (`194c03a`) automatically started after CB2
  acceptance. Fresh full gates precede the real-model selective-recompute check;
  32 focused component tests passed before launch. No CB3 result admitted yet.
- The earlier CB3 queue A was interrupted while waiting, before model evaluation,
  to add an explicit independent-reuse replay arm. Its terminal record is kept.
- CPU live routing was 55/100 at the latest check. The retrieved-R numerical
  failure and held-out quality gates remain unpassed.

## Scope/execution update — 2026-09-26 19:38 UTC

CacheBlend is now required for the MVP. [CB-MVP1](CACHEBLEND_MVP.md) freezes
the integration steps. New run `rg-blend-storage-20260926-a`, source `6ee50b6`,
is running on one Saturn GPU after 25 focused tests passed. Its inherited suite,
coverage and actual GPU transport tests precede the ten-case document storage/
composition checkpoint. No new result is admitted yet. This is preparation for
selective recomputation, not a completed CacheBlend/R path. CPU live routing is
54/100. The results snapshot below remains explicitly dated.

This is a manually verified snapshot from the original research workspace.
Public component tests are separate; historical raw artifacts are not included.

| Run | Verified outcome | Claim boundary |
|---|---|---|
| `rg-retrieved-cache-20260926-b` | Ten-case smoke, 100-case development, and independent 100-case repeat raw audits PASS; four arms match across deployments | Stored retrieved-document KV fidelity, not evidence-quality improvement or live arbitrary-prompt serving |
| `rg-retrieved-r-20260926-b` | Ten-case smoke PASS; 100-case development numerical audit FAIL; repeat not launched | Excluded from admitted R-quality results |
| `rg-live-route-20260926-a` | 49/100 queries completed; ten-case raw delivery audit PASS; process active | Routing/storage only; no new generation |
| `rg-tiled-preflight-20260926-a` | All 14 native attention/probability/replay GPU tensor checks PASS | Not full-model or long-context admission |
| `rg-tiled-prefill-20260926-a` | Failed startup reserve check before first point | No new timing point or 8,192-token result |
| `rg-tiled-prefill-20260926-b` | 1,024/4,096 raw audits and exact native-overlap proofs PASS; 8,192 failed a 2 GiB QK/scaling allocation under the unchanged 12 GiB allocator cap | Partial diagnostic sweep; no admitted 8,192 point or crossover |
| `rg-retrieved-outliers-20260926-a` | Three captured first failures replay exactly, including their original guard decisions | Successful diagnosis/replay, not a numerical correction or R-quality pass |
| `rg-prefix-sweep-20260925-b` | Native 1,024/2,048/4,096 per-length raw audits PASS; 8,192 exceeded the 12 GiB job allocator budget | Shared-resource, staged-eager, HDD diagnostics only; no practical crossover established |

Retrieved R failure: five failing layer-step guard records among 121,356 recorded
layer-step checks. Two actual-output failures are the same event in R and R-repeat
(case index28, decode step9, layer35). This count is a diagnosis, not an acceptance
rate that excuses the failures. Captured R/R-repeat output reaches a BF16
rounding midpoint; the binary64 reference falls just below it. One rounded
element differs by 0.03125, producing normalized L2 0.0010474 above the frozen
0.001 limit. Further precision-stage attribution and a passing correction remain
required; neither tolerance nor acceptance was relaxed.

The fresh tiled sweep observed 73,563 MiB free against a fixed 73,728 MiB startup
minimum: 165 MiB short. It failed before inherited tests or model execution. The
candidate's mathematics was not rejected by this startup failure. The subsequent
B run passed exact native overlap at 1,024/4,096 but failed at 8,192 inside the
allocator budget, despite approximately 82.8 GiB physical GPU memory free.

The full-corpus retrieved native development evaluation completed with 14% EM
without evidence versus 11% with retrieved evidence (100 cases,
`rg-retrieval-queue-20260924-f`). Its raw execution passed, but evidence-quality
improvement did not. Storage fidelity does not turn this into a quality gain.

Historical CacheBlend reconciliation: `cb-repro-artifact-full-20260919-b`
completed September 20 with full-set repeats and recorded passing execution
guards. It is an original-layout **corrected port**, not paper-number replication.
See [CacheBlend history](CACHEBLEND.md) for the adapted study, later port,
raw-result hashes, measured quality and remaining limitations.

Completed cache, failed R, and failed tiled runs have checksum-verified raw
archives in the research storage system. Run IDs and hashes are retained there;
no private hostnames, paths, credentials, prompts, or raw tensors are published.
The quality figures here are explicitly scoped historical results, not new
held-out R-quality confirmation. The [README checklist](../README.md#mvp-checklist)
tracks unfinished acceptance gates.
