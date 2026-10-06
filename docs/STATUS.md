# Verified checkpoint — 2026-09-26 18:33 UTC

## Superseding update — 2026-10-06 23:16 UTC

Request-time selection `rg-blend-live-20261005-a` passed100development and
fresh100repeat; raw35,138-file NAS archive verified. Exact GPU/R tokens and
step logits are against matched FP64 GPU arithmetic, not nativeBF16. The
evidence-benefit gate remains unpassed; no held-out or speedup claim.

Top-two routing and isolated rank-execution optimization each passed100queries.
Original fresh-routing generation `rg-blend-online-20261005-a` passed development
but its repeat exhausted bounded host-pressure retries. It is not admitted as
a complete integration result. Recovery `rg-ranked-online-20261006-a` /862d62d
is running with qualified exact-rank search and120-second stable admission;
runtime thresholds unchanged.76focused and770inherited tests passed before
model smoke; no full successor result yet.

12K failure root cause:0.5GiB L1 crossed80% eviction watermark, yielding39RAM+
9disk chunks on a supposed warm request. A separately declared1GiB capacity
within the same40GiB host budget passed actual CPU transport:48disk then48RAM,
identical KV.54-file diagnostic archive verified. Source00bd331; full8K..20K
capacity series queued after quality and will not be merged with0.5GiB timings.
No GPU OOM was responsible for this delivery guard failure.

Archive verification repaired to permit long scans that keep producing checksum
output and avoid recopying matching data. The large failed integration archive
is being reverified; not yet claimed complete. Failed raw evidence preserved.

## Superseding update — 2026-10-05 05:12 UTC

`rg-blend-live-20261005-a` smoke now has raw admission:10cases,160complete
generations,30,960 numerical checks. Development has progressed beyond26cases;
full100 and fresh100repeat remain pending. No new aggregate quality claim.

Fresh question-to-CacheBlend GPU/R generation implemented in research revision
`6dd3e58`, with76focused tests passing onSaturn. The successor
`rg-blend-online-20261005-a` is queued after the independent longer-prefill sweep
and requires live selector and top-two routing admission. It reconstructs
tokens from freshly retrieved text and uses resolved keys for actual LMCache
delivery. Full/zero/missing/numerical/output guards remain mandatory. This is
not yet an HTTP endpoint or a held-out quality result.

Full-corpus BM25 routing is slow. An isolated execution-only candidate,
`rg-rank-exact-20261005-a` / `24c1b7a`, uses native rank ordering with exact
boundary ties.56focused tests passed;100 frozen full-record/score comparisons
are running. No retriever substitution, ranking change or speedup claim.
Raw archive/report workers are queued; older failure records remain unchanged.

## Superseding update — 2026-10-05 04:54 UTC

Combined fixed-row `rg-blend-r-20260928-b` completed smoke, development100 and
fresh100repeat. Numerical/ownership/raw checks PASS; W/R tokens and all-step
logits agree exactly against the matched FP64 GPU reference. Native BF16 is a
separate bridge, not bitwise-equivalent R arithmetic. Development evidence gain
does not pass the frozen quality gate. Raw34927-file NAS archive verified:
`14555cb526a9a6318d807f5f6241acceb48f02881c8941a00fd84feaff5638bd`.

Request-time selection checkpoint `rg-blend-live-20261005-a`, source`ce2109f`,
is now active on SaturnGPU3. A nativeBF16 layer0/layer1-V scout selects using
actual request and loaded KV; fixed0.16 policy, shared map for W/R. Historical
rows are post-selection regression controls only.48focused tests and715inherited
tests pass; full-model smoke, development100 and fresh100 remain pending.
No ratio tuning, new tolerance, confirmation-data access, or speed claim.
Live request-time selection does not itself constitute a live serving endpoint.

The prior24GiB recovery stopped during inherited tests when GPU free memory
fell below48GiB; no accepted points and no automatic non-host retry. Earlier
host-pressure stop and12GiB allocation failures remain failed. New independent
`rg-sweep-recovery-20261005-a` is serialized behind quality, exact8K/10K anchors
before12K/16K/20K, bounded host-only retries and separate attempt reports.

Archival gateway repaired using an already authorized second host with the
same NAS mount. Old failed sweep9files and failed report1file are now verified;
new-run terminal report and raw archive workers are queued. Recorded reviews
remain labeled recorded. Earlier progress paragraphs below are historical.

## Progress update — 2026-09-28 23:13 UTC

Direct service/state inspection confirms the recovery chain is progressing.
`rg-blend-r-20260928-b` (`4cb1ac1`) passed its fresh10-case smoke:160 complete
generations and30,960 numerical checks. Source/cache restart/CPU RoPE/ownership/
numerical/named full-zero/native bridge/repeat/raw-output guards PASS.
Development advanced to59/100; no captured numerical failure or host-pressure
retry so far. Its prerequisites passed682 inherited tests, unchanged coverage,
all9 actual GPU transport tests, synthetic and model checks. Full100 audit and
independent100repeat remain pending; no new aggregate quality claim.

Allocator B is intentionally waiting behind quality, with both report workers
waiting for prerequisites. Alive services do not imply concurrent GPU work or
accepted results. At inspection host available~194GiB, system/user pressure0%,
GPUfree~88GiB. These are instantaneous readings, not a guarantee against OOM.

Interrupted combined run A raw archive is now checksum-verified:13,916files,
manifest `2f6878eb2b212acbd7d5c44b9919f7bfa8d8c2f73bbe5c77bb6752ab3a0db08c`.
It remains interrupted/unresolved, not successful. Successor archives still
wait for terminal completion. Last accepted timing length remains10,240; no
new timing point, practical crossover or physical NAND/R claim.

## Recovery update — 2026-09-28 22:33 UTC

The earlier four-service queue did not finish. Combined CB5 run
`rg-blend-r-20260928-a` was killed by systemd-oomd at21:38:39UTC after74 cases;
user-cgroup memory pressure exceeded50% for20seconds. Nearby kernel logs show
global OOM and exhausted swap. Its stale running state is not a completed run.
Allocator run A failed its inherited GPU-sharing tests: expandable-segment CUDA
import was denied `pidfd_getfd`. No new sweep points. Its13-file failed archive
is verified, manifest `6a3dc74a8883bf9770b3af5fc9403447b476f21b4875dfee016d02a8f44b7c6e`.
Reports failed on missing admission/accepted points. These are not quality passes.

Recovery source `4cb1ac1`:24 focused tests PASS. New actual services:
`rg-blend-r-20260928-b` runs fresh inherited tests before smoke10/full100/repeat100;
`rg-allocator-sweep-20260928-b` waits behind it; both corresponding report-b
workers are waiting. All successor admissions remain pending.
Host available-memory and system/user pressure now gate admission and stop owned
children early; at most two retries apply only to explicit host stops, with full
guards rerun and separate attempt directories. Numerical failures never auto-retry.
The original74 cases are diagnostic, not spliced into the new evaluation.

The bounded allocator successor disables expandable segments and sets
`max_split_size_mb:128`, retaining ordinary CUDA sharing. This is not yet a
validated allocation fix. All inherited GPU tests, exact8K/10K overlaps,12GiB
model cap and scientific thresholds remain. No security controls disabled or
system OOM policy changed. Reports now record a clear failure when zero points
are accepted. No unqualified claim of uninterrupted GPU work or automatic repair.

## Superseding update — 2026-09-28 21:05 UTC

- `rg-fp64-validation-20260927-a` (`d8e2645`): retrieved-prefix smoke,
  100-case development and fresh 100-case repeat PASS; 122,796 numerical checks
  per full run. Unchanged numerical/placement budgets; all 13 arms repeat exactly.
  Development EM: native/LMCache/GPU-wide/R each 11%, absent 14%.
  Numerical admission is not evidence benefit or native-bitwise equivalence.
  Raw archive verified: 32,086 files, manifest
  `69d106ec645cfe80cd255e8329f2ac5abd878ddca1e64bc681b772f4a5505f96`.
- `rg-live-route-20260926-a` (`914976f`): 100/100 fresh retrieval/delivery cases
  and all ten ten-case raw audits PASS. Not a live answer endpoint. Archive verified:
  313 files, manifest `a87716b917ea946d6aa360e5ed71f114d3a8146d51abe3e2efd4f96a30adfdff`.
- `rg-blend-r-20260928-a` (`9e86066`): combined R/CacheBlend smoke PASS,
  10 cases, 160 complete generations, 30,960 numerical checks. Full 100-case
  development is running; automatic fresh repeat follows only after audit.
  Selected rows are frozen from admitted CB4 traces to isolate placement.
  Named wide-native full/zero controls and exact native F/B/A bridges pass
  smoke; adaptive live selection and full combined quality remain unpassed.
  Raw archive is pending terminal completion, not yet verified.
- `rg-prefix-extended-20260927-a` (`b71299c`): row-tiled 8,192 and 10,240
  raw audits PASS, including exact repeated 8K tokens/logits/stored KV.
  At 10,240: median recompute/RAM/filesystem TTFT = 1,698.60/2,221.07/2,664.40 ms.
  These are shared-resource staged-HF-eager diagnostics on Qwen2.5-3B BF16,
  RTX PRO 6000 Blackwell Workstation and scratch HDD, not NAND/serving results.
  12,288 failed a 258 MiB MLP allocation under the 12 GiB job cap; 16,384 was
  not attempted. No measured crossover. Raw archive verified: 6,402 files,
  manifest `49f28901761223b50970a6653d5278d9960b7ed9febd07010a1d52f2213b8818`.
- `rg-allocator-sweep-20260928-a` (`051c4f1`): queued behind combined quality.
  Only allocation layout changes (`expandable_segments:True`); caps/math stay
  fixed. Fresh exact 8K/10K anchors precede 12K/16K attempts; stop on failure.
- `rg-blend-r-report-20260928-a` (`fd46121`): waiting for combined development
  and repeat. Will export paired analysis and a recorded HTML case review only
  after all raw audits pass. It is not already available or a live endpoint.

Earlier running/failed statuses below are historical snapshots. Old failed runs
remain failed; only their explicitly named successors may pass. No held-out
quality, practical speed, original CacheBlend paper-number, or physical R claim.

## Superseding update — 2026-09-27 01:22 UTC

`rg-blend-development-20260926-a` (source `f1a72a3`) completed matched smoke,
100-case storage, 100-case GPU quality and exact fresh-engine 100-case repeat.
Inherited/raw checks PASS; 37,186-file archive checksum verified, manifest SHA256
`241dcc1bcfc50c77a36becab1f79ea6751f17c733e036a1eea26a31c92c836cf`.

| Development arm | EM | Contains |
| --- | ---: | ---: |
| No evidence | 11% | 21% |
| Coherent full prefill | 12% | 14% |
| Selective recomputation | 15% | 16% |
| Independent document reuse | 17% | 18% |

All 100 frozen queries are included. New paired analysis from instrument
`78a5df2` verifies receipt hashes and the fresh repeat, retaining ten complete
printed generations per arm. Blend versus absent: six wins/two losses, +4pp EM;
exploratory paired bootstrap 95% interval [-1,+10]pp (10,000 draws, fixed seed).
Blend versus independent reuse: one win/three losses, -2pp, interval [-6,+2]pp.
These intervals are not multiplicity-adjusted or confirmation tests. The +5pp
evidence-gain gate is not passed; no R, held-out, performance or paper-number
claim. Contains is separate from EM. Failure-case evidence sufficiency remains
to be reviewed. The new analysis artifact archive is pending; source raw archive
is already verified. Earlier “running” entries below are historical snapshots.

Live routing is82/100. A persistent eight-hour queue `rg-window-20260927-a`
is active: CPU analysis finished; GPU-safe admission watcher will launch exact
captured R precision attribution, followed by an independent allocation-only
long-prefill repair/overlap/8192 chain. Thirteen focused tests passed. Unchanged
72/48 GiB reserves; no GPU currently qualifies. It does not select a numerical
repair or silently admit failed gates. The existing R100 failure remains failed.

## CacheBlend milestone update — 2026-09-26 20:00 UTC

Both first integration checkpoints now have passing raw audits and verified
durable archives: CB2 storage/composition (10 cases, 70 generations, 883 files)
and CB3 GPU selective recomputation (10 cases, 100 generations, 1,058 files).
CB3 source `194c03a`; raw-state SHA-256
`a914306f8bd4d4442db55769ba8a1a0d22988a31bac9d75995fd4862d5949299`.
These are correctness checkpoints, not admitted quality benefit, R placement,
paper-number reproduction or performance results.

Next run `rg-blend-development-20260926-a`, source `f1a72a3`, is active on
Saturn GPU 2. After 35 focused tests passed, it started the automatic sequence:
fresh ten-case matched no-evidence/miss smoke, 100-case storage, 100-case GPU
selective/reuse/full quality, independent fresh-engine 100-case repeat. Each
stage requires fresh inherited tests and raw acceptance; any failure stops it.
No confirmation data, ratio tuning, new resources or tolerance changes.
All queries, including retrieval misses, remain in the denominator. Development
metrics do not establish the final evidence-benefit/R non-inferiority gates.

CPU routing is 56/100. The retrieved-R numerical failure remains unpassed.

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
