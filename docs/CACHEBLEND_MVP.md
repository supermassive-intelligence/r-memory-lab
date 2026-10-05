# Required CacheBlend + LMCache + R-machine MVP

Scope frozen September 26, 2026. This extends the quality-first MVP at the user's
request. Strict-prefix KV reuse remains a baseline; it cannot complete this MVP.

## Intended request path

```text
Question -> answer-blind ranked documents -> compatible document KV keys
                                                |
                             LMCache persistent store -> R-side DRAM
                                                |
                         request-position remapping + selective recomputation
                                                |
                               +----------------+----------------+
                               |                                 |
                         GPU-only control                 matched R/G attention
                               +----------------+----------------+
                                                |
                                   complete answers + guard audit
```

Document KV is computed independently before the request. Store each block's
pre-RoPE K and V with model/tokenizer/token/context/layout identity. Apply native
RoPE at the requested positions on load; do not reinterpret post-RoPE prefix KV
as relocatable document KV. Question KV is not persisted in this document store.

Position remapping alone does not restore missing inter-document interactions.
CacheBlend's selective recomputation is an approximation to repair those
interactions; its quality must be measured. Recompute selected rows on G and
use absolute-position causal masks. The R branch must separately validate
external-KV ownership, partial attention and merging against its GPU control.
R is emulated; current HDD-backed storage is not advertised as NAND hardware.

## Fine-grained execution and acceptance

| Step | Deliverable | Required gate | State |
|---|---|---|---|
| CB0 | Recover prior corrected CacheBlend port and full repeats | Archive/source identity; preserve original causal failure | Complete; [evidence](CACHEBLEND.md) |
| CB1 | Independent pre-RoPE document encoder and stable identity | No query/answer dependence; all-layer capture; exact native-position reconstruction; incompatible KV rejected | Ten-case checkpoint passed; 20 independent documents |
| CB2 | Real LMCache persistence, restart/reload and ordered composition | Byte-identical payloads; valid lengths; remapped resident/reloaded outputs and every-step logits identical; miss/zero controls | `rg-blend-storage-20260926-b`: raw audit PASS, 70 complete generations; 883-file raw archive verified |
| CB3 | GPU causal-corrected selective recomputation | Fixed layer-1 selection and 0.16 value-difference ratio initially; sparse row trace; future-value causality; exact named full-recompute endpoint | `rg-blend-prefill-20260926-b`: ten-case/100-generation raw audit PASS; 1,058-file raw archive verified; not quality confirmation |
| CB4 | Matched retrieved GPU quality | No evidence / coherent full prefill / independent reuse / selective reuse; same inputs and positions; ten-case smoke then 100-case all-query development | `rg-blend-development-20260926-a`: all stages and exact fresh repeat completed; raw/archive PASS. EM absent11/full12/blend15/reuse17%; development experiment complete, quality-benefit gate unpassed |
| CB5 | Same composed KV and selected rows on R/G | Existing numerical budgets, no persistent external GPU duplicate, limit/zero controls, independent repeat | Fixed-row100+fresh100 passed in rg-blend-r-20260928-b; request-time selection validation active in rg-blend-live-20261005-a; quality gate still unpassed |
| CB6 | Live demo, untouched confirmation and reproduction | Evidence-benefit and R non-inferiority gates; live/replay labels; cache/recomputed rows; reproducible complete outputs | Pending |
| CB7 | Delayed-consumption study | All K=0…36 with fixed exposure/history; at most two frozen confirmation choices | After immediate composition is admitted |

CB1–CB6 are mandatory for the integrated MVP. A negative result is valid research,
but cannot be checked off as a successful quality-preserving demo.

## First bounded executable checkpoint

Qwen2.5-3B, BF16, existing frozen BM25 development manifest. Use its first ten
queries without correctness filtering and first two ranked chunks, independently
tokenized with a trailing separator and capped at 256 tokens each. This is a
declared new multi-document workload, not a silent replacement of the old
top-one result. Empty retrieval remains in the denominator. No retriever sweep,
gold-conditioned filtering or confirmation-set access.

Capture raw K/V projections through all 36 blocks, verify original-position
RoPE against the native cache, persist through LMCache, restart and reload.
Compare these named controls, each with ten complete printed generations:

| Arm | Meaning | Exact-match requirement |
|---|---|---|
| U | Resident independent-document composition | Named storage control, not presumed equal to coherent full prefill |
| L / L-repeat | Persisted/reloaded composition, including restart | Complete tokens and every-step logits equal U |
| F / F-zero | Coherent full prefill and cache bypass | F-zero equals F |
| A / A-miss | No evidence and forced missing-key fallback | A-miss equals A at matched reserved question positions |

This first checkpoint does **not** implement selective recomputation, establish
answer-quality benefit, or validate R attention. It prepares CB3. Existing
Mistral results cannot be transferred to this Qwen/storage/retrieval workload.

The first attempt (`rg-blend-storage-20260926-a`) failed before generation
because the Transformers cache iterator returned auxiliary metadata after K/V.
The successor extracts K/V explicitly and adds a regression test. The failed
attempt remains archived; no tolerance or token workload was changed.

CB3 uses the same ten token layouts and a verified private copy of CB2's actual
LMCache files. The 0.16 fraction applies to document rows; all uncached
instruction/question rows remain fresh. This is a declared document-only
adaptation of the corrected algorithm, not original question-cache semantics.
Its ten arms include native/repeat, ratio-1 full-limit, resident/reloaded/repeated
blend, independent reuse/repeat and blend/reuse disabled-cache controls. Raw
tokens/logits, sparse row traces and all 100 complete generations are required.
Disabled-cache controls execute the same full prompt and must match native;
they are not operational cache-miss fallback, which remains the named no-evidence
control. No R attention or aggregate quality claim from this ten-case check.

Fresh inherited tests, unchanged coverage floors and actual GPU transport tests
precede model execution. Keep two CPUs, 40 GiB host RAM/no swap, a 12 GiB model
allocator, 24 GiB test allocator, 72 GiB initial free-device reserve and 48 GiB
stop reserve. No wall-clock cap or competing latency benchmark. Preserve failures
under their original run IDs and checksum-archive raw payloads/logits/source.

## Combined R checkpoint — 2026-09-28

At23:13UTC, successor B passed fresh10-case/160-generation smoke and reached
59/100 development with no captured numerical failure or host retry. Full raw
audit, fresh100repeat and final quality admission remain pending.

Recovery at22:33UTC: first full development interrupted by host oomd at74cases.
Successor `rg-blend-r-20260928-b` (`4cb1ac1`) restarts smoke/full/repeat from case1,
adding host-pressure checks and bounded host-stop-only retries. Historical partial
results are not spliced or admitted; model/evidence/arithmetic/contracts unchanged.

`rg-blend-r-20260928-a` (`9e86066`, CB5-FIXED1) passed its 10-case smoke:
160 complete generations and 30,960 numerical checks. The automatic 100-case
development and independent repeat are not yet admitted. The same row selection
from each admitted CB4 trace is used for GPU and CPU/GPU attention, isolating
placement from changes in selection. This is not an adaptive live selector.

Actual LMCache restart/reload supplies CPU-owned external KV. Selected/fresh
rows remain on G through prefill and decode; absolute causal maps account for
noncontiguous positions. CPU RoPE is checked bitwise against GPU. Temporary
auditor reference copies are not production ownership or timing evidence.
FP64 accumulation uses unchanged numerical and placement budgets. Full/zero
endpoints name independent wide-native H; native F/B/A must retain exact CB4
outputs. Partial FP64 attention is not claimed bitwise identical to native BF16.
Failed historical paths remain failed. A terminal-only recorded review/report
is queued after the full repeat; live generation and confirmation remain work.

## Scientific gates remain unchanged

The completed 100-case continuation froze the same top-two selection, truncation,
token construction and recompute policy before outputs. A fresh ten-case smoke
adds A, A-repeat and real missing-key fallback; question and decode positions
remain matched when evidence is absent. It then executes 100-case storage,
100-case selective/full/reuse quality and an independent fresh-model repeat.
All stages require fresh inherited tests and raw audits. Complete sample records
precede development EM/containment reporting. Development scores are not held-out
confirmation, and no parameter sweep is hidden in this sequence.

Apply [all existing acceptance contracts](ACCEPTANCE.md). Every new arm needs
named zero/limit/repeat controls; unresolved or failed guards exclude headline
results. Do not relax tolerances to admit the current R numerical outlier.

After development, freeze untouched confirmation and configuration. Require
at least +5 percentage points EM evidence benefit with the paired 95% interval
excluding zero, and R non-inferiority within 1 percentage point against both
matched native and LMCache GPU controls under the predeclared one-sided bounds.
Report selective-versus-full and selective-versus-independent-reuse comparisons
separately; storage identity is not quality parity. No ratio/layer tuning on
confirmation, no inherited paper-performance claim, and no physical R speed claim.
