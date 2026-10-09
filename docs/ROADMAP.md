# Quality-first MVP roadmap

Snapshot: 2026-10-09; PAUSED at user request. The [README task list](../README.md#mvp-checklist)
is the compact completion checklist; [STATUS.md](STATUS.md) carries run identities.

September 26 scope amendment: CacheBlend is required, not a post-MVP option.
The [CB0–CB7 integration plan](CACHEBLEND_MVP.md) adds M2b without changing
the numerical, quality, repetition or evidence-admission contracts.

| Milestone | Gate | Current state |
|---|---|---|
| Stored KV fidelity | Byte-identical reload; named native/zero/miss generation controls; independent repeat | Historical retrieved-cache 100-case development + repeat passed |
| Live retrieval routing | Question-only retrieval, compatible stable key, actual LMCache delivery, raw payload audit | 100/100 and all ten ten-case raw audits PASS; not live generation |
| Live generation | Connect fresh routing to generation; complete controls and raw logit audit | Development10/100/fresh100 PASS; bounded HTTP10+fresh10 PASS; services stopped |
| Emulated R attention | Matched numerical checks, no persistent GPU duplicate of R-owned KV, repeat | FP64 retrieved-prefix 100 cases + fresh repeat PASS; unchanged bounds; evidence quality negative |
| Required CacheBlend composition (M2b) | Independent document KV persistence, position remapping, causal-corrected selective recomputation, matched GPU/R controls | Fresh retrieval/selection100+repeat passed; matched FP64 W/R exact; quality-benefit gate unpassed |
| Quality confirmation | Frozen untouched data, declared evidence gain and R non-inferiority bounds | Not passed |
| Practical serving | End-to-end workload, matched uncached/cached controls, TTFT/completion/throughput | Not established |
| Longer prefill | Exact native overlap before admitting memory-bounded attention | Separate24GiB/1GiB L1 sweep through20,480 PASS; diagnostic cache advantage first sampled16K; older failures preserved |
| Portable public harness | Explicit model/data paths, upstream-compatible integration, reproducible accepted runs | Component release available; full harness pending |

No work is queued. If resumed, validate a fresh manual browser session, prepare
the untouched sample/exposure audit and freeze confirmation design. Do not tune
the selector on confirmation data. The selector adds a GPU scout and does not
claim a speed improvement. Keep the measured sweep's resource series separate.
Strict-prefix correctness alone cannot complete the MVP. The long-prefill
allocator issue is a separate diagnostic task, not grounds to relax quality gates.

Keep earliest retrieval launch and delayed consumption separate. A delayed
launch is not mandatory. A negative result, ordinary cached RAG winning, or a
mechanism-only result must narrow the claim rather than trigger unlimited tuning.

Optional precision, scheduling, kernel, and storage branches require a measured
bottleneck. Do not attribute an eager/serving-stack difference solely to graphs.
There is no physical R-machine performance claim in this roadmap.
