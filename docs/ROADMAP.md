# Quality-first MVP roadmap

Snapshot: 2026-09-28 23:13 UTC. The [README task list](../README.md#mvp-checklist)
is the compact completion checklist; [STATUS.md](STATUS.md) carries run identities.

September 26 scope amendment: CacheBlend is required, not a post-MVP option.
The [CB0–CB7 integration plan](CACHEBLEND_MVP.md) adds M2b without changing
the numerical, quality, repetition or evidence-admission contracts.

| Milestone | Gate | Current state |
|---|---|---|
| Stored KV fidelity | Byte-identical reload; named native/zero/miss generation controls; independent repeat | Historical retrieved-cache 100-case development + repeat passed |
| Live retrieval routing | Question-only retrieval, compatible stable key, actual LMCache delivery, raw payload audit | 100/100 and all ten ten-case raw audits PASS; not live generation |
| Live generation | Connect fresh routing to generation; complete controls and raw logit audit | Not complete |
| Emulated R attention | Matched numerical checks, no persistent GPU duplicate of R-owned KV, repeat | FP64 retrieved-prefix 100 cases + fresh repeat PASS; unchanged bounds; evidence quality negative |
| Required CacheBlend composition (M2b) | Independent document KV persistence, position remapping, causal-corrected selective recomputation, matched GPU/R controls | GPU100-case development+repeat raw checks/archive verified; blend15% versus absent11% EM; quality gate and R integration unpassed |
| Quality confirmation | Frozen untouched data, declared evidence gain and R non-inferiority bounds | Not passed |
| Practical serving | End-to-end workload, matched uncached/cached controls, TTFT/completion/throughput | Not established |
| Longer prefill | Exact native overlap before admitting memory-bounded attention | Row-tiled through 10,240 PASS; 12,288 MLP allocation failed under 12 GiB cap; allocator-layout successor queued |
| Portable public harness | Explicit model/data paths, upstream-compatible integration, reproducible accepted runs | Component release available; full harness pending |

Immediate order: finish combined fixed-row R/CacheBlend development and repeat,
review paired quality and failures, then validate adaptive live selection and
fresh routing-to-generation. Only then freeze untouched quality confirmation.
The fixed-row smoke passed; full combined admission is pending.
The first full attempt was host-OOM interrupted; guarded successor B is active.
Its fresh smoke passed and development reached59/100 without retries; full
development/repeat admission remains pending.
The first allocator successor failed CUDA IPC prerequisites; a separately
declared standard-allocator candidate is queued with all guards unchanged.
Strict-prefix correctness alone cannot complete the MVP. The long-prefill
allocator issue is a separate diagnostic task, not grounds to relax quality gates.

Keep earliest retrieval launch and delayed consumption separate. A delayed
launch is not mandatory. A negative result, ordinary cached RAG winning, or a
mechanism-only result must narrow the claim rather than trigger unlimited tuning.

Optional precision, scheduling, kernel, and storage branches require a measured
bottleneck. Do not attribute an eager/serving-stack difference solely to graphs.
There is no physical R-machine performance claim in this roadmap.
