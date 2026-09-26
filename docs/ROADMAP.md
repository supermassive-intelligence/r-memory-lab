# Quality-first MVP roadmap

Snapshot: 2026-09-26 18:33 UTC. The [README task list](../README.md#mvp-checklist)
is the compact completion checklist; [STATUS.md](STATUS.md) carries run identities.

| Milestone | Gate | Current state |
|---|---|---|
| Stored KV fidelity | Byte-identical reload; named native/zero/miss generation controls; independent repeat | Historical retrieved-cache 100-case development + repeat passed |
| Live retrieval routing | Question-only retrieval, compatible stable key, actual LMCache delivery, raw payload audit | Running, 49/100; first ten-case audit passed |
| Live generation | Connect fresh routing to generation; complete controls and raw logit audit | Not complete |
| Emulated R attention | Matched numerical checks, no persistent GPU duplicate of R-owned KV, repeat | Retrieved 100-case numerical gate failed; captured failures replayed, correction pending |
| Quality confirmation | Frozen untouched data, declared evidence gain and R non-inferiority bounds | Not passed |
| Practical serving | End-to-end workload, matched uncached/cached controls, TTFT/completion/throughput | Not established |
| Longer prefill | Exact native overlap before admitting memory-bounded attention | Tiled 1,024/4,096 native overlap passed; 8,192 allocation failed under 12 GiB cap |
| Portable public harness | Explicit model/data paths, upstream-compatible integration, reproducible accepted runs | Component release available; full harness pending |

Immediate order: finish live-routing acceptance; isolate the R rounding failure
and validate a bounded correction; wire fresh routing to GPU generation before
R generation; then freeze untouched quality confirmation. The long-prefill
allocator issue is a separate diagnostic task, not grounds to relax quality gates.

Keep earliest retrieval launch and delayed consumption separate. A delayed
launch is not mandatory. A negative result, ordinary cached RAG winning, or a
mechanism-only result must narrow the claim rather than trigger unlimited tuning.

Optional precision, scheduling, kernel, and storage branches require a measured
bottleneck. Do not attribute an eager/serving-stack difference solely to graphs.
There is no physical R-machine performance claim in this roadmap.
