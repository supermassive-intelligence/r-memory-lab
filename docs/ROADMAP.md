# Quality-first MVP roadmap

| Milestone | Gate | Current state |
|---|---|---|
| Stored KV fidelity | Byte-identical reload; named native/zero/miss generation controls; independent repeat | Historical retrieved-cache 100-case development + repeat passed |
| Live retrieval routing | Question-only retrieval, compatible stable key, actual LMCache delivery, raw payload audit | Running; first ten-case audit passed |
| Live generation | Connect fresh routing to generation; complete controls and raw logit audit | Not complete |
| Emulated R attention | Matched numerical checks, no persistent GPU duplicate of R-owned KV, repeat | Retrieved 100-case numerical gate failed; diagnose captured inputs |
| Quality confirmation | Frozen untouched data, declared evidence gain and R non-inferiority bounds | Not passed |
| Practical serving | End-to-end workload, matched uncached/cached controls, TTFT/completion/throughput | Not established |
| Longer prefill | Exact native overlap before admitting memory-bounded attention | Candidate tensor smoke passed; full sweep pending |
| Portable public harness | Explicit model/data paths, upstream-compatible integration, reproducible accepted runs | Component release available; full harness pending |

Keep earliest retrieval launch and delayed consumption separate. A delayed
launch is not mandatory. A negative result, ordinary cached RAG winning, or a
mechanism-only result must narrow the claim rather than trigger unlimited tuning.

Optional precision, scheduling, kernel, and storage branches require a measured
bottleneck. Do not attribute an eager/serving-stack difference solely to graphs.
There is no physical R-machine performance claim in this roadmap.
