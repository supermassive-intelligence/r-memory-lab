# Reproduction boundaries

## Included and runnable

The component tests exercise grouped-query attention splitting/merging, masks,
empty partitions, exact endpoints, numerical budgets, prefix packing/unpacking,
stable cache identities, frozen BM25 selection, routing rejection rules, BF16
reference conversion and persistence false-success regressions.

`scripts.rg_retrieval_manifest` accepts explicit `--index`, `--development`,
`--index-audit`, `--out`, and `--count` paths. It requires a completed, independently
audited compatible index; this release does not supply or rebuild that corpus.
Only question text enters the search function. Corpus and model licenses must be
handled separately by the operator.

## LMCache transport

`scripts.rg_lmcache_prefix.PrefixStore` is the extracted engine-driven transport
used by the experiments, including checksum-verified offline filesystem
population. It launches an owned **loopback-only** service, uses an explicitly
owned output directory, and supports a configurable pair of unprivileged ports.
The historical L1 capacity is 0.5 GiB and transport chunks are 256 tokens.

It requires the experimental LMCache 0.5.5 multiprocess engine-driven interface,
including `RegisterEngineDrivenContextPayload`, `IPCCacheServerKey`, and the
`lmcache.v1.multiprocess.http_server` entry point. The historical runtime was a
pinned research checkout; an arbitrary current `pip install lmcache` is **not**
asserted to be compatible. Integration on a clean public dependency set remains
a milestone. The component persistence tests mock the transport and are not
represented as a fresh live-LMCache integration run.

Transport uses Python pickle only with its owned trusted loopback service.
Do not expose that service publicly or load untrusted serialized payloads.
Persistent KV is sensitive derived data and should be protected like its source.

## Not included in this release

- Historical fleet SSH/systemd/Kubernetes launchers and private infrastructure paths.
- Model weights, tokenized corpora, query/answer manifests, or raw KV/logit tensors.
- The full model-generation/scoring harness and its inherited model/GPU suite.
- An arbitrary-prompt live demo endpoint or production-serving integration.

Consequently, the historical 100-query checkpoints cannot be reproduced from
this release alone. Portable model/data configuration, public-compatible LMCache
integration and a complete guarded harness are upcoming work, not finished claims.

## Historical model context

The recent checkpoints used Qwen2.5-3B base, BF16, 36 layers, eager attention;
model snapshot `3aab1f1954e9cc14eb9509a215f9e5ca08227a9b`.
The measured workstation GPU was an RTX PRO 6000 Blackwell Workstation Edition,
not a simulated Max-Q node. Correctness checks used shared allocated compute;
timing diagnostics do not imply isolated throughput or production latency.
