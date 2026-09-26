# CacheBlend reproduction history

Reconciled against archived execution and analysis artifacts on **2026-09-26**.
We did attempt reproduction; describing CacheBlend only as related work omitted
substantial completed experiments. However, completed quality experiments are
not the same as reproducing the paper's exact quality or performance numbers.

## Experiments completed

| Track | Workload and evidence | Supported conclusion |
|---|---|---|
| Adapted modern-runtime quality | Mistral-7B-Instruct-v0.2; all 150 MuSiQue and 200 WikiMQA examples; full run, independent repeat, zero/endpoint controls and inherited suite | Admitted adapted quality checkpoint; both frozen ±0.02 tokenizer-F1 point targets passed, but confidence intervals did not establish equivalence |
| Adapted independent-reuse comparison | Same 350 examples; independent document reuse versus selective recomputation; admitted source runs and matched default outputs | Selective recomputation improved tokenizer F1 over independent reuse on both datasets; this is a mechanism result, not exact paper numbers |
| Original-layout corrected Blackwell port | Original-example token layouts; all 150/200 cases and fresh-engine full repeats; corrected absolute-position causal mask | Completed with recorded `ALL_PASS` execution guards; full-recompute endpoint matches native output tokens and complete first-token logits; quality results below |
| Unmodified legacy-stack probe | Original PyTorch 2.2.1/CUDA 12.1 stack on allocated Blackwell | Failed GPU tensor creation with `no kernel image`; no unmodified-stack quality result |
| First-output instrumentation | Ten-case-per-dataset smoke, matched outputs and cache controls | Instrumentation validated; not a paper TTFT or throughput reproduction |

### Original-layout corrected port

Run **`cb-repro-artifact-full-20260919-b`**, source `ac3dc43`, completed
**2026-09-20 08:42 UTC**. The old local state file still described it as running;
the terminal archive supersedes that stale entry. Admission status is explicitly
`admitted-corrected-port-quality-not-paper-numbers`.

Ten deterministically selected complete generations per arm/dataset were printed
in the archived analysis log before aggregate reporting. Raw generations,
first-token logit vectors, cache traces, inherited-suite results and repeats
remain in the research archive; this public note does not redistribute them.

| Dataset | Cases | Native full F1 | Corrected blend F1 | Independent reuse F1 |
|---|---:|---:|---:|---:|
| MuSiQue | 150 | 0.276829 | 0.263222 | 0.157277 |
| WikiMQA | 200 | 0.239646 | 0.213261 | 0.192147 |

These are the pinned example's tokenizer-F1 scores, not EM. Compared with
independent reuse, corrected blend's paired F1 difference is +0.105944
(95% CI +0.057727 to +0.154717) on MuSiQue and +0.021114
(−0.023233 to +0.065453) on WikiMQA. The latter is inconclusive.
Compared with native full prefill, the differences are −0.013607 and −0.026385;
neither interval establishes equivalence. In particular, the WikiMQA point
loss exceeds 0.02; the earlier adapted study's point-target pass must not be
transferred to this different workload. Execution guards passing is not a
quality-parity guarantee.

## Why exact paper-number reproduction remains unestablished

- The executable port uses modern Blackwell-compatible software, not the
  unmodified legacy dependency stack.
- Original bottom-right sparse-query masking failed the absolute-position
  causal guard. Its results remain excluded; the reported port uses a declared
  correction rather than silently calling the original algorithm validated.
- Matching the supplied examples is not matching every figure-specific input,
  retrieval manifest, load trace, model revision and hardware condition.
- The admitted port is resident-cache quality work. It does not establish
  storage/NAND behavior, request-visible latency gains or throughput gains.
- The historical CacheBlend experiment harness is not included in this first
  public component extraction; its results are provenance, not a claim that
  running the public unit tests reproduces them.

## Evidence identity

| Checkpoint | Run identity |
|---|---|
| Adapted quality | `cb-repro-qa-full-20260917-a`; repeat `cb-repro-qa-full-20260918-b` |
| Adapted zero controls / suite | `cb-repro-zero-full-20260918-a`; `cb-repro-inherited-suite-20260918-c` |
| Independent reuse | `cb-repro-reuse-full-20260918-a`; derived comparison reports `ALL_PASS` |
| Original-layout full/repeat | `cb-repro-artifact-full-20260919-b`; all four admitted arms report `ALL_PASS` |
| Legacy compatibility failure | `cb-repro-legacy-gpu-20260919-a` |
| Timing instrumentation smoke | `cb-repro-ttft-instrument-20260919-a` |

The four original-layout raw JSON hashes were recomputed against the durable
archive and matched the admission report on September 26:

| Artifact | SHA-256 |
|---|---|
| `first-musique.json` | `85b7144093235ad532b11022136de9b6ad8cfef5f9755a23c8afffd07d128c1e` |
| `repeat-musique.json` | `5537d3a784f3cfc13b1dddcbbf2f8e3c791ee3d565a54d179299b42c3c68594e` |
| `first-wikimqa.json` | `bfdb3b48f83b4a623bd7919b2202cdef4fe86e0fefd667bb1b3db0c412bed399` |
| `repeat-wikimqa.json` | `39de9c3a6b4013f9ebf903b76acfebec74972047756d6c940e3b2b2231d78bca` |

This reconciliation checked terminal state, the inherited-suite receipt,
analysis code/report, printed-sample log and raw-file hashes. It did not rerun
generation or reinterpret an archive checksum as a new scientific acceptance.

## Relevance to the R-machine MVP

CacheBlend motivates selective recomputation when independently cached document
KV is reused beyond a strict prefix. It does not by itself implement R-side
attention. The current MVP first validates compatible prefix-KV storage/reload,
question-only retrieval and R/G placement. Following the user's September 26
scope amendment, CacheBlend-style composition is a **required integration
milestone**, not an optional extension. It is not yet completed. See the
[execution plan and gates](CACHEBLEND_MVP.md).
