# Acceptance contracts

Component tests, experiment execution, raw-evidence acceptance, and scientific
claims are different gates. Passing one does not imply the others.

1. **Reproduce at limit:** every new path and arm must reproduce its explicitly
   named limiting control with matched model, data, positions, history and exposure.
2. **Zero-point identity:** name the no-effect control and require exact identity.
   Do not replace a bitwise contract with a tolerance after observing failure.
3. **Printed evidence first:** ten deterministically selected complete generations
   per arm precede aggregate quality reporting. Audit printed records against raw
   output tokens and per-step logits.
4. **Fail closed:** missing, failed, or unresolved required guards exclude the run
   from headline summaries. Preserve its original run ID and failure diagnosis.
5. **Matched retrieval comparisons:** include misses and wrong evidence in the
   denominator; never use gold answers to filter selected passages.
6. **No confirmation tuning:** freeze configuration and confirmation allocation
   before seeing confirmation outputs. Development is not held-out validation.
7. **Raw evidence:** bind run identity, source/model/data/instrument revisions,
   commands, and guard status to raw artifacts. A checksum-copy receipt is not a
   scientific acceptance receipt.

## Numerical tracks

The historical bitwise split-attention track and the separately declared
`RG-NE-v1` / `RG-placement-v1` tracks are distinct. The exact constants remain in
`experimental/rg_equivalent.py`; this public extraction does not change them.
`RG-RNE-VALID1` changes the reference conversion instrument, not model arithmetic.
Its earlier passing supplied-evidence study does not excuse failures on retrieved
evidence. Small numerical error is not automatically harmless to generation.

The explicit QK64 and row-tiled-softmax candidates are research alternatives, not
silent default repairs. Their names and admission requirements must accompany
results. Synthetic tests cannot substitute for model-level validation.

## Latency and resource scope

Input-known QA requires request-visible TTFT and completion measurements. A
decode-shaped diagnostic alone does not establish a serving claim. Account for
retrieval, queueing, I/O, transfer, reconstruction, synchronization and graph costs.

Historical timing diagnostics used shared GPUs and a filesystem backed by HDD;
restarting LMCache cleared L1, not the OS page cache. They are not cold-NAND or
isolated serving benchmarks. Analytical hardware ceilings are not measured curves.

The public component test suite is explicitly a subset of the original research
suite. The full original inherited/coverage/model/GPU contracts remain required
for research acceptance and have not been weakened or replaced by this extraction.
