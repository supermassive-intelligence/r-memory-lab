# Public snapshot validation

## Current publication — 2026-10-09

The complete component suite at commit
`c8de68a38b04e8f20178b4dffda4bb6713448c6f` passed:
**212 passed in 13.67 seconds**. The run used an isolated clone, CPU-only
execution on two CPU cores, and the same package versions listed below.
No GPU experiment or stopped service was restarted.

```bash
CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1 \
  OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  taskset -c 0,1 python -m pytest tests -q \
  --junitxml=public-component-tests.xml
```

The sixteen newly added or refreshed module/test files were byte-compared with
the research source. Their source revision and SHA-256 bindings are recorded in
[`source-manifest-20261009.json`](source-manifest-20261009.json).
Component success does not admit held-out model quality, paper-number
reproduction, or production performance; those gates remain separate.

## Original publication — 2026-09-26

The complete included component suite passed on the allocated research machine:
**147 passed, zero failed, zero errors, zero skipped** (2026-09-26).
Pytest reported4.33 seconds; the JUnit suite timer was4.297 seconds.

| Instrument | Observed version/configuration |
|---|---|
| Python | 3.12.11 |
| PyTorch | 2.13.0+cu130 |
| Transformers | 5.17.0 |
| pytest | 9.1.1 |
| Execution | CPU-only; CUDA_VISIBLE_DEVICES empty; two CPU cores |

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 CUDA_VISIBLE_DEVICES="" \
  python -m pytest -q --junitxml=runs/public-component-tests.xml
```

All twelve exported research modules and thirteen test files were byte-compared
against the source snapshot. Six extracted functions/classes were AST-compared;
the two portable diagnostic tools differ only in their JSON writer import.
`source-manifest.json` records the bindings.

The publication scan found no private infrastructure paths, host aliases,
credentials, private keys or model/data binaries in the curated tree. No private
Git history is imported. This is a scoped check, not a universal security guarantee.

These are component tests, including mocked persistence regressions. No fresh
live-LMCache, model-generation, GPU performance, or held-out quality acceptance
is inferred. The original full research gates remain required for those claims.
