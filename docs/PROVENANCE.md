# Source provenance and publication scope

This repository starts with a curated snapshot, not the private workspace's full
Git history. The source workspace revision is
`f683642` (2026-09-26). The existing repositories/worktrees were not reset,
rewritten, deleted, or merged during publication.

- All twelve `experimental/rg_*.py` modules are copied without modification.
- Thirteen selected `tests/test_rg_*.py` files are copied without modification.
- `PrefixStore`, its port/delivery helpers, and `load_selected_cases` are extracted
  unchanged from the original `scripts/rg_lmcache_prefix.py`. Model-generation
  and scoring entry points are deliberately excluded, not stubbed as working.
- `boundary_cases` is extracted unchanged from `scripts/rg_reference_conversion.py`.
- Retrieval-manifest and tiled-attention diagnostic modules retain their logic;
  their JSON writer import now points to a portable local helper.
- New documentation and package configuration describe this narrower release.

See `source-manifest.json` for SHA256 bindings of exported original modules,
tests, and extracted function bodies. This is a traceability record, not a claim
that the entire original research suite is public or has been rerun here.

The publication excludes raw model/data artifacts, corpus/query examples,
private infrastructure configuration, unrelated project code and work logs.
Historical failed results remain failures. No research acceptance threshold has
been changed by this extraction. Dependency licensing is not transferred by
including an import or reference to another project.
