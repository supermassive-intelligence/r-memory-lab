# Verified checkpoint — 2026-09-26 16:17 UTC

This is a manually verified snapshot from the original research workspace.
Public component tests are separate; historical raw artifacts are not included.

| Run | Verified outcome | Claim boundary |
|---|---|---|
| `rg-retrieved-cache-20260926-b` | Ten-case smoke, 100-case development, and independent 100-case repeat raw audits PASS; four arms match across deployments | Stored retrieved-document KV fidelity, not evidence-quality improvement or live arbitrary-prompt serving |
| `rg-retrieved-r-20260926-b` | Ten-case smoke PASS; 100-case development numerical audit FAIL; repeat not launched | Excluded from admitted R-quality results |
| `rg-live-route-20260926-a` | 39/100 queries completed; ten-case raw delivery audit PASS; process active | Routing/storage only; no new generation |
| `rg-tiled-preflight-20260926-a` | All 14 native attention/probability/replay GPU tensor checks PASS | Not full-model or long-context admission |
| `rg-tiled-prefill-20260926-a` | Failed startup reserve check before first point | No new timing point or 8,192-token result |
| `rg-prefix-sweep-20260925-b` | Native 1,024/2,048/4,096 per-length raw audits PASS; 8,192 exceeded the 12 GiB job allocator budget | Shared-resource, staged-eager, HDD diagnostics only; no practical crossover established |

Retrieved R failure: five failing layer-step guard records among 121,356 recorded
layer-step checks. Two actual-output failures are the same event in R and R-repeat
(case index28, decode step9, layer35). This count is a diagnosis, not an acceptance
rate that excuses the failures. Root-cause analysis remains required.

The fresh tiled sweep observed 73,563 MiB free against a fixed 73,728 MiB startup
minimum: 165 MiB short. It failed before inherited tests or model execution. The
candidate's mathematics was not rejected by this startup failure.

Completed cache, failed R, and failed tiled runs have checksum-verified raw
archives in the research storage system. Run IDs and hashes are retained there;
no private hostnames, paths, credentials, prompts, or raw tensors are published.
No new aggregate answer-quality metric is asserted by this public snapshot.
