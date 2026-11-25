# Case pa-decode-as-submitted: FP8 paged decode fix and aiter routing, as submitted for review

Base: upstream MSLK `BASE`. Patches: `patches/` — the same change as
`pa-decode-pre-review`, after the earlier review rounds, as it was submitted
as two pull requests (a kernel fix with tests, and the aiter routing). The
known problems are the ones human reviewers found in those pull requests.

Score a report by how many known problems it finds (a suspected finding with
the right cause counts), and note false claims separately.

| # | Known problem | Kind |
|---|---|---|
| 1 | The new block-table bound is `grid x * row stride * 4` bytes, which is 0 for a table with row stride 0 (e.g. built with `expand`): every read returns 0, every sequence reads block 0, and results are wrong (introduced) | bug |
| 2 | Masked tile slots now read block 0, so block 0 must hold finite data even when no sequence uses it: NaN in an unused block 0 gives NaN output, where the base did not (introduced) | bug |
| 3 | The CSR entry point converts page ids to int32 but not `context_lengths`, so int64 lengths, which it used to accept, now raise on the new int32 check (introduced) | bug |
| 4 | A comment and padding workaround in `fp8_paged_cache.py` (padding each sequence to a full partition to avoid an out-of-bounds fault) are obsolete once the fault is fixed | doc |
| 5 | `random_fp8_paged_cache` raises on an empty list of lengths (`max(1, *pages)`) | bug |
| 6 | Under TheRock, the first aiter call after any MSLK FlyDSL kernel segfaults (Triton loaded after the ROCm SDK's LLVM); both test files the change adds to CI crash. The repo's other tests import Triton first for this | bug |
| 7 | The routing rule sums full-length and short-context times, so long contexts dominate and aiter gets shapes that run at 0.5-0.75x with short contexts; the trade-off is not an explicit, bounded choice | design |
| 8 | The aiter adapter builds a new work plan (allocation plus planning kernel) on every call; aiter's documented use is to prepare one outside graph capture, with its tuned budget, and refresh it in place | perf |
| 9 | The tuning script's correctness check is weaker than the op's tests (elementwise only, no per-sequence scale and relative-error checks), though most aiter-routed shapes are only checked by the tuner | test gap |
| 10 | Per-run tuning tables and logs (~6,000 lines) are committed though the combined table already records both runs | nit |
