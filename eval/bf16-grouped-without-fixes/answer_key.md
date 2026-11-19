# Case bf16-grouped-without-fixes: BF16 grouped GEMM routed to aiter, without its MSLK fixes

Base: upstream MSLK `BASE`. Patch: `patches/` — `bf16bf16bf16_grouped_stacked`
and `bf16bf16bf16_grouped_grad` routed to aiter's Triton `gmm` through tuning
tables, as first written, applied without the MSLK fixes it was later stacked
on (a dgrad CUDA-graph fix and a test-gating fix).

Score a report by how many known problems it finds (a suspected finding with
the right cause counts), and note false claims separately.

| # | Known problem | Kind |
|---|---|---|
| 1 | MSLK's dgrad (`mslk/gemm/triton/grouped_gemm.py`, `m_sizes.tolist()`) host-syncs, so it fails under CUDA-graph capture (pre-existing) | bug |
| 2 | The dgrad tuning table cannot come from this commit's tuning script: the driver times under CUDA graphs, where MSLK's dgrad fails | bug |
| 3 | MSLK's dgrad rejects non-empty groups of 1-7 rows (pre-existing), so the new tests with small groups fail on the MSLK and default paths | bug |
| 4 | `test/gemm/dispatch_test.py` gates its routing checks on `is_aiter_available`, so the new entries fail with `MSLK_DISABLE_AITER=1` | bug |
| 5 | The same gating in `MXFP4Tests` (`test/gemm/gemm_test.py`), pre-existing | bug |
| 6 | The new dgrad tests sit in a class skipped on gfx950, the arch the tables cover | test gap |
| 7 | aiter's `gmm` switches config at M = 256·G inside some buckets, so part of those buckets runs a config that was never timed and is slower than MSLK | perf |
| 8 | `num_sms=0` on the stacked op is routed to aiter, which raises | bug |
| 9 | With aiter off, a 0-d `X` raises a different error than the base (routing key computed first) | bug |
