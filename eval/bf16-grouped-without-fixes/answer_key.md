# Case bf16-grouped-without-fixes: BF16 grouped fwd/dgrad routed to aiter, without its MSLK fixes

Base: upstream MSLK `BASE` (188df20). Patch: `patches/` — the BF16 grouped aiter routing as first written, applied without the MSLK fixes it was later stacked on (a dgrad CUDA-graph fix and a test-gating fix), and without the dgrad graph test from those fixes.

| # | Known problem | Kind |
|---|---|---|
| 1 | MSLK dgrad (mslk/gemm/triton/grouped_gemm.py, m_sizes.tolist()) host-syncs: fails under CUDA-graph capture, so routed/MSLK calls are not graph safe (pre-existing) | bug |
| 2 | The dgrad tuning table can't be produced by this commit: the driver times under CUDA graphs and MSLK's dgrad fails capture | bug |
| 3 | MSLK dgrad rejects non-empty groups of 1-7 rows (pre-existing); new op tests with small groups fail on the MSLK/default path | bug |
| 4 | test/gemm/dispatch_test.py gates routing checks on is_aiter_available: the new entries fail with MSLK_DISABLE_AITER=1 | bug |
| 5 | Same is_aiter_available gating in MXFP4Tests (gemm_test.py ~3461), pre-existing | bug |

## v2 (held out): 4/5
Caught 1-4 (4 mislabelled pre-existing: the new dispatch entries are introduced). Missed 5.
Extra true findings: BF16Tests skipped on gfx950 by its class decorator; aiter's gmm switches
to large_kn at M = 256*G inside a bucket, so parts of 3 buckets run an untimed config and
are slower (confirmed real: fixed later in the op's PR); num_sms=0 raises on
aiter-routed stacked calls ; 0-d X error type with aiter off.
