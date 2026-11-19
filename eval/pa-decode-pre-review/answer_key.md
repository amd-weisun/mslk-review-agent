# Case pa-decode-pre-review: FP8 paged decode routed to aiter, before review

Base: upstream MSLK `BASE` (188df20). Patches: `patches/` — the paged-decode MSLK fix and its aiter routing, as first written, before any review.

| # | Known problem | Kind | v1 |
|---|---|---|---|
| 1 | FP8 cache dtype other than e4m3fn (uint8 / fnuz) routed to aiter, raises | bug | caught |
| 2 | Scales of a larger block pool routed to aiter, raises | bug | caught |
| 3 | Non-3-D output routed to aiter, raises | bug | caught |
| 4 | Block table with spare rows routed to aiter, raises | bug | caught (by reading) |
| 5 | Scales on CPU routed to aiter, raises | bug | missed |
| 6 | f16 queries routed though only bf16 was tuned | risk | missed |
| 7 | 4-D (non-transposed) V cache routed to aiter | risk | missed |
| 8 | block_tables=None / bad query rank change error type with aiter off | bug | caught |
| 9 | Zero-width block table faults (pre-existing) | bug | missed |
| 10 | Block-table descriptor unbounded, reads past last row (pre-existing) | bug | caught (suspected) |
| 11 | int64 lengths/tables silently misread by MSLK (pre-existing) | bug | caught |
| 12 | Block 0 must hold finite values after the fix; undocumented | doc | missed |
| 13 | aiter JIT-compiles per batch size | risk | caught |
| 14 | Input builder copied in test, tuner, dispatch test | nit | missed |
| 15 | MSLK needs preallocated scratch to capture, aiter doesn't | risk | missed |

v1: 9/15. Extra true findings: V-scale errors < ~30% pass the tolerance; 2-D query
error type; return value differs per backend. False claim: lint not installed.
Not held out from v2 on: the agent's checklist was revised after missing items here.
