# Case pa-decode-pre-review: FP8 paged decode routed to aiter, before review

Base: upstream MSLK `BASE`. Patches: `patches/` — an MSLK fix to the FP8
paged-attention decode kernel, and that op routed to aiter's FlyDSL `pa_decode`
through a tuning table, as first written, before any review.

Score a report by how many known problems it finds (a suspected finding with
the right cause counts), and note false claims separately.

| # | Known problem | Kind |
|---|---|---|
| 1 | FP8 cache dtype other than e4m3fn (`uint8`, `fnuz`) routed to aiter, which raises | bug |
| 2 | Scales covering a larger block pool than the cache routed to aiter, which raises | bug |
| 3 | Output not shaped like the query (2-D, 4-D) routed to aiter, which raises | bug |
| 4 | Block table with spare rows (more rows than sequences) routed to aiter, which raises | bug |
| 5 | Scales on the CPU routed to aiter, which raises | bug |
| 6 | f16 queries routed to aiter though the table was tuned with bf16 only | risk |
| 7 | 4-D (non-transposed) V cache routed to aiter | risk |
| 8 | With aiter off, `block_tables=None` or a bad query rank raises a different error than the base (routing key computed first) | bug |
| 9 | Zero-width block table faults the GPU (pre-existing) | bug |
| 10 | Block-table buffer descriptor unbounded: tile reads past the last row (pre-existing) | bug |
| 11 | int64 context lengths or block tables silently misread by MSLK's kernel (pre-existing) | bug |
| 12 | After the fix, masked reads go to block 0, which must hold finite values; not documented | doc |
| 13 | aiter JIT-compiles per batch size: first call at a new batch is slow in eager serving | risk |
| 14 | Cache-building input code copied in the test, the tuning script and the dispatch test | nit |
| 15 | MSLK's kernel needs preallocated scratch to be captured in a CUDA graph, aiter's does not: capture works on some shapes, fails on others | risk |
| 16 | Test tolerance (atol = rtol = 0.3) lets V-scale errors under ~30% pass | test gap |
| 17 | The op's return string differs by backend (`"aiter"` vs `"ps_small_block"`) | nit |

Note: this case was used while the agent's checklist was written, so scores on
it overstate how the agent does on unseen changes.
