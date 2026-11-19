# Example review: `pa-decode-pre-review`

What `/mslk-rocm-review:review` returned for the eval case
`eval/pa-decode-pre-review` (an MSLK fix to the FP8 paged-attention decode
kernel, then that op routed to aiter, as first written). The report below is
unedited.

- Plugin version: 0.6.1
- Run: unattended (`claude -p "/mslk-rocm-review:review <range> ..."`), with
  the range, what the change claims, PR #574 as the reference, and GPU 5 in
  the prompt
- Wall time: 6 min 22 s on gfx950 (MI350X)

## How it scored against the answer key

9 of the 17 known problems found, no false claims.

| Known problem (key #) | Report |
|---|---|
| Cache dtype other than e4m3fn routed to aiter (1) | #1, by reading |
| Scales of a larger block pool routed to aiter (2) | #1 |
| Output not shaped like the query routed to aiter (3) | #1 |
| With aiter off, `block_tables=None` raises a different error (8) | #5 |
| Block-table reads unbounded past the last row (10) | #9 |
| Masked reads go to block 0, which must hold finite values (12) | #3 |
| Graph capture without scratch works on aiter shapes only (15) | #4 |
| Tolerance lets small K/V scale errors pass (16) | #7 |
| Return string depends on the backend (17) | #10 |
| Spare table rows, CPU scales, 4-D V cache routed to aiter (4, 5, 7) | missed |
| f16 queries routed though only bf16 was tuned (6) | missed |
| Zero-width block table faults (9) | missed |
| int64 lengths or tables misread (11) | missed |
| aiter compiles per batch size (13) | missed (listed as not checked) |
| Input-building code copied in three files (14) | missed |

Real problems it found that the key did not list: short real lengths in a
large block table are routed to aiter where it is slower (#2), a V cache with
more blocks than the K cache (#1), and a different error when `query` has
more rows than the batch (#6).

---

I reviewed `188df20..52949df` (7e262dc, the padding fix and tests; 52949df, routing to aiter) and found **no High findings and four Medium ones**. The tests pass with aiter on and off, lint is clean, and the CI wiring is complete.

The reviewer ran its probes on GPU 5 in the existing worktree at `~/scratch/eval/case-pa`. I didn't create or change anything. I checked findings 1, 2, 3 and 5 against the code; the rest are as the reviewer reported them.

## Summary

| # | Severity | Finding | Evidence | Where |
|---|---|---|---|---|
| 1 | Medium | `_aiter_accepts` lets through calls that aiter rejects but MSLK used to serve | reproduced; **verified** | `mslk/attention/dispatch/paged_attention_decode_fp8.py:85-112` |
| 2 | Medium | Routing looks only at how many pages the block table can hold, not the real lengths, so short sequences in a large table go to aiter, which is slower there | reproduced; **verified** | `dispatch/paged_attention_decode_fp8.py:115-130`, tune script `_lengths` |
| 3 | Medium | The padding fix points the masked tail at block 0. If block 0 holds NaN, every output row is NaN on both backends, so "padding can hold any value" is not fully true | reproduced; **verified** | `mslk/attention/flydsl/decode/pa_decode_fp8.py:1604-1620` |
| 4 | Medium | CUDA-graph capture without scratch buffers works on shapes routed to aiter but raises ValueError on shapes routed to MSLK | reproduced | aiter adapter vs `pa_decode_fp8.py:2290` |
| 5 | Low | `_key` runs even with aiter off, so `block_tables=None` now raises AttributeError, not ValueError, which breaks "behaves as before" | reproduced; **verified** | `dispatch/paged_attention_decode_fp8.py:125-126` |
| 6 | Low | When `query` has more rows than the batch, the error changes from NotImplementedError to ValueError on shapes routed to aiter | reproduced | `dispatch/…:97` |
| 7 | Low | The 0.3/0.3 test tolerance would miss a K/V scale a few percent off or a few dropped tokens | suspected | `test/attention/flydsl/pa_decode_fp8_test.py:35` |
| 8 | Low | The adapter calls `plan_pa_decode` on every call, so it allocates during capture; aiter's docstring says to allocate outside capture (capture still passed) | read | `mslk/attention/aiter/paged_attention_decode_fp8.py:47` |
| 9 | Low | The block-table `s_buffer_load` has no size bound and reads past the last row when the table width isn't a multiple of 256 tokens; the values are masked but the read is still out of range | suspected, pre-existing | `pa_decode_fp8.py:1351, 1616-1626` |
| 10 | Low | The backend name the op returns (`"aiter"` or `"ps_small_block"`) now depends on the shape, and this isn't documented | read | `pa_decode_fp8_dispatch.py:141` |

## Medium findings

**1. The aiter gate is looser than aiter's own checks.** On a shape the table sends to aiter (32, 8, 128, 16, context 256, batch 256), three cases raise ValueError in aiter but ran correctly on the MSLK kernel:
- a 2D `output`;
- scale tensors with more blocks than the cache;
- a `value_cache` with more blocks than `key_cache`.

`_aiter_accepts` checks none of these. It also doesn't check that `query` is 3D, that the cache dtype is `float8_e4m3fn` on gfx950, or that the block table is int32.
- **Fix:** add those conditions to `_aiter_accepts`.

**2. Routing ignores the real lengths.** `_key` sets `context_max` from table width × block size, rounded up to a power of two. The tuner only measures lengths between half of that and all of it. Servers often size block tables for the model's maximum length, so short real lengths get aiter's long-context choice. Timed in a CUDA graph on GPU 5:

| shape (H, KVH, bs, capacity, B) | real length | routed to | mslk | aiter |
|---|---|---|---|---|
| 128,8,64,131072,1 | 131072 | aiter | 159.6 µs | 56.5 µs |
| 128,8,64,131072,1 | 2048 | aiter | 7.4 µs | 12.7 µs |
| 128,8,64,131072,1 | 512 | aiter | 7.3 µs | 11.5 µs |

- **Fix:** do one of these:
  - tune over several real-length/capacity ratios;
  - add a key the caller supplies (like aiter's `max_context_length`);
  - document that the table assumes tables sized close to the real lengths.

**3. Block 0 must hold finite values.** I confirmed in the code that pages past a sequence's length are replaced with block 0. The reviewer filled an unused block 0 with FP8 NaN and set the padding to -1: all four sequences came out NaN on both backends, because the masked lanes still compute `0 * NaN`. aiter documents the same requirement; MSLK's comment only says block 0 "always exists".
- **Fix:** do one of these:
  - state the requirement in the docstring and add a test for it;
  - better, set the probabilities or V to zero for masked positions, not just the scores.

**4. Whether CUDA-graph capture works depends on which backend the table picks.** Without scratch buffers:
- (32,8,128,16, batch 256 at 256) goes to aiter and captures fine.
- (8,1,128,16, batch 8 at 1024) goes to MSLK and raises `ValueError CUDA graph capture requires preallocated exp_sums…`.

So a caller that tested one shape can fail on another, or after the table is regenerated.
- **Fix:** require scratch buffers under capture whichever backend serves the call, or document it.

## Not checked
- The base commit itself was not run. "Worked before" was inferred by forcing the `mslk` backend at the head, which is the base code plus the padding fix.
- Whether `mslk.aiter.tuning` reproduces the committed table row by row.
- The CPU cost per call outside graphs, and whether aiter recompiles across batch sizes.
- Whether any path other than the small-block stage reads the block table.
- fp16 output on shapes routed to aiter, beyond the one test case.

## Test and lint results
- **aiter on:** 36 passed. `lazy_import_test` fails only when run in the same process as the other tests; on its own it passes.
- **`MSLK_DISABLE_AITER=1`:** 24 passed, 12 skipped, all of them cases that force or need aiter.
- **Lint:** flake8 and ufmt are clean.
- **Table and packaging:** the table covers exactly the shapes that were left untuned, and `setup.py` packages it.

## Reference the reviewer asked for
The reviewer had no worktree for the base commit (188df20) or for PR #574. Giving it one would let it test "behaves as before" directly. It also couldn't save its probe scripts because writing to `~/scratch/review/case-pa/` needed approval. They can be rerun from the inputs above.
