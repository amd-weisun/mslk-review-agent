# Example review: `pa-decode-pre-review`

What `/mslk-rocm-review:review` returned for the eval case
`eval/pa-decode-pre-review` (an MSLK fix to the FP8 paged-attention decode
kernel, then that op routed to aiter, as first written). The report below is
unedited. The probe scripts it cites are in `pa-decode-pre-review-probes/`
(the container's home path replaced by a placeholder).

- Plugin version: 0.6.2
- Run: unattended (`claude -p "/mslk-rocm-review:review <range> ..."`), with
  the range, what the change claims, PR #574 as the reference, and GPU 5 in
  the prompt
- Wall time: 6 min 15 s on gfx950 (MI350X)

## How it scored against the answer key

7 of the 17 known problems found, no false claims. (A run of 0.6.1 on the same
case found 9: expect a couple of problems either way between runs.)

| Known problem (key #) | Report |
|---|---|
| Cache dtype other than e4m3fn routed to aiter (1) | #6, suspected |
| Scales of a larger block pool routed to aiter (2) | #6, suspected |
| f16 queries routed though only bf16 was tuned (6) | #3 |
| 4-D V cache routed to aiter (7) | #3 |
| With aiter off, malformed inputs raise a different error (8) | #5 |
| Masked reads go to block 0, which must hold finite values (12) | #1 |
| Tolerance lets small K/V scale errors pass (16) | #4 |
| Non-3-D output, spare table rows, CPU scales routed to aiter (3, 4, 5) | missed |
| Zero-width block table faults (9) | missed |
| Block-table reads unbounded past the last row (10) | listed as not checked |
| int64 lengths or tables misread by MSLK (11) | missed (#6 notes aiter's int32 check only) |
| aiter compiles per batch size (13) | missed |
| Input-building code copied in three files (14) | missed |
| Graph capture without scratch works on aiter shapes only (15) | missed |
| Return string depends on the backend (17) | missed |

Real problems it found that the key did not list: with a block table much
wider than the contexts, routing can also pick MSLK where aiter is faster
(#2, measured 29 vs 20 µs).

---

I checked the reviewer's findings against the code. Now the report.

# ROCm review: 188df20..201985c (FP8 paged-attention decode: padding fix plus aiter routing)

The reviewer found no High-severity bugs. The change's tests pass on GPU 5 with aiter on (36 passed) and with aiter off (24 passed, 12 skipped). There are four Medium findings: two are about how the routing picks a backend, one is a new requirement the padding fix puts on block 0, and one is the test tolerance.

## Summary

| # | Severity | Finding | Evidence | Where |
|---|---|---|---|---|
| 1 | Medium | Masked tile slots now read block 0, so NaN or Inf in block 0 makes every sequence's output NaN | reproduced on the head; base not run | `mslk/attention/flydsl/decode/pa_decode_fp8.py:1604-1637` |
| 2 | Medium | The routing key uses the block-table width instead of the real context length, so some calls get the wrong backend | reproduced | `mslk/attention/dispatch/paged_attention_decode_fp8.py:125-129` |
| 3 | Medium | fp16 queries and 4-D V are sent to aiter, but tuning only measured bf16 queries with 5-D V | reproduced (routing decision only) | `dispatch/paged_attention_decode_fp8.py:85-112` |
| 4 | Medium | The test tolerance `atol = rtol = 0.3` on outputs of size about 1 would not catch scale errors of a few percent | suspected | `test/attention/flydsl/pa_decode_fp8_test.py:35` |
| 5 | Low | With aiter off, `_key` still runs before the aiter-enabled check, so malformed inputs fail with a different error than on base | suspected | `mslk/aiter/routing.py:206` |
| 6 | Low | `_aiter_accepts` does not check int32 table and length dtypes, the KV dtype, or the scale shape, so aiter raises where MSLK's kernel did not | suspected | `dispatch/...:85-112` |
| 7 | Low | `test_routes_by_its_tuning_table` mostly repeats what `dispatch_test.py` checks | read | `pa_decode_fp8_test.py:146` |

## Medium findings

**#1: block 0 must now hold finite values.**
- **What changed:** the fix replaces every table entry past `num_context_blocks` with `zero_block`. I confirmed this at lines 1609-1636. The comment only says block 0 "always exists"; it does not say block 0 must be finite.
- **What the reviewer ran:** it filled an unused block 0 with 0x7F, which is NaN in e4m3fn. All output rows came out NaN, both with -1 padding and with padding set to the sequence's first page. With a finite block 0 the error was 0.125.
- **Why it may be a regression:** on base, padding with a valid page would have read that finite page. The reviewer did not run base, because no base worktree was given.
- **aiter has the same behaviour:** it also returns NaN in this case.
- **Suggested fix:** document the requirement on the op and add a test for it. Better, zero the masked P·V term with a select instead of relying on 0·x, so block 0's contents no longer matter.

**#2: routing by table width.**
- **What the code does:** `_key` computes `context_max` from `block_tables.shape[1] * block_size` (confirmed at lines 125-129). Tuning made the table exactly as wide as the context, but serving stacks often size the table for the model's maximum length.
- **Measured case where it picks wrong:** batch 64, length 512, table width 32768. The call goes to MSLK at about 29 µs, while aiter takes about 20 µs.
- **Measured case where it picks right:** batch 8, length 1000, table width 131072.
- **Suggested fix:** the GPU lengths can't be read on the host without breaking CUDA graphs. So either add a host-side `max_context_length` argument (aiter's `pa_decode` takes one), or tune with wide tables and short lengths. At minimum, document the assumption.

**#3: calls that were never measured go to aiter.** The table key has no query dtype or V layout, and `_aiter_accepts` doesn't check either (confirmed in the code). aiter does accept both, so this is a speed risk, not a correctness bug. Suggested fix: require `query.dtype == torch.bfloat16 and value_cache.dim() == 5` in `_aiter_accepts`, or add both to the key.

**#4: loose test tolerance.** The current bound catches a wrong head, page or length. It would miss a softmax or V scale that is 5–20% off, or a few dropped tokens. Suggested fix: add a relative-norm check, for example `(out - ref).norm() / ref.norm() < 0.05`.

## Checked with no problems found
- Every shape in the untuned list has a row in the tuning table.
- CUDA-graph capture works on the aiter path.
- CI enables both new test files, the path filter covers the new paths, and `setup.py` packages the table.
- flake8 and ufmt are clean.
- `lazy_import_test` fails only when run in the same process after other test files; it passes on its own, and CI runs each file separately.

## Not checked
- **Base side of #1:** whether base really behaved better here was not run.
- **Eager host cost of the aiter path:** each call runs `plan_pa_decode` and three scratch allocations; nobody measured what that costs.
- **Block-table reads near the end:** whether masked reads can go past the end of the block table was not checked. That code (`max_size=False`) was the same before this change.
- **Tuning table accuracy:** it was not checked beyond the two wide-table points in #2.
- **4-D V in MSLK's kernel:** whether MSLK's own kernel handles 4-D V correctly was not checked.

## What I verified
- I read the code behind #1, #2 and #3 and it matches what the reviewer describes.
- I did not rerun the GPU probes. The NaN result and the timings are the reviewer's numbers, from `~/scratch/review/case-pa/probe_b.py`.
- #4–#7 are as the reviewer reported them. I did not check them myself.

**References the reviewer asked for:**
- A base worktree, to confirm whether #1 is a regression.
- How the serving stack sizes its block tables and initializes block 0. This decides how much #1 and #2 matter in practice.

I didn't create any worktree, so there is nothing to clean up. The reviewer's probe scripts are in `~/scratch/review/case-pa/`.
