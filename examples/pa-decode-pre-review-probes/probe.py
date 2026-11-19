"""Probes for 188df20..201985c (FP8 paged decode routed to aiter).

A. CUDA-graph capture on the aiter path at a batch size not run before capture
   (warm-up at another batch). MSLK's kernel specializes on neither.
B. Block 0 holding NaN (unused by any sequence), padding -1 or a valid non-zero
   page: does the masked tail now poison the output?
C. Routing of calls the tuning never measured: fp16 query, 4-D (non-transposed)
   V, and a block table much wider than the lengths.
"""

import sys
import traceback

import torch

sys.path.insert(0, "<home-in-container>/scratch/eval/case-pa/test/attention/flydsl")
from mslk.aiter.routing import force_backend  # noqa: E402
from mslk.attention.dispatch.paged_attention_decode_fp8 import OP, ROUTING  # noqa: E402
from mslk.attention.flydsl.decode.pa_decode_fp8_dispatch import (  # noqa: E402
    paged_attention_decode_fp8 as op,
)
from pa_decode_fp8_test import make_inputs, reference  # noqa: E402


def scratch(batch, kv, heads, hd, parts=8):
    s = (batch, kv, parts, heads // kv)
    return dict(
        max_context_partition_num=parts,
        exp_sums=torch.empty(s, device="cuda"),
        max_logits=torch.empty(s, device="cuda"),
        temporary_output=torch.empty(*s, hd, dtype=torch.bfloat16, device="cuda"),
    )


print("== A: graph capture after warm-up at another batch")
for backend in ("mslk", "aiter"):
    torch.manual_seed(0)
    try:
        with force_backend(OP, backend):
            a1, _ = make_inputs(32, 8, 128, 16, [256] * 256)
            op(*a1, **scratch(256, 8, 32, 128))
            torch.cuda.synchronize()
            a2, kv2 = make_inputs(32, 8, 128, 16, [256] * 128)
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                op(*a2, **scratch(128, 8, 32, 128))
            a2[0].fill_(float("nan"))
            g.replay()
            torch.cuda.synchronize()
            err = (a2[0].float() - reference(a2, kv2)).abs().max().item()
            print(f"  {backend}: captured, replay max err {err:.3f}")
    except Exception as e:
        print(f"  {backend}: {type(e).__name__}: {str(e)[:300]}")

print("== B: block 0 NaN, unused by any sequence")
for pad in ("-1", "first page"):
    for backend in ("mslk", "aiter"):
        torch.manual_seed(0)
        args, kv = make_inputs(32, 8, 128, 16, [300, 17, 1000])
        bt = args[5]
        # Make block 0 unused: swap it with the spare block (num_blocks-1 is
        # not necessarily spare, so find the unreferenced one).
        used = set(bt[bt >= 0].tolist())
        spare = next(i for i in range(args[2].shape[0]) if i not in used)
        bt[bt == 0] = spare
        kv[0][spare] = kv[0][0].clone()
        kv[1][spare] = kv[1][0].clone()
        args[2][spare] = args[2][0]
        args[3][spare] = args[3][0]
        args[7][spare] = args[7][0]
        args[8][spare] = args[8][0]
        if pad == "first page":
            for b in range(bt.shape[0]):
                bt[b][bt[b] < 0] = bt[b, 0]
        args[2][0].view(torch.uint8).fill_(0x7F)  # e4m3fn NaN
        args[3][0].view(torch.uint8).fill_(0x7F)
        with force_backend(OP, backend):
            op(*args)
        torch.cuda.synchronize()
        out = args[0].float()
        print(
            f"  pad={pad:10s} {backend}: nan rows "
            f"{torch.isnan(out).flatten(1).any(1).tolist()}, "
            f"max err {(out - reference(args, kv)).abs().nan_to_num(1e9).max().item():.3g}"
        )

print("== C: routing of unmeasured calls")
torch.manual_seed(0)
args, _ = make_inputs(32, 8, 128, 16, [256] * 256)
print("  tuned bf16, transposed V:", ROUTING.backend_for(*args))
a16, _ = make_inputs(32, 8, 128, 16, [256] * 256, dtype=torch.float16)
print("  fp16 query:", ROUTING.backend_for(*a16))
v4 = args[3].permute(0, 1, 3, 2, 4).reshape(args[3].shape[0], 8, 128, 16)
print("  4-D V:", ROUTING.backend_for(*args[:3], v4.contiguous(), *args[4:]))


def bench(fn, n=50):
    fn()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(10):
            fn()
    g.replay()
    torch.cuda.synchronize()
    ts = []
    for _ in range(5):
        s, e = torch.cuda.Event(True), torch.cuda.Event(True)
        s.record()
        for _ in range(n):
            g.replay()
        e.record()
        torch.cuda.synchronize()
        ts.append(s.elapsed_time(e) * 1000 / n / 10)
    return min(ts), max(ts)


for batch, length, width_tokens in ((64, 512, 32768), (8, 1000, 131072)):
    torch.manual_seed(0)
    args, _ = make_inputs(32, 8, 128, 16, [length] * batch)
    bt = torch.full((batch, width_tokens // 16), -1, dtype=torch.int32, device="cuda")
    bt[:, : args[5].shape[1]] = args[5]
    args = (*args[:5], bt, *args[6:])
    sc = scratch(batch, 8, 32, 128)
    routed = ROUTING.backend_for(*args, **sc)
    res = {}
    for backend in ("mslk", "aiter"):
        with force_backend(OP, backend):
            res[backend] = bench(lambda: op(*args, **sc))
    print(
        f"  batch {batch} len {length} table {width_tokens}: routed {routed}; "
        + ", ".join(f"{b} {lo:.2f}-{hi:.2f} us" for b, (lo, hi) in res.items())
    )
