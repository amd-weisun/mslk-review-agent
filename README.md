# MSLK ROCm review agent

A Claude Code plugin for reviewing changes to
[MSLK](https://github.com/meta-pytorch/MSLK) on AMD GPUs (gfx942, gfx950):

- `mslk-rocm-review:reviewer`, an independent, read-only review agent. It
  reviews a branch or commit range, or checks whether an aiter op can serve an
  MSLK op, and backs its findings with probes on an idle GPU in your dev
  container.
- `/mslk-rocm-review:review`, a skill that asks you for what the reviewer
  needs, launches it, and checks its findings before reporting.

The reviewer is self-contained: it works from upstream MSLK and from the
references you give it (PRs, commits, files, docs). It needs nothing from any
particular fork.

What it checks, in priority order:

1. Guards and fallback: routing predicates against the validation of the code
   they route to, an argument matrix (dtype, device, rank, shape, emptiness),
   and calls routed to a backend that was never tuned for them.
2. Behaviour with aiter off, which must match the base commit.
3. Kernel memory safety: unbounded buffer descriptors, padding used as
   addresses, empty tensors, index dtype misreads.
4. CUDA-graph safety: host syncs, capture-time allocations, per-size JIT.
5. Tests: every backend forced and routed, aiter on and off, arch skips, and
   tolerance checked by mutation.
6. ROCm CI wiring: test lists, path filters, `package_data`.
7. Performance: A/B under CUDA graphs; tuning buckets where a backend changes
   config inside the bucket.
8. Completeness of an aiter integration.
9. Style, against the surrounding code.

Every finding is ranked High, Medium or Low by its impact if merged
(definitions in the agent), and tagged reproduced (with the probe script) or
suspected, and pre-existing or introduced.

## Install

```bash
claude plugin marketplace add amd-weisun/mslk-review-agent
claude plugin install mslk-rocm-review@mslk-rocm-tools
```

Claude Code asks for your settings when the plugin is enabled; change them
later with `/config`:

| Setting | Example |
|---|---|
| Dev container | `alice_mslk` |
| Repo path in the container | `/workspace/MSLK` |
| Home path in the container | `/mnt/host_home` |
| Probe directory (host) | `~/scratch/review` |
| Python setup command | `export PATH=/opt/venv/bin:$PATH` |
| aiter source path in the container | `/opt/venv/lib/python3.12/site-packages/aiter` |
| Lint tools directory in the container | `/opt/lintenv/bin` |

The container must be running, with your MSLK checkout and your home directory
mounted.

Update: `claude plugin marketplace update mslk-rocm-tools`, then
`claude plugin update mslk-rocm-review@mslk-rocm-tools`.

## Use

Run `/mslk-rocm-review:review`, optionally with what to review:

> /mslk-rocm-review:review main..my-branch

It asks for whatever is missing:

- the range, branch or PR;
- what the change claims, and what must not change;
- references you have (PRs, commits, files, tickets, design notes or earlier
  reviews it follows), which are optional but sharpen the review;
- a GPU, if you have a preference.

Then it runs the reviewer and checks its findings. If the reviewer lacked
something that would have changed the review, its report starts with
**References that would help**.

You can also call the agent directly. Give it facts, not your conclusions:

> Review `main..my-branch` with mslk-rocm-review:reviewer. The change routes
> `<op>` to aiter's `<op>` through a tuning table and should behave as before
> with aiter off. It follows PR #574. Use GPU 6.

To let it run code from a commit other than your checkout, make a worktree
under your home (the skill does this for you):

```bash
git worktree add --detach ~/scratch/wt/review <commit>
ln -s <clone>/mslk/mslk.so ~/scratch/wt/review/mslk/mslk.so   # reuse your build if C++ is unchanged
```

Check its findings before acting on them: it separates reproduced from
suspected, but it can still be wrong.

## Eval set

`eval/` holds changes with known problems and their answer keys, to measure
the agent before and after you change it. Each case is self-contained: an
upstream MSLK base commit (`BASE`) and patches on top of it.

```bash
# In a clone of upstream MSLK that contains the base commit:
eval/setup_case.sh <case> <mslk-clone> ~/scratch/eval/<case>
```

Then run the reviewer on the printed range with that worktree, without hints
and without showing it the answer key, and score its report against
`eval/<case>/answer_key.md`.

| Case | Known problems | Scores |
|---|---|---|
| `pa-decode-pre-review` | 15 | v1: 9/15 (no longer held out) |
| `bf16-grouped-without-fixes` | 5 | v2: 4/5 (held out) |

A case is only a fair test of agent versions written before anyone looked at
the agent's results on it. When the agent misses something real, add the
change as a new case: `git format-patch <base>..<head> -o eval/<case>/patches`,
the base commit in `eval/<case>/BASE`, and an answer key.

## Changing the agent

Edit `plugins/mslk-rocm-review/agents/reviewer.md`, bump `version` in
`plugins/mslk-rocm-review/.claude-plugin/plugin.json`, run
`claude plugin validate ./plugins/mslk-rocm-review`, and record the new
version's eval scores in this README.
