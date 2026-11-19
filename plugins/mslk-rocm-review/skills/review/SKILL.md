---
name: review
description: Review an MSLK change on the ROCm side with the independent mslk-rocm-review:reviewer agent. Collects the range, what the change claims, and the user's references first, then launches the reviewer and checks its findings. Use when the user asks for a ROCm/AMD review of an MSLK branch, commit range or PR, or a fit check of an aiter op for an MSLK op.
---

# MSLK ROCm review

Run the `mslk-rocm-review:reviewer` agent on a change, with the inputs it needs.

## 1. Collect the inputs

Take what the user already said or what the arguments give; ask only for what
is missing, in one message:

- **What to review**: a branch, a commit range (`base..head`), a PR, or, for a
  fit check, the MSLK op and the aiter op.
- **What the change claims**: what it should do and what must not change
  (e.g. "behaves as before with aiter off").
- **References** they have: PRs, commits, files, tickets, design notes or
  earlier reviews it follows or relates to. Say these are optional but make
  the review sharper.
- **GPU** to use, if they have a preference.

If the user has no references, go ahead without them.

## 2. Prepare

- Resolve the range to commits in the user's MSLK clone (`git log --oneline
  <range>`). If the head is not the checked-out commit, create a worktree of
  it under the user's home so the container can import it:
  `git worktree add --detach ~/scratch/wt/review-<name> <head>`, and link a
  built `mslk/mslk.so` from the clone if the change does not touch C++. Ask
  before creating it if the user has not asked you to work in their clone.
- If no GPU was named, pick one with ~0% GFX activity
  (`docker exec <container> amd-smi metric --usage`).

## 3. Launch the reviewer

Launch `mslk-rocm-review:reviewer` with a prompt that contains only facts:
the mode (diff review or fit check), the range and worktree path, what the
change claims, the references, and the GPU. Do not add your own opinion of
the change: the review is meant to be independent.

## 4. Check and report

The reviewer separates reproduced from suspected findings, but it can be
wrong. Before you present a confirmed bug as fact, check its evidence (read
the cited code, or rerun its probe if cheap). Then report to the user:
confirmed bugs, risks, nits, what was not checked, and any references the
reviewer asked for. Say which findings you verified.

Remove any worktree you created once the user is done with it.
