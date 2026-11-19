#!/usr/bin/env bash
# Build an eval case as a git worktree: upstream base commit + the case's patches.
#   eval/setup_case.sh <case> <mslk-clone> <worktree-dir>
# <mslk-clone> must contain the base commit (a clone of upstream MSLK, fetched).
# Put <worktree-dir> under your home so the dev container can see it. If the
# clone has a built mslk/mslk.so, it is linked in (fine while the case does not
# change C++).
set -euo pipefail
case_dir="$(cd "$(dirname "$0")" && pwd)/$1"
clone="$2"
dest="$3"
[ -f "$case_dir/BASE" ] || { echo "no such case: $1" >&2; exit 1; }
base="$(cat "$case_dir/BASE")"
git -C "$clone" cat-file -e "$base^{commit}" 2>/dev/null \
  || { echo "base $base not in $clone; fetch upstream MSLK first" >&2; exit 1; }
git -C "$clone" worktree add --detach "$dest" "$base"
git -C "$dest" -c user.name=eval -c user.email=eval@localhost am -q "$case_dir"/patches/*.patch
if [ -e "$clone/mslk/mslk.so" ]; then
  ln -s "$(cd "$clone" && pwd)/mslk/mslk.so" "$dest/mslk/mslk.so"
fi
echo "Review range: $base..$(git -C "$dest" rev-parse --short HEAD) in $dest"
echo "Answer key (do not show the reviewer): $case_dir/answer_key.md"
