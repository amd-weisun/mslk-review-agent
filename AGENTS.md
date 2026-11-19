# Maintaining the MSLK ROCm review plugin

This repo is a Claude Code marketplace (`mslk-rocm-tools`) with one plugin,
`mslk-rocm-review`: the `reviewer` agent and the `review` skill. See
`README.md` for what they do and how people install and use them.

## Layout

- `.claude-plugin/marketplace.json`: the marketplace; lists the plugin.
- `plugins/mslk-rocm-review/.claude-plugin/plugin.json`: the plugin manifest:
  `version` and the per-user settings (`userConfig`).
- `plugins/mslk-rocm-review/agents/reviewer.md`: the review agent.
- `plugins/mslk-rocm-review/skills/review/SKILL.md`: the skill that collects
  inputs, launches the agent and checks its findings.
- `eval/<case>/`: a change with known problems: `BASE` (an upstream MSLK
  commit), `patches/`, and `answer_key.md`. `eval/setup_case.sh` builds a
  case; `eval/RESULTS.md` logs eval runs.

## Rules for changes

- **Keep the agent general.** It reviews any MSLK change on the ROCm side.
  When it misses something, add or sharpen a general principle that would
  also catch the same kind of problem in other changes. Never name a case's
  files, ops, values or bugs in the agent or the skill.
- **Keep it self-contained.** The agent may rely only on upstream MSLK and on
  references the user gives. Nothing about a particular fork, branch, person
  or machine: per-user details belong in `userConfig`.
- **Keep it read-only.** The agent writes only probe files, with the Write
  tool, in the user's probe directory; never in the repo under review.
- **Keep it fast.** Reviews should stay around 10 minutes. A new check should
  say when it applies, so the agent can skip it otherwise.
- **No internal labels.** Refer to agent versions only by the plugin
  `version`.
- **Bump `version`** in `plugin.json` with every change under `plugins/`
  (patch for wording or small fixes, minor for new checks or behaviour).
  Installed copies update only when the version changes.
- **Validate** before committing: `claude plugin validate .` and
  `claude plugin validate ./plugins/mslk-rocm-review` must pass.
- **Measure** changes to the agent or the skill: run at least one eval case
  (see README, Eval set) and add a row to `eval/RESULTS.md` with the plugin
  version, case, problems found, false claims and wall time. Say in the PR
  how the result compares with the previous row for that case.
- **Never edit an answer key to match a review.** Add a problem to a key only
  when it is confirmed real in that change, whoever found it.
- **Add cases** for real problems the agent missed in a review: `BASE`, the
  patches (`git format-patch <base>..<head>`), and an answer key, as the
  existing cases do. Cases whose misses shaped the checklist overstate later
  scores; new cases keep the eval honest.
- **Changes reach `main` through pull requests**, with the validation and
  eval results in the description. Do not push to `main` directly.
