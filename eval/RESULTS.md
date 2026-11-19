# Eval results

One row per run: the reviewer on a case, without hints, scored against the
case's answer key.

| Plugin version | Case | Found | False claims | Wall time | Notes |
|---|---|---|---|---|---|
| 0.6.0 | bf16-grouped-without-fixes | 6/9 (missed 5, 8, 9) | 0 | 6 min 52 s | Case was used while writing the checklist. The review cited later commits from other branches (since disallowed in 0.6.1). 88 tool calls. |
| 0.6.1 | pa-decode-pre-review | 9/17 (missed 4, 5, 6, 7, 9, 11, 13, 14) | 0 | 6 min 22 s | Case was used while writing the checklist. Also found 3 real problems not in the key. Report: examples/pa-decode-pre-review.md. Probe scripts not saved: writes to the probe directory still needed approval in the unattended run. |
| 0.6.2 | pa-decode-pre-review | 7/17 (missed 3, 4, 5, 9, 10, 11, 13, 14, 15, 17) | 0 | 6 min 15 s | Same case as the 0.6.1 run: about ±2 problems between runs. Probe scripts saved (Edit rule for the probe directory). Report and probes in examples/. |
