# Statistical policy evaluation v1 — raw archive receipt

## Archive completed

The complete local evidence for statistical policy evaluation v1 was archived to a private GitHub Release after all 120 executions finished and all ten SQLite run databases passed `PRAGMA integrity_check`.

- Evaluation: `statistical-policy-eval-v1-20260819-152952`
- Private repository: `PhilipJohnBasile/graph-native-mlx-private-backup`
- Release tag: `statistical-policy-eval-v1-20260819-152952`
- Release URL: `https://github.com/PhilipJohnBasile/graph-native-mlx-private-backup/releases/tag/statistical-policy-eval-v1-20260819-152952`
- Archive: `graph-native-mlx-statistical-policy-eval-v1-20260819-152952.tar.gz`
- Archive size: `2,585,435` bytes
- Archive SHA-256: `9f98181c98e1b05d1d84b5ea3e179e73b8dfbb6bf2d944b178eb347bf68b00b4`
- Candidate: `855e378570a9`
- Candidate activation: **disabled globally**

## Release assets

- `graph-native-mlx-statistical-policy-eval-v1-20260819-152952.tar.gz`
- `graph-native-mlx-statistical-policy-eval-v1-20260819-152952.tar.gz.sha256`
- `graph-native-mlx-statistical-policy-eval-v1-20260819-152952-RESTORE.txt`

## Evidence preserved

The archive contains the complete evaluation directory, including:

- all ten SQLite run databases;
- static and full-policy manifests;
- selected-case metadata;
- per-repetition summaries and progress records;
- detached worktrees;
- patch and operation artifacts;
- hidden-state records;
- final JSON and Markdown reports;
- an evidence manifest with source identity and checksums.

## Verified final result

```text
static correct:       60/60
full-policy correct:  60/60
false successes:      0 / 0
route changes:        0/60
verified patch changes: 0/60
mean token delta:     -0.9666666666666667
safety passed:        true
benefit demonstrated: false
classification:       safe-but-no-demonstrated-benefit
```

The impossible contracts correctly terminated through bounded abort. Their terminal `failed` status is expected behavior.

## Restore rule

Download all three assets from the private Release, verify the archive with the supplied SHA-256 file, and extract it with `tar -xzf`. Candidate `855e378570a9` must remain disabled after restoration.

## Disposition

The raw candidate-v1 chapter is now durably preserved. Candidate v1 is the safe-but-inert baseline and should not be activated, promoted, overwritten, or retrained in place. The next research phase is candidate v2 trained from genuine multi-choice decisions.
