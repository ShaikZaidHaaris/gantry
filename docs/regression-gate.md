# Running the gate on every checkpoint

The habit worth having: every checkpoint gets paired against the last accepted
one automatically, so a regression is caught by a red build rather than by
somebody noticing months later.

## Locally

```bash
# after a run has been recorded into history
gantry ci-pin <run-key> --task lift_cube        # once, to establish the reference
gantry ci <run-key> --task lift_cube            # on every candidate after that
```

Exit codes: `0` pass, `1` significant regression, `2` the comparison could not
be made (no such run, no shared scenes). A drop that is not separable at the
trial count exits `0` and says so — a gate that blocks on noise gets switched
off within a week and then protects nothing.

## In GitHub Actions

```yaml
name: evaluation
on: [pull_request]

jobs:
  regression:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"
      - run: pip install -e ".[dev]" -e plugins/feedback_power -e plugins/feedback_rank

      # However your evaluation runs — the point is that it ends by recording
      # the run into history, which is what makes the next step possible.
      - name: evaluate and record
        run: python your_eval.py --record history/
        id: eval

      - name: compare against the pinned baseline
        run: |
          gantry ci "${{ steps.eval.outputs.run_key }}" \
            --history history/ \
            --summary "$GITHUB_STEP_SUMMARY"
```

Writing the report to `$GITHUB_STEP_SUMMARY` puts the table in the run's summary
page, so a reviewer sees the paired comparison without opening logs.

## What it reports

```
### lift_cube: mh_official vs baseline ph_official

| | rate | n |
|---|---|---|
| baseline | 56.0% | 50 |
| candidate | 66.0% | 50 |

paired on 50 shared scene(s): won 5, lost 0, p=0.0625
threshold p<0.0167 (3 run(s) recorded on this task)

Up +10.0%.
```

The threshold tightens as a task accumulates runs: the tenth checkpoint on a task
is not making the first one's claim, and the arithmetic says so rather than
leaving it to whoever reads the number.

## Why pinning rather than "the most recent run"

Because the most recent run is sometimes the broken one. A gate that silently
re-baselines on whatever ran last will accept an arbitrarily large regression as
long as it arrives one commit at a time.
