# The runner: how rollouts get produced

Gate 3 says *what* it needs: these arms, this many scenes. This directory says
*how*, for one particular machine: openpi to train and serve, RoboTwin to
evaluate.

The split is the point. Producing rollouts means choosing a trainer, a
checkpoint format, a policy server and a simulator, four choices the gate must
not make, because the moment it imports one of them the framework has a
favourite model and "no preference to any specific model" stops being true. The
gate invokes an executable, hands it a JSON job, and reads run records back.

## The contract

    run.sh <job.json>

Job in:

    {"dataset": "<lerobot dir>", "trials": 20, "task": "pick_dual_bottles",
     "arms": ["your data", "shuffled control"], "baseline": "baseline",
     "out": "<dir>", "progress": "<file.jsonl>"}

Out, in `out/`: one run record per arm, and `arms.json` naming which file is
which arm. Progress is appended to `progress` as one JSON object per line,
`{"phase", "current", "total", "note"}`, which the gate tails and forwards.

Progress goes to a file rather than stdout because the trainer and the simulator
both write freely to stdout in formats that are not ours. A file the runner owns
is a contract; scraping somebody else's log is a guess that breaks when they
reformat.

## Wiring it up

    BENCH_RUNNER=/path/to/run.sh          # the gate stays inert without this
    BENCH_TRAIN_STEPS=3000                # smaller for a smoke run

A worker with no `BENCH_RUNNER` reports that it cannot produce rollouts, which
is `failed`, our machinery, and never `refused`, which would blame the
contributor's data for our gap.

## What this one does, in order

1. build both arms, cut to identical lengths
2. register each in openpi's own config list
3. compute norm stats, train, serve, evaluate, one arm at a time
4. write run records and the manifest

One arm at a time, and the checkpoint is deleted after its evaluation: a
checkpoint here is 8.5 GB against 13 GB free, so two cannot coexist. A previous
hand-run of this pipeline died mid-write with ENOSPC and lost a training run.
