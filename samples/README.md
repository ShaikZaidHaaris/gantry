# Samples

The three training sets the experiment actually ran, so a fresh clone has
something to upload. Without these, nothing in the repository can exercise the
product's own path: `bench/data/` is where uploads land and it is gitignored, so
a clone arrives with an empty bench and no way to fill it.

| file | episodes | frames | what it is |
|---|---|---|---|
| `baseline.zip` | 50 | 6,129 | RoboTwin's own demonstrations, alone |
| `baseline_plus_ego_two_handed.zip` | 58 | 8,423 | the same 50, plus ego clips where both hands were tracked |
| `baseline_plus_ego_one_handed.zip` | 58 | 8,512 | the same 50, plus ego clips where one hand was mostly absent |

All three are `pick_dual_bottles` on an `aloha-agilex`, 25 fps, 16-wide state and
action, head camera as mp4, LeRobot layout. Built by
`experiments/robotwin_ego/build_arms.py` and `build_ab.py`.

    087d4916ec1f994d61fe4883926402977bfbee5e8741757ae1f26cfef5746f9b  baseline.zip
    238d60037ef61a7ce60003d9c10f82458e9b28d6d1d167b6642386e1d979ffeb  baseline_plus_ego_one_handed.zip
    1a89c900fb4d2648d15e0d951fd850c11eb0cd2fe49466f5b4bc6408803619c1  baseline_plus_ego_two_handed.zip

## What the three are for

The 50 RoboTwin episodes are **identical in all three**. The only thing that
differs is which ego footage was added on top, and in the baseline's case that
nothing was. That is what makes the comparison mean something: any difference in
what a policy learns has one candidate cause.

The ego clips come from EPIC-Kitchens, put through hand-pose estimation and
retargeting, then converted into RoboTwin's own action space and world frame at
build time so the policy is trained in the frame it will be run in. They are
split by a real property of the footage, the fraction of frames in which both
hands were moving, and not by a corruption invented for the purpose. An arm the
tracker never found is held at its last value by the connector, so it sits in
the data as a well-formed, right-dtype, motionless block.

The task is bimanual, so footage where both hands were visible **should** teach
it better than footage where one was not. That ordering is predicted by the task
rather than asserted, which is what makes this a validation of the benchmark and
not only a measurement. If it cannot rank these two, it cannot rank two real
contributors either.

## What happened when they ran

Each trained as a π₀.₅ LoRA and evaluated closed-loop on expert-screened scenes.

    baseline                        12 / 100    12%
    baseline + ego two-handed        4 /  50     8%    vs baseline p=0.75
    baseline + ego one-handed        0 /  50     0%    vs baseline p=0.031, worse

**Neither ego addition improved on the baseline.** The two-handed half is
indistinguishable from it and the one-handed half is significantly worse. That
is the honest headline and it is not the flattering one.

The finding is in the comparison between the two additions, on the ladder:

    rung             two-handed      one-handed     paired test
    moved            50/50  100%     50/50 100%     p=1.0
    lifted           50/50  100%     49/50  98%     p=1.0
    moved all 2      44/50   88%     34/50  68%     p=0.002    separated
    lifted all 2     41/50   82%     31/50  62%     p=0.0063   separated
    solved            4/50    8%      0/50   0%     p=0.125    not separated

Read the bottom row before the others. Binary success, the number everybody
reports, **cannot tell the two additions apart**: 4 against 0 is four
disagreements, all favouring the two-handed half, which is the most extreme
outcome available at that count, and it still only reaches p=0.125. The rungs
above it are measured on the same rollouts for no extra money and do separate. A
leaderboard carrying only the success column would call these two equal.

The baseline is missing from that table because its record predates the stage
instrumentation, so every rung reads as not comparable against it rather than as
a zero. That is the machinery admitting what it did not measure, and it is a
known gap: it needs one re-run.

## What a clone can actually run

Uploading any of them gets you through the first two gates on a laptop:

- **Intake** reads the archive and reports what is in it. Seconds, free, CPU.
- **Data report** runs every installed feedback check over the clips. About a
  minute, free, CPU.

All three pass both, with one finding at strong severity on each: every clip
carries the same instruction, which is true of the footage and is the report
doing its job rather than a defect in the sample.

The two gates after that need hardware. The signal check fits a probe and wants
a GPU to be quick. The robot test needs a simulator and a checkpoint, and only
trains when `BENCH_RUNNER` is set. Left unset it reads rollouts already on disk,
which is the right default for anything shared, since a stray click should not
start a run that costs a day of GPU.

## Samples that need no download

`src/gantry/fixtures/` generates episodes with planted defects: 24 in a suite,
seven defect kinds, plus decoys that must **not** fire. No GPU, no simulator, no
checkpoint, and it is what the test suite is judged against.

    python -c "from gantry.fixtures import make_suite; print(len(make_suite().episodes))"

Use these to work on a detector. Use the zips above to see the product.

## Not included

The raw ego dataset before it was mixed in, `/home/ubuntu/egorun/ego_lerobot` on
the box: 18 clips, 10 fps, 1080x1920, in a camera-mounted frame. It is 99 MB at
full resolution and it is not a benchmark submission, it is an ingredient. The
earlier run that evaluated ego on its own scored 0/10 against a shuffled control
that also scored 0/10, and refused to name a winner: see
`experiments/robotwin_ego/RESULTS.md`. Adding a base arm that can actually do
the task is what turned that refusal into an answerable question, and is why
these three exist in this shape.

## Why the full datasets are committed rather than trimmed

Twenty-three megabytes is a real cost and it is permanent in the history. A
trimmed sample would be smaller, and it would also no longer be the data that
produced the tables above. A sample whose results cannot be reproduced from it
demonstrates the pipeline rather than the finding, and the finding is the part
worth showing.
