# Samples

Two real datasets, so a fresh clone has something to upload. Without these,
nothing in the repository can exercise the product's own path: `bench/data/` is
where uploads land and it is gitignored, so a clone arrives with an empty bench
and no way to fill it.

| file | what it is | size |
|---|---|---|
| `two_handed_58clips.zip` | both arms working together | 9.4 MB |
| `one_handed_58clips.zip` | one arm doing the work | 9.6 MB |

Both are RoboTwin `pick_dual_bottles` on an `aloha-agilex`, 58 clips each, 25
fps, 16-wide state and action, head camera as mp4, in LeRobot layout. They are
byte-identical to what the live deployment holds:

    1a89c900fb4d2648d15e0d951fd850c11eb0cd2fe49466f5b4bc6408803619c1  two_handed_58clips.zip
    238d60037ef61a7ce60003d9c10f82458e9b28d6d1d167b6642386e1d979ffeb  one_handed_58clips.zip

## Upload both, not one

One upload only shows that the pipeline runs. The pair is the point.

These two datasets are indistinguishable through Intake and through the data
report. They stay indistinguishable through the signal check. They separate only
at the top of the ladder, on the rungs that need both hands at once, which is
the capability the one-handed footage never demonstrates:

    rung             two-handed      one-handed     paired test
    moved            50/50  100%     50/50 100%     p=1.0
    lifted           50/50  100%     49/50  98%     p=1.0
    moved all 2      44/50   88%     34/50  68%     p=0.002    separated
    lifted all 2     41/50   82%     31/50  62%     p=0.0063   separated
    solved            4/50    8%      0/50   0%     p=0.125    not separated

Read the bottom row before the others. Binary success, the number everybody
reports, **cannot tell these two apart**: 4 against 0 is four disagreements, all
of them favouring the two-handed data, which is the most extreme outcome
available at that count, and it still only reaches p=0.125. The rungs above it
are measured on the same rollouts for no extra money and do separate. A
leaderboard carrying only the success column would call these datasets equal.

## What a clone can actually run

Uploading either one gets you through the first two gates on a laptop:

- **Intake** reads the archive and reports what is in it. Seconds, free, CPU.
- **Data report** runs every installed feedback check over the clips. About a
  minute, free, CPU.

Both samples pass both gates, with one finding at strong severity on each:
every clip carries the same instruction, which is true of the footage and is
the report doing its job rather than a defect in the sample.

The two gates after that need hardware. The signal check fits a probe and needs
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

## Why the full datasets are committed rather than trimmed

Nineteen megabytes is a real cost and it is permanent in the history. A trimmed
sample would be smaller, and it would also no longer be the data that produced
the table above. A sample whose results cannot be reproduced from it is a
demonstration of the pipeline rather than of the finding, and the finding is the
part worth showing.
