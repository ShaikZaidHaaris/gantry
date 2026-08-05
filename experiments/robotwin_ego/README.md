# The ego checkpoint, closed-loop in RoboTwin

The run `feedback_control` had been refusing to write. It would not say whether
ego data helped, on the grounds that no evaluator could execute what the policy
emits — everything installed was single-arm and joint-space, and
`retargeter_hands` refuses to produce joint positions because inverse kinematics
needs link lengths, joint limits and a choice of elbow configuration that a
retargeter does not have.

RoboTwin takes absolute end-effector poses. That is what the retargeter *does*
produce, so the chain runs with no IK step and nothing relaxed to make it fit.

## The two arms

    ego        fine-tuned on the real hand trajectories
    shuffled   the same frames and the same actions, with the actions detached
               from the frames they belong to

A model fine-tuned on `shuffled` has had exactly as much fine-tuning, on exactly
as many frames, with the same action distribution, and no relationship between
what it saw and what it did. Beating it is evidence the ego actions carried
information. Not beating it means the gain was from fine-tuning at all — a
different and much less interesting claim, and one that reproduces for every
contributor including the ones whose data is worthless.

## What gets converted, and where

The checkpoint speaks **14-wide, euler_xyz**; RoboTwin reads **16-wide,
quat_wxyz**. Both directions go through the adapter plane rather than through
arithmetic written here:

    state    RoboTwin  endpose.vector (16, quat)  ->  checkpoint (14, euler)
    action   checkpoint (14, euler)               ->  RoboTwin  (16, quat)

`adapt_policy` plans both from whatever adapters are installed and refuses in
the constructor if nothing closes the gap — before a simulator is built and
before a checkpoint is loaded.

## Seeds are screened first

RoboTwin randomises object placement per seed and not every arrangement is
solvable. Scoring a policy on one that is not charges it for the sampler, and
because the unsolvable fraction moves with the seed range, two runs over
different ranges are not comparable even on the same task. The screen runs
RoboTwin's own scripted expert and keeps the seeds it solved; the expert's rate
is the ceiling and is recorded on every episode.

The screen is also where the language goal comes from. RoboTwin fills the
sentence's placeholders with what the randomiser actually placed, and those
parameters only exist once the expert has run.

## What this run cannot tell you

The ego trajectories were recovered in a camera-mounted frame and scaled by a
hand span. RoboTwin's arms live in its own world frame with their own reach.
Nothing here aligns the two, because nothing legitimately can — a fitted
transform between them would be tuned on the thing being measured.

So the absolute positions are expected to be largely unreachable, and the honest
reading of a low number is "these workspaces do not overlap", not "the data was
bad". `run_robotwin.py` measures the per-arm reachable fraction and reports it
next to the success rate so the two cannot be confused.

## Running it

    bash install.sh          # once, on a box with the simulator's dependencies
    bash paired.sh pick_dual_bottles 10

`paired.sh` serves each checkpoint in turn — a π₀.₅ LoRA server holds ~22.5 GB
and the L40S has 46, so two at once would leave the renderer nothing — then runs
`compare.py`, which hands both records to `feedback_control`. The records go in,
not the summary numbers: whether a difference between two success rates means
anything depends on how many trials there were, and a module shown only
"0.4 and 0.2" cannot tell 2-of-5 from 200-of-500.
