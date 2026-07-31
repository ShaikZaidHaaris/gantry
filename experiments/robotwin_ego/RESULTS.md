# pick_dual_bottles, 10 screened scenes, 2026-07-31

    ego        0 / 10        success rate 0 [0, 0.278]
    shuffled   0 / 10        success rate 0 [0, 0.278]
    expert    10 / 12        the ceiling on these same scenes (83%)

`feedback_control` returned `control.not_separated` and refused to name a
winner. `feedback_power` says why: ten trials at 0% can only separate an effect
of about **+0.275**; a +0.100 claim would need **28**. Reporting the larger of
two zeros would have been noise with a decimal point on it.

## The chain ran

Everything the run was built to exercise worked against the real simulator:

- The expert screened 10 solvable seeds out of 12 tried, rejecting 5 and 10.
- `rotation@0.1.0.dev0` was in the action chain, converting the checkpoint's
  14-wide `euler_xyz` to RoboTwin's 16-wide `quat_wxyz` every step, and the
  16-wide state back to 14 every step.
- Each scene carried the sentence RoboTwin generated for the bottles the
  randomiser actually placed, and both arms were given the identical wording.
- Episodes ran to RoboTwin's own 400-step budget for this task.

So the zero is not a plumbing failure. The policy was asked, in the right space,
in the right words, on solvable arrangements, and did not solve them.

## Why the zero says little about the data

The policy is being evaluated far outside the distribution it was trained on,
and the state it receives is the clearest measure of that:

                        training (ego)          RoboTwin gives it
    left  xyz      [-0.006, 0.107, 0.449]   [-0.001, -0.164, 0.927]
    right xyz      [ 0.103, 0.094, 0.543]   [ 0.137, -0.050, 0.911]

The height is off by about 0.45 m and the sign of *y* is flipped. On top of that
the camera is a tabletop head view rather than egocentric kitchen video. The
outputs are unanchored to match: the ego model commands its right arm to
x = -0.486 where training had +0.104.

This does **not** bias the comparison — both arms are identically
disadvantaged, so the contrast remains fair. It removes its power. More trials
would mostly buy more zeros.

## The one asymmetry worth noting

The control's commands are *more* reachable than the treatment's:

                    within reach, left / right
    ego                  34.6%  /   4.0%
    shuffled             88.9%  /  71.4%

That is the expected signature of a model trained on detached labels: with no
correspondence between frames and actions to learn, it regresses toward the
mean of the action distribution and emits small, central motions. The ego model
emits larger, more structured ones. That is evidence it learned *something*
other than the marginal — it is not evidence that what it learned helps here.

## Frame offsets

    ego        left  0.302 m  [ 0.121,  0.238, -0.139]
               right 0.649 m  [-0.623,  0.148,  0.107]
    shuffled   left  0.458 m  [ 0.027,  0.189, -0.416]
               right 0.276 m  [-0.027,  0.143, -0.235]

Measured from where each arm actually works, not from the world origin.

A left/right arm swap was hypothesised from the inverted x-separation and
**tested against the recorded commands: it does not hold.** Swapping improves
the right arm (0.649 -> 0.177 m) and worsens the left (0.302 -> 0.558 m), with
total offset falling only from 0.95 to 0.74 m. The accurate statement is
narrower: the left block lands roughly where it should and the right block is
displaced about 0.6 m in -x from both arms.

No transform was fitted. Estimating an alignment from these commands and then
scoring under it would tune the correction on the very episodes being scored.

## What would make this measure data quality

In rough order of how much each would buy:

1. **Train on RoboTwin's own frame.** The ego chain currently emits poses in a
   camera-mounted frame; RoboTwin's arms have bases 0.6 m apart in its world
   frame. Until those agree, this measures the gap between them.
2. **Then re-run the pair.** With both policies in-distribution, 28 trials
   would support a +0.100 claim, per the power module.
3. **Meanwhile, use held-out open-loop action error** on the ego test split,
   which measures fit rather than capability but does not require the frames to
   agree — `feedback_control` falls back to it when there are no outcomes.
