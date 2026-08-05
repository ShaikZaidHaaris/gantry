# The matrix with a baseline column

Four checkpoints × 6 single-arm bodies × 13 tasks = 312 cells that ran. One trial
at an eight-step horizon — reachability, not measurement. Every cell says so.

```
392 cells: 312 ran, 80 refused, 0 failed, 0 in-trial errors    (4.8 min)
```

## The baseline

`step0` is the pretrained GR00T-N1.7-3B with a `new_embodiment` projector after
**one gradient step** on `lift_train_ph`, otherwise identical to the fine-tune
protocol used for ft_ph/mh/mg — same LR (1e-4), batch (2), grad accum (8), seed
(42), projector-only. That checkpoint stands in for "an untrained projector on
the pretrained backbone."

## Why this baseline and not "no fine-tune at all"

The pretrained model does not ship a projector for the lift data's shape.
`libero_sim` is a *posttrain* tag — the id exists so a fine-tune can be pointed
at it, but the pretrained checkpoint carries no trained weights for it. Every
projector the base model actually has (`oxe_droid_relative_eef_relative_joint`,
`real_g1`, `real_r1_pro_sharpa` variants, `xdof`) is a different state shape
than the 7-wide lift observation, so the base model cannot consume lift data
directly.

The step-0 baseline is the closest legitimate thing: the same pretrained
backbone, a projector with essentially no gradient information, run through the
same server on the same wire as the three fine-tuned checkpoints. What every
non-Panda number then means is precisely what a lab asks about a fine-tune:

    "our fine-tune adds N points over an untrained projector with matched
     backbone, matched protocol, matched scenes."

Rather than the weaker naive framing:

    "our fine-tune scores X on this arm."

## What this smoke does and does not show

Every cell was one trial at 8 steps, so success rates here are essentially
random. All 312 running cells classify, no in-trial errors, no crashes when the
sweep swaps between checkpoints — that is the point of the smoke. The number
that matters (baseline delta per cell) requires 20+ trials at a real horizon.

To turn this into measurement:

    CKPTS=step0,ph,mh,mg ARMS=panda,sawyer,iiwa,kinova3,jaco,ur5e \
    TRIALS=20 EXECUTE=16 MUJOCO_GL=egl python sweep.py

Estimated cost: 4 × 6 × 13 × 20 trials at ~4 s/trial = about 7 hours, resumable.

## Files

    train_step0.sh    the checkpoint's training command (exit 0)
    sweep.py          the harness with a baseline column
    *.json            312 ran + 80 refused
