# The MG checkpoint was trained on 52 failed demonstrations and no successful ones

Found by the lineage machinery on its first real use, 30 July 2026.

## What was measured

`lift_train_mg` — the LeRobot dataset the `ft_mg` checkpoint was fine-tuned on —
was reconnected to `robomimic_data/lift/mg/low_dim.hdf5` by content: every
episode's actuation channel hashed and matched exactly. All 52 linked, none
ambiguous.

Then each linked episode's success label was read from the robomimic source,
where the labels live:

```
robomimic mg:       244/1500 succeed (16%)

lift_train_mg:  52 episodes,  52 linked,   0/52  succeeded   (0%)
lift_train_ph: 180 episodes, 180 linked, 180/180 succeeded (100%)
lift_train_mh:  75 episodes,  75 linked,  75/75  succeeded (100%)
```

## What it means

The three-way comparison was never going to measure data quality. `ph` and `mh`
are all-success collections; the `mg` slice is all-failure. That is not a
gradient in demonstration quality — it is one categorically different thing, and
any ranking of the three would have been reported as a finding about data
collection method.

`ft_mg` was trained for 600 steps on demonstrations in which the cube is never
lifted.

## How the curation layer responds

The drop-list built from the robomimic labels, translated into the LeRobot
dataset's own names, then applied:

```
[curation.empties_the_dataset] applying this leaves 0 of 52 episodes
```

Which is the correct answer, and the reason the refusal exists.

## What to do

Building an `mg` training set from the 244 successful demonstrations gives a
comparison that is about collection method rather than about whether the
episodes worked. Until then, `ft_mg` should not appear in a data-quality claim.
