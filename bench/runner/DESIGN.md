# The robot test, as it should be built

What `bench/runner` does today, why it cannot answer the question it is asked,
and the shape that can.

Nothing here is implemented yet. It is written down first because the change
touches what a verdict *means*, and that is worth agreeing before it is worth
coding.

## What happens now

```python
source = Path(job["dataset"])                          # the upload, and only that
build_arms(source, real_dir, control_dir, progress)    # copy twice, derange one
train(real_repo); train(control_repo)                  # two arms, two policies
```

Two arms are trained on the contributor's clips: one with the actions where they
belong, one with each clip's actions donated to a different clip. Both are then
evaluated on the same scenes. Whatever fine-tuning-in-general buys, the control
buys too, so beating the control is evidence the *correspondence* carried
information. That part is right and stays.

What is missing is that **nothing else is in the training set**.

## Why that cannot work for egocentric video

A policy that has only ever seen a person's hands in a kitchen has no idea how
to drive an aloha-agilex. Both arms score zero, the paired test has nothing to
separate, and the run costs hours of GPU to report that it learned nothing.

This is not a prediction. It is `experiments/robotwin_ego/RESULTS.md`:

```
ego        0 / 10     success rate 0 [0, 0.278]
shuffled   0 / 10     success rate 0 [0, 0.278]
expert    10 / 12     the ceiling on these same scenes (83%)
```

`control.not_separated`, and a refusal to name a winner. The experiment learned
the lesson and wrote it into `build_arms.py`:

> The first run of this experiment had only the last two, and both scored zero,
> because a policy trained on kitchen video has no way to pick up a bottle in a
> simulator it has never seen. Zero minus zero is not a measurement. `base` is
> the arm that can actually do the task.

The product never implemented it. The two samples that pass today only pass
because `build_ab.py` mixed the benchmark's demonstrations into them weeks ago,
outside the product, by hand. A contributor uploading their own footage gets
none of that.

## The shape

Two phases per arm. The contributor's data is pretraining; the benchmark's own
demonstrations are a fixed finetune that every arm receives identically.

```
treatment    your clips, real pairing      ->  the benchmark's demonstrations
control      your clips, actions deranged  ->  the benchmark's demonstrations
                    (pretrain)                          (finetune)
```

Then both are evaluated closed-loop on the same scenes, as now.

### Why sequential rather than mixed

Mixing the two into one training set was the obvious alternative and it fails on
volume. One contributor uploads 100 clips, another 1,000, and the benchmark's
100 demonstrations are 50% of one training set and 9% of the other. At 9% the
task competence the base exists to provide can be lost entirely: both arms floor
at zero and the comparison, while still perfectly fair, has no power. The
contributor with more data scores worse on identical-quality footage, and the
verdict reports that as a fact about their data.

Sequential removes it. The finetune is the same demonstrations, the same steps,
the same schedule, for every submission, so every arm arrives at the evaluation
with task competence restored from a common footing. How much ego came before
changes what the policy brings to that finetune; it no longer changes whether
the policy can do the task at all.

### Why a null result still means one thing

The obvious objection to a finetune phase is catastrophic forgetting: the
second phase can wash out whatever the first taught, and then a null says
nothing about the data.

The shuffled control answers it. Both arms receive the **identical** finetune,
so forgetting applies to both equally. If they converge, the finding is:

> At this finetune budget, the correspondence in your footage left no trace.

That is a statement about this pipeline at this budget, it is true, and it is
actionable: a shorter finetune, or more pretraining, are things a contributor
or an operator can change. It is not the ambiguity it looks like from outside,
because the control absorbs the forgetting.

### Fix the steps, not the epochs

The one way volume creeps back in. Train 1,000 clips and 100 clips for the same
number of *epochs* and the larger upload receives ten times the gradient steps:
the run then measures compute, not data.

So the pretraining budget is a step count, fixed across submissions. More clips
then means more diversity at the same compute, which is the real advantage of
having more data, and it shows up as an effect rather than as a bigger budget.

This also retires the idea of capping an upload and subsampling it. Nothing
needs to be thrown away, and no sampler has to be defended.

## The job spec

Today:

```json
{"dataset": "<lerobot dir>", "trials": 20, "task": "pick_dual_bottles",
 "arms": ["your data", "shuffled control"], "baseline": "baseline",
 "out": "<dir>", "progress": "<file.jsonl>"}
```

Intended:

```json
{"dataset": "<lerobot dir>",
 "finetune": {"dataset": "<the benchmark's demonstrations>", "steps": 1500},
 "pretrain": {"steps": 3000},
 "trials": 50, "task": "pick_dual_bottles",
 "arms": ["your data", "shuffled control"], "baseline": "baseline",
 "out": "<dir>", "progress": "<file.jsonl>"}
```

The gate still says *what* it needs and knows nothing about trainers or
checkpoints. `finetune.dataset` is named by the gate because it is a property of
the benchmark, not a choice the runner makes.

`build_arms` is unchanged: it still copies the upload twice and deranges one
copy, and it still cuts both arms to identical lengths so the control is not
also shorter. Only the training sequence around it changes.

## What the verdict has to say

Every one of these changes the answer, so none of them may be implicit:

- **the pretraining step budget**, and that it is fixed rather than per-epoch
- **the finetune step budget**, and which demonstration set it used
- **how many clips the contributor supplied**, since at a fixed step budget that
  is a statement about diversity
- **that the finetune was identical across both arms**, which is what makes a
  null readable

A verdict that omits the budgets is not reproducible, and two submissions run at
different budgets are not comparable however similar the numbers look.

## What this still does not answer

Worth stating so nobody reads more into a passing result than it carries.

**Whether ego data can replace teleoperation.** This design asks whether your
footage helps *on top of* a fixed set of demonstrations. The commercially
interesting question is whether it lets you collect fewer of them, and that is a
different run: hold the ego side constant and vary the finetune set downward.
The design here is what that experiment would be built from.

**Anything about a real robot.** The evaluation is a simulator, and the frames
of the ego corpus and the simulator's world do not align. `RESULTS.md` is blunt
about it: absolute positions are expected to be largely unreachable, and the
honest reading of a low number is "these workspaces do not overlap", not "the
data was bad".

**Whether the retargeting is right.** Hand span is a stand-in, `measured_by`
says so on every episode, and a wrong span is a smoothly wrong dataset rather
than a broken one.

## Cost

Two phases per arm rather than one, but the total step count can be held where
it is, so GPU is roughly unchanged: the pretraining budget comes out of what is
currently spent training on the upload alone.

The measured evaluation cost is unchanged at about 95 seconds per scene per arm,
so 50 scenes across two arms remains around 2.6 hours, and the earlier sizing
table still applies.
