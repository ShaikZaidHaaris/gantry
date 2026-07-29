# Gantry

**A measurement layer for robot learning.** Datasets, embodiments, policies, evaluators and
feedback are independent planes — swap any one without touching the other four.

A gantry is the fixed frame that lets arbitrary payloads move over a workspace. The frame is
what this builds. The payloads are everyone else's.

```bash
pip install -e .                      # core: numpy, nothing else
pip install -e plugins/connector_lerobot -e plugins/feedback_core
python -m gantry.cli list
```

---

## The problem

Teams spend their budget collecting demonstrations and training on them, and then cannot say
which data was worth collecting or why a policy failed. The usual answer — collect more — is
the most expensive possible guess.

Gantry measures the things that are cheap to measure and easy to skip: whether a dataset is
any good before you train on it, where in a task a policy actually breaks, and how much of a
result came from the execution protocol rather than the model.

## What it found

Pointed at three public collections of the same task, with **no training at all**:

| | PH | MH | MG |
|---|---|---|---|
| path efficiency | **0.583** | 0.490 | 0.134 |
| direction change rate | **0.537** | 0.593 | 2.331 |
| actuation jerk | 0.041 | **0.032** | 0.724 |
| episode length | 48 | 56.5 | 150 |

That reproduces the published quality ordering straight from the raw recordings. Through the
RoboMimic reader, the cause of the MG collapse takes seconds to find: **244 of 1500
demonstrations succeed.** Sixteen percent.

---

## The idea

Nine planes, each an independent axis of variation. A plugin implements one contract, declares
what it needs and what it provides, and never learns anything about the others.

| Plane | What it is | Shipped |
|---|---|---|
| `dataset` | where episodes come from | `lerobot` · `robomimic` · `csv` · `evallog` |
| `task` | what is being attempted, and how anyone decides it was done | `declared` |
| `curation` | what to do to the data, said precisely enough to be wrong | `labels` · `collect` |
| `policy` | anything that turns observations into actions | `gr00t` · `served` · `replay` · `constant` · `noisy_replay` |
| `evaluation` | running a policy and recording what happened | `robosuite` · `libero` · `gym` · `offline` · `waypoint` |
| `feedback` | records in, findings out | `screen` · `compare` · `funnel` · `attribution` · `harden` · `protocol` |
| `embodiment` | what a robot is, physically | `declared` |
| `adapter` | one correct answer: units, rates, rotations | `unit_convert` · `resample` · `rotation` · `permute` |
| `retargeter` | declared-judgement conversions between bodies | `pose_to_position` |

Planes are themselves a registry: a plugin can add one without editing core.

**Any plane can be the axis.** A manifest names one component per plane and says
which one varies. Three datasets under one policy and three policies in one world
are the same shape of question, so `varies` is a field rather than an assumption:

```json
{ "varies": "policy",
  "cohorts": { "ph": {"name": "gr00t", …}, "mg": {"name": "gr00t", …} },
  "dataset": {"name": "robomimic", …}, "evaluation": {"name": "robosuite"} }
```

Whatever varies is the one thing not held. A feedback module declares the mirror
image with `holds`, and the two are checked against each other from provenance —
so a comparison that claims to measure the data, while the policy also changed,
is refused rather than reported.

## What makes it different

**It describes rather than normalises.** A connector reports what a format actually declares
and nothing more. LeRobot names its columns whatever the author liked, so this reader declares
no units and no meanings — a plausible guess is how a millimetre reads as a metre.

**It refuses rather than coerces.** Every compatibility answer is a `Verdict` carrying stable
dotted codes, not a bool. `units.scale`, `rate.mismatch`, `metadata.mismatch` — a resolver that
says "incompatible" is useless; one that says "30 Hz vs 20 Hz, needs a resampler" is a
diagnosis. Refusals are a product surface here, and most of the test suite is about them.

**No naked floats.** Every number carries its n, its interval and its method, and provenance
records what produced it. Two runs are comparable exactly when their provenance matches on the
axes being compared, which is what stops a simulated arm and a real humanoid being averaged
into one meaningless table.

**Statistics that survive small n.** Wilson and Newcombe intervals, McNemar for paired
outcomes, Cliff's delta, tie-corrected Mann-Whitney, Benjamini-Hochberg, and robust bounds that
fall back to IQR rather than a standard deviation.

**Isolation is first-class.** A plugin whose dependencies conflict with the host declares so
and runs in a subprocess speaking JSON lines. That is how a model with a hard CUDA pin gets
evaluated by a process that installed none of it.

---

## A worked example

Two runs of one policy at two execution settings, ranked and tested:

```
protocol (execute-1, execute-8)
  execute-1.success_rate: 0.117 fraction [0.058, 0.222] (n=60)
  execute-8.success_rate: 0.017 fraction [0.003, 0.089] (n=60)
  [strong] execute-1 is the best setting (execute=1) at 11.7%, +10.0% over execute-8;
           paired on 60 shared scenes, won 6 lost 0, p=0.0312
    -> Run at execute=1. It costs nothing: same policy, same data, same task.
```

Nothing was retrained. How much of a predicted action chunk gets executed before re-planning
is one of the largest levers on a measured result and the one most often left implicit, so it
lives in the record with everything else that changes an answer.

## Tasks are files, not code

A task says what is being attempted, where things start, and what counts as done —
naming no robot and no simulator:

```json
{ "name": "lift_cube",
  "instruction": "lift the cube off the table",
  "objects": [{"id": "cube", "kind": "cube_20mm",
               "start": {"surface": "table", "x": [-0.03, 0.03], "y": [-0.03, 0.03]}}],
  "success": [{"check": "lifted", "args": {"object": "cube", "height": 0.04},
               "rubric": "The cube is clear of the table by at least 4 cm and held in
                          the gripper. A cube nudged off the edge does not count."}],
  "staging": {"robosuite": {"env_name": "Lift", "places": {"cube": "cube"}}} }
```

**Every success criterion is written twice** — once machine-checkable, once as a rubric
precise enough that two people watching a video agree. A criterion with no rubric is
refused. In simulation success is a free, exact pose check; on a real bench it is a person
watching, and the rubric is the part that survives the move. The 14 tasks in
`manifests/tasks/` are all scorable by hand today, with no simulator.

**The rectangle in the file is the rectangle in the simulator.** A `start` region becomes
robosuite's placement sampler, so the numbers you would tape onto a real table are the ones
that ran:

```
lift_cube        file says x=[-0.03, 0.03]   sim gives x=[-0.030, +0.030]
lift_cube_wide   file says x=[-0.10, 0.10]   sim gives x=[-0.092, +0.096]
```

Same file, six bodies — Panda, Sawyer, IIWA, Kinova3, Jaco, UR5e — each rebuilt around its
own measured description, none of them normalised to a shared one.

And where a world *cannot* be driven from a file, it says so instead of pretending. Some
robosuite environments build their own layout and discard the one they are handed; that is
detected by checking the sampler survived, not by a list of environment names, so an
environment added later is judged on what it does:

```
pick_place_can   REFUSED: PickPlaceCan exposes no surface origin …
hang_tool        REFUSED: ToolHang takes no placement at all …
```

## Curation: telling you what to do to the data

The other eight planes describe or measure. This one **prescribes** — and every
prescription is an object that can turn out to be wrong:

```
[labels@screening] drop 1256 -> success_rate +0.042
```

That plan says which episodes, from which signal, at what claim strength, and
**what it predicts**. Then it gets tested rather than believed:

```
20 trials  -> refused: separating +0.042 from a 35% baseline needs about 66 paired trials
200 trials -> proceed

curation.verified  observed +0.400 over 20 shared scenes (won 8, lost 0, p=0.0078)
```

Prescriptions also cover data that does not exist yet. A failed rollout is a
seed, a seed re-stages the scene exactly, and the task's region is the same
rectangle on a simulated table or a real one — so "collect more data" becomes a
work order:

```
collect 5 x lift_cube, from 5 failed scene(s)
  seeds: (3000, 3001, 3002, 3003, 3004)
  note : failed scenes sit toward the low end of x (median +0.068, over +0.060..+0.076)
```

**Proposing and judging are different planes on purpose.** Every curation method
in the literature reports its own wins, because evaluation is normally too
expensive to run twice. Here a curator proposes and the feedback plane judges,
with three gates the published methods do not have:

- **Leakage** — a signal that read rollouts to choose what to drop cannot be
  scored on those same scenes. It declares the seeds it consumed; the overlap is
  a refusal.
- **Power** — a plan predicting +3pp verified on 20 trials is refused *before*
  the retrain, not discovered after it.
- **Selection** — the tenth plan from one signal faces a corrected threshold. The
  same result that passes at p=0.031 on its first try is refuted as the best of
  forty.

Verdicts are kept either way, in a ledger addressed by content. After enough of
them it answers a question nobody has published: **which curation signals
actually work, on which tasks.**

```
  labels    2/3 held, median +0.120 when it did   [132 gpu-min]
  deminf    0/1 held

  lift_cube            held: labels    refuted: deminf
  assemble_square_nut  held: -         refuted: labels
```

## GR00T N1.7

`plugins/policy_gr00t` speaks GR00T's inference protocol — ZeroMQ, msgpack — and **never
imports GR00T**. The model keeps its torch, its CUDA build and its pinned numpy; the process
measuring it installs `pyzmq` and `msgpack`. Start the server the way its own docs say:

```bash
python gr00t/eval/run_gr00t_server.py --model-path <ckpt> --embodiment-tag <tag>
```

```python
from gantry_policy_gr00t import Gr00tPolicy, Endpoint

policy = Gr00tPolicy("/path/to/lerobot/dataset", Endpoint(port=5555))
```

What the model reads is asked of the server. How wide each field is comes from the dataset's
`meta/modality.json`. The two are checked against each other before an episode is opened, so a
checkpoint pointed at the wrong dataset refuses in a second rather than producing a night of
plausible numbers.

---

## Layout

```
src/gantry/          core — depends on numpy and nothing else
  spine/               what an episode, a run, a channel and a verdict are
  contracts/           the nine plane interfaces
  conformance/         one kit per contract, returning verdicts, importing no test framework
  resolve/             binding, adapters, retargeters, requirements
  isolate.py           subprocess boundary for conflicting dependencies
plugins/             25 plugins, each installable on its own
docs/                writing-a-plugin.md
manifests/           a run described as a file, so it can be committed and diffed
reference/           earlier implementations, kept to check this one against
ARCHITECTURE.md      the whole design, contracts first
```

## Testing

```bash
python -m pytest tests --ignore=tests/integration   # core alone, no plugins
python -m pytest                                   # 285, plugins installed
python -m pytest plugins/*/tests                   # 673 across 25 plugins
python tools/isolation_check.py       # each plugin alone, in a fresh venv
```

The last one is the modularity claim made mechanical. A monorepo's shared environment quietly
satisfies every import whether it was declared or not, so a plugin can lean on a sibling it
never mentions and nobody finds out until someone installs it alone. This builds a fresh
interpreter per plugin and installs only core, that plugin, and what it actually declares:

```
25/25 plugins install and pass on their own
```

CI runs core on four Python versions with no plugin present, every plugin in isolation, and
everything together.

## Status

Working: the nine planes, 25 plugins, 960 tests. Verified against real LeRobot and
RoboMimic data; against a GR00T N1.7 checkpoint fine-tuned on this data and served over its
real protocol; and in closed loop against MuJoCo, where replaying recorded actions into the
rebuilt world lifts the cube 3/3, which is the check that proves the wiring rather than
asserting it.

Thin: evaluations so far are tens of trials, not hundreds. A fine-tune that freezes the
diffusion head proves the pipeline and is not a capability result.

## License

Apache-2.0.
