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

Seven planes, each an independent axis of variation. A plugin implements one contract, declares
what it needs and what it provides, and never learns anything about the others.

| Plane | What it is | Shipped |
|---|---|---|
| `dataset` | where episodes come from | `lerobot` · `robomimic` · `csv` · `evallog` |
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
  contracts/           the five plane interfaces
  conformance/         one kit per contract, returning verdicts, importing no test framework
  resolve/             binding, adapters, retargeters, requirements
  isolate.py           subprocess boundary for conflicting dependencies
plugins/             20 plugins, each installable on its own
docs/                writing-a-plugin.md
manifests/           a run described as a file, so it can be committed and diffed
reference/           earlier implementations, kept to check this one against
ARCHITECTURE.md      the whole design, contracts first
```

## Testing

```bash
python -m pytest tests --ignore=tests/integration   # core alone, no plugins
python -m pytest                                   # 285, plugins installed
python -m pytest plugins/*/tests                   # 564 across 20 plugins
python tools/isolation_check.py       # each plugin alone, in a fresh venv
```

The last one is the modularity claim made mechanical. A monorepo's shared environment quietly
satisfies every import whether it was declared or not, so a plugin can lean on a sibling it
never mentions and nobody finds out until someone installs it alone. This builds a fresh
interpreter per plugin and installs only core, that plugin, and what it actually declares:

```
20/20 plugins install and pass on their own
```

CI runs core on four Python versions with no plugin present, every plugin in isolation, and
everything together.

## Status

Working: the seven planes, 20 plugins, 849 tests. Verified against real LeRobot and
RoboMimic data; against a GR00T N1.7 checkpoint fine-tuned on this data and served over its
real protocol; and in closed loop against MuJoCo, where replaying recorded actions into the
rebuilt world lifts the cube 3/3, which is the check that proves the wiring rather than
asserting it.

Thin: evaluations so far are tens of trials, not hundreds. A fine-tune that freezes the
diffusion head proves the pipeline and is not a capability result.

## License

Apache-2.0.
