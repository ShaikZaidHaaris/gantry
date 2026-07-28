# Gantry — architecture

A framework where datasets, embodiments, policies, evaluation, and feedback are
five independent planes. Any one of them can be swapped without touching the
other four. This document is the whole design: contracts first, no code.

The name: a gantry is the fixed frame that lets arbitrary payloads move over a
workspace. The frame is what we build. The payloads are everyone else's.

---

## 0. Feasibility, stated up front

Graded, not hand-waved:

| Goal | Feasibility | Why |
|---|---|---|
| Swap datasets (LeRobot, HDF5, RLDS, rosbag, CSV…) | **High** | Format plumbing is a solved-shape problem. Connectors + one canonical view. |
| Swap policies (GR00T, Octo, OpenVLA, anything) | **High** | Already proven in this repo: `model_backends/` runs three foreign policies behind one descriptor + serve contract. We generalize that pattern. |
| Swap sim evaluators (robosuite, LIBERO, Isaac…) | **High** | One interface, one adapter per sim. Dependency conflicts are the real fight, solved by process isolation. |
| Swap feedback modules | **High** | Feedback already consumes only records (proven by `metrics.py` → `feedback_v2.py` → `harden.py` on the L4). It never needs to know what produced them. |
| Swap embodiments | **Medium** | Describing an embodiment is easy. *Retargeting between* embodiments is lossy and semantic. Feasible as explicit, declared-loss adapters; infeasible as a silent universal space. |
| Real-world evaluation as a drop-in | **Medium** | The interface slots in cleanly. The hard part is operational (resets, safety, small n, humans labeling stages), not architectural. Design for it now, staff it later. |
| "Fully future proof, craziest embodiments" | **Asymptotic** | No schema survives contact with a soft robot or a swarm. What survives is a *descriptor + adapter + conformance-test* discipline, so new weirdness costs one adapter, not a rewrite. That is the strongest honest version of future-proof, and it's what this design buys. |

Overall: the frame is weeks of work for a first working spine (two connectors,
one sim evaluator, one policy backend, the existing feedback math). The long
tail is adapters, forever. That's not a flaw — adapters *are* the product.

---

## 1. The one design principle

> **Describe, don't normalize.**

Every failure mode we hit this month was silent coercion: an absolute asset
path baked into stored XML, `objects[0]` being Milk instead of Can, a
demo-name collision across sources, a metric confounded by episode duration.
Each one was data pretending to be compatible when it wasn't.

So the rule: no component ever guesses. Every artifact carries a
machine-readable **descriptor** of what it is, needs, and provides. A
**resolver** checks descriptor pairs before anything runs, and either finds an
explicit adapter chain or **refuses loudly with the reason**. Incompatibility
is a first-class, well-reported outcome — never a runtime surprise.

Corollaries:

- Channels are semantic, not positional. There is no "column 7". There is
  `{name: eef_pos, kind: cartesian_position, frame: base, units: m, rate: 20Hz}`.
- Every number carries provenance. No naked floats: a success rate knows what
  policy, embodiment, task, seeds, and evaluator produced it.
- Lossy adapters declare their loss ("drops wrist DoF", "resamples 30→20 Hz").
  The loss travels with the provenance.

---

## 2. The two spines

Everything hangs off two record types. They are the *only* things all five
planes agree on, which is exactly why the planes can be independent.

### Spine A — EpisodeRecord (data at rest)

What a demonstration *is*, storage-agnostic:

```
EpisodeRecord
├─ meta: {id, source, embodiment_ref, task_ref, collected_by, license}
├─ schema: [ChannelSpec]           # name, kind, dtype, shape, units, frame, rate
├─ steps: streamable table         # the channels, lazily read
├─ media: [MediaRef]               # video/depth streams, referenced not copied
└─ labels: {success?, stage_events?, annotations?}   # all optional, all declared
```

Key decisions:
- **View, don't convert.** Connectors expose foreign formats *as* EpisodeRecords
  lazily. No mass conversion step, no second copy of a 2 TB dataset. An
  optional materializer caches hot views.
- `stage_events` (reached, grasped, lifted, …) are optional and *declared*.
  The feedback plane's funnel needs them; the resolver knows which datasets
  and evaluators can supply them and which can't.

### Spine B — RunRecord (data in motion)

What an evaluation *produced*:

```
RunRecord
├─ provenance: {policy@ver, checkpoint_hash, embodiment@ver, task@ver,
│               evaluator@ver, protocol: {n, seeds, horizon, chunking},
│               adapters_used: [with declared losses], timestamp, host}
├─ episodes: [EpisodeRecord]       # the rollouts, same spine as data at rest
├─ metrics: per-episode + aggregate, each value provenance-stamped
└─ events: stage_events per episode, if the evaluator can emit them
```

A rollout is just an EpisodeRecord that a policy generated instead of a human.
This single decision is what lets the feedback plane treat "screen a dataset"
and "diagnose an eval run" as the same computation — which we already proved
on the L4, where the same 42 metrics ran on both.

---

## 3. The five planes

```
┌────────────┐   ┌─────────────┐   ┌──────────┐   ┌────────────┐   ┌──────────┐
│  DATASETS  │   │ EMBODIMENTS │   │ POLICIES │   │ EVALUATION │   │ FEEDBACK │
│ connectors │   │ descriptors │   │ backends │   │  runners   │   │ modules  │
└─────┬──────┘   └──────┬──────┘   └────┬─────┘   └─────┬──────┘   └────┬─────┘
      │                 │               │               │               │
      ▼                 ▼               ▼               ▼               ▼
   EpisodeRecord ◄── resolver checks descriptors, builds adapter chains ──►
                                RunRecord
```

### Plane 1 — Datasets (connectors)

Interface: `enumerate() → episode ids`, `schema(id)`, `open(id) → EpisodeRecord`.

First-class connectors: LeRobot v2.x, robomimic HDF5, RLDS/TFDS, rosbag2,
flat CSV/parquet. Each connector is a plugin that must pass the **connector
conformance suite** (section 5) before it's trusted.

The connector's only job is faithful *description*. It never retargets, never
resamples, never renames semantically. That's the adapters' job, downstream
and explicit.

### Plane 2 — Embodiments (descriptors + retargeters)

An embodiment is **data, not code**:

```
EmbodimentDescriptor
├─ identity: {name, version}
├─ kinematics: ref to URDF/MJCF, joint names, chains
├─ state_space:  [ChannelSpec]
├─ action_space: [ChannelSpec] + semantics {absolute|delta, gripper: binary|continuous}
├─ control: {rate, expected_latency, chunking_conventions}
└─ sensors: camera rig, mounts, intrinsics refs
```

Between any two space specs there may exist **retargeters** — declared
transforms (joint→EE, absolute→delta, 30 Hz→20 Hz, 7-DoF→6-DoF) each carrying
a machine-readable loss statement. The resolver composes chains and reports
total loss. If no chain exists, the run refuses with "no path from X to Y",
which is a feature: a silent wrong-space rollout is worse than no rollout.

This is how "craziest embodiments" get in: a soft robot or a hand with 21 DoF
writes one descriptor and whatever retargeters make sense. Nothing upstream or
downstream changes.

### Plane 3 — Policies (backends)

Interface, transport-agnostic:

```
init(embodiment_descriptor, task_context) → capability report
act(observation window) → action chunk + info
reset(episode context)
```

Two transports: in-process (dev), and served over socket/HTTP with the backend
in its **own container/venv** (prod). This is lifted directly from this repo's
`model_backends/` pattern — `backend.descriptor.json` + `backend.launch.json`
+ serve — which already runs Octo, OpenVLA, and SpatialVLA behind one
contract. That pattern is proven; Gantry generalizes it rather than inventing
a new one.

The policy's descriptor declares its needs (which channel kinds, rates, image
sizes, which embodiments it claims to support). The resolver matches needs
against what the evaluator provides *before* anything boots.

One deliberate scope cut: **training/fine-tuning is not a plane in v1.** A
trainer is just something that produces a new policy version. Orchestrating
training is a later, separate concern; wiring it in now would triple the
surface area.

### Plane 4 — Evaluation (runners)

Interface: `run(policy_handle, task_spec, protocol) → yields RunRecords`.

Three backend families, one contract:

1. **Sim** — adapters for robosuite, LIBERO, ManiSkill, Isaac. Each in its own
   process/container because their dependency trees conflict (we could not
   even co-install some of these on the L4 — isolation is not optional).
2. **Real** — an operator console implementing the same interface: guided
   resets from the task spec's initial-state distribution, human-labeled
   stage events (buttons), safety interlocks. Emits identical RunRecords.
3. **Offline** — no environment at all: held-out open-loop action error
   against a reference dataset. Cheap, GPU-only, no sim. Declares that it
   cannot emit stage events, so the resolver knows funnel-feedback won't run
   on it.

The **TaskSpec** is its own small artifact: initial-state distribution,
horizon, success detector, and optional **stage detectors**. In sim, stage
detectors are state predicates. In real, they're human inputs or a perception
plugin. Stage detectors are the oxygen for the funnel — see caveat 7.

Protocol config (n, seed policy, pairing, chunk size) lives in the RunRecord.
The n_action_steps sweep on the L4 (26 → 41 % from chunking alone) is the
proof that execution protocol is a *result-changing variable* and must be a
recorded, first-class part of every run, never a hidden default.

### Plane 5 — Feedback (modules)

Modules consume **records only**. They never import a sim, a policy, or a
connector. Each declares requirements the resolver checks:

| Module (already built & validated on the L4) | Requires |
|---|---|
| **Screen** — dataset statistics vs reference/comparative thresholds | EpisodeRecords only |
| **Funnel diagnosis** — conditional stage rates, Wilson CIs, counterfactual uplift | RunRecords **with stage events** |
| **Attribution** — behavior↔outcome discriminative stats (Cliff's δ, BH-FDR, logistic) | episode metrics + outcomes |
| **Hardening** — universal vs data-attributable classification | **≥ 2 datasets'** worth of runs |
| **Prescription** — ranked data-collection directives | output of the above |

The screen runs in three modes, in honesty order:
- **Comparative** (default): rank 2+ candidate sets against each other. No
  absolute thresholds needed — this mode matches what the benchmark actually
  proved.
- **Reference**: fit thresholds from a user-supplied known-good set, screen
  others against them.
- **Absolute**: only demonstration success rate, the one statistic that
  survives a change of task. The Lift-specific constants (path ≥ 0.45,
  jerk ≤ 0.10, …) are *reference-mode fits*, never universal claims.

---

## 4. Cross-cutting machinery

- **Registry & plugins.** Every implementation on every plane registers via
  entry points with its descriptor. `gantry list` shows what's installed and
  what combinations resolve.
- **Contract versioning.** The five interfaces + two spines are semver'd
  independently. A plugin pins the contract versions it implements. Breaking
  a contract means a major bump, and old plugins keep working against the old
  contract until dropped explicitly.
- **Conformance suites** (section 5) — the actual enforcement mechanism.
- **Process isolation as a first-class citizen.** Any plugin may declare
  `isolation: container|venv|in-process`. The runner honors it. This is how
  robosuite, Isaac, and a JAX policy coexist without a dependency war.
- **Run manifests.** A whole experiment is one declarative file: which
  dataset(s), embodiment, policy@checkpoint, evaluator, protocol, feedback
  modules. Re-running the manifest reproduces the experiment. The manifest,
  not the CLI invocation, is the unit of reproducibility.
- **Refusal quality.** Resolver errors are a product surface: "Funnel
  diagnosis unavailable: evaluator `offline-action-error` emits no stage
  events. Runnable alternatives: screen, attribution." That sentence is the
  difference between a framework and a pile of interfaces.

---

## 5. How this stays trustworthy: conformance, not combinatorics

With D datasets × E embodiments × P policies × V evaluators × F feedback
modules, testing combinations is hopeless (5 of each is 3,125 paths). We
never test combos. We test **contracts**:

- Every connector passes the connector suite (schema fidelity, lazy reads,
  unit/frame declarations present, round-trip stability).
- Every policy backend passes the backend suite (capability report honest,
  chunk semantics correct, latency reported).
- Every evaluator passes the evaluator suite (seed determinism where claimed,
  stage events well-formed, protocol echoed into RunRecords).
- Every retargeter passes property tests (invertibility where claimed,
  declared loss actually bounds observed loss on fixtures).

Plus **synthetic fixtures with known defects**: generated trajectory sets
where the ground truth is planted (a known grasp-timing flaw, a known jerk
excess), so the whole feedback stack is CI-testable with no GPU, no sim, no
checkpoint. If the screen can't find the planted defect, the build fails.

If every plugin passes its suite, any resolvable combination is trustworthy
by construction. That is the only testing strategy that scales with a plugin
ecosystem, and it's how the "craziest" future plugin gets held to the same
bar as the first-party ones.

---

## 6. Problems and caveats — the honest list

1. **Universal abstraction is a mirage; adapters are the product.** Anyone
   promising one schema for all embodiments is selling something. The
   realistic maximum is: describing a new thing costs one descriptor, and
   connecting it costs one adapter with declared loss. Plan the roadmap as an
   adapter pipeline, not a schema quest.

2. **Cross-embodiment numbers are not comparable, and the framework must
   refuse to pretend otherwise.** A 43 % on a Franka in robosuite and a 43 %
   on a G1 in the real world are different quantities. Aggregation across
   runs is only allowed when provenance matches on the axes being compared;
   otherwise Gantry declines to put them in one table. This will annoy people
   who want a single leaderboard number. It is correct anyway.

3. **The rule of three, or death by premature abstraction.** Freeze v1 of
   each contract only after **three real implementations** of that plane
   exist and fit it. Interfaces designed from imagination (for hypothetical
   swarms and soft robots) rot instantly. Interfaces distilled from three
   working cases survive the fourth.

4. **Dependency hell is guaranteed, so isolation is load-bearing.** Sim
   stacks conflict with each other and with policy stacks (CUDA, JAX/torch,
   mujoco versions). The per-plugin container/venv decision is not
   optional hygiene; it's the reason plane-swapping works at all. Cost:
   serialization overhead over IPC and slightly slower dev loops. Accept it.

5. **Stage events are the feedback layer's oxygen.** The funnel — the most
   valuable diagnosis we have (it found the grasp bottleneck and the 41.4-pt
   uplift) — degrades to outcome-only statistics without staged events. Sim
   gives them nearly free; offline eval never has them; real needs humans or
   perception. The degraded mode must be explicit and visible, not silent.

6. **Real-world evaluation is organizationally hard, not architecturally
   hard.** Resets, safety, operator time, and tiny n. The interface slots in
   now; the statistics must be small-n honest from day one (Wilson, paired
   bootstrap, McNemar — already built). Do not let the roadmap pretend a
   real-eval plugin makes real evaluation cheap.

7. **Timing semantics change results and must live in the contract.**
   Chunking, control rate, latency. We measured +14.7 points from chunk size
   alone. Any framework that leaves execution protocol out of the record is
   generating irreproducible numbers.

8. **Units and frames are the silent killer.** The Milk-object bug cost a
   day and nearly poisoned every spatial metric. Every channel carries units
   and frame; the resolver validates dimensional sanity; fixtures plant
   frame bugs to prove the checks fire.

9. **Storage gravity.** Robot datasets are TB-scale video. The view-don't-
   convert decision exists so data stays where it lives (S3, disk, HF hub)
   and only metadata + windows move. Materialize caches deliberately, never
   implicitly. (We lost 141 GB to an ephemeral disk this month; storage
   discipline is in the design for a reason.)

10. **Scope discipline.** v1 explicitly excludes: training orchestration,
    a hosted multi-tenant service, automatic reward learning, and universal
    retargeting. Each is a separate product. The plugin seams leave room for
    all of them later.

---

## 7. Scaling path

**Stage 0 — one machine (v1).** Pure library + CLI. Manifests run locally;
plugins isolate via venv/container; records land as parquet + JSON on disk.
Everything below is additive — nothing at this stage gets rewritten later.

**Stage 1 — one lab.** A job runner executes manifests on a queue (the box, a
second box, a SLURM/cloud burst). Records land in an artifact store (S3) with
a small metadata DB (runs, provenance, metric aggregates) for querying.
Plugins already being containerized makes workers trivial.

**Stage 2 — many users.** The web tier: the existing demo becomes a real
front-end over the RunRecord store — browse runs, compare arms, read
prescriptions. Screen-in-browser stays client-side (it already is). Anything
needing GPUs stays a submitted manifest, never an upload-and-hope service.

**Stage 3 — ecosystem.** Third parties publish connectors/backends/evaluators
as pip packages against the semver'd contracts; conformance suites gate a
community index. This is where descriptor discipline pays out: the framework
never needs to know about a plugin to run it, only to resolve it.

Cost note: stages 0–1 are compute you already have. The recurring cost driver
is GPU eval time, not the framework.

---

## 8. Build order (when we start building — not yet)

1. Spines: EpisodeRecord + RunRecord + ChannelSpec + provenance model.
2. Resolver + registry + refusal reporting.
3. Two connectors (LeRobot, robomimic HDF5) — forces spine A to be real.
4. One policy backend (GR00T via the existing serve pattern) + one sim
   evaluator (robosuite) — forces spine B to be real.
5. Port the proven feedback math (screen, funnel, attribution, hardening)
   onto records; synthetic-defect fixtures in CI.
6. Offline evaluator (action error) — first no-sim eval, proves plane 4 has
   two genuinely different implementations.
7. Third implementation of each plane, then freeze contracts at v1 (rule of
   three satisfied).
8. Real-eval operator console — last, because it's the most operationally
   expensive and the contracts must be stable first.

---

*Everything in section 3–5 borrows deliberately from patterns already proven
in this repo (`model_backends/` descriptors + conformance, embodiment tags)
and from this month's measured results on the L4. Nothing here is speculative
plumbing; the speculative parts are labeled as caveats.*
