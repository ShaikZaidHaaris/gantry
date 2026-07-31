# Gantry v2 — the full architecture

One sentence: **a measurement system for robot learning whose every number is
refusable, reproducible, and compounding.** Refusable — it can say no, and its
refusals are the product surface. Reproducible — anyone can re-derive any
published number mechanically. Compounding — every run, anywhere, makes the next
report smarter.

Layers 1–2 below exist today (51 plugins, 10 planes). Layers 3–6 are the build.
The constitution applies to all six and never changes.

---

## 0. The constitution

These invariants are the moat. Everything else is implementation and may be
rewritten; these may not. Each is already enforced somewhere in the codebase,
and the enforcement mechanism is named because an invariant nobody checks is a
comment.

| # | invariant | enforced by |
|---|---|---|
| 1 | Core names no implementation. No robot, no simulator, no model | `test_neutrality` — has caught violations twice |
| 2 | Describe, don't coerce. A mismatch is a refusal with a code, never a silent conversion | `Verdict` + discriminators |
| 3 | Abstention is first-class. `None` is never folded into `False` | scorer contract, `outcomes_of`, rollout metrics |
| 4 | No naked floats. Every number carries n, interval, method | `Measurement` |
| 5 | Severity belongs to the module that found it. Nothing upgrades or blocks in transit | the assembler bug of 2026-07-31, now a test |
| 6 | Proposing and verifying are never the same component | curation/feedback split, control arm |
| 7 | Licences dominate through lineage; unknown outranks restricted | `feedback_provenance` |
| 8 | Every declared capability is checked against a real record | conformance kits |
| 9 | One plane varies; every hold is verified from provenance, not trusted | manifest + feedback `holds` |
| 10 | What cannot be answered is said out loud, prominently | "What we could not tell you" section |

---

## 1. The six layers

```
┌──────────────────────────────────────────────────────────────────┐
│ 6  PRODUCT      manifest chains · report assembler · submit API  │
│                 gantry verify                        (the offer) │
├──────────────────────────────────────────────────────────────────┤
│ 5  KNOWLEDGE    shared History · fidelity ledger · calibration   │
│                 corpus · effect-size priors        (compounding) │
├──────────────────────────────────────────────────────────────────┤
│ 4  RECORD       content-addressed store · Claims · hash chain    │
│                 append-only runs                        (trust)  │
├──────────────────────────────────────────────────────────────────┤
│ 3  EXECUTION    scheduler · dumb workers · batched rollout       │
│                 shards · resume · cache                 (scale)  │
├──────────────────────────────────────────────────────────────────┤
│ 2  CONTRACTS    10 planes · policy@2.0 · connector@1.2 chains    │
│                 conformance kits                   (modularity)  │
├──────────────────────────────────────────────────────────────────┤
│ 1  SPINE        channels · episodes · verdicts · measurements    │
│                 fingerprints · inference           (epistemics)  │
└──────────────────────────────────────────────────────────────────┘

each layer speaks only to the one below it; plugins live at layer 2
and never see layers 3+ (a plugin cannot know whether it is running
on a laptop or shard 412 of a fleet job — that is the scaling trick)
```

---

## 2. Layer 1 — the spine (exists; one addition)

What exists: `ChannelSpec` with kinds, units, frames, semantics,
discriminators; `EpisodeRecord` with lazy `StepSource`s; `Verdict` with dotted
codes; `Measurement`; `StageEvent`; the numpy-only `inference` module (Barnard,
Holm, confidence sequences, IQM, stratified bootstrap, MMRV, PPI, κ, α).

**Addition — `Fingerprint`.** One canonical content address for every kind of
artifact, because layers 3–5 all key on it:

```python
fingerprint(episode)   # by SEMANTICS, not channel names (lineage.py has the seed of this)
fingerprint(dataset)   # Merkle root of its episodes' fingerprints
fingerprint(weights)   # file hash + declared recipe hash
fingerprint(manifest)  # canonical-JSON hash
fingerprint(code)      # git sha + resolved dependency lock
```

Rule: two things with one fingerprint are interchangeable everywhere; two
things with different fingerprints are never silently treated as the same.
This is invariant 2 extended to artifacts.

---

## 3. Layer 2 — contracts (exists; two upgrades)

The ten planes stand: dataset, task, scorer, curation, embodiment, policy,
evaluation, feedback, adapter, retargeter. Contract versioning discipline
stands: minor bump = new optional method with a refusing default; major bump =
breaking, refused by name at resolve time.

### Upgrade A — `connector@1.2`: chains

The flagship pipeline is connectors composed over connectors, and manifests
cannot say so — the crown jewel lives in scripts. Fix at the contract:

```json
"dataset": { "chain": [
  {"name": "egovideo",   "config": {"path": "uploads/42"}},
  {"name": "handpose",   "config": {"estimator": "rtm+mediapipe",
                                     "rig": "gopro_hero5_wide"}},
  {"name": "egoactions", "config": {"hold_missing": true, "size": [224, 224]}},
  {"name": "worldframe", "config": {"trajectory": "device"}, "optional": true}
]}
```

Resolution walks the chain: stage *n*'s connector is constructed with stage
*n−1*'s as its source; each link's requirements are checked at plan time
(`handpose` needs a video channel; `egoactions` needs metric wrists — a chain
that cannot work is refused before a frame is decoded). Lineage falls out for
free: each stage already writes `derived_from`.

### Upgrade B — `policy@2.0`: the batched contract

The single highest-leverage change in the codebase. Today's contract is one
observation in, one chunk out, which caps every evaluator at one environment
and every comparison at the n where p=0.33 lives.

```python
class Policy(ABC):                        # 2.0
    def act(self, observation) -> chunk                 # unchanged
    def act_batch(self, observations) -> list[chunk]:   # NEW
        return [self.act(o) for o in observations]      # default: loop
```

Every existing policy keeps working untouched. Paired with:

- `CAP_VECTORIZED` on the evaluator contract — a world that can step k scenes
  at once declares it, honestly (ManiSkill: 4096; robosuite: 1).
- `BatchedClosedLoop` in core — drives k scenes against `act_batch`,
  preserving every rollout invariant (obs-before-action, per-trial failure
  isolation, milestone indexing). Written once, like `rollout.py` was.

Effect: n=50 evenings become n=5000 evenings. Every statistical refusal in the
feedback layer is a function of n; this is the knob.

---

## 4. Layer 3 — execution (new)

Design rule: **a run is data, and workers are dumb.** All intelligence stays in
the resolver and the record; a worker pulls a job, executes it, writes a
record, and heartbeats. Nothing about correctness lives in the scheduler.

```
            manifest (fingerprinted)
                     │  plan
                     ▼
              ┌─────────────┐   shards: eval → by (scene, seed) block
              │  Scheduler  │           ingest → by clip
              └──────┬──────┘           train → opaque (recorded, not run)
        ┌────────────┼────────────┐
        ▼            ▼            ▼
   worker(local) worker(ssh)  worker(cloud)     ← same binary everywhere
        │            │            │
        └────────────┴────────────┘
                     ▼
          Record store (layer 4)
```

- **Job** = `(manifest_fp, shard_spec, seed_block)`. Deterministic identity ⇒
  a completed job is cached by its key and never re-run. Resume after a crash
  is "skip the shards whose records exist" — for free.
- **Sharding is the parallelism model.** Evaluation is embarrassingly parallel
  across scenes and seeds; the 9 idle g5 boxes become 9 workers with zero
  plugin changes, because plugins cannot see the layer.
- **Memory discipline is contractual**: streaming `StepSource`s are the
  default, `cache=False` for batch builds, workers declare a memory budget and
  the scheduler packs within it. (The 44 GB wedge becomes a scheduling error,
  not an outage.)
- **Training is scheduled but opaque.** The scheduler launches a declared
  recipe (an openpi command, hashed) and records what came out. Core never
  interprets it — invariant 1.

---

## 5. Layer 4 — the record (new): trust as protocol

Append-only, content-addressed store. Local directory first, S3 later — the
layout is the API, the backend is boring.

```
store/
  objects/<fp>            RunRecords, reports, dataset manifests, weight pointers
  claims/<fp>             the thing below
  chain.jsonl             append-only: every object, its parents, a timestamp
```

**The Claim** is the unit of trust — what a published number actually is:

```python
Claim:
  measurement: Measurement          # value, n, interval, method
  holds: {plane: component_ref}     # verified from provenance, not asserted
  varies: str
  cohorts: {name: dataset_fp}
  evidence_seeds: tuple[int, ...]   # the leakage guard, promoted to the record
  manifest_fp / code_fp / weights_fp
  parents: tuple[claim_fp, ...]     # what this builds on
```

**`gantry verify <claim-fp>`** re-resolves the manifest, replays cached shards
or re-runs live, and reports agreement within the claim's own interval. The
sentence this buys: *"here is our number, and here is the command that
regenerates it on your hardware."* No benchmark ships this. If labs begin
citing re-runnable claims, the verifier becomes the standard, and protocol
moats outlive feature moats.

The hash chain makes the corpus tamper-evident: every record names its
parents, so a rewritten history is detectable by anyone holding any suffix.

---

## 6. Layer 5 — knowledge (new): the compounding corpus

History stops being local JSON and becomes a **synced view over the record
store**. Every module that currently starves for a prior reads from it:

| table | keyed by | feeds | today |
|---|---|---|---|
| baselines | (task, embodiment, policy family) | `power` — real effect sizes, not guesses | empty |
| capture priors | capture signal | `capture` — "uploads above you average 96%" | empty |
| judge calibration | (judge, task family) | `calibrate` — κ that persists across runs | empty |
| **fidelity ledger** | (evaluator, task family) | which sim may stand in for hardware: accumulated MMRV | empty |
| effect priors | (intervention, task family) | `power`, `control` | empty |

The fidelity ledger deserves its own line, because it is the dataset nobody
else can build:

```
             lift   pick-place   kitchen   long-horizon
robosuite    0.04     0.11         —           —
robocasa      —        —          0.09?        —          MMRV vs real,
maniskill    0.08     0.13         —           —          accumulated across
simpler      0.02     0.03         —           —          every run, forever
```

It requires running sim and hardware through one spine with one scorer
discipline — which is exactly what exists — and it grows with every
`evaluator_bench` session and every SimplerEnv run. It is the table labs will
consult before believing any simulator, and it cannot be scraped, only earned.

**Privacy tiers** make compounding safe: every row is `private | org |
global`. A contributor's episodes never leave their tier; the *aggregates*
(priors, ledger rows) compound globally. The corpus is the network effect —
**a measurement system whose refusals shrink with use** — and corpora take
calendar time, which is why the write-everything policy starts now, not at
scale.

---

## 7. Layer 6 — product

Thin by design; every hard decision lives below it.

- **Submit API**: upload → manifest chain instantiated → job graph → report
  URL. The GUI is a manifest editor and a report renderer, nothing more.
- **The report** (exists): licence → scope → control → effect → filming advice
  → our-side advice → *what we could not tell you*. Blockers suppress without
  deleting; nothing upgrades in transit.
- **`gantry verify`**: the public trust surface, same binary customers run.
- **Capture tiers as product truth**: bronze (bare video → shape + grasp
  timing, position refused), silver (calibrated capture → full trajectories),
  gold (UMI-style gripper → no retargeting gap). The report states the tier
  and what it can support — describe, don't coerce, applied to capture.

---

## 8. The three flows

**Upload → report** (product):
```
upload ─▶ chain resolves (refusals here are instant and free)
       ─▶ ingest shards fan out ─▶ dataset_fp + control derived
       ─▶ train jobs (recorded recipes): ego / shuffled / base
       ─▶ eval shards fan out across fleet, batched
       ─▶ feedback modules read records ─▶ assembler ─▶ report + Claims
```

**Claim → verification** (trust):
```
claim_fp ─▶ fetch manifest/code/weights by fingerprint
         ─▶ replay cached shards or re-run
         ─▶ compare within the claim's own interval ─▶ verdict
```

**Fleet evaluation** (scale):
```
1 manifest ─▶ 200 shards × (4096 batched envs where the world allows)
           ─▶ 9 workers drain the queue ─▶ records land content-addressed
           ─▶ n goes from 50 to 5000; p=0.33 becomes an answer
```

---

## 9. Scaling model — what each axis costs

| axis | mechanism | ceiling |
|---|---|---|
| data volume | streaming sources, content-address dedup, sidecar metadata | disk, not RAM |
| trial count | act_batch × vectorized worlds × shards | GPU-hours, linearly |
| fleet size | dumb idempotent workers | queue throughput (trivial) |
| corpus | append-only store, synced views | none — it only grows |
| plugin count | contracts + conformance keep integration O(1) | none observed at 51 |
| contributors | tiers isolate data; aggregates compound | trust, which the Claims protocol addresses |

---

## 10. Why this is the moat

1. **The epistemics are expensive to copy** — not technically, commercially. A
   competitor adopting refusal-first reporting must first stop publishing the
   numbers their customers currently enjoy. Honesty has a switching cost that
   protects the honest.
2. **The corpus cannot be scraped, only accumulated.** Fidelity rows,
   calibration κ, effect priors — each requires running real evaluations
   through one spine. Every customer run widens the gap.
3. **The protocol outlives the product.** If re-runnable Claims become how
   results are cited, whoever runs the verifier defines the standard.
4. **The constitution keeps it coherent at scale.** Fifty-one plugins already
   integrate at O(1) cost because contracts + conformance carry the burden.
   Layer 3's plugin-blindness extends the same property to fleets and teams.

---

## 11. Migration from today

| step | what | size | unblocks |
|---|---|---|---|
| 1 | `connector@1.2` chains in manifests | ~1 wk | the product; kills the SSH scripts |
| 2 | `policy@2.0` + `BatchedClosedLoop` | ~1 wk | decisive n |
| 3 | record store + Claims + `verify` | ~2 wk | trust protocol |
| 4 | scheduler + workers over the store | ~2 wk | the fleet |
| 5 | History as synced view + tiers | ~1 wk | compounding starts |
| — | write-everything policy | now | the corpus clock starts today |
| — | fidelity rows on every bench/simpler run | ambient | the ledger |

Nothing in the migration touches a plugin. That is the test that the layering
is right.

---

## 12. What stays out — permanently

- **No training plane.** Recipes are recorded and hashed, never executed by
  core. The day core interprets a training run, neutrality dies.
- **No universal action space.** Per-pair retargeters with declared losses are
  the honest design; a canonical space is a lie with a schema.
- **No more simulators** until Isaac Lab gets a dedicated machine. Twelve
  evaluators already cover every measurement axis identified.
- **No unglated model judgement.** The κ gate on scorers is load-bearing;
  cheap judgement is only safe because it can be refused.
