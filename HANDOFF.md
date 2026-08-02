# Gantry and Gantry Bench: a full map

Written for somebody (or some model) picking this up cold. It says where every
part lives, what it does, how the parts connect, what is finished, what is not,
and which mistakes have already been made so they are not made again.

Repository: `github.com/ShaikZaidHaaris/gantry`, branch `closed-loop`.
At the time of writing: 88 commits, 58 plugins, ~64k lines of Python, ~2.6k of
TypeScript, 1,854 test functions across 99 files, all passing except the ones
marked `gpu`.

---

## 1. What this is

Two things in one repository, and it matters that they are separate.

**Gantry** is a measurement framework for robot-learning data. It is a library
of contracts and plugins. It answers questions like "did this dataset make the
policy better, and how would you know". It has no user interface and no opinion
about who is asking.

**Gantry Bench** (`bench/`) is a product built on top of Gantry. A contributor
uploads a dataset, and it runs a sequence of increasingly expensive checks and
returns a report. It is the thing with a web page and a database.

Gantry does not know Bench exists. Bench imports Gantry the way any other
consumer would. Every rule below about planes and refusals belongs to Gantry;
Bench inherits them by using it.

---

## 2. The one idea

Almost every design decision follows from a single claim:

> Fine-tuning a large pretrained model on *anything* moves it.

Show a model five thousand frames of kitchen video with the action labels
attached to the wrong frames, and the loss still falls, the behaviour still
changes, and a before-and-after table still shows a difference. Reported as
"their data helped", that is false in the most expensive possible way: it is a
result that reproduces for **every** contributor, including the ones whose
footage is worthless.

So the comparison that means anything is not *baseline vs treatment*. It is
**treatment vs its own shuffled control**: the same clips, the same actions, the
same number of gradient steps, with each clip's actions belonging to a different
clip. Whatever fine-tuning-in-general buys, the control buys too. Beating the
baseline says the model changed. Beating the shuffle says the change came from
the correspondence between what the camera saw and what the hands did, which is
the only thing a contributor's data can actually be selling.

Everything else, the ladder and the paired tests and the abstentions, exists to
keep that claim honest at the sample sizes a robot evaluation can afford.

---

## 3. Rules that are enforced, not aspired to

These recur everywhere. When reading code that seems over-careful, it is
usually one of these.

**Absent is not zero.** A property nobody measured is reported as not measured,
never as a score of zero. This has been violated twice and both times produced a
confident, decisive, entirely fabricated result:
- `feedback_capture` once told a corpus filmed in ten kitchens that it was filmed
  in zero, at strong severity, and prescribed filming somewhere else. The scene
  labels had been lost passing through LeRobot. Blaming a contributor for our own
  bug is the worst thing the feedback layer can do.
- The robot gate once reported the baseline reaching 0/100 on every ladder rung
  and the contributor's data sweeping all of them at p=0.0. The baseline had
  simply been evaluated before the stage instrumentation existed.

**No naked floats.** Every rate carries its `n` and a Wilson 95% interval. A
rate of 1.0 from four trials and from four hundred are the same float and not
the same claim.

**Describe, do not coerce. Refuse rather than convert.** A component that cannot
answer returns a `Verdict` with stable dotted reason codes, rather than
degrading into a plausible-looking answer.

**Only disagreements carry information.** Every comparison is paired on the same
scenes. Scenes where both arms succeed or both fail count in the rates and are
excluded from the test. This is why the paired test can see a difference at 50
scenes that an unpaired comparison would need several hundred for.

**Our failure is never their refusal.** `failed` means our worker died or our
disk filled. `refused` means the data was judged and did not pass. A gate that
failed can be retried; a gate that refused cannot, because re-rolling until the
answer is liked is the one thing a benchmark must not offer.

---

## 4. Gantry: the library

### 4.1 The spine: `src/gantry/spine/`

Shared vocabulary. No plugin logic.

| file | what it holds |
|---|---|
| `channel.py` | `ChannelSpec`: one named stream, its shape, units, frame, `dim_labels`, `discriminators` |
| `episode.py` | `EpisodeRecord`, `EpisodeMeta`, `EpisodeLabels`, `StepSource`, `ArraySource` |
| `provenance.py` | `Measurement` (value + n + ci + method), `proportion()` (Wilson) |
| `verdict.py` | `Verdict` / `Reason`: the refusal type, with dotted codes |
| `descriptor.py` | `Descriptor`: what a component declares about itself |
| `inference.py` | the statistics: `trials_needed`, `confidence_sequence`, `mcnemar`, `barnard`, `iqm`, `stratified_bootstrap`, `ppi_mean`, `cohen_kappa`, `krippendorff_alpha` |
| `text.py` | `plural`, `count_of`, `readable`, `without_codes`. Sentence construction, because the product's output is prose |
| `units.py`, `plane.py`, `run.py` | unit handling, plane names, run records |

`inference.py` is implemented rather than imported on purpose: the alternative is
that installing the core pulls scipy, statsmodels and rliable, and the promise
that describing an experiment costs nothing to install stops being true.

**A live bug fixed here, worth knowing about.** `trials_needed` accepted a
`power` argument and ignored it, returning the count at which the *expected*
result was barely significant, which is a coin flip sold as a plan. And the share of
disagreements favouring the better arm was computed as `gain/discordant + 0.5`,
which equals exactly 1.0 (every disagreement going the right way) while the
comment above it said the opposite. Together these took "+8 points on a 12%
baseline" from 35 trials to 210. Fifty trials catch a +8pt effect **17%** of the
time.

### 4.2 The contracts: `src/gantry/contracts/`

One file per plane. A plane is a swappable responsibility. Nothing in the
framework prefers a specific model, simulator or dataset format; each is a
plugin behind one of these.

| plane | contract | what a plugin does |
|---|---|---|
| dataset | `connector.py` | read some format as `EpisodeRecord`s |
| task | `task.py` | what is being asked, and how it is scored |
| scorer | `scorer.py` | decide whether an attempt succeeded |
| curation | `curation.py` | choose or transform which episodes are used |
| embodiment | `embodiment.py` | describe a robot |
| policy | `policy.py` | `Policy` (inference) and `Learns` (optional: can be fitted) |
| evaluation | `evaluator.py` | run a policy against a world |
| feedback | `feedback.py` | turn records into findings and prescriptions |
| adapter | (in `resolve/`) | convert between channel representations |
| retargeter | (in plugins) | map one body's motion onto another's |

`Learns` is a `runtime_checkable` Protocol, not a method on `Policy`, and that is
deliberate: a policy served over HTTP cannot be fitted and a checkpoint loaded
from disk was fitted elsewhere. Neither should have to implement or refuse a
`fit`. Anything fittable declares it by having the method.

### 4.3 The rest of core

- `rollout.py`: the closed-loop driver. `World` protocol, plus optional
  `Speaking` so a policy is prompted with the sentence the world is scoring.
- `resolve/`: requirement matching and adapter insertion. This is how a policy
  that wants quaternions gets them from a dataset that stores euler angles.
- `manifest.py`, `plan_io.py`, `runner.py`: declaring and executing a run.
- `store.py`, `ledger.py`, `lineage.py`, `history.py`: recording what happened.
- `conformance/`: a test harness plugin authors run against their own plugin.
- `cli.py`: command line entry.

### 4.4 Plugins: `plugins/`, 58 of them

Discovered by entry point, never imported by name. Groups:
`gantry.connectors` (9), `gantry.evaluators` (13), `gantry.feedback` (16),
`gantry.policies` (5), `gantry.scorers` (3), `gantry.retargeters` (3),
`gantry.curators` (2), `gantry.adapters` (2), `gantry.tasks` (1),
`gantry.embodiments` (1).

The ones that matter most for Bench:

- **`connector_lerobot`**: reads LeRobot v2.x, which is what contributors upload.
- **`evaluator_robotwin`**: RoboTwin 2.0, dual-arm, 50 tasks. Reads RoboTwin's
  own YAMLs rather than inventing a partial config. Emits per-object stage
  events, which is what the ladder is built from.
- **`evaluator_offline`**: open-loop action error, no simulator, runs on a laptop.
- **`policy_pi0`**: serves π₀/π₀.₅ over openpi's websocket.
- **`policy_probe`**: a ridge regression on downsampled pixels. Deliberately
  incapable of flattering the data: no capacity to memorise, one hyperparameter,
  no initialisation or schedule, closed form. It exists so the two arms of the
  signal check differ in *nothing* except which frames sit beside which actions.
- **`curate_control`**: builds the shuffled control. Uses a *derangement*, not a
  shuffle: an ordinary shuffle leaves some episodes matched with themselves, and
  those are training data sitting inside the control arm.
- **`curate_split`**: `Split` (cohorts on a measured property), `hold_out`
  (train/test), `match_frames`. Splits on whole episodes and refuses when a
  participant or scene straddles the split.
- **`feedback_*`**: 16 modules. `capture` (filming advice), `extraction`
  (whether the pipeline read the footage), `provenance` (licensing), `coverage`
  (is this data about the task), `stillness` (frozen dimensions), `control` (the
  shuffle comparison), `power` (sizing), `compare`/`rank` (across policies),
  `calibrate` (judge agreement), `verify` (curation plans).

---

## 5. Gantry Bench: the product

### 5.1 The gauntlet

Four gates, cheap to expensive. This is simultaneously the UX, the cost control
and the epistemics.

| gate | question | cost | can it stop here |
|---|---|---|---|
| **G0 Intake** | Can we read this at all? | seconds | yes, refuses with codes |
| **G1 Data report** | What is this footage like? | ~1 min | no, but delivers real value |
| **G2 Signal check** | Is there anything learnable here? | ~10 min | **yes, the money gate** |
| **G3 Robot test** | Does the robot actually get better? | hours | final verdict |

All four are currently **free** (`cost_cents: 0`). The prices are still computed
and displayed, because what a run costs is a real number and hiding it makes the
trial-count choice look arbitrary.

Free gates queue themselves when the previous one passes. Paid gates never do:
a contributor is never billed for work they did not ask for.

### 5.2 `bench/api/`: FastAPI + SQLAlchemy

`app/db.py` holds the models and the storage seam.

Tables: `orgs`, `users`, `memberships`, `benchmarks`, `submissions`,
`dataset_versions`, `gates`, `jobs`, `events`.

Two things to understand:

- **Gates and jobs belong to a *version*, not a submission.** A second upload
  gets its own set of gates and the previous version keeps its verdicts. This is
  the whole reason resubmission is worth doing. A worker writes to the gate for
  *its own job's version*, so a long v1 run finishing after v2 was uploaded
  cannot stamp its verdict onto footage it never saw.
- **`sweep_stale()`** fails jobs whose worker stopped heartbeating (90s). A gate
  stuck on "running" forever is the worst failure mode: it looks like patience is
  the answer, so nobody investigates.

Routes:

```
GET  /api/me                                  who is this, which org
GET  /api/benchmarks                          benchmarks + the gate spec
GET  /api/benchmarks/{key}/plan               what a trial count can conclude, and cost
GET  /api/submissions                         list
POST /api/submissions                         create
GET  /api/submissions/{id}                    one, deep, with events
POST /api/submissions/{id}/dataset            upload (creates a new version)
POST /api/submissions/{id}/meaning            confirm what the channels mean
GET  /api/submissions/{id}/versions           every upload + what changed between them
GET  /api/submissions/{id}/events             SSE: durable log + live progress
POST /api/submissions/{id}/gates/{key}/start  buy a gate
POST /api/submissions/{id}/gates/{key}/retry  re-run one that failed on our side
GET  /api/compare                             leaderboard: ranked, paired, letters
POST /api/jobs/claim                          worker: take a job          [token]
GET  /api/jobs/{id}/archive                   worker: fetch the dataset   [token]
POST /api/jobs/{id}/heartbeat                 worker: alive + progress    [token]
POST /api/jobs/{id}/finish                    worker: verdict             [token]
GET  /{path}                                  serves the built SPA
```

**The SSE stream has two frame types.** `message` frames are the durable log and
carry an `id:`, so a browser that loses its connection reconnects with
`Last-Event-ID` and is sent exactly what it missed. `progress` frames carry the
running gate's position, have **no** id, and are not replayable, nobody wants a
replay of a progress bar, and advancing the resume cursor on one would skip
durable events. Progress is read off the gate row rather than a queue, so the
stream coalesces at its own rate no matter how fast a worker reports.

**Progress never enters the event log.** A 3,000-step run would put 3,000 rows
into a log whose whole value is that a person can read it. It overwrites a column
on the gate, which also means a reload an hour into a long run shows where the
run actually is.

### 5.3 `bench/worker/`: claims jobs, runs gates

`run.py` is deliberately dumb: it knows how to talk to the API and which function
handles which gate. Everything else is the gate's business.

- **The heartbeat is a thread.** Gates block for minutes or hours inside one
  call. A worker beating between steps would be swept as dead mid-run, throwing
  away an honest result and reporting it as our failure. The sender wakes on
  movement with a floor between sends, so a phase change appears at once while a
  per-step counter cannot become a request per step.
- **`fetch.py` / `ensure_dataset`** returns where the dataset actually is: the
  local path when it exists, a streamed download otherwise, written to a
  `.partial` and moved into place. The job carries both a path and a URL; the
  path is an optimisation for a co-located worker, not the mechanism.
- `--work` gives a remote worker its own scratch root, because the job's workdir
  is a path on the API's machine.

Gate contract, all four identical:

```python
def run(archive: Path, workdir: Path, report: Report, params: Mapping) -> dict
# returns {status, summary, findings, measures, abstained, detail, detected}
# status ∈ passed | refused | abstained | failed
```

`gates/intake.py` (G0), unpacks, refuses unsafe archive paths, non-zips, missing
LeRobot metadata, no episodes, no action channel, no camera, missing videos, and
videos that will not decode (sampled, because opening ten thousand files would
stop the free gate being free).

`gates/report.py` (G1), finds every installed feedback module by entry point and
routes each by what it *declares* it needs. Modules that read outcomes are not
offered a training set: a dataset has no rollouts, and one handed demonstrations
as though they were attempts once reported two *training sets* as having "solved
100%". Also owns the translation layer: `LABELS` (module name → what it checks),
`plainly()` (strips dotted codes and the cohort prefix), and the missing-module
check.

`gates/signal.py` (G2), the money gate. Holds episodes back, fits the probe to
the rest, fits the same probe to the same episodes with actions detached, scores
both on the held-out set, compares with an exact sign test. **Images only**, proprioceptive state predicts the next action in every dataset, so a probe
allowed to read it finds signal in footage whose pixels say nothing.
`smallest_conclusive()` derives the minimum held-out count from the test itself,
so the floor and the test cannot drift apart.

`gates/robot.py` (G3), two separable halves. `assemble()` reads finished
rollouts and produces the verdict, costing nothing. `produce()` invokes a runner
to make them, costing a day of GPU. Rungs are read off whatever stage events the
world emitted; nothing here knows what a bottle is.

### 5.4 `bench/runner/`: how rollouts get produced

The gate says *what* it needs. This says *how*, for one machine: openpi to train
and serve, RoboTwin to evaluate. It is a seam, not an import, because the moment
the gate imports a trainer the framework has a favourite model.

Contract: `run.sh job.json`, writes run records plus `arms.json` into `out/`, and
appends progress to a file the gate tails. Progress goes to a file rather than
stdout because the trainer and simulator both write freely in formats that are
not ours.

Stages: build both arms (cut to identical lengths), register each in openpi's
config list, compute norm stats, train, serve, evaluate, write records. One arm
at a time, deleting each checkpoint before the next, because a checkpoint is
8.5 GB against 13 GB free.

Set `BENCH_RUNNER` to enable it. Left unset the gate reads rollouts and says
plainly that it cannot produce them, which is the right default for a shared
deployment.

### 5.5 `bench/web/`: React + TypeScript + TanStack Query

```
src/App.tsx                    routes and nav
src/api/client.ts              every hook; the only thing that fetches
src/api/types.ts               the shapes the API returns
src/lib/tokens.css             the whole design system
src/routes/Submissions.tsx     the list
src/routes/NewSubmission.tsx   the upload wizard
src/routes/SubmissionDetail.tsx  the hero screen
src/routes/Compare.tsx         the leaderboard
src/routes/VerdictPage.tsx     the verdict as a document + PDF
src/components/GateTimeline.tsx  the gauntlet, live
src/components/DataReport.tsx    G1's report, four sections
src/components/Verdict.tsx       G3's ladder and refusals
src/components/BudgetPanel.tsx   what a trial count can conclude
src/components/Versions.tsx      resubmit + what changed
src/components/Channels.tsx      what is in the upload
src/components/ui.tsx            shared: FindingRow, StatusPill, readable, plural
```

Design rules that were arrived at by getting them wrong first:

- **A finding is two sentences**, not a card. The machine code, the module
  credit, the severity dot and the tinted "WHAT TO DO" box were all the system
  talking about itself. Codes live on hover.
- **Grouping carries severity.** A red dot next to items in a section called
  "What to fix" says the same thing twice.
- **The em dash appears in exactly one place**: the placeholder in a table cell
  with no value. Everywhere else it was rewritten as prose.
- **Machine names are readable, raw on hover.** `observation.images.head` shows
  as "Head camera". The five LeRobot bookkeeping columns are not shown at all.
- **The leaderboard's letter is the column to read.** At these sample sizes the
  ordering is mostly noise, so entries nothing separates share a letter.

### 5.6 `bench/deploy/`

`deploy.sh HOST KEY` builds the frontend, copies the pipeline, plugins, API,
bundle and worker, creates the virtualenv, installs everything, and enables two
systemd units with `Restart=always`.

It binds the API to loopback and does **not** open a port. Exposure is a tunnel,
or whoever owns the firewall.

---

## 6. How a submission flows

```
browser ──POST /api/submissions──────────────► API ── creates submission
        ──POST .../dataset──────────────────► API ── new DatasetVersion, new
                                                     gate rows, queues a G0 job
                                                          │
worker ──POST /api/jobs/claim──────────────────────────►  │ (token)
       ◄── {archive_url, workdir, params, version} ───────┘
       ── GET /api/jobs/{id}/archive ─► streams the zip
       ── runs the gate, calling report(...) as it moves
       ── POST /api/jobs/{id}/heartbeat  {progress}  ~every 10s or on movement
       ── POST /api/jobs/{id}/finish     {status, findings, measures, detail}
                                                          │
browser ◄── SSE: message frame (durable) ─────────────────┘
        ◄── SSE: progress frame (live, no id)
        ── refetches the record on any durable event
```

G0 passing auto-queues G1 (free). G2 and G3 wait for `POST .../gates/{key}/start`.

---

## 7. The experiment this was built against

`experiments/robotwin_ego/` holds the scripts. `RESULTS.md` holds the numbers.

Two datasets of RoboTwin footage, 58 clips each, differing in one thing: **A**
(`rt_two_handed`) has both arms moving; **B** (`rt_one_handed`) has one arm
effectively frozen. Both trained as π₀.₅ LoRA for 3,000 steps, both evaluated
closed-loop on the same 50 expert-screened scenes.

```
                 A two-handed      B one-handed      paired
moved             50/50 100%        50/50 100%       p=1.0
lifted            50/50 100%        49/50  98%       p=1.0
moved all 2       44/50  88%        34/50  68%       p=0.002   separated
lifted all 2      41/50  82%        31/50  62%       p=0.0063  separated
solved             4/50   8%         0/50   0%       p=0.125   not separated
```

**This table is the argument for the whole product.** The two datasets are
indistinguishable until the task needs both hands at once, which is exactly the
capability B's footage never demonstrated. And binary success, the metric
everybody reports, *cannot separate them*: 4/50 against 0/50 is four discordant
pairs, all favouring A, the most extreme outcome available, and it still only
reaches p=0.125. Same rollouts, same scenes, same money. One rung concludes and
the other cannot.

Baseline: the stock model solves 12/100. A does not beat it (p=0.75). B is
significantly **worse** than it (p=0.031).

---

## 8. What is running right now

- **Box**: an AWS L40S at `44.203.100.77`, user `ubuntu`. Security group opens
  **port 22 only**, which is why there is a tunnel.
- **API + web**: `gantry-api.service`, uvicorn on `127.0.0.1:8090`, serving both
  the API and the built SPA. Data in `/home/ubuntu/gantry_bench/data`.
- **Worker**: `gantry-worker.service`, gates `g0,g1,g2,g3`, `BENCH_RUNNER`
  **unset** so it reads rollouts and does not train.
- **Public URL**: a Cloudflare quick tunnel, `*.trycloudflare.com`. Ephemeral,
  since it changes whenever `cloudflared` restarts.
- **Three submissions** seeded from the real experiment records.
- Secrets: `/home/ubuntu/gantry_bench/env`, mode 600, generated on the host,
  never in the repository.

---

## 9. What is not done

**The biggest gap: no submission has produced its own shuffled control through
the product.** G3 can read rollouts and can train, serve and evaluate, but a full
two-arm run has never completed through the gate. Both live submissions carry
`robot.no_control_arm` at strong severity for exactly this reason. Until one
completes, every verdict is a *ranking* and not an *attribution*, and attribution
is the product's central claim.

Also open:

- The baseline record has no stage events, so every ladder rung reads "not
  comparable" against it. It needs one re-run with instrumentation.
- The `reached` rung never fires, because of a 6 cm threshold against a
  wrist-link pose.
  Deferred mid-experiment so A and B stayed comparable; still uncalibrated.
- No tests for `bench/runner`. `assemble` is tested, `produce` is not.
- **No authentication.** `viewer()` reads an `x-demo-user` header and trusts it.
  That function is the single seam real auth goes through.
- SQLite, local disk, no billing. All deliberate single-point seams (`db.py`
  plus two storage helpers), but no external lab can use it until they change.
- The "confirm what your channels mean" step exists in the API and no screen
  calls it.
- One benchmark only (`pick_dual_bottles`). A second would prove the plumbing is
  really task-agnostic.
- No methodology page, no operator console, no notifications.
- The openpi config injection edits their source file. Guarded by an anchor check
  that fails loudly, but an upgrade will break it.
- The leaked AWS key from early in the project was never rotated.

---

## 10. Traps, in the order they will bite

1. **`pkill -f <pattern>` matches its own command line.** This killed four SSH
   sessions and, later, a runner *after* its training had succeeded. Use the
   bracket trick (`'[s]erve_policy.py'`) or match on something the invocation
   does not contain.
2. **`pgrep` patterns that do not match how a process was actually started.**
   A systemd unit with `WorkingDirectory` runs `python run.py`, so nothing in its
   command line contains the path. Three stray workers survived every cleanup
   because of this, and one of them was old enough to still be writing old
   wording.
3. **Two processes on one port.** Cleanup by name left a stale tokenless API
   bound alongside the new one; requests hit whichever won, which looks exactly
   like an auth bug. Kill by `lsof -ti tcp:PORT -sTCP:LISTEN`.
4. **Stored text does not change when the code does.** Findings are written when
   a gate runs. A copy change only reaches the page after the gates that wrote
   those sentences run again.
5. **Disk.** 13 GB free, 8.5 GB per checkpoint. A training run already died
   mid-write with ENOSPC and lost the whole thing.
6. **The worker's `--work` root is not the API's storage path.** Placing rollouts
   in the wrong one silently triggers a real GPU training run. This happened
   twice.
7. **Backgrounding over SSH** with `nohup` dies with the shell. Use `setsid ...
   < /dev/null` or, better, systemd.

---

## 11. Running it

```bash
# install
uv sync --all-extras          # or: pip install -e . && pip install -e plugins/*

# tests
python -m pytest tests/ plugins/*/tests bench/worker/tests -m "not gpu"
cd bench/api && python -m pytest tests/

# local stack
cd bench/api && BENCH_DATA=../data uvicorn app.main:app --port 7920
cd bench/worker && python run.py --api http://127.0.0.1:7920 --gates g0,g1,g2
cd bench/web && BENCH_API=http://127.0.0.1:7920 npm run dev

# deploy
bench/deploy/deploy.sh ubuntu@HOST /path/to/key.pem
cloudflared tunnel --url http://127.0.0.1:8090   # on the host
```

Environment: `BENCH_DATA`, `BENCH_WORKER_TOKEN`, `BENCH_REQUIRED_MODULES`,
`BENCH_RUNNER`, `BENCH_TRAIN_STEPS`, `BENCH_API`, `BENCH_WEB_PORT`.

---

## 12. Where to look first

- The argument for the product: section 7 above, and
  `experiments/robotwin_ego/RESULTS.md`.
- The rules: `src/gantry/contracts/feedback.py` docstring.
- The money gate: `bench/worker/gates/signal.py` docstring.
- What the product refuses to claim: `bench/worker/gates/robot.py` docstring.
- The design of the UI: `webui/ARCHITECTURE.md` and `webui/PLAN.md`, written
  before the build. `bench/web` is the result.

Most files open with a docstring explaining why they are shaped the way they
are, usually naming the specific failure that shaped them. Those are the fastest
way in.
