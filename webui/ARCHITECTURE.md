# Gantry — full product & UI architecture

This is the blueprint: how the product is organised, what every screen looks
like and does, the rules that keep it coherent, and the technical structure
underneath. Written to be buildable screen by screen.

---

# PART 1 — THE PRODUCT

## 1.1 One sentence

A user anywhere in the world uploads robot-training data; Gantry tests it and
returns a verdict — *did this data make the robot better?* — with the evidence
and a fix list.

## 1.2 The three moments that matter

Every good data product is designed around its emotional peaks. Ours are:

1. **The first minute.** The user has just uploaded. They must immediately see
   that something real is happening to *their* data — not a spinner, but their
   own numbers appearing: "18 clips found · 10 kitchens · hands visible 69% of
   the time." First trust is won here.

2. **The wait.** Testing takes hours. The user will close the tab. The product
   must be glanceable from a phone: where is my submission, what stage, how
   long left, has anything gone wrong. Email/Slack ping on verdicts.

3. **The verdict.** One screen the user screenshots and sends to their team.
   It has to stand alone: the answer, the evidence, the confidence, and what
   it does *not* prove — all readable by someone who never saw the product.

Everything else exists to serve these three.

## 1.3 UX principles (house style, no jargon)

1. **One thing at a time.** A screen answers one question. Detail is opened,
   never pre-opened. If a screen needs a scroll map, it's three screens.
2. **Show, then explain.** The number first, the meaning one line under it,
   the method behind a "how was this measured?" link. Three depths, always.
3. **Say how sure.** Every rate has its sample size and range. If the honest
   answer is "can't tell yet," the product says that — as a result, styled
   like a result, never as an error or an empty box.
4. **Never blame the user for our bugs.** If we couldn't measure something,
   we say "not measured," never zero. (This already bit us once: we told a
   ten-kitchen dataset it was filmed in zero locations.)
5. **The state is always visible.** At any moment the user can tell: what is
   done, what is running right now, what comes next, what it will cost.
6. **Spend before you ask.** Free checks run first and deliver real value
   before any paid step is offered. The upgrade is earned, not gated.

---

# PART 2 — STRUCTURE

## 2.1 Site map

```
/                       Landing + methodology (public, static)
/login                  Auth

APP (signed in, org-scoped)
/submissions            Home — every submission, filterable
/submissions/new        The wizard (4 steps)
/submissions/:id        THE HERO — gate timeline for one submission
/submissions/:id/verdict     The shareable verdict document
/submissions/:id/report      Full findings & prescriptions
/submissions/:id/data        The dataset view (schema, clips, measures)
/compare                Pick 2+ submissions on one benchmark → ranking
/benchmarks             Catalogue of tasks you can test against
/settings               Org, members, API keys, billing

OPERATOR (staff only)
/ops                    Queue, workers, failures, versions
```

Shallow on purpose: a user is never more than two clicks from home, and the
submission page is the centre of gravity — everything links back to it.

## 2.2 The object everyone is looking at

```
Submission #S-0142  "kitchen-ego-v2"
  dataset      v2 (18 clips, 5,265 frames)      ← versioned; v1 linked
  benchmark    pick_dual_bottles @ robotwin-2.0
  status       Running — Gate 3, episode 61/100
  gates        G0 ✓  G1 ✓ (3 findings)  G2 ✓ (+12% over control)  G3 ●
  verdict      —  (arrives when G3 ends)
```

A submission moves through four gates, cheap → expensive:

| | Name | Question it answers | Cost | Time |
|--|------|--------------------|------|------|
| G0 | **Intake** | Can we read this at all? | free | seconds |
| G1 | **Data report** | What is this footage like? | free | ~1 min |
| G2 | **Signal check** | Is there anything learnable here? | ~$1 | ~10 min |
| G3 | **Robot test** | Does the robot actually get better? | $20–200 | hours |

A submission can stop at any gate — refused (our checks failed it, with codes),
abstained (we can't tell, and say why), or simply not purchased further.

---

# PART 3 — THE SCREENS

Conventions used below: `[...]` button · `(...)` pill/status · `▸` expandable
row · `┆` live/streaming element.

## 3.1 Submissions (home)

The multi-customer view. A lab sees all their work in one table.

```
┌────────────────────────────────────────────────────────────────────┐
│ Submissions                                     [ New submission ] │
│ (All) (Running 2) (Verdict ready 5) (Refused 1)      🔍 filter     │
├────────────────────────────────────────────────────────────────────┤
│ NAME             BENCHMARK           STATUS         VERDICT     n  │
│ kitchen-ego-v2   pick_dual_bottles   (● G3 61/100)  —          50  │
│ kitchen-ego-v1   pick_dual_bottles   (✓ done)       (Helped +9%)50 │
│ warehouse-a      stack_blocks        (✓ done)       (No signal) 30 │
│ forklift-cam     stack_blocks        (✗ refused G0) —           —  │
└────────────────────────────────────────────────────────────────────┘
```

- Status pill shows the *current gate and live progress* — the table itself is
  glanceable, no need to open anything.
- Verdict chips use plain words: `Helped +9%` · `No signal` · `Can't tell yet`.
- Row click → submission page. Empty state teaches: one illustration, one
  sentence, one button.

## 3.2 New submission — the wizard

Four steps, one decision per step, progress bar across the top. The user can
leave and resume; the draft persists.

**Step 1 · Source**
```
  Where is your data?
  ┌──────────────────────────────┐
  │  ⬆  Drop a .zip here         │   or   [ HuggingFace repo ]
  │     LeRobot format           │        [ S3 / GCS bucket  ]
  └──────────────────────────────┘
  ┆ uploading… 240 MB / 1.2 GB — resumable
```
The instant the header of the archive is readable, detected facts stream in
below the drop zone — episodes, frames, fps, cameras — so the wait is already
informative.

**Step 2 · Confirm meaning**
The one irreducibly human step. We show what we detected; the user confirms
what it *means*. Wrong answers here are the #1 cause of garbage results, so
this step exists to catch them before they cost money.

```
  We found these channels — tell us what they are:
  action (16 numbers/frame)
     detected: two arm poses, quaternion rotation      [ looks right ▾ ]
     frame:    ( camera-relative | robot-relative | world )   ← must pick
  observation.images.head  (224×224 video)             [ head camera ▾ ]

  ⚠ Your poses look camera-relative. The benchmark executes in world
    coordinates — we'll convert using the camera pose you provide, or
    refuse if we can't. This is the single most common silent error.
```

**Step 3 · Benchmark & budget** — the honesty slider
```
  Test against:  [ pick_dual_bottles — dual-arm, RoboTwin ▾ ]

  How many robot trials?
  30 ─────●───────────────── 300
  ┌───────────────────────────────────────────────┐
  │ 50 trials   →  detects a difference ≥ 27 pts  │
  │ est. $34 · ~6 h                               │
  │ To detect ≥ 8 pts you'd need ~300 ($180).     │
  └───────────────────────────────────────────────┘
  ☑ Stop after Signal check if nothing is learnable (saves the full fee)
```
The panel converts money into *what the answer can resolve*. No one else does
this; it prevents every "you charged me and told me nothing" conversation.

**Step 4 · Review** — everything above on one card, plus in plain words what
the test *cannot* conclude at the chosen size. `[ Submit ]`.

## 3.3 Submission page — the hero

One vertical timeline. Done gates are one collapsed line each. The running
gate is expanded and live. Future gates are dim placeholders. Nothing below
the line of "now" pretends to exist.

```
┌────────────────────────────────────────────────────────────────────┐
│ kitchen-ego-v2        pick_dual_bottles       (● Running — Gate 3) │
│ v2 · uploaded 2h ago · compare to v1 ↗                             │
├────────────────────────────────────────────────────────────────────┤
│ ✓ Intake          18 clips · 5,265 frames · schema confirmed   12s │
│ ✓ Data report     3 findings — 1 to fix before refilming ▸     48s │
│ ✓ Signal check    learnable: error 31% below shuffled control ▸ 9m │
│ ● Robot test                                              1h 12m ┆ │
│   ├ ✓ train treatment (yours)      38 min                          │
│   ├ ✓ train control (scrambled)    36 min                          │
│   └ ● evaluating   episode 61/100  ▓▓▓▓▓▓▓░░░  ~39 min left        │
│       reached 61 · lifted-both 44 · solved 9                     ┆ │
│ ○ Verdict         when the robot test completes                    │
└────────────────────────────────────────────────────────────────────┘
```

- The live counters during evaluation are the product's heartbeat — the user
  watches *their* ladder climb in real time.
- A failed gate turns the row red with the reason and one action: `[ Retry ]`
  or `[ See what to fix ]`. Failures of *ours* say so explicitly and never
  count against the user's data.

## 3.4 Verdict — the document

Designed like a one-page report, not a dashboard. Permalinked, exportable to
PDF, readable by someone who has never seen Gantry.

```
┌────────────────────────────────────────────────────────────────────┐
│  VERDICT                                              #S-0142 · v2 │
│                                                                    │
│  Your data helped.                                                 │
│  +9 points over the scrambled control (14% → 23%), on 100          │
│  paired trials. Range on that gain: +2 to +16.                     │
│                                                                    │
│      baseline (no ego data)   12%  ▐▓▓░░░░░░░░│                    │
│      scrambled control        14%  ▐▓▓▓░░░░░░░│                    │
│      YOUR DATA                23%  ▐▓▓▓▓▓░░░░░│  expert 89%        │
│                                                                    │
│  How far the robot got (per 100 attempts)                          │
│      reached → 96   grasped → 82   lifted both → 71   solved → 23  │
│                                                                    │
│  Why we believe it: same scenes, same training recipe, same        │
│  budget for all three arms; only the pairing of video to actions   │
│  differs. Paired exact test p = 0.021.                             │
│                                                                    │
│  What this does not say: one training run per arm — the gain       │
│  could partly be luck of the seed. To claim it firmly, rerun       │
│  with 3 seeds (+$96). [ Strengthen this verdict ]                  │
│                                                                    │
│  Top fixes for v3:  1. Keep both hands in frame (…)  2. …          │
│  [ Download PDF ]   [ Share link ]   [ Compare to v1 ]             │
└────────────────────────────────────────────────────────────────────┘
```

Three verdict archetypes, each with its own opening line and colour:
**Helped** (green) · **No detectable effect** (neutral — with the "what it
would take to detect" line) · **Can't tell** (amber, with the exact blocker).
"No signal" is presented as a *finding about the data*, never as failure noise.

## 3.5 Report — findings & prescriptions

The full diagnosis, but paced. Left: a short nav of the four question groups
(Usable at all? · Footage quality · Did it help? · How solid is that?).
Right: one group at a time, findings as rows (severity dot · plain-English
title · which clips), each expanding to evidence + the fix. The dotted codes
exist but live inside the expanded view — labs get precision, contributors
get sentences first.

Every fix names its evidence: *"Clips 13 and 14: left hand absent for 99% of
frames (moved 0.6 cm total). Re-film keeping both hands in view — this
footage trains the one-handed failure we measured in Gate 3."*

## 3.6 Compare

Pick 2+ submissions on the same benchmark. Output is one table and one chart:
rate + interval per arm, pairwise "is this difference real?" (✓/✗/–), and the
same ladder side by side. If the submissions were tested on different scene
sets, the screen says the comparison is weakened and by how much — it never
silently pretends.

## 3.7 Benchmarks catalogue

A card per task: what the robot must do (short clip), embodiment, price/time
at reference sizes, and the current baseline so users can calibrate hope
before uploading. This is also the marketing surface.

## 3.8 Ops console (staff)

Queue table (submission, gate, worker, elapsed, cost so far), worker health,
failure log with the exact refusal codes, and pinned plugin versions per run
(every verdict must be reproducible to a version set).

---

# PART 4 — THE SYSTEM OF PARTS (design system)

Small on purpose. Every screen above is built from ~12 parts:

| Part | Rule |
|---|---|
| **Status pill** | One vocabulary product-wide: Queued / Running (with live sub-state) / Passed / Refused / Abstained / Failed-ours. Colour = state, nowhere else. |
| **Measure** | A number never travels alone: value + n + range, one component, three sizes (tile / inline / table cell). |
| **Range bar** | The interval drawn, with ceiling tick. Same component everywhere so users learn to read it once. |
| **Gate row** | Collapsed (icon · name · one-line result · duration) / expanded (live) / dim (future). |
| **Ladder** | Horizontal funnel, five rungs, same order always. |
| **Finding row** | severity dot · sentence · scope chips · expand → evidence + fix. |
| **Verdict banner** | The three archetypes; used only on verdict surfaces. |
| **Table** | 40px rows, right-aligned numerals, sortable, one expandable pattern. |
| **Wizard shell** | Steps across the top, one question per screen, draft-saved. |
| **Drawer** | Detail slides over the table; the list never navigates away. |
| **Empty/Loading/Error** | Every data region ships all three, written, not defaulted. |
| **Toast + activity feed** | Transient events; the feed is the durable copy. |

Tokens: keep the current set (near-white canvas, one blue, red/amber/green as
status only, system sans, tabular numerals, 8px grid). Add nothing until a
screen forces it.

Motion: only on state change (gate completes, counter ticks, row expands).
150ms. Never ambient.

Voice: verdict lines in plain English ("Your data helped", "We couldn't read
this"), evidence in precise English, codes for the API. Never "oops", never
blame, never exclamation marks.

---

# PART 5 — TECHNICAL ARCHITECTURE

## 5.1 Shape

```
Browser (React SPA)
   │  HTTPS JSON + SSE (live events)
   ▼
API server (FastAPI)  ── Postgres (all state)
   │                  ── S3 (datasets, checkpoints, run records, PDFs)
   ▼
Job queue (Postgres-backed)
   ▼
Workers
   ├─ intake worker (CPU)   G0/G1 — runs gantry connectors + feedback modules
   └─ GPU workers           G2/G3 — train/serve/evaluate via gantry pipeline
```

One repo, three deployables: `web/`, `api/`, `worker/`. The existing gantry
plugins are the workers' library — the product is a thin, honest shell around
the pipeline that already exists.

## 5.2 Frontend

- **React + TypeScript + Vite.** Component model kills the DOM-by-hand bug
  class we already hit; types keep 30 screens coherent.
- **TanStack Query** for all server data (this app *is* server state: polling,
  caching, invalidation). **SSE** feeds gate progress; events invalidate
  queries so every surface updates from one source of truth.
- **No component framework.** Our 12 parts, styled with our tokens (CSS
  variables + CSS modules). Charts hand-drawn as now — range bars and funnels
  are 30 lines each and always on-brand.
- Folder shape: `web/src/{app,routes,components,lib,api}` with generated API
  types from the OpenAPI schema so frontend and backend cannot drift.

## 5.3 Backend

- **FastAPI + Postgres + SQLAlchemy/Alembic.** Core tables:
  `orgs, users, memberships, api_keys, benchmarks, datasets, dataset_versions,
  submissions, gates, jobs, runs, findings, artifacts, events`.
- `events` is append-only per submission and drives both SSE and the activity
  feed — one history, two consumers.
- **Jobs**: a `jobs` table claimed with `SELECT … FOR UPDATE SKIP LOCKED` —
  boring, transactional, no extra infra. Workers heartbeat; a dead worker's
  job is re-queued or marked Failed-ours. Every job records the gantry plugin
  versions it ran with (reproducibility is a product feature).
- **Storage**: uploads go direct-to-S3 with presigned multipart URLs (the API
  never proxies gigabytes); artifacts (records, ladders, PDFs) written by
  workers, read by API.
- **API surface** (first cut):
  `POST /submissions` · `POST /submissions/:id/dataset` (presigned) ·
  `POST /submissions/:id/confirm-schema` · `POST /submissions/:id/advance`
  (buy next gate) · `GET /submissions[/:id]` · `GET /submissions/:id/events`
  (SSE) · `GET /submissions/:id/verdict|report` · `GET /benchmarks` ·
  `POST /compare` · plus the same under `/v1` with API-key auth for CI.

## 5.4 The gates as code

Each gate is a worker function with one contract:
`run(submission) → verdict(Passed|Refused|Abstained) + findings + artifacts`.
Internally they call what already exists:

- **G0** → `connector_lerobot` open + schema checks (today's upload inspector)
- **G1** → feedback data-side modules (capture, extraction, provenance,
  coverage) + the split measures
- **G2** → probe train + held-out action error vs `detach()` control
  (`feedback_control`'s offline path)
- **G3** → the full pipeline this week proved end-to-end: build arms → train →
  evaluate with the ladder → paired comparison → feedback layer

The bench never computes in the UI; the UI renders worker artifacts. Same rule
as today, now enforced by the architecture.

## 5.5 Build order

1. **Skeleton** — auth, orgs, submissions table, wizard steps 1–2, G0 live.
   *(A user can upload and see intake results.)*
2. **G1 + report page** — the free data report. First real value shipped.
3. **SSE + gate timeline** — the hero page comes alive.
4. **G2 + budget panel** — the money gate and the honesty slider.
5. **G3 + verdict page + PDF** — the full product for one benchmark.
6. **Compare, API keys, ops console** — the lab product.

Each step is shippable; nothing waits on the step after it.
