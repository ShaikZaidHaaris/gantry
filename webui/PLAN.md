# Gantry Bench — product plan

## 0. What is wrong today

The current page is a **viewer for one experiment that already happened**. It reads
JSON off local disk and renders all of it at once. There is no user, no
submission, no time, no state. A second customer cannot exist in it.

The product is not a report. It is a **service that answers one question, with
evidence, for anyone in the world who uploads data**:

> Did my data actually make the robot better — and how do you know?

Everything below follows from taking that seriously.

---

## 1. Who it is for

**P1 · Contributor** — collects data, uploads it, wants a verdict and a fix list.
Waits hours. Cares about turnaround, cost, and not being blamed for our bugs.

**P2 · Lab lead** (the main client) — has twelve datasets and a training budget.
Wants a defensible ranking with intervals they can put in front of a colleague
who will attack it. Cares about comparability and methodology transparency.

**P3 · Operator** (us) — keeps the bench honest. Cares about queue, GPUs,
failures, plugin versions.

Today's UI serves none of them. It serves *the author of one experiment*.

---

## 2. The object model

Everything hangs off one hero object.

```
Organisation
 └─ Project                      a lab's workspace, one benchmark focus
     ├─ Submission               ← THE HERO OBJECT
     │   ├─ Dataset v1, v2…      versioned artifact
     │   ├─ Benchmark @ version  task + embodiment + protocol, pinned
     │   ├─ Gates  G0 G1 G2 G3   the gauntlet, each with verdict + cost
     │   ├─ Arms                 treatment · control · baseline
     │   ├─ Runs                 per arm: rollouts, ladder, records
     │   ├─ Verdict              did it help + how sure
     │   └─ Prescriptions        ranked, each tied to a measurement
     └─ Leaderboard              submissions ranked on one benchmark
```

A **Submission** is what a user creates, watches, shares, and iterates on. The
whole UI is its lifecycle. Versioning is first-class because the product loop is
*submit → fix → resubmit → see the delta*.

---

## 3. The gauntlet — the central idea

Four gates, **cheap to expensive**, each with an explicit verdict and a price.
This is simultaneously the UX, the cost control, and the epistemics.

| Gate | What it does | Time | Cost | Can it stop here? |
|---|---|---|---|---|
| **G0 Accept** | Readable? schema, episodes, channels, widths | seconds | free | yes — refuses with codes |
| **G1 Describe** | Licence, hand visibility, instruction & scene variety, extraction health | < 1 min | free | no, but delivers real value |
| **G2 Fit** | Short probe train; held-out action error vs shuffled control | ~10 min | ~$1 | **yes — the money gate** |
| **G3 Capability** | Full train + closed-loop, treatment vs control vs baseline, ladder, paired test | hours | $20–200 | final verdict |

**Why this is the design.** A contributor gets a real filming report from G1
before we spend a cent of GPU. G2 is the gate that matters commercially: if the
data shows no learnable structure against its own shuffled control *offline*, we
stop and say so, instead of burning eight hours to produce two zeros — which is
exactly what happened in the RoboTwin work. Only data that survives G2 earns G3.

Each gate renders as a row in a vertical timeline that fills in over time. That
is the "step by step" the UI currently lacks.

---

## 4. Screens

**1 · Submissions** (home)
Table of every submission: name, benchmark, status pill, verdict chip, n,
submitted-at. Filter by benchmark and status. This is the multi-customer view
that does not exist today.

**2 · New Submission** — a real four-step wizard
- **Source** — drag a zip, or connect a HuggingFace repo / S3 prefix
- **Schema** — we show what we detected; the user confirms what the channels
  *mean*. This surfaces the semantics plane, and it is where a contributor
  learns their poses are camera-frame before it costs them a run.
- **Benchmark & budget** — pick the task, then choose trials. The panel shows,
  live: *"50 trials detects a +27pt difference · 300 detects +8pt · est. $34 /
  6h."* Nobody in this space prices statistical power. It is a genuine
  differentiator and it sets expectations before the money is spent.
- **Review** — what will run, what it will cost, what it can and cannot conclude.

**3 · Submission detail** — the hero screen
Vertical gate timeline. Completed gates collapse to a one-line result; the
running gate expands with live progress (ingest → convert → train → evaluate,
with the current episode counter). Everything below the current gate is dimmed
and unopened. Nothing is shown before it exists.

**4 · Verdict** — the shareable artifact
One statement, large. Then the three-arm comparison, the ladder, the interval,
and — always — *what this does not say*. This is the screen a lab lead
screenshots into Slack. It should be designed as a document, exportable to PDF,
with a permalink.

**5 · Prescriptions**
Ranked fix list. Each item: the measurement behind it, the fix, and the expected
effect. Not generic advice — *"episodes 13 and 14 have a left arm that moved
0.66 units in 300 frames; re-film with both hands in frame."*

**6 · Compare / Leaderboard**
N submissions on one benchmark, ranked with intervals, pairwise significance,
and a compact letter display so "these two are indistinguishable" reads at a
glance. This is what P2 buys.

**7 · Methodology** (public)
Static, linkable: how the gauntlet works, why the shuffled control, what the
ladder is, why we abstain. Trust with labs is won here, not in the UI chrome.

**8 · Operator console**
Queue depth, GPU utilisation, failed jobs with logs, plugin versions per run.

---

## 5. What "not static" means concretely

- **Job states**: queued → running → passed/failed/abstained, per gate
- **Live progress**: SSE stream of gate events; the episode counter ticks
- **Skeletons** while loading; real empty states; real error states
- **Optimistic upload** with progress and resumability for large archives
- **Motion only for state change** — a gate turning green. Nothing decorative.

---

## 6. Design language

Keep the current tokens (light canvas, single blue, one typeface, tabular
numerals). They are right. What is missing is not colour, it is **state and
hierarchy over time**.

Three additions:
- **The status pill** is the most-read element in the product. One vocabulary
  everywhere: `Queued · Running · Passed · Refused · Abstained · Failed`.
- **The verdict** gets its own visual weight and its own page. It is the product.
- **Density scales with expertise**: contributor view is spacious and prose-led;
  the leaderboard is dense and tabular. Same tokens, different register.

---

## 7. Technical shape

Today: FastAPI reading local JSON, one experiment, no state.

Needed:
- **Postgres** — orgs, submissions, gates, runs, findings, artifacts
- **Object storage** — datasets, checkpoints, records (S3)
- **Job queue + GPU workers** — the gauntlet as a DAG per submission
- **SSE** for live gate progress
- **Auth** — org, members, API keys (labs will want CI submission)

**The one real fork: the frontend.**

- *(a) Stay vanilla*, restructured into modules with a small client-side store
  and router. No build step, fast, but the wizard + live state + leaderboard
  will strain it.
- *(b) React + TypeScript + TanStack Query.* Server state *is* this app —
  polling, caching, invalidation, optimistic updates are the whole job, and
  TanStack does exactly that. Adds a build step.

My recommendation is **(b)**. The current page already hit the ceiling: I have
been hand-writing DOM and hit a bug where a chained `append().style` silently
killed the renderer. That class of bug disappears with a real component layer,
and the product needs live state everywhere.

---

## 8. Phasing

**Phase 1 — the spine.** Object model, auth, submissions list, upload, G0 + G1.
A contributor can submit and get a real filming report. *No GPU required.* This
alone is a shippable product and it de-risks everything downstream.

**Phase 2 — the money gate.** G2 offline fit against the shuffled control, plus
the budget/power calculator in the wizard. Now we can decline work honestly.

**Phase 3 — the verdict.** G3 closed-loop, three arms, ladder, paired test,
verdict page, prescriptions, PDF export.

**Phase 4 — the lab product.** Leaderboard, cross-submission comparison, API,
operator console.

Phase 1 is roughly where the effort should go first, and almost none of it is
the current page.

---

## 9. What stays from today

- The design tokens and the list-table/disclosure pattern
- `/api/diagnosis` grouping and the prescription rendering
- The upload inspector (it is already G0 + half of G1)
- The rule that the page never computes its own statistics

Everything else is a viewer for one experiment and should be replaced.
