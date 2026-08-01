"""Gate 3 — Robot test. Does the robot actually get better?

The gate has two halves and they are separable on purpose.

**Producing rollouts** costs a day of GPU: convert, train the contributor's arm,
train its shuffled control, and run both closed-loop against the same scenes.

**Reading them** costs nothing, and is where every claim the product makes gets
made. :func:`assemble` is that half. It takes finished run records and returns
the verdict, so the reasoning can be built, tested and argued with against real
rollouts without a GPU anywhere near it -- and so a run that has already been
paid for can be re-read when the reasoning improves, rather than re-run.

The ladder
----------
Binary success is the metric everyone reports and it is nearly useless at the
sample sizes a robot evaluation can afford. In this project's own A/B, two
datasets differed by 20 points on two-handed coordination -- clearly, at
p=0.006 -- while their success rates, 4/50 against 0/50, could not be separated
at all (p=0.125). Same rollouts, same scenes, same money. So the rungs below
success are reported as first-class results rather than as diagnostics.

Rungs are read off whatever stage events the world emitted, not from a fixed
list. A world that reports ``object0.lifted`` gets a "lifted" rung with an
``any`` and an ``all`` reading; a world that reports nothing gets no ladder and
says so. Nothing here knows what a bottle is.

Absent is not zero, and this is where it bites
-----------------------------------------------
Rungs are measured *per arm*. An arm whose rollouts carry no stage events did
not fail those rungs -- nobody looked. The first version of this file did not
distinguish the two, and on the project's own records it reported the baseline
as reaching 0/100 on every rung and the contributor's data as sweeping all of
them at p=0.0. The baseline had simply been evaluated before the instrumentation
existed. A confident, decisive, entirely fabricated result, in the shape a real
one would arrive in. So a rung an arm never reported is ``None`` here, it is not
compared, and the report says which arms were unable to answer.

Pairing
-------
Every comparison is between arms on *the same scenes*, and only the scenes where
they disagree carry information. Scenes where both succeed or both fail are
counted in the rates and excluded from the test, which is what makes it paired
and is why it can see a difference the unpaired version cannot.

What it refuses to claim
------------------------
Without a shuffled control arm, this gate will report a *ranking* and refuse to
report an *attribution*. Fine-tuning on anything moves a model, so beating the
baseline says the model changed, not that the change came from the
contributor's footage. That refusal is the honest core of the product and it is
enforced here rather than left to whoever writes the summary.
"""

from __future__ import annotations

import json
import math
import os
import subprocess
import time
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

Report = Callable[..., None]


def _quiet(*_args, **_kwargs) -> None:
    """Default report: say nothing. A gate must run without a worker."""


ALPHA = 0.05

#: The arm names this gate understands, and what each one is for. Names rather
#: than positions, because "arm 2" in a table six weeks later is not a claim
#: anybody can check.
TREATMENT = "your data"
CONTROL = "shuffled control"
BASELINE = "baseline"


# -- reading rollouts -------------------------------------------------------


def scene_of(episode: Mapping[str, Any]) -> str | None:
    return ((episode.get("labels") or {}).get("annotations") or {}).get("scene")


def _events(episode: Mapping[str, Any]) -> set[str]:
    return {e.get("name", "") for e in (episode.get("labels") or {}).get("stage_events") or []}


def kinds_in(episodes: Iterable[Mapping[str, Any]]) -> tuple[tuple[str, ...], int]:
    """The stage kinds this world reported, and how many things it tracked.

    Both read from the events rather than configured. ``objects`` is the most
    any single episode mentioned: an object that no episode ever touched emits
    no event at all and is invisible here, which understates the denominator of
    an "all" rung rather than inventing one.
    """
    kinds: list[str] = []
    objects = 0
    for episode in episodes:
        names = _events(episode)
        objects = max(objects, len({n.split(".")[0] for n in names if "." in n}))
        for name in sorted(names):
            _, _, kind = name.partition(".")
            if kind and kind not in kinds:
                kinds.append(kind)
    return tuple(kinds), objects


def kinds_reported_by(episodes: Iterable[Mapping[str, Any]]) -> set[str]:
    """Stage kinds this *arm* ever reported.

    Per arm, not per cohort. A kind another arm reported and this one never did
    is a kind this arm cannot be scored on, and scoring it anyway is how an
    uninstrumented run comes out as a clean sweep against.
    """
    found: set[str] = set()
    for episode in episodes:
        for name in _events(episode):
            _, _, kind = name.partition(".")
            if kind:
                found.add(kind)
    return found


def rungs_of(
    episode: Mapping[str, Any],
    kinds: Sequence[str],
    objects: int,
    reported: set[str] | None = None,
) -> dict[str, bool | None]:
    """Which rungs this episode reached. ``None`` where nobody looked.

    ``all`` rungs are only offered when the world tracked more than one thing;
    with a single object "any" and "all" are the same rung under two names, and
    a table with both reads as more evidence than there is.
    """
    names = _events(episode)
    out: dict[str, bool | None] = {}
    for kind in kinds:
        if reported is not None and kind not in reported:
            out[kind] = None
            if objects > 1:
                out[f"{kind} all {objects}"] = None
            continue
        touched = {n.split(".")[0] for n in names if n.endswith("." + kind)}
        out[kind] = bool(touched)
        if objects > 1:
            out[f"{kind} all {objects}"] = len(touched) >= objects
    # Success comes from the outcome, not from the events, so it is measured
    # even for an arm that reported no stages at all.
    out["solved"] = bool((episode.get("labels") or {}).get("success"))
    return out


def ladder_order(
    kinds: Sequence[str], objects: int, reached: Mapping[str, Mapping[str, Any]] | None = None
) -> list[str]:
    """Rungs in the order they are climbed, easiest first.

    Sorted by how many scenes actually reached each rung, rather than by a list
    written here -- a hard-coded order would be wrong for the first world that
    reports something this file has never heard of, and reading the events in
    the order a set iterates gave "lifted" before "moved", which is backwards
    and made the drop between them look like a gain.

    ``solved`` is pinned last. It is the rung the whole ladder leads to, and a
    task where more scenes solve than reach some intermediate would otherwise
    reorder the story around a measurement artefact.
    """
    rungs: list[str] = []
    for kind in kinds:
        rungs.append(kind)
        if objects > 1:
            rungs.append(f"{kind} all {objects}")
    if reached:
        def climbed(rung: str) -> int:
            return sum(1 for v in reached.values() if v.get(rung))

        rungs.sort(key=lambda r: -climbed(r))
    return [*rungs, "solved"]


# -- comparing --------------------------------------------------------------


def wilson(wins: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n <= 0:
        return (0.0, 1.0)
    p = wins / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, centre - half), min(1.0, centre + half))


def sign_test(left: int, right: int) -> float:
    """Two-sided exact test on the disagreements. McNemar, computed exactly.

    Exact rather than the chi-square form, which needs more disagreements than a
    robot evaluation usually produces and is anti-conservative when it does not
    have them -- it reports significance that is not there, at exactly the
    sample size where somebody is most tempted to believe it.
    """
    n = left + right
    if n == 0:
        return 1.0
    top = max(left, right)
    tail = sum(math.comb(n, i) for i in range(top, n + 1)) / (2**n)
    return min(1.0, 2 * tail)


def compare(a: Mapping[str, dict], b: Mapping[str, dict], rung: str) -> dict:
    """One rung, two arms, on the scenes both ran and both measured.

    A scene either arm did not measure is dropped from the pair, not counted as
    a failure for that arm. If that leaves nothing, the comparison says it could
    not be made -- which is a different sentence from "no difference found" and
    must never be rendered as one.
    """
    shared = [s for s in sorted(set(a) & set(b)) if a[s].get(rung) is not None and b[s].get(rung) is not None]
    if not shared:
        return {"rung": rung, "scenes": 0, "measured": False}
    a_only = sum(1 for s in shared if a[s][rung] and not b[s][rung])
    b_only = sum(1 for s in shared if b[s][rung] and not a[s][rung])
    return {
        "rung": rung,
        "measured": True,
        "scenes": len(shared),
        "a_only": a_only,
        "b_only": b_only,
        "agreed": len(shared) - a_only - b_only,
        "p_value": round(sign_test(a_only, b_only), 4),
        "separated": sign_test(a_only, b_only) <= ALPHA,
    }


def rates(reached: Mapping[str, dict], rung: str) -> dict:
    """This arm's rate on one rung, or a statement that it was never measured."""
    values = [v.get(rung) for v in reached.values()]
    measured = [v for v in values if v is not None]
    if not measured:
        return {"measured": False, "n": 0, "unmeasured": len(values)}
    wins = sum(1 for v in measured if v)
    low, high = wilson(wins, len(measured))
    return {
        "measured": True,
        "wins": wins,
        "n": len(measured),
        "rate": round(wins / len(measured), 4),
        "ci": [round(low, 4), round(high, 4)],
        # Reported even when zero, so a rate over part of an arm is never
        # mistaken for a rate over all of it.
        "unmeasured": len(values) - len(measured),
    }


# -- the verdict ------------------------------------------------------------


def assemble(arms: Mapping[str, Sequence[Mapping[str, Any]]], report: Report = _quiet) -> dict:
    """Finished rollouts in, gate result out. No GPU, no world, no policy.

    ``arms`` maps an arm name to its episodes. :data:`TREATMENT` is required;
    :data:`CONTROL` and :data:`BASELINE` are each optional and their absence
    changes what may be concluded rather than being quietly ignored.
    """
    if TREATMENT not in arms:
        return {
            "status": "failed",
            "summary": f"no {TREATMENT!r} arm was recorded, so there is nothing to report on",
            "findings": [],
        }

    report("reading the rollouts")
    everything = [e for episodes in arms.values() for e in episodes]
    kinds, objects = kinds_in(everything)

    reached = {}
    unreported: dict[str, list[str]] = {}
    for name, episodes in arms.items():
        reported = kinds_reported_by(episodes)
        missing = [k for k in kinds if k not in reported]
        if missing:
            unreported[name] = missing
        reached[name] = {
            scene_of(e): rungs_of(e, kinds, objects, reported)
            for e in episodes
            if scene_of(e)
        }

    # Ordered by what the treatment arm actually reached, so the ladder reads
    # as the funnel it is.
    order = ladder_order(kinds, objects, reached.get(TREATMENT))

    report("scoring the ladder", total=len(order))
    ladder = []
    for index, rung in enumerate(order):
        report("scoring the ladder", current=index + 1, total=len(order), note=rung)
        row: dict[str, Any] = {"rung": rung, "arms": {}}
        for name in arms:
            row["arms"][name] = rates(reached[name], rung)
        for other in (CONTROL, BASELINE):
            if other in reached:
                row[f"vs {other}"] = compare(reached[TREATMENT], reached[other], rung)
        ladder.append(row)

    findings: list[dict] = []
    measures: dict[str, dict] = {}
    for row in ladder:
        for name, cell in row["arms"].items():
            if not cell["measured"]:
                continue
            measures[f"{row['rung']} · {name}"] = {
                "value": cell["rate"],
                "n": cell["n"],
                "ci": cell["ci"],
                "units": "fraction of scenes",
                "method": "Wilson 95%",
                "module": "robot",
            }

    # -- what may be concluded, in order of what is missing ------------------

    solved = next(r for r in ladder if r["rung"] == "solved")
    yours = solved["arms"][TREATMENT]

    if CONTROL not in reached:
        # The refusal that matters. Fine-tuning on anything moves a model, so
        # without this arm a difference against the baseline is a difference,
        # not an attribution -- and reporting it as one is a result that
        # reproduces for every contributor including the worthless ones.
        findings.append(
            {
                "code": "robot.no_control_arm",
                "severity": "strong",
                "summary": (
                    "no shuffled control was run, so any difference here cannot be "
                    "attributed to the correspondence in your footage"
                ),
                "prescription": (
                    "Run the shuffled control: the same clips, the same actions, the same "
                    "number of gradient steps, with each clip's actions belonging to a "
                    "different clip. Fine-tuning on anything moves a model, so beating the "
                    "baseline only shows the model changed. Beating the shuffle is what shows "
                    "the change came from what your camera saw."
                ),
                "module": "robot",
            }
        )

    if unreported:
        # Named, and at strong severity. This is the difference between a
        # comparison and a fabrication, and it is invisible in every table.
        findings.append(
            {
                "code": "robot.rungs_unmeasured",
                "severity": "strong",
                "summary": (
                    "; ".join(
                        f"the {name!r} arm reported no {', '.join(kinds)} events"
                        for name, kinds in unreported.items()
                    )
                    + ". Those rungs are unmeasured for it, not failed"
                ),
                "prescription": (
                    "Re-run that arm with the same instrumentation as the others, or compare "
                    "only on the rungs both measured. An arm evaluated before the stage "
                    "events existed scores zero on every rung, which reads as a total loss "
                    "and is really an absence of measurement."
                ),
                "module": "robot",
            }
        )

    separated = [r for r in ladder if r.get(f"vs {BASELINE}", {}).get("separated")]
    if BASELINE in reached and solved.get(f"vs {BASELINE}", {}).get("measured"):
        against = solved[f"vs {BASELINE}"]
        base = solved["arms"][BASELINE]
        if against["separated"]:
            direction = "more" if against["a_only"] > against["b_only"] else "fewer"
            summary = (
                f"solved {yours['wins']}/{yours['n']} scenes, {direction} than the "
                f"baseline's {base['wins']}/{base['n']} (p={against['p_value']})"
            )
        else:
            summary = (
                f"solved {yours['wins']}/{yours['n']} scenes against the baseline's "
                f"{base['wins']}/{base['n']}, which these {against['scenes']} paired "
                f"scenes cannot separate (p={against['p_value']})"
            )
            findings.append(
                {
                    "code": "robot.not_separated_on_success",
                    "severity": "info",
                    "summary": (
                        f"success rate did not separate from the baseline on "
                        f"{against['scenes']} paired scenes"
                    ),
                    "prescription": (
                        "Success is the coarsest rung and the hardest to move. Look at the "
                        "rungs below it: they are measured on the same rollouts at no extra "
                        "cost, and they separate at sample sizes success never will."
                    ),
                    "module": "robot",
                }
            )
    else:
        summary = f"solved {yours['wins']}/{yours['n']} scenes, with nothing to compare against"

    if separated:
        names = ", ".join(r["rung"] for r in separated if r["rung"] != "solved")
        if names:
            summary += f" · separated from the baseline on {names}"
            findings.append(
                {
                    "code": "robot.separated_below_success",
                    "severity": "info",
                    "summary": f"the ladder separates from the baseline at: {names}",
                    "prescription": (
                        "This is where your data changed the behaviour. The rung a policy "
                        "stops at is what its training data did not demonstrate, so this "
                        "names the capability to film next."
                    ),
                    "module": "robot",
                }
            )

    # Where the treatment arm loses most of its scenes: the rung with the
    # largest drop is the one to film for, and it is a fact about the data
    # rather than an opinion about the policy.
    drops = []
    for before, after in zip(ladder, ladder[1:]):
        a, b = before["arms"][TREATMENT], after["arms"][TREATMENT]
        if not (a["measured"] and b["measured"]):
            continue
        drops.append((a["rate"] - b["rate"], before["rung"], after["rung"]))
    if drops:
        size, before_rung, after_rung = max(drops)
        if size > 0.2:
            findings.append(
                {
                    "code": "robot.bottleneck",
                    "severity": "strong",
                    "summary": (
                        f"{size:.0%} of scenes are lost between '{before_rung}' and "
                        f"'{after_rung}', which makes that step the bottleneck"
                    ),
                    "prescription": (
                        f"The policy gets to '{before_rung}' and stops. Footage that "
                        f"demonstrates '{after_rung}' clearly and repeatedly is what would "
                        "move this; more clips of the part it already does will not."
                    ),
                    "module": "robot",
                }
            )

    return {
        "status": "passed",
        "summary": summary,
        "findings": findings,
        "measures": measures,
        "detail": {
            "ladder": ladder,
            "order": order,
            "arms": list(arms),
            "objects_tracked": objects,
            # Per scene, per rung, for the treatment arm. Kept because ranking
            # two submissions against each other is a *paired* comparison on the
            # scenes they both ran, and aggregates cannot be re-paired: two
            # arms at 8% might agree on every scene or disagree on sixteen, and
            # only the second is evidence. Stored once here rather than
            # recomputed from run records the leaderboard would have to reopen.
            "reached": {
                scene: {rung: value for rung, value in rungs.items()}
                for scene, rungs in reached.get(TREATMENT, {}).items()
            },
            "unreported": unreported,
            "has_control": CONTROL in reached,
            "has_baseline": BASELINE in reached,
            "alpha": ALPHA,
        },
    }


def from_records(paths: Mapping[str, Path], report: Report = _quiet) -> dict:
    """Assemble a verdict from run records already on disk."""
    arms = {}
    for name, path in paths.items():
        report("reading", note=str(path.name))
        arms[name] = json.loads(Path(path).read_text())["episodes"]
    return assemble(arms, report)


#: Where a submission's finished rollouts live, and the manifest naming which
#: arm each file is. Written by whatever produced them -- a GPU worker today, a
#: re-read of an earlier paid run tomorrow.
ROLLOUTS = "rollouts"
MANIFEST = "arms.json"

#: The executable that turns a dataset into rollouts, if this worker has one.
#: Set by the deployment, not by this file.
#:
#: A seam rather than an import, and deliberately so. Producing rollouts means a
#: trainer, a checkpoint format, a policy server and a simulator -- four choices
#: this gate must not make, because the moment it imports one of them the
#: framework has a favourite model and every claim about being model-agnostic
#: becomes marketing. The gate says what it needs; a runner says how. The
#: contract is a JSON job in and run records out, which is the same shape the
#: rest of this repo's backends already use.
RUNNER = os.environ.get("BENCH_RUNNER", "")

#: How long to wait for a runner before giving up on it. A full two-arm run is
#: hours; this is a ceiling on hanging, not a budget.
RUNNER_TIMEOUT = float(os.environ.get("BENCH_RUNNER_TIMEOUT", 12 * 3600))


def produce(job: dict, folder: Path, report: Report) -> tuple[bool, str]:
    """Run the configured runner until it has written rollouts. Returns (ok, why).

    Progress is read from a file the runner appends to rather than from its
    stdout: the runner drives a trainer and a simulator that both write freely
    to stdout, and picking progress out of that would mean parsing somebody
    else's log format and re-parsing it whenever they changed it. A file the
    runner owns is a contract; scraping is a guess.
    """
    if not RUNNER:
        return False, (
            "no rollouts exist for this submission, and this worker has no runner "
            "configured to produce them. Set BENCH_RUNNER to something that can train "
            "and evaluate"
        )
    runner = Path(RUNNER)
    if not runner.exists():
        return False, f"the configured runner does not exist: {runner}"

    folder.mkdir(parents=True, exist_ok=True)
    spec = folder / "job.json"
    ticks = folder / "progress.jsonl"
    ticks.write_text("")
    spec.write_text(json.dumps({**job, "out": str(folder), "progress": str(ticks)}, indent=2))

    report("starting the run", note=runner.name)
    process = subprocess.Popen(
        [str(runner), str(spec)],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    seen = 0
    started = time.monotonic()
    while process.poll() is None:
        time.sleep(2.0)
        if time.monotonic() - started > RUNNER_TIMEOUT:
            process.kill()
            return False, f"the run passed {RUNNER_TIMEOUT / 3600:.0f}h without finishing"
        try:
            lines = ticks.read_text().splitlines()
        except OSError:
            continue
        for line in lines[seen:]:
            seen += 1
            try:
                at = json.loads(line)
            except ValueError:
                continue
            report(
                at.get("phase", "running"),
                current=at.get("current"),
                total=at.get("total"),
                note=at.get("note", ""),
            )

    if process.returncode != 0:
        tail = (process.stdout.read() or "").strip().splitlines()[-8:]
        return False, "the runner failed: " + (" / ".join(tail) or f"exit {process.returncode}")
    if not (folder / MANIFEST).exists():
        return False, "the runner finished without writing a manifest of what it produced"
    return True, ""


def run(
    archive: Path,
    workdir: Path,
    report: Report = _quiet,
    params: Mapping[str, Any] | None = None,
) -> dict:
    """The gate contract.

    Rollouts already present are read; otherwise they are produced, which is
    what the configured runner is for. Either way the reading is the same code,
    so a run that has already been paid for can be re-read when the reasoning
    improves rather than re-run.

    ``params["trials"]`` is how many scenes the user bought. It decides what the
    run can conclude, so a rollout set that does not match it is reported rather
    than quietly accepted: charging for two hundred scenes and reporting fifty
    is the same failure as running fifty and calling it two hundred.
    """
    trials = int((params or {}).get("trials") or 0)
    folder = workdir / ROLLOUTS
    manifest = folder / MANIFEST
    if not manifest.exists():
        root = next((workdir / "unpacked").rglob("meta/info.json"), None)
        ok, why = produce(
            {
                "dataset": str(root.parents[1]) if root else "",
                "trials": trials,
                "task": str((params or {}).get("task") or ""),
                "arms": [TREATMENT, CONTROL],
                "baseline": BASELINE,
            },
            folder,
            report,
        )
        if not ok:
            # `failed`, not `refused`. Our machinery could not produce the
            # evidence; nothing about the contributor's data was judged, and
            # saying otherwise would bill somebody for our gap.
            return {"status": "failed", "summary": why, "findings": []}

    report("reading the manifest")
    named = json.loads(manifest.read_text())
    bought = int((params or {}).get("trials") or 0)
    missing = [name for name, file in named.items() if not (folder / file).exists()]
    if missing:
        return {
            "status": "failed",
            "summary": f"the manifest names arms with no rollouts on disk: {', '.join(missing)}",
            "findings": [],
        }
    out = from_records({name: folder / file for name, file in named.items()}, report)

    if trials:
        ran = max(
            (row["arms"].get(TREATMENT, {}).get("n", 0) for row in out.get("detail", {}).get("ladder", [])),
            default=0,
        )
        out.setdefault("detail", {})["trials_bought"] = trials
        out["detail"]["trials_run"] = ran
        if ran < trials:
            out.setdefault("findings", []).insert(
                0,
                {
                    "code": "robot.short_of_budget",
                    "severity": "strong",
                    "summary": (
                        f"{trials} scenes were bought and {ran} were run, so this says less "
                        "than it was paid to say"
                    ),
                    "prescription": (
                        "Every conclusion here was sized against the number of scenes "
                        f"bought. Run the rest, or read this against what {ran} scenes can "
                        f"actually separate, which is a larger difference than {trials} "
                        "would have been."
                    ),
                    "module": "robot",
                },
            )
    return out
