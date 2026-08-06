"""Gate 1 -- Data report. What is this footage like?

Free, about a minute, and it earns trust: a contributor gets a real report on
their filming before anything paid is offered, and everything in it is a
measurement something took rather than an opinion.

This file is deliberately almost empty. It finds the installed feedback modules
by entry point, offers each one the dataset, and passes on what they say. It
names no module and knows what none of them measure -- a module added to the
pipeline appears in the product without an edit here, and nothing in the product
can quietly become the module that matters. When the two-handedness measure was
first written it lived in this file; it is now
``gantry-feedback-stillness``, general over any embodiment, and this gate did
not need to learn about it.

Two rules it inherits from the pipeline and must not break:

* **Absent is not zero.** A module that declines is recorded with its reason,
  never dropped. A report that silently omits what it could not judge reads as
  a clean bill of health, which is the most expensive way this could be wrong.
* **Only the data side.** Modules that read outcomes are not offered a training
  set. A dataset has no rollouts, and a module handed demonstrations as though
  they were attempts reports nonsense with a straight face -- two *training
  sets* once came back as having "solved 100%".
* **A module that is not installed is a gap, not a silence.** This gate reports
  what the modules it found had to say. A worker with fewer of them installed
  produces a shorter report on the same footage, and a shorter report reads as
  a cleaner dataset. That happened here: a GPU worker with two feedback plugins
  returned "1 finding, none blocking" on a dataset that produced five findings
  on a machine with six. Nothing was wrong with the data and nothing said so.

  So the inventory is recorded on every report, and where a deployment declares
  what it expects, anything missing is named at strong severity and listed
  among the checks that could not run. Absent is not zero, applied to the
  checker rather than to the data.
"""

from __future__ import annotations

import os

from pathlib import Path
from typing import Any, Callable, Mapping

from gantry.spine import count_of, plural, readable, without_codes

#: What a gate is handed to say where it is. See ``intake.Report``.
Report = Callable[..., None]


def _quiet(*_args, **_kwargs) -> None:
    """Default report: say nothing. A gate must run without a worker."""


#: What each check is called on the page. Module names are package names, chosen
#: by whoever wrote the plugin and meaningful to them; a contributor reading
#: "extraction" has to guess. A module with no entry here keeps its own name,
#: which is the safe default rather than a good one, and adding a line is the fix.
LABELS = {
    "capture": "Filming",
    "extraction": "Hand tracking",
    "provenance": "Licensing",
    "coverage": "Task coverage",
    "stillness": "Motion",
    "screen": "Cohort screening",
    "control": "Shuffled control",
    "compare": "Comparison",
    "power": "Sample size",
    "protocol": "Protocol",
    "calibrate": "Judge agreement",
    "rank": "Ranking",
    "verify": "Split checking",
    "funnel": "Stage funnel",
    "harden": "Robustness",
    "attribution": "Attribution",
}


def label(module: str) -> str:
    return LABELS.get(module, readable(module))


#: What the single cohort in a data report is called. Modules prefix their
#: sentences with the cohort name because most of them compare two or more, and
#: a report on one upload then reads "your data: your data did X" once the page
#: has already said whose data it is.
COHORT = "your data"


def plainly(text: str) -> str:
    """One line of internal explanation, as a sentence for the page.

    Modules name their own checks when they explain themselves, and a verdict
    prefixes its code in brackets. That is right for a log and wrong here: a
    reader who sees `capture.hands_offscreen` in the middle of a sentence learns
    that they are reading machine output, and stops trusting the rest of the
    page to be written for them. The codes stay in the record next to this, so
    nothing is lost.
    """
    out = without_codes(text)
    for prefix in (f"{COHORT}: ", f"{COHORT.capitalize()}: "):
        if out.startswith(prefix):
            out = out[len(prefix):]
    if out[:1].islower():
        out = out[0].upper() + out[1:]
    return out.rstrip(". ") + "." if out and not out.endswith((".", "!", "?")) else out


#: Modules are routed by what they declare they need, not by name. ``outcomes``
#: or ``stage_events`` means the module reads rollouts, which an upload has
#: none of.
OUTCOME_CAPS = ("outcomes", "stage_events")

#: What this deployment expects to be installed, comma-separated. Unset means
#: the inventory is still recorded but nothing is called missing -- a developer
#: with a partial checkout should not be told their laptop is broken, and a
#: production worker should never quietly run with half its checks.
REQUIRED = tuple(
    name.strip() for name in os.environ.get("BENCH_REQUIRED_MODULES", "").split(",") if name.strip()
)


def inventory() -> dict[str, str]:
    """Every installed feedback module and its version, whatever side it reads.

    The whole set, not just the ones this gate uses: a report that lists two
    modules is ambiguous between "two exist" and "two applied", and the
    difference is exactly what makes drift invisible.
    """
    from importlib.metadata import entry_points, version

    found: dict[str, str] = {}
    for entry in sorted(entry_points(group="gantry.feedback"), key=lambda e: e.name):
        try:
            found[entry.name] = version(entry.dist.name) if entry.dist else "?"
        except Exception:  # noqa: BLE001 - a module that will not say is still present
            found[entry.name] = "?"
    return found


def _cohort(root: Path, name: str):
    from gantry_connector_lerobot import LeRobotConnector

    from gantry.contracts.feedback import Cohort

    connector = LeRobotConnector(str(root))
    episodes = tuple(connector.open(i) for i in connector.episode_ids())
    return Cohort(name=name, episodes=episodes, metadata={"side": "data"})


def data_side_modules():
    """Every installed feedback module that reads data rather than rollouts."""
    from importlib.metadata import entry_points

    found = []
    for entry in sorted(entry_points(group="gantry.feedback"), key=lambda e: e.name):
        try:
            module = entry.load()()
            caps = dict(getattr(module.requirement(), "capabilities", {}) or {})
        except Exception:  # noqa: BLE001 - a module we cannot construct is skipped
            continue
        if any(caps.get(cap) for cap in OUTCOME_CAPS):
            continue
        found.append((entry.name, module))
    return found


def run(
    archive: Path,
    workdir: Path,
    report: Report = _quiet,
    params: Mapping[str, Any] | None = None,
) -> dict:
    """The gate contract. Intake already unpacked; read what it left."""
    report("finding the dataset")
    root = next((workdir / "unpacked").rglob("meta/info.json"), None)
    if root is None:
        return {
            "status": "failed",
            "summary": "the unpacked dataset is missing. Intake did not leave it where this gate looks for it",
            "findings": [],
        }

    report("opening the clips")
    # A ConfigError here is the connector judging the DATA -- a version it
    # does not read, a schema it refuses -- and its message is already a
    # sentence written for a person. Let it through as a refusal; the bare
    # exception used to fall to the runner's catch-all and be reported as our
    # machinery breaking, on a dataset whose problem had a fix.
    from gantry.errors import ConfigError

    try:
        cohort = _cohort(root.parents[1], COHORT)
    except ConfigError as error:
        return {
            "status": "refused",
            "summary": str(error),
            "findings": [{"code": "report.unreadable", "severity": "strong",
                          "summary": str(error), "prescription": None}],
        }
    findings: list[dict] = []
    measures: dict[str, dict] = {}
    abstained: list[dict] = []

    modules = data_side_modules()
    for index, (name, module) in enumerate(modules):
        # Named, not just counted. A module that hangs is findable this way and
        # invisible the other, and on a large upload the slow one is the whole
        # question.
        report("running checks", current=index, total=len(modules), note=name)
        try:
            check = getattr(module, "check_inputs", None)
            if check is not None:
                verdict = check([cohort])
                if not verdict.ok:
                    # The reasons themselves, not ``explain()``. That method
                    # formats for a log: it opens with "refused" and prefixes
                    # each code in brackets, which on a page reads as a stack
                    # trace with a sentence in it.
                    abstained.append({
                        "module": label(name),
                        "reason": " ".join(plainly(r.message) for r in verdict.reasons),
                        "codes": [r.code for r in verdict.reasons],
                    })
                    continue
            # ``said``, not ``report``: ``report`` is the progress callable this
            # gate was handed, and rebinding it here meant the second module in
            # the loop called a dataclass. Two different things called the same
            # word, one of them the gate contract.
            said = module.analyse([cohort])
        except Exception as error:  # noqa: BLE001 - one module, not the gate
            abstained.append({
                "module": label(name),
                "reason": "This check could not run on your data. That is our fault, not yours.",
                "codes": [f"{type(error).__name__}: {error}"],
            })
            continue

        for finding in said.findings:
            findings.append(
                {
                    "code": finding.code,
                    "severity": finding.severity,
                    "summary": plainly(finding.summary),
                    "prescription": getattr(finding, "prescription", None),
                    "evidence": dict(getattr(finding, "evidence", {}) or {}),
                    "module": name,
                }
            )
        # Kept apart from findings: a measurement is not a complaint, and a
        # module that measured a lot while flagging nothing has said something.
        for key, value in said.measurements.items():
            measures[key] = {**value.as_dict(), "module": name}
        for note in getattr(said, "notes", ()) or ():
            abstained.append({"module": label(name), "reason": plainly(note), "codes": []})

    installed = inventory()
    missing = [name for name in REQUIRED if name not in installed]
    if missing:
        findings.append(
            {
                "code": "report.modules_missing",
                "severity": "strong",
                "summary": (
                    f"{count_of(len(missing), 'check')} this deployment expects "
                    f"{plural(len(missing), 'is', 'are')} not installed on the machine that "
                    f"ran it: {', '.join(label(m) for m in missing)}"
                ),
                "prescription": (
                    "Install them and run the report again. Until then this report is shorter "
                    "than it should be, and a short report is easy to mistake for a clean "
                    "one. Nothing here says anything about the footage those checks would have "
                    "looked at."
                ),
                "module": "report",
            }
        )
        for name in missing:
            abstained.append(
                {
                    "module": label(name),
                    "reason": "This check is not installed on the machine that ran the report.",
                    "codes": [name],
                }
            )

    # G1 never refuses. Its job is to describe; whether the footage is worth
    # spending on is the next gate's question, and a strong finding here is
    # advice rather than a door closing.
    strong = [f for f in findings if f["severity"] == "strong"]
    if strong:
        summary = f"{count_of(len(findings), 'finding')}, {len(strong)} worth fixing before you refilm"
    elif findings:
        summary = f"{count_of(len(findings), 'finding')}, none blocking"
    else:
        summary = "nothing to flag in the footage"
    summary += (
        f" · {count_of(len(cohort.episodes), 'clip')} read by "
        f"{count_of(len(modules), 'check')}"
    )

    return {
        "status": "passed",
        "summary": summary,
        "findings": findings,
        "measures": measures,
        "abstained": abstained,
        # Recorded on every report, configured or not. Two runs of the same
        # dataset that disagree should be explainable from the record rather
        # than from remembering which machine each ran on.
        "detail": {
            "modules_installed": installed,
            "modules_applied": [name for name, _ in modules],
            "modules_required": list(REQUIRED),
            "modules_missing": missing,
        },
    }
