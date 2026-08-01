"""Gate 1 — Data report. What is this footage like?

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
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

#: What a gate is handed to say where it is. See ``intake.Report``.
Report = Callable[..., None]


def _quiet(*_args, **_kwargs) -> None:
    """Default report: say nothing. A gate must run without a worker."""


#: Modules are routed by what they declare they need, not by name. ``outcomes``
#: or ``stage_events`` means the module reads rollouts, which an upload has
#: none of.
OUTCOME_CAPS = ("outcomes", "stage_events")


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


def run(archive: Path, workdir: Path, report: Report = _quiet) -> dict:
    """The gate contract. Intake already unpacked; read what it left."""
    report("finding the dataset")
    root = next((workdir / "unpacked").rglob("meta/info.json"), None)
    if root is None:
        return {
            "status": "failed",
            "summary": "the unpacked dataset is missing — intake's output was not where this gate expected it",
            "findings": [],
        }

    report("opening the clips")
    cohort = _cohort(root.parents[1], "your data")
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
                    abstained.append({"module": name, "reason": verdict.explain()})
                    continue
            # ``said``, not ``report``: ``report`` is the progress callable this
            # gate was handed, and rebinding it here meant the second module in
            # the loop called a dataclass. Two different things called the same
            # word, one of them the gate contract.
            said = module.analyse([cohort])
        except Exception as error:  # noqa: BLE001 - one module, not the gate
            abstained.append({"module": name, "reason": f"{type(error).__name__}: {error}"})
            continue

        for finding in said.findings:
            findings.append(
                {
                    "code": finding.code,
                    "severity": finding.severity,
                    "summary": finding.summary,
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
            abstained.append({"module": name, "reason": note})

    # G1 never refuses. Its job is to describe; whether the footage is worth
    # spending on is the next gate's question, and a strong finding here is
    # advice rather than a door closing.
    strong = [f for f in findings if f["severity"] == "strong"]
    if strong:
        summary = f"{len(findings)} finding(s), {len(strong)} worth fixing before you refilm"
    elif findings:
        summary = f"{len(findings)} finding(s), none blocking"
    else:
        summary = "nothing to flag in the footage"
    summary += f" · {len(cohort.episodes)} clip(s) read by {len(modules)} module(s)"

    return {
        "status": "passed",
        "summary": summary,
        "findings": findings,
        "measures": measures,
        "abstained": abstained,
    }
