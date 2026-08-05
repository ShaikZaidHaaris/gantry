"""Every feedback module over the three arms, assembled into one report.

Not just the control. The control answers "did the data carry information";
the rest answer whether that answer can be trusted and what to do about it —
how many trials it took, whether the arms were separated at all, what the
licences allow, where the extraction lost frames, how the filming could be
better. A single verdict with none of that around it is a number without an
error bar and without a next step.

Modules that have nothing to say abstain, and an abstention is reported as one
rather than dropped. A report that silently omits what it could not judge reads
as a clean bill of health.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from gantry_feedback_report import as_markdown, assemble

from gantry.contracts.feedback import Cohort
from gantry.store import read_run

TASK = sys.argv[1] if len(sys.argv) > 1 else "pick_dual_bottles"
HERE = Path("/home/ubuntu/egorun")

#: Which training set each arm read, and what it plays in the comparison. The
#: names on the left are what feedback_control looks for: it wants a treatment,
#: a control, and optionally a baseline, and it will not guess which is which.
ARMS = {
    "two_handed": "rt_two_handed",
    "one_handed": "rt_one_handed",
    "two_handed_shuffled": "rt_two_handed_shuf",
    "one_handed_shuffled": "rt_one_handed_shuf",
    "base": "rt_base",
}


def modules():
    """Every feedback module installed, found by entry point.

    Loaded rather than listed, so this file names none of them and a module
    added later is included without editing anything here.
    """
    from importlib.metadata import entry_points

    found = []
    for entry in entry_points(group="gantry.feedback"):
        try:
            found.append((entry.name, entry.load()()))
        except Exception as error:  # noqa: BLE001
            print(f"  ! {entry.name} would not load: {error}")
    return sorted(found, key=lambda pair: pair[0])


def cohorts() -> list[Cohort]:
    out = []
    for name, dataset in ARMS.items():
        path = HERE / f"abl_run_{dataset}_{TASK}.json"
        if not path.exists():
            print(f"  ! no record for {name} ({path.name})")
            continue
        record = read_run(path)
        out.append(
            Cohort(
                name=name,
                episodes=tuple(record.episodes),
                metadata={"task": TASK, "dataset": dataset},
            )
        )
    return out


def main() -> None:
    arms = cohorts()
    if not arms:
        raise SystemExit("no records to read")

    print("=" * 74)
    print(f"  A vs B: two contributor cohorts, closed-loop in RoboTwin: {TASK}")
    print("=" * 74)
    for arm in arms:
        scored = [o for o in arm.outcomes if o is not None]
        print(f"  {arm.name:10s} {sum(scored):2d}/{len(scored):2d}   ({arm.metadata['dataset']})")
    print()

    reports = []
    for name, module in modules():
        try:
            report = module.analyse(arms)
        except Exception as error:  # noqa: BLE001 - one module, not the run
            print(f"  ! {name} raised: {type(error).__name__}: {error}")
            continue
        reports.append(report)

    assembled = assemble(reports, dataset=f"two ego cohorts split on hand visibility, against {TASK}")
    print(as_markdown(assembled))

    (HERE / f"ab_report_{TASK}.md").write_text(as_markdown(assembled))
    (HERE / f"ab_report_{TASK}.json").write_text(
        json.dumps(
            {
                "task": TASK,
                "arms": {
                    arm.name: {
                        "dataset": arm.metadata["dataset"],
                        "successes": sum(1 for o in arm.outcomes if o),
                        "scored": sum(1 for o in arm.outcomes if o is not None),
                    }
                    for arm in arms
                },
                "modules": [
                    {
                        "module": r.module,
                        "measurements": {k: str(v) for k, v in r.measurements.items()},
                        "findings": [
                            {"code": f.code, "severity": f.severity, "summary": f.summary}
                            for f in r.findings
                        ],
                    }
                    for r in reports
                ],
            },
            indent=2,
        )
    )
    print(f"\nwritten to {HERE}/ab_report_{TASK}.md")


if __name__ == "__main__":
    main()
