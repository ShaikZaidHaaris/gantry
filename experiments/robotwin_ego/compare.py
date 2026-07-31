"""The two arms, side by side, judged by the feedback layer rather than by eye.

`feedback_control` has been refusing to answer since it was written, because no
evaluator could execute what the ego policy emits and it will not read an
absent outcome as a failure. This is the first time it has both arms.

It is handed the records, not the summary numbers. Whether the difference
between two success rates means anything depends on how many trials there were
and how they were paired, and a module that only ever sees "0.4 and 0.2" cannot
tell 2-of-5 from 200-of-500.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from gantry_feedback_control import Control

from gantry.contracts.feedback import Cohort
from gantry.store import read_run

TASK = sys.argv[1] if len(sys.argv) > 1 else "pick_dual_bottles"
HERE = Path("/home/ubuntu/egorun")


def path_for(arm: str) -> Path:
    # write_run appends .json and drops a .npz sidecar beside it.
    return HERE / f"robotwin_run_{arm}_{TASK}.json"


def load(arm: str) -> Cohort:
    record = read_run(path_for(arm))
    return Cohort(name=arm, episodes=tuple(record.episodes), metadata={"task": TASK})


def main() -> None:
    cohorts = []
    for arm in ("ego", "shuffled"):
        path = path_for(arm)
        if not path.exists():
            print(f"missing {arm}: {path}")
            continue
        cohorts.append(load(arm))

    if len(cohorts) < 2:
        raise SystemExit("both arms are needed; the control is the whole point")

    report = Control().analyse(cohorts)

    print("=" * 74)
    print(f"  ego vs shuffled control, closed-loop in RoboTwin: {TASK}")
    print("=" * 74)
    for arm in cohorts:
        outcomes = [o for o in arm.outcomes if o is not None]
        print(f"  {arm.name:10s} {sum(outcomes)}/{len(outcomes)} trials")
    print()
    for name, measurement in sorted(report.measurements.items()):
        print(f"  {name:34s} {measurement}")
    print()
    for finding in report.findings:
        print(f"  [{finding.severity}] {finding.code}")
        print(f"      {finding.summary}")
        if finding.prescription:
            print(f"      -> {finding.prescription}")
        print()

    payload = {
        "task": TASK,
        "arms": {
            arm.name: {
                "successes": sum(1 for o in arm.outcomes if o),
                "scored": sum(1 for o in arm.outcomes if o is not None),
            }
            for arm in cohorts
        },
        "measurements": {k: str(v) for k, v in report.measurements.items()},
        "findings": [
            {"code": f.code, "severity": f.severity, "summary": f.summary}
            for f in report.findings
        ],
    }
    (HERE / f"robotwin_verdict_{TASK}.json").write_text(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
