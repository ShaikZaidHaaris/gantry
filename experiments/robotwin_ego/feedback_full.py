"""The whole feedback layer, over both halves of what a run produces.

A contributor's question is "did my data help, and what should I film
differently". Answering it needs two different kinds of record and they are not
interchangeable:

* the **data** side — the episodes as the ingest produced them, carrying what
  the estimator, the retargeter and the assembly measured. This is where
  "your hands were out of shot 31% of the time" lives.
* the **outcome** side — the closed-loop rollouts, carrying successes and
  stages. This is where "it did not beat its own shuffled control" lives.

Each module is offered the side it declares it needs, not both. Offering both
looked harmless and was not: RoboTwin's demonstrations carry ``success=True``
meaning "this is a demonstration", and a module that reads outcomes was handed
them as though they were rollouts -- reporting two *training sets* as having
"solved 100%" and agreed on every scene. A training set has no outcome to
compare. The modules already declare ``capabilities={"outcomes": True}``, so the
routing asks them rather than guessing.

A module that refuses is recorded as an abstention with its reason, never
dropped: a report that silently omits what it could not judge reads as a clean
bill of health, which is the single most expensive way this could be wrong.
"""

from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path

from gantry.contracts.feedback import Cohort
from gantry.store import read_run

HERE = Path("/home/ubuntu/egorun")
LEROBOT = Path("/home/ubuntu/.cache/huggingface/lerobot/gantry")
TASK = sys.argv[1] if len(sys.argv) > 1 else "pick_dual_bottles"

#: Datasets to read as the data side, by the arm name the modules expect.
#: feedback_control looks for exactly these names and will not guess.
DATASETS = {
    "ego": "ego",
    "shuffled": "shuffled",
    "two_handed": "rt_two_handed",
    "one_handed": "rt_one_handed",
}

#: Eval records to read as the outcome side.
OUTCOMES = {
    "base": "abl_run_rt_base100_pick_dual_bottles",
    "ego": "robotwin_run_ego_pick_dual_bottles",
    "shuffled": "robotwin_run_shuffled_pick_dual_bottles",
    "two_handed": "abl_run_rt_two_handed_pick_dual_bottles",
    "one_handed": "abl_run_rt_one_handed_pick_dual_bottles",
}


def modules():
    """Every installed feedback module, found by entry point.

    Loaded rather than listed, so this file names none of them and one added
    later is included without editing anything here.
    """
    from importlib.metadata import entry_points

    found = []
    for entry in sorted(entry_points(group="gantry.feedback"), key=lambda e: e.name):
        try:
            found.append((entry.name, entry.load()()))
        except Exception as error:  # noqa: BLE001
            print(f"  ! {entry.name} would not load: {error}", flush=True)
    return found


def data_cohorts() -> list[Cohort]:
    from gantry_connector_lerobot import LeRobotConnector

    out = []
    for arm, folder in DATASETS.items():
        path = LEROBOT / folder
        if not path.exists():
            continue
        try:
            connector = LeRobotConnector(str(path))
            episodes = tuple(connector.open(i) for i in connector.episode_ids())
        except Exception as error:  # noqa: BLE001
            print(f"  ! {arm} dataset unreadable: {error}", flush=True)
            continue
        out.append(Cohort(name=arm, episodes=episodes, metadata={"side": "data"}))
        print(f"  data    {arm:12s} {len(episodes)} episodes", flush=True)
    return out


def outcome_cohorts() -> list[Cohort]:
    out = []
    for arm, stem in OUTCOMES.items():
        path = HERE / f"{stem}.json"
        if not path.exists():
            continue
        try:
            record = read_run(path)
        except Exception as error:  # noqa: BLE001
            print(f"  ! {arm} record unreadable: {error}", flush=True)
            continue
        out.append(
            Cohort(name=arm, episodes=tuple(record.episodes), metadata={"side": "outcome", "task": TASK})
        )
        scored = [o for o in out[-1].outcomes if o is not None]
        print(f"  outcome {arm:12s} {sum(scored)}/{len(scored)}", flush=True)
    return out


def side_for(module) -> str:
    """Which half of the record this module is asking for.

    ``outcomes`` or ``stage_events`` mean it reads rollouts; anything else reads
    the data as ingested. Read off the module's own requirement so a module
    added later is routed without editing this file.
    """
    try:
        caps = dict(getattr(module.requirement(), "capabilities", {}) or {})
    except Exception:  # noqa: BLE001 - a module that cannot say gets the data side
        return "data"
    return "outcome" if (caps.get("outcomes") or caps.get("stage_events")) else "data"


def run(name, module, cohorts, side):
    """One module against one side, keeping the refusal if it refuses."""
    if not cohorts:
        return None
    try:
        check = getattr(module, "check_inputs", None)
        if check is not None:
            verdict = check(cohorts)
            if not verdict.ok:
                return {
                    "module": name,
                    "side": side,
                    "abstained": True,
                    "reason": verdict.explain(),
                    "codes": [r.code for r in verdict.reasons],
                }
        report = module.analyse(cohorts)
    except Exception as error:  # noqa: BLE001 - one module, not the run
        return {
            "module": name,
            "side": side,
            "abstained": True,
            "reason": f"{type(error).__name__}: {error}",
            "codes": ["module.raised"],
            "traceback": traceback.format_exc(limit=3),
        }
    return {
        "module": name,
        "side": side,
        "abstained": False,
        "measurements": {k: str(v) for k, v in report.measurements.items()},
        "findings": [
            {
                "code": f.code,
                "severity": f.severity,
                "summary": f.summary,
                "prescription": getattr(f, "prescription", None),
            }
            for f in report.findings
        ],
    }


def main() -> None:
    print("reading cohorts", flush=True)
    data = data_cohorts()
    outcomes = outcome_cohorts()

    results = []
    for name, module in modules():
        side = side_for(module)
        cohorts = outcomes if side == "outcome" else data
        got = run(name, module, cohorts, side)
        if got is not None:
            results.append(got)
            mark = "abstained" if got["abstained"] else f"{len(got['findings'])} finding(s)"
            print(f"  {name:12s} [{side:7s}] {mark}", flush=True)

    payload = {
        "task": TASK,
        "cohorts": {
            "data": [{"arm": c.name, "episodes": len(c.episodes)} for c in data],
            "outcome": [
                {
                    "arm": c.name,
                    "wins": sum(1 for o in c.outcomes if o),
                    "scored": sum(1 for o in c.outcomes if o is not None),
                }
                for c in outcomes
            ],
        },
        "results": results,
    }
    out = HERE / f"feedback_full_{TASK}.json"
    out.write_text(json.dumps(payload, indent=2))
    print(f"\nwrote {out}", flush=True)

    spoke = [r for r in results if not r["abstained"]]
    quiet = [r for r in results if r["abstained"]]
    print(f"{len(spoke)} module-run(s) produced findings, {len(quiet)} abstained")


if __name__ == "__main__":
    main()
