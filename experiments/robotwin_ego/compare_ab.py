"""Dataset A against dataset B, on the ladder and paired scene by scene.

Both arms faced the same fifty screened arrangements, so most scenes tell you
nothing — the easy ones both manage and the hard ones both fail. Only the
disagreements carry information, which is why this is paired rather than a
comparison of two marginal rates.

The ladder, not the binary
--------------------------
The baseline solves this task twelve times in a hundred. At that rate a binary
outcome needs 326 trials per arm to resolve an eight-point difference, and the
run that was queued at ten trials each would have needed the other arm at 63%.
Reaching an object, disturbing it and lifting it happen far more often, so each
rung separates policies the binary cannot.

Reported as rungs rather than as one weighted score, because the weights would
be mine and the rungs are facts. A reader who wants a single number can pick the
rung that matters for their purpose; a reader handed a score cannot take it
apart again.

What this will not say
----------------------
That dataset A is better than dataset B. One training run each cannot separate
the data from the training seed, and `feedback_compare` refuses that conclusion
by construction — it reports which checkpoint won. Turning "this checkpoint won"
into "this data is better" needs several seeds per dataset, paired on
(seed, scene). The pairing key already supports it; the GPU hours are the cost.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from gantry.contracts.feedback import Cohort
from gantry.spine import mcnemar, proportion
from gantry.store import read_run

TASK = sys.argv[1] if len(sys.argv) > 1 else "pick_dual_bottles"
HERE = Path("/home/ubuntu/egorun")

ARMS = {"two_handed": "rt_two_handed", "one_handed": "rt_one_handed"}

#: In the order a policy would climb them.
RUNGS = ("reached_any", "moved_any", "lifted_any", "lifted_both", "solved")


def reached(episode, rung: str) -> bool:
    """Whether one episode got at least this far."""
    names = set(episode.labels.stages)
    lifted = sum(1 for name in names if name.endswith(".lifted"))
    return {
        "reached_any": any(name.endswith(".reached") for name in names),
        "moved_any": any(name.endswith(".moved") for name in names),
        "lifted_any": lifted >= 1,
        "lifted_both": lifted >= 2,
        "solved": bool(episode.labels.success),
    }[rung]


def scene_of(episode) -> str:
    return str(episode.labels.annotations.get("scene", episode.meta.id))


def main() -> None:
    cohorts, records = {}, {}
    for name, dataset in ARMS.items():
        path = HERE / f"abl_run_{dataset}_{TASK}.json"
        if not path.exists():
            raise SystemExit(f"missing {name}: {path}")
        record = read_run(path)
        records[name] = record
        cohorts[name] = Cohort(name=name, episodes=tuple(record.episodes))

    a, b = records["two_handed"], records["one_handed"]
    print("=" * 78)
    print(f"  dataset A (two-handed footage) vs dataset B (one-handed): {TASK}")
    print("=" * 78)
    print(f"  {len(a.episodes)} vs {len(b.episodes)} episodes\n")

    by_scene_a = {scene_of(e): e for e in a.episodes}
    by_scene_b = {scene_of(e): e for e in b.episodes}
    shared = sorted(set(by_scene_a) & set(by_scene_b))
    print(f"  {len(shared)} scenes attempted by both, paired on\n")

    print(f"  {'rung':14s} {'A':>12s} {'B':>12s} {'A only':>7s} {'B only':>7s} {'p':>8s}")
    out = {"task": TASK, "paired_scenes": len(shared), "rungs": {}}
    for rung in RUNGS:
        wins_a = [reached(by_scene_a[s], rung) for s in shared]
        wins_b = [reached(by_scene_b[s], rung) for s in shared]
        only_a = sum(1 for x, y in zip(wins_a, wins_b) if x and not y)
        only_b = sum(1 for x, y in zip(wins_a, wins_b) if y and not x)
        pa, pb = proportion(sum(wins_a), len(shared)), proportion(sum(wins_b), len(shared))
        # McNemar: only the scenes they disagreed on carry information.
        verdict = mcnemar(only_a, only_b)
        p = getattr(verdict, "value", verdict)
        print(
            f"  {rung:14s} {sum(wins_a):>3}/{len(shared):<8} {sum(wins_b):>3}/{len(shared):<8}"
            f" {only_a:>7} {only_b:>7} {float(p):>8.3f}"
        )
        out["rungs"][rung] = {
            "A": {"wins": sum(wins_a), "n": len(shared), "rate": str(pa)},
            "B": {"wins": sum(wins_b), "n": len(shared), "rate": str(pb)},
            "A_only": only_a,
            "B_only": only_b,
            "p": round(float(p), 5),
        }

    print()
    print("  A only / B only are the scenes they disagreed on -- the only ones that")
    print("  carry information. p is McNemar's exact test on those disagreements.")
    print()
    print("  This says which checkpoint won, not which dataset is better: one")
    print("  training run each cannot separate the data from the training seed.")

    (HERE / f"ab_ladder_{TASK}.json").write_text(json.dumps(out, indent=2))
    print(f"\nwritten to {HERE}/ab_ladder_{TASK}.json")


if __name__ == "__main__":
    main()
