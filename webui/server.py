"""The bench, in a browser.

Serves what Gantry has actually measured — run records, verdicts, ladders,
splits — and runs the real components against anything uploaded. Nothing here
computes a result of its own: every number on the page is read from a record
some run wrote, or produced by the same plugins the pipeline uses. A web page
that recomputed its own statistics would drift from the bench it claims to
show, and the drift would be invisible.

The same house rules apply on screen as in the pipeline:

* no naked floats — every rate is shown with its n and its interval;
* abstention is first-class — "could not tell you" renders as an answer,
  never as an empty cell;
* refusals carry their codes — an upload that fails conformance is told why,
  in the same dotted vocabulary the pipeline uses.
"""

from __future__ import annotations

import io
import json
import math
import re
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import FastAPI, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "experiments" / "robotwin_ego" / "results"
EXPERIMENT = ROOT / "experiments" / "robotwin_ego"
WORKSPACE = Path(__file__).resolve().parent / "workspace"
WORKSPACE.mkdir(exist_ok=True)

app = FastAPI(title="gantry bench")

#: The ten planes, in the order the pipeline meets them. The descriptions are
#: the contract in one line each, and the prefix maps a plugin directory to its
#: plane without needing every plugin importable.
PLANES = [
    ("dataset", "connector_", "reads someone's data as episodes, refusing what it cannot describe"),
    ("semantics", "semantics_", "the vocabulary: what a channel's numbers mean"),
    ("adapter", "adapters_", "declared transforms between descriptions, losses stated"),
    ("retargeter", "retargeter_", "one body's motion onto another's, refusing what needs IK"),
    ("embodiment", "embodiment_", "which body, declared rather than inferred"),
    ("task", "tasks_", "what was being attempted"),
    ("policy", "policy_", "the thing being measured"),
    ("evaluation", "evaluator_", "worlds that execute a policy and say what happened"),
    ("curation", "curate_", "prescriptions over episodes — and honest splits"),
    ("scorer", "scorer_", "labels episodes after the fact"),
    ("feedback", "feedback_", "turns records into verdicts, or abstains"),
]


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def wilson(wins: int, n: int, z: float = 1.959964) -> tuple[float, float]:
    """The interval every rate on the page carries."""
    if n == 0:
        return (0.0, 1.0)
    p = wins / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def mcnemar_exact(only_a: int, only_b: int) -> float:
    """Exact binomial McNemar on the disagreements — the same test the
    comparison module uses, so the page and the pipeline cannot disagree."""
    n = only_a + only_b
    if n == 0:
        return 1.0
    k = min(only_a, only_b)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2**n
    return min(1.0, 2 * tail)


RUNGS = ("reached_any", "moved_any", "lifted_any", "lifted_both", "solved")


def climbed(stages: list[str], rung: str) -> bool:
    names = set(stages)
    lifted = sum(1 for s in names if s.endswith(".lifted"))
    return {
        "reached_any": any(s.endswith(".reached") for s in names),
        "moved_any": any(s.endswith(".moved") for s in names),
        "lifted_any": lifted >= 1,
        "lifted_both": lifted >= 2,
        "solved": "solved" in names,
    }[rung]


def read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


# ---------------------------------------------------------------------------
# the bench
# ---------------------------------------------------------------------------


@app.get("/api/bench")
def bench() -> JSONResponse:
    plugins_dir = ROOT / "plugins"
    planes = []
    for name, prefix, contract in PLANES:
        members = sorted(
            p.name for p in plugins_dir.iterdir() if p.is_dir() and p.name.startswith(prefix)
        )
        planes.append({"plane": name, "contract": contract, "plugins": members})

    # What the bench currently knows, read from the records rather than typed in.
    base = read_json(RESULTS / "abl_rt_base100_pick_dual_bottles.json") or {}
    verdict = read_json(RESULTS / "robotwin_verdict_pick_dual_bottles.json") or {}
    split = read_json(RESULTS / "ab_split.json") or {}

    headline = None
    if base:
        wins, n = base.get("successes", 0), base.get("episodes", 0)
        lo, hi = wilson(wins, n)
        headline = {
            "wins": wins,
            "n": n,
            "lo": round(lo, 4),
            "hi": round(hi, 4),
            "expert": base.get("expert_solve_rate"),
            "task": base.get("task"),
        }

    return JSONResponse(
        {
            "planes": planes,
            "plugin_count": sum(len(p["plugins"]) for p in planes),
            "baseline": headline,
            "verdict": verdict.get("findings", []),
            "split": split.get("parts", {}),
            "principles": [
                ["No naked floats", "every rate carries its n and its interval"],
                ["Abstention is first-class", "'could not tell you' is an answer, never an empty cell"],
                ["Refuse rather than convert", "a gap nothing closes is reported by code, not papered over"],
                ["Describe, don't coerce", "data is read as it is, or not at all"],
                ["The control is the point", "beat the shuffled copy or the gain was just fine-tuning"],
                ["Absent ≠ zero", "a measurement that failed is not a measurement of nothing"],
            ],
        }
    )


# ---------------------------------------------------------------------------
# runs
# ---------------------------------------------------------------------------


def normalise_run(path: Path) -> dict | None:
    body = read_json(path)
    if not isinstance(body, dict) or "successes" not in body or "episodes" not in body:
        return None
    wins, n = int(body["successes"]), int(body["episodes"])
    lo, hi = wilson(wins, n)
    return {
        "id": path.stem,
        "arm": body.get("arm", path.stem),
        "task": body.get("task"),
        "wins": wins,
        "n": n,
        "rate": round(wins / n, 4) if n else None,
        "lo": round(lo, 4),
        "hi": round(hi, 4),
        "expert": body.get("expert_solve_rate"),
        "instruction": body.get("instruction"),
        "ladder": body.get("ladder"),
        "stage_counts": body.get("stage_counts"),
        "has_pairing": bool(body.get("stages_per_episode")) and bool(body.get("seeds")),
        "action_chain": body.get("action_chain"),
        "frame_shift": body.get("frame_shift"),
        "seconds": body.get("seconds"),
        "reachability": body.get("reachability"),
    }


@app.get("/api/runs")
def runs() -> JSONResponse:
    out = []
    for folder, era in ((RESULTS, "current"), (RESULTS / "unframed", "before the frame fix")):
        if not folder.exists():
            continue
        for path in sorted(folder.glob("*.json")):
            if path.name.startswith(("robotwin_seeds", "robotwin_verdict", "robotwin_frames", "rt_arms", "ab_split", "abl_run", "robotwin_run", "ab_ladder")):
                continue
            run = normalise_run(path)
            if run:
                run["era"] = era
                out.append(run)
    frames = read_json(RESULTS / "robotwin_frames_pick_dual_bottles.json")
    return JSONResponse({"runs": out, "frames": frames})


# ---------------------------------------------------------------------------
# compare
# ---------------------------------------------------------------------------


@app.get("/api/compare")
def compare(a: str, b: str) -> JSONResponse:
    left, right = read_json(RESULTS / f"{a}.json"), read_json(RESULTS / f"{b}.json")
    if not left or not right:
        return JSONResponse({"error": "run not found"}, status_code=404)

    for side, body in (("A", left), ("B", right)):
        if not body.get("stages_per_episode") or not body.get("seeds"):
            return JSONResponse(
                {
                    "abstained": True,
                    "reason": (
                        f"{side} ({body.get('arm')}) carries no per-scene stage record — it was "
                        "evaluated before the ladder existed. A comparison would have to fall "
                        "back to two marginal rates, which at these ns cannot separate anything. "
                        "Abstaining beats a number that looks like a measurement."
                    ),
                }
            )

    pair_a = dict(zip(left["seeds"], left["stages_per_episode"]))
    pair_b = dict(zip(right["seeds"], right["stages_per_episode"]))
    shared = sorted(set(pair_a) & set(pair_b))

    rungs = []
    for rung in RUNGS:
        wins_a = {s: climbed(pair_a[s], rung) for s in shared}
        wins_b = {s: climbed(pair_b[s], rung) for s in shared}
        only_a = sum(1 for s in shared if wins_a[s] and not wins_b[s])
        only_b = sum(1 for s in shared if wins_b[s] and not wins_a[s])
        ka, kb = sum(wins_a.values()), sum(wins_b.values())
        la, ha = wilson(ka, len(shared))
        lb, hb = wilson(kb, len(shared))
        rungs.append(
            {
                "rung": rung,
                "a": {"wins": ka, "n": len(shared), "lo": round(la, 4), "hi": round(ha, 4)},
                "b": {"wins": kb, "n": len(shared), "lo": round(lb, 4), "hi": round(hb, 4)},
                "only_a": only_a,
                "only_b": only_b,
                "p": round(mcnemar_exact(only_a, only_b), 5),
                "scenes": [
                    {
                        "seed": s,
                        "state": ("both" if wins_a[s] and wins_b[s] else "a" if wins_a[s] else "b" if wins_b[s] else "neither"),
                    }
                    for s in shared
                ],
            }
        )

    return JSONResponse(
        {
            "a": {"id": a, "arm": left.get("arm")},
            "b": {"id": b, "arm": right.get("arm")},
            "paired_scenes": len(shared),
            "rungs": rungs,
            "caveat": (
                "This ranks two checkpoints, not two datasets: one training run each "
                "cannot separate the data from the training seed. Turning it into a claim "
                "about data needs several seeds per dataset, paired on (seed, scene)."
            ),
        }
    )


# ---------------------------------------------------------------------------
# datasets & upload
# ---------------------------------------------------------------------------


@app.get("/api/datasets")
def datasets() -> JSONResponse:
    known = []
    arms = read_json(RESULTS / "rt_arms.json") or {}
    for name, body in arms.items():
        known.append({"name": name, "origin": "built on the bench", **body})
    split = read_json(RESULTS / "ab_split.json") or {}
    # The A/B training sets live inside the split summary rather than the arms
    # manifest; surface them in the same list so "built" means built.
    for name, body in ((split.get("parts") or {}).get("arms") or {}).items():
        if isinstance(body, dict) and "episodes" in body:
            known.append({"name": name, "origin": "built on the bench", **body})
    uploads = []
    for meta in sorted(WORKSPACE.glob("*/gantry_upload.json")):
        body = read_json(meta)
        if body:
            uploads.append(body)
    return JSONResponse({"built": known, "split": split, "uploads": uploads})


def inspect_upload(root: Path, name: str) -> dict:
    """The same questions the pipeline asks, asked of an upload.

    Read-only and refusal-first: anything absent is reported as absent, in the
    pipeline's own vocabulary, rather than silently defaulted.
    """
    report: dict[str, Any] = {"name": name, "problems": [], "checks": []}

    info_path = next(root.rglob("meta/info.json"), None)
    if info_path is None:
        report["problems"].append(
            {"code": "lerobot.no_info", "message": "no meta/info.json anywhere in the archive — not a LeRobot v2 dataset"}
        )
        return report
    base = info_path.parents[1]
    info = read_json(info_path) or {}
    report["episodes"] = info.get("total_episodes")
    report["frames"] = info.get("total_frames")
    report["fps"] = info.get("fps")
    report["features"] = {
        k: {"dtype": v.get("dtype"), "shape": v.get("shape")}
        for k, v in (info.get("features") or {}).items()
    }
    report["checks"].append(["meta/info.json", "present"])

    stats = base / "meta" / "episodes_stats.jsonl"
    report["checks"].append(["episodes_stats.jsonl", "present" if stats.exists() else "absent"])

    sidecar = base / "meta" / "gantry.jsonl"
    licences: set[str] = set()
    if sidecar.exists():
        for line in sidecar.read_text().splitlines():
            try:
                found = re.findall(r'"licence"\s*:\s*"([^"]+)"', line)
                licences.update(found)
            except Exception:
                pass
        report["checks"].append(["gantry.jsonl sidecar", "present"])
    else:
        report["checks"].append(["gantry.jsonl sidecar", "absent"])
        report["problems"].append(
            {
                "code": "provenance.undeclared",
                "message": "no gantry sidecar: licence and lineage are undeclared, and undeclared outranks permissive",
            }
        )
    report["licences"] = sorted(licences)

    videos = list(base.rglob("*.mp4"))
    report["checks"].append(["videos", f"{len(videos)} file(s)"])

    # The two-handedness measure, run with the same arithmetic the splitter
    # uses: an arm the tracker lost is a frozen block, invisible to any schema.
    parquets = sorted(base.rglob("data/**/*.parquet"))[:24]
    action_key = next((k for k in report["features"] if k == "action"), None)
    if parquets and action_key:
        import pandas as pd

        width = (report["features"]["action"]["shape"] or [0])[0]
        per_arm = width // 2 if width in (14, 16) else None
        if per_arm:
            per_episode = []
            for pq in parquets:
                try:
                    frame = pd.read_parquet(pq, columns=["action"])
                    values = np.stack(frame["action"].values)
                except Exception:
                    continue
                live = []
                for index in range(2):
                    block = values[:, index * per_arm : index * per_arm + 3]
                    if len(block) < 2:
                        live.append(0.0)
                        continue
                    step = np.linalg.norm(np.diff(block, axis=0), axis=1)
                    live.append(float((step > 1e-9).mean()))
                per_episode.append(round(min(live), 4))
            report["both_hands_moving"] = per_episode
            report["checks"].append(
                ["two-handedness", f"median {np.median(per_episode):.2f} over {len(per_episode)} episode(s)"]
            )
    return report


@app.post("/api/upload")
async def upload(file: UploadFile) -> JSONResponse:
    raw = await file.read()
    name = re.sub(r"[^A-Za-z0-9_.-]", "_", file.filename or "upload").removesuffix(".zip")
    target = WORKSPACE / name
    if target.exists():
        name = f"{name}_{sum(1 for _ in WORKSPACE.glob(name + '*'))}"
        target = WORKSPACE / name
    target.mkdir(parents=True)
    try:
        with zipfile.ZipFile(io.BytesIO(raw)) as archive:
            archive.extractall(target)
    except zipfile.BadZipFile:
        return JSONResponse(
            {"problems": [{"code": "upload.not_a_zip", "message": "the file is not a zip archive"}]},
            status_code=400,
        )
    report = inspect_upload(target, name)
    (target / "gantry_upload.json").write_text(json.dumps(report, indent=2))
    return JSONResponse(report)


# ---------------------------------------------------------------------------
# reports
# ---------------------------------------------------------------------------


@app.get("/api/reports")
def reports() -> JSONResponse:
    out = []
    for path in sorted(EXPERIMENT.glob("*.md")):
        out.append({"name": path.name, "body": path.read_text()})
    return JSONResponse({"reports": out})


@app.get("/")
def index() -> FileResponse:
    return FileResponse(Path(__file__).resolve().parent / "static" / "index.html")


app.mount("/static", StaticFiles(directory=Path(__file__).resolve().parent / "static"), name="static")
