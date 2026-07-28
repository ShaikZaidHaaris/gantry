#!/usr/bin/env python3
"""Feedback layer: turn eval failures into concrete data-collection prescriptions.

Answers "what training data would make this policy better?" rather than merely
"which dataset won". Five analyses feed a rule-based recommender:

  1. FAILURE TAXONOMY  - which stage of the task actually breaks
  2. SPATIAL COVERAGE  - where (in object-position space) failures cluster,
                         cross-referenced against TRAINING data density.
                         Failure-dense + data-sparse = the highest-value gap.
  3. BEHAVIOURAL       - how failing episodes act differently from succeeding ones
  4. TEMPORAL          - when in the episode progress stalls
  5. RECOMMENDATIONS   - prioritised, quantified data prescriptions

Usage:
  python feedback_layer.py --eval ~/rm_eval/k1_ph_1000.json \
      [--train-hdf5 ~/robomimic_data/lift/ph/low_dim.hdf5] [--task lift]
"""
from __future__ import annotations

import argparse
import json
import os
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


# ----------------------------------------------------------------- helpers
def _mean(xs):
    xs = [x for x in xs if x is not None]
    return float(np.mean(xs)) if xs else None


def _fmt(v, nd=3):
    return "n/a" if v is None else f"{v:.{nd}f}"


def load_eval(path: Path):
    d = json.load(open(path))
    tp = path.with_suffix(".traces.json")
    traces = json.load(open(tp))["episodes"] if tp.exists() else []
    return d, traces


def training_object_positions(h5_path: str, task: str) -> np.ndarray:
    """Initial object XY of every training demo (coverage map)."""
    try:
        import h5py
    except ImportError:
        return np.empty((0, 2))
    if not os.path.exists(h5_path):
        return np.empty((0, 2))
    out = []
    with h5py.File(h5_path, "r") as f:
        for name in f["data"].keys():
            obs = f["data"][name]["obs"]
            # robomimic low-dim "object" = [pos(3), quat(4), gripper_to_obj(3)]
            if "object" in obs:
                out.append(np.asarray(obs["object"][0][:2], dtype=np.float64))
    return np.asarray(out) if out else np.empty((0, 2))


# ----------------------------------------------------------------- analyses
def failure_taxonomy(records):
    n = len(records)
    modes = Counter(r["failure_mode"] for r in records)
    stages = Counter(r["stage_reached"] for r in records)
    lines = ["1. FAILURE TAXONOMY", "-" * 70]
    lines.append(f"   episodes: {n}   success: {sum(r['success'] for r in records)}")
    lines.append("   failure modes (share of all episodes):")
    for m, c in modes.most_common():
        lines.append(f"     {m:32s} {c:3d}  {100*c/n:5.1f}%")
    lines.append("   furthest stage reached:")
    for s, c in stages.most_common():
        lines.append(f"     {s:32s} {c:3d}  {100*c/n:5.1f}%")
    dominant = modes.most_common(1)[0] if modes else (None, 0)
    return "\n".join(lines), {"modes": modes, "stages": stages,
                              "dominant": dominant[0], "dominant_frac": dominant[1]/n if n else 0}


def spatial_analysis(records, traces, train_xy, bins=3):
    """Success vs initial object position, cross-referenced with train density."""
    lines = ["2. SPATIAL COVERAGE (object start position)", "-" * 70]
    by_ep = {t["episode"]: t for t in traces}
    pts = []
    for r in records:
        tr = by_ep.get(r["episode"])
        if not tr or not tr["steps"]:
            continue
        s0 = tr["steps"][0]
        if s0.get("obj_x") is None:
            continue
        pts.append((s0["obj_x"], s0["obj_y"], bool(r["success"]), r["progress_pct"]))
    if not pts:
        lines.append("   (no trace data - re-run eval to capture per-step traces)")
        return "\n".join(lines), {}

    P = np.array([(p[0], p[1]) for p in pts])
    succ = np.array([p[2] for p in pts])
    prog = np.array([p[3] for p in pts])
    xe = np.linspace(P[:, 0].min() - 1e-6, P[:, 0].max() + 1e-6, bins + 1)
    ye = np.linspace(P[:, 1].min() - 1e-6, P[:, 1].max() + 1e-6, bins + 1)

    lines.append(f"   eval object-X range [{P[:,0].min():.3f}, {P[:,0].max():.3f}]  "
                 f"Y range [{P[:,1].min():.3f}, {P[:,1].max():.3f}]")
    lines.append(f"   {'cell (x,y)':22s} {'n':>3s} {'succ%':>6s} {'prog%':>7s} {'train%':>7s}  verdict")
    cells = []
    for i in range(bins):
        for j in range(bins):
            m = ((P[:, 0] >= xe[i]) & (P[:, 0] < xe[i+1]) &
                 (P[:, 1] >= ye[j]) & (P[:, 1] < ye[j+1]))
            if m.sum() == 0:
                continue
            tr_share = None
            if train_xy.size:
                tm = ((train_xy[:, 0] >= xe[i]) & (train_xy[:, 0] < xe[i+1]) &
                      (train_xy[:, 1] >= ye[j]) & (train_xy[:, 1] < ye[j+1]))
                tr_share = 100.0 * tm.sum() / len(train_xy)
            sr, pr = 100*succ[m].mean(), prog[m].mean()
            verdict = ""
            if tr_share is not None:
                if sr < 100*succ.mean() and tr_share < 100.0/(bins*bins):
                    verdict = "<< GAP: weak here AND under-represented in training"
                elif sr < 100*succ.mean():
                    verdict = "weak (but training coverage is adequate)"
            cells.append(dict(x=(xe[i]+xe[i+1])/2, y=(ye[j]+ye[j+1])/2, n=int(m.sum()),
                              succ=sr, prog=pr, train_share=tr_share, verdict=verdict))
            lines.append(f"   ({cells[-1]['x']:+.3f},{cells[-1]['y']:+.3f})    {int(m.sum()):3d} "
                         f"{sr:6.1f} {pr:7.1f} {_fmt(tr_share,1):>7s}  {verdict}")
    return "\n".join(lines), {"cells": cells, "overall_succ": float(succ.mean())}


def behavioural_analysis(records):
    lines = ["3. BEHAVIOURAL SIGNATURE (failures vs successes)", "-" * 70]
    ok = [r for r in records if r["success"]]
    bad = [r for r in records if not r["success"]]
    keys = ["gripper_switches", "gripper_closed_frac", "action_saturation_frac",
            "action_jerk_mean", "path_efficiency", "eef_path_length",
            "min_gripper_to_can", "reach_closure_frac", "max_can_height_gain"]
    lines.append(f"   {'metric':24s} {'success':>10s} {'failure':>10s}   delta")
    flags = {}
    for k in keys:
        a, b = _mean([r.get(k) for r in ok]), _mean([r.get(k) for r in bad])
        if a is None or b is None:
            continue
        d = b - a
        lines.append(f"   {k:24s} {a:10.4f} {b:10.4f}   {d:+.4f}")
        flags[k] = d
    if not ok:
        lines.append("   (no successful episodes yet - comparison unavailable)")
    return "\n".join(lines), flags


def temporal_analysis(records, traces):
    """When does progress stop improving?"""
    lines = ["4. TEMPORAL STALL (when progress plateaus)", "-" * 70]
    by_ep = {t["episode"]: t for t in traces}
    stalls, finals = [], []
    for r in records:
        tr = by_ep.get(r["episode"])
        if not tr or not tr["steps"]:
            continue
        p = np.array([s["progress_pct"] for s in tr["steps"]])
        if p.size < 5:
            continue
        peak = p.max()
        # first step reaching 95% of this episode's peak progress
        idx = int(np.argmax(p >= 0.95 * peak)) + 1 if peak > 0 else len(p)
        stalls.append(idx)
        finals.append(peak)
    if not stalls:
        lines.append("   (no trace data)")
        return "\n".join(lines), {}
    lines.append(f"   episodes analysed      : {len(stalls)}")
    lines.append(f"   median stall step      : {np.median(stalls):.0f} of {len(p)}")
    lines.append(f"   mean peak progress     : {np.mean(finals):.1f}%")
    lines.append(f"   episodes stalling <25% : {100*np.mean(np.array(stalls) < 0.25*len(p)):.0f}% "
                 f"(early stall => policy commits to a bad plan quickly)")
    return "\n".join(lines), {"median_stall": float(np.median(stalls)),
                              "frac_early": float(np.mean(np.array(stalls) < 0.25*len(p)))}


# ----------------------------------------------------------------- recommender
def recommend(tax, spat, beh, temp, records):
    """Map observed patterns -> concrete, quantified data prescriptions."""
    recs = []
    n = len(records)
    dom, frac = tax["dominant"], tax["dominant_frac"]

    if dom == "reached_but_no_grasp" and frac > 0.3:
        recs.append((
            "HIGH", "Grasp initiation is the bottleneck",
            f"{100*frac:.0f}% of episodes reach the object but never close a grasp. "
            "The policy servos to the object then stalls.",
            "Collect/upweight demos that emphasise the GRASP MOMENT: slow final approach "
            "(last ~5cm), an unambiguous open->closed gripper transition while in contact, "
            "and a brief pause holding the object before moving. Segment-level upweighting "
            "of the frames spanning gripper closure is usually more effective than adding "
            "whole new episodes."))
    if dom in ("never_reached_cube", "never_reached_can") and frac > 0.3:
        recs.append((
            "HIGH", "Reaching / visual servoing is the bottleneck",
            f"{100*frac:.0f}% never get near the object at all.",
            "Add demos with DIVERSE START POSES and approach directions; the policy has not "
            "learned a general reach controller. Vary initial arm configuration, not just "
            "object position."))
    if dom in ("grasped_but_no_lift", "lifted_but_below_threshold") and frac > 0.2:
        recs.append((
            "HIGH", "Grasp is unstable or lift is incomplete",
            f"{100*frac:.0f}% grasp but fail to lift clear.",
            "Add demos with FIRM grasps and a decisive vertical lift well above threshold; "
            "include recovery from partial slips so the policy learns to re-grip."))

    gs = beh.get("gripper_switches")
    if gs is not None and gs > 5:
        recs.append((
            "MEDIUM", "Gripper oscillation",
            f"Failing episodes toggle the gripper {gs:+.0f} more times than successful ones - "
            "the policy is unsure when to close.",
            "Prefer demos with CRISP binary gripper signals. Audit the training set for "
            "noisy/chattering gripper labels and consider binarising the gripper channel."))
    sat = beh.get("action_saturation_frac")
    if sat is not None and sat > 0.02:
        recs.append((
            "MEDIUM", "Control saturation in failures",
            f"Failing episodes saturate action limits {sat:+.3f} more of the time.",
            "Training data may lack FINE, low-magnitude corrections near the object. Add "
            "demos with slow precise alignment rather than fast large motions."))
    pe = beh.get("path_efficiency")
    if pe is not None and pe < -0.05:
        recs.append((
            "MEDIUM", "Inefficient wandering paths",
            f"Failing episodes are {abs(pe):.3f} less direct.",
            "Add demos with clean, direct approach trajectories; consider filtering out "
            "meandering demonstrations from the training mix."))

    gaps = [c for c in spat.get("cells", []) if c["verdict"].startswith("<<")]
    for c in sorted(gaps, key=lambda c: c["succ"])[:3]:
        recs.append((
            "HIGH", f"Spatial coverage gap near object (x={c['x']:+.3f}, y={c['y']:+.3f})",
            f"Success {c['succ']:.0f}% here vs {100*spat['overall_succ']:.0f}% overall, "
            f"yet only {_fmt(c['train_share'],1)}% of training demos start in this region.",
            f"Collect demos with the object placed around (x={c['x']:+.3f}, y={c['y']:+.3f}). "
            "This is the highest-value gap: the policy is weak exactly where data is thin."))

    if temp.get("frac_early", 0) > 0.5:
        recs.append((
            "MEDIUM", "Early commitment to a bad plan",
            f"{100*temp['frac_early']:.0f}% of episodes plateau in the first quarter.",
            "Add demos showing RECOVERY behaviour (failed reach -> re-approach). Purely "
            "successful demos never teach the policy how to correct itself."))

    if not recs:
        recs.append(("INFO", "No dominant failure pattern detected",
                     "Failures look diffuse rather than systematic.",
                     "Scale data volume broadly, or increase eval episodes to resolve patterns."))
    return recs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", required=True, help="path to an eval JSON")
    ap.add_argument("--train-hdf5", default=None)
    ap.add_argument("--task", default="lift")
    ap.add_argument("--bins", type=int, default=3)
    args = ap.parse_args()

    path = Path(os.path.expanduser(args.eval))
    d, traces = load_eval(path)
    records = d["records"]
    train_xy = training_object_positions(
        os.path.expanduser(args.train_hdf5), args.task) if args.train_hdf5 else np.empty((0, 2))

    print("=" * 74)
    print(f"FEEDBACK REPORT  |  {d['label']}  |  n={d['n_episodes']}  |  task={args.task}")
    print(f"mean progress {d['HEADLINE']['mean_progress_pct']}%  "
          f"success {100*d['rates']['success']['rate']:.0f}%  "
          f"grasp {100*d['rates']['grasped']['rate']:.0f}%")
    if train_xy.size:
        print(f"training demos analysed for coverage: {len(train_xy)}")
    print("=" * 74)

    t_txt, tax = failure_taxonomy(records);           print(t_txt, "\n")
    s_txt, spat = spatial_analysis(records, traces, train_xy, args.bins); print(s_txt, "\n")
    b_txt, beh = behavioural_analysis(records);       print(b_txt, "\n")
    p_txt, temp = temporal_analysis(records, traces); print(p_txt, "\n")

    print("5. DATA PRESCRIPTIONS (what to collect next)")
    print("-" * 70)
    for i, (pri, title, finding, action) in enumerate(recommend(tax, spat, beh, temp, records), 1):
        print(f"\n  [{pri}] {i}. {title}")
        print(f"      finding : {finding}")
        print(f"      action  : {action}")
    print("\n" + "=" * 74)
    print("FEEDBACK_DONE")


if __name__ == "__main__":
    main()
