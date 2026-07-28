#!/usr/bin/env python3
"""Cross-dataset prescription hardening.

A prescription derived from ONE model is a hypothesis. Comparing several models
trained on differently-flawed data turns it into an evidence-backed claim:

  1. UNIVERSAL vs SPECIFIC
     A bottleneck present in every dataset is most likely a task/recipe limit
     (horizon, action space, training budget) and will NOT be fixed by curating
     that data. A bottleneck present in only some datasets is data-attributable.

  2. EXISTENCE PROOF
     If dataset B clears a transition that dataset A stalls on, then that
     transition is achievable at this budget - proven, not assumed. Two-proportion
     tests with CIs on the difference decide whether the gap is real.

  3. TARGET VALUES FROM THE WINNER
     For a data-attributable bottleneck we compute the same statistic on the raw
     TRAINING demonstrations of every dataset, and read off the winner's value as
     the concrete target a data generator should hit.

Output: hardened prescriptions carrying evidence class, existence proof,
measured target values, and the residual risk of each claim.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
from pathlib import Path

import numpy as np

TRANSITIONS = ["P(reach)", "P(grasp|reach)", "P(lift|grasp)", "P(success|lift)"]


# ---------------------------------------------------------------- statistics
def wilson(k, n, z=1.96):
    if n == 0:
        return 0.0, 0.0, 0.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(max(p * (1 - p) / n + z * z / (4 * n * n), 0.0)) / d
    return p, max(0.0, c - h), min(1.0, c + h)


def two_prop(k1, n1, k2, n2):
    """Newcombe difference CI + z p-value for p1 - p2."""
    if n1 == 0 or n2 == 0:
        return None
    p1, l1, h1 = wilson(k1, n1)
    p2, l2, h2 = wilson(k2, n2)
    diff = p1 - p2
    lo = diff - math.sqrt((p1 - l1) ** 2 + (h2 - p2) ** 2)
    hi = diff + math.sqrt((h1 - p1) ** 2 + (p2 - l2) ** 2)
    pp = (k1 + k2) / (n1 + n2)
    se = math.sqrt(max(pp * (1 - pp) * (1 / n1 + 1 / n2), 1e-12))
    z = diff / se if se > 0 else 0.0
    pval = 2 * (1 - 0.5 * (1 + math.erf(abs(z) / math.sqrt(2))))
    return {"diff": diff, "lo": max(-1, lo), "hi": min(1, hi), "p": pval}


# ------------------------------------------------- data-side demo statistics
def demo_statistics(h5_path: str, max_demos: int = 400) -> dict:
    """Statistics on the RAW demonstrations - the levers a data team controls."""
    try:
        import h5py
    except ImportError:
        return {}
    if not os.path.exists(h5_path):
        return {}
    out = {k: [] for k in ["len", "grip_transitions", "grip_close_frac",
                           "dwell_near_frames", "terminal_speed", "path_efficiency",
                           "action_jerk", "time_to_close_frac"]}
    with h5py.File(h5_path, "r") as f:
        names = list(f["data"].keys())[:max_demos]
        for nm in names:
            g = f["data"][nm]
            a = np.asarray(g["actions"][:], float)
            if a.ndim != 2 or a.shape[0] < 3:
                continue
            obs = g["obs"]
            eef = np.asarray(obs["robot0_eef_pos"][:], float) if "robot0_eef_pos" in obs else None
            objr = None
            if "object" in obs:
                o = np.asarray(obs["object"][:], float)
                if o.shape[1] >= 10:      # [pos3, quat4, gripper_to_obj3]
                    objr = np.linalg.norm(o[:, 7:10], axis=1)
                elif eef is not None and o.shape[1] >= 3:
                    objr = np.linalg.norm(o[:, :3] - eef, axis=1)
            gr = a[:, 6]
            sgn = np.sign(gr)
            out["len"].append(len(a))
            out["grip_transitions"].append(int(np.sum(np.abs(np.diff(sgn)) > 0)))
            out["grip_close_frac"].append(float(np.mean(gr > 0)))
            out["action_jerk"].append(float(np.mean(np.abs(np.diff(a, axis=0)))))
            close = np.where(np.diff(sgn) > 0)[0]
            out["time_to_close_frac"].append(
                float((close[0] + 1) / len(a)) if close.size else 1.0)
            if objr is not None:
                out["dwell_near_frames"].append(int(np.sum(objr < 0.03)))
            if eef is not None:
                seg = np.linalg.norm(np.diff(eef, axis=0), axis=1)
                path = float(seg.sum())
                disp = float(np.linalg.norm(eef[-1] - eef[0]))
                out["path_efficiency"].append(disp / path if path > 1e-9 else 0.0)
                k = max(int(0.2 * len(eef)), 1)
                idx = close[0] if close.size else len(eef) - 1
                lo = max(idx - k, 0)
                out["terminal_speed"].append(
                    float(seg[lo:idx].mean()) if idx > lo else float(seg.mean()))
    return {k: (float(np.mean(v)) if v else None) for k, v in out.items()}


# ---------------------------------------------------------------- loading
def load_feedback(d):
    j = json.load(open(d))
    rows = {r[0]: {"k": r[1], "n": r[2], "p": r[3], "lo": r[4], "hi": r[5]}
            for r in j["funnel"]}
    return {"label": j["label"], "funnel": rows,
            "disc": j.get("discriminative", []), "spatial": j.get("spatial")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feedback-dir", default="/home/ubuntu")
    ap.add_argument("--data-dir", default="/home/ubuntu/robomimic_data/lift")
    ap.add_argument("--datasets", default="ph,mh,mg")
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    dss = [s.strip() for s in args.datasets.split(",")]
    fb, dstats = {}, {}
    for s in dss:
        p = Path(args.feedback_dir) / f"feedback_{s}.json"
        if p.exists():
            fb[s] = load_feedback(p)
        dstats[s] = demo_statistics(os.path.join(args.data_dir, s, "low_dim.hdf5"))

    W = 82
    print("=" * W)
    print(f"HARDENED PRESCRIPTIONS  |  datasets: {', '.join(dss)}  |  available feedback: {', '.join(fb) or 'none'}")
    print("=" * W)
    if len(fb) < 2:
        print("\nNeed >=2 dataset feedback files to harden. Currently have: "
              f"{list(fb) or 'none'}")
        print("(each is written as ~/feedback_<ds>.json by the eval pipeline)")

    # ---------------- 1. funnel comparison ----------------
    print("\n1. FUNNEL COMPARISON ACROSS DATASETS")
    print("-" * W)
    if fb:
        print(f"   {'transition':18s} " + "".join(f"{s.upper():>18s}" for s in fb))
        for t in TRANSITIONS:
            cells = []
            for s in fb:
                r = fb[s]["funnel"].get(t)
                cells.append(f"{r['p']:.2f} [{r['lo']:.2f},{r['hi']:.2f}]" if r else "n/a")
            print(f"   {t:18s} " + "".join(f"{c:>18s}" for c in cells))

    # ---------------- 2. classify each transition ----------------
    print("\n2. EVIDENCE CLASS PER TRANSITION")
    print("-" * W)
    classes = {}
    for t in TRANSITIONS:
        vals = {s: fb[s]["funnel"].get(t) for s in fb if fb[s]["funnel"].get(t)}
        if len(vals) < 2:
            continue
        ps = {s: v["p"] for s, v in vals.items()}
        best = max(ps, key=ps.get); worst = min(ps, key=ps.get)
        tp = two_prop(vals[best]["k"], vals[best]["n"], vals[worst]["k"], vals[worst]["n"])
        weak_everywhere = all(p < 0.5 for p in ps.values())
        separated = tp and tp["lo"] > 0
        if weak_everywhere and not separated:
            cls = "UNIVERSAL (task/recipe limit - data curation unlikely to fix)"
        elif separated:
            cls = f"DATA-ATTRIBUTABLE (existence proof: {best.upper()} clears it)"
        else:
            cls = "UNRESOLVED (underpowered - needs more eval episodes)"
        classes[t] = {"class": cls, "best": best, "worst": worst, "ps": ps, "test": tp}
        print(f"   {t:18s} {cls}")
        if tp:
            print(f"      {best.upper()} {ps[best]:.2f} vs {worst.upper()} {ps[worst]:.2f}  "
                  f"diff {tp['diff']:+.2f} [{tp['lo']:+.2f},{tp['hi']:+.2f}] p={tp['p']:.4f}")

    # ---------------- 3. training-data statistics ----------------
    print("\n3. TRAINING-DATA STATISTICS (the levers a data team controls)")
    print("-" * W)
    keys = ["len", "grip_transitions", "grip_close_frac", "dwell_near_frames",
            "terminal_speed", "path_efficiency", "action_jerk", "time_to_close_frac"]
    print(f"   {'statistic':22s} " + "".join(f"{s.upper():>12s}" for s in dss))
    for k in keys:
        row = "".join(
            f"{dstats[s][k]:12.4f}" if dstats.get(s, {}).get(k) is not None else f"{'n/a':>12s}"
            for s in dss)
        print(f"   {k:22s} {row}")

    # ---------------- 4. hardened prescriptions ----------------
    print("\n" + "=" * W)
    print("HARDENED PRESCRIPTIONS")
    print("=" * W)
    STAT_FOR = {
        "P(grasp|reach)": ["dwell_near_frames", "grip_transitions", "terminal_speed",
                           "time_to_close_frac"],
        "P(reach)": ["path_efficiency", "action_jerk"],
        "P(lift|grasp)": ["grip_close_frac", "action_jerk"],
        "P(success|lift)": ["len", "grip_close_frac"],
    }
    out = []
    for t, c in classes.items():
        if c["class"].startswith("UNIVERSAL"):
            out.append(dict(
                transition=t, klass="UNIVERSAL", priority="P3-DEFER",
                claim=f"{t} is weak in every dataset ({', '.join(f'{s}={p:.2f}' for s,p in c['ps'].items())}).",
                implication="No dataset clears this step, so it is most likely a TASK or RECIPE "
                            "limit (training budget, action horizon, control mode) rather than a "
                            "data-quality gap. Collecting more of this behaviour is unlikely to pay "
                            "off until the recipe changes.",
                action="Do NOT commission new data for this. Instead vary training budget / "
                       "action chunk / control mode and re-measure."))
            continue
        if not c["class"].startswith("DATA-ATTRIBUTABLE"):
            out.append(dict(
                transition=t, klass="UNRESOLVED", priority="P2-MEASURE",
                claim=f"{t} differs across datasets but CIs overlap "
                      f"(diff {c['test']['diff']:+.2f} [{c['test']['lo']:+.2f},{c['test']['hi']:+.2f}]).",
                implication="Underpowered: cannot yet attribute to data.",
                action="Increase eval episodes (n=30 -> 100) before commissioning data."))
            continue

        best, worst, tp = c["best"], c["worst"], c["test"]
        # Rank ALL demo statistics by relative gap rather than a hand-picked
        # subset: which lever separates the winner is an empirical question, and
        # a narrow prior can hide the biggest difference (e.g. episode length).
        preferred = STAT_FOR.get(t, [])
        stats = preferred + [k for k in (dstats.get(best) or {}) if k not in preferred]
        targets = []
        for k in stats:
            bv, wv = dstats.get(best, {}).get(k), dstats.get(worst, {}).get(k)
            if bv is None or wv is None:
                continue
            if abs(wv) > 1e-9:
                rel = (bv - wv) / abs(wv)
                if abs(rel) >= 0.10:      # only report materially different levers
                    targets.append((k, wv, bv, rel, k in preferred))
        # mechanistically-linked statistics first, then by effect size
        targets.sort(key=lambda r: (not r[4], -abs(r[3])))
        out.append(dict(
            transition=t, klass="DATA-ATTRIBUTABLE", priority="P0-ACT",
            claim=(f"{worst.upper()} stalls at {t}={c['ps'][worst]:.2f} while {best.upper()} "
                   f"reaches {c['ps'][best]:.2f} (diff {tp['diff']:+.2f} "
                   f"[{tp['lo']:+.2f},{tp['hi']:+.2f}], p={tp['p']:.4f})."),
            implication=(f"EXISTENCE PROOF: this transition IS achievable at the current budget - "
                         f"{best.upper()} demonstrates it. The gap is therefore attributable to "
                         f"data, not to the task or recipe."),
            action=("Move the training corpus toward the winner on these measured levers:\n"
                    + ("\n".join(
                        f"          - {k}: {worst.upper()}={wv:.4f} -> target {best.upper()}={bv:.4f} "
                        f"({rel:+.0%}){'  [mechanistically linked]' if pref else ''}"
                        for k, wv, bv, rel, pref in targets[:5])
                       if targets else "          (no single demo statistic separates them; "
                                       "inspect trajectories directly)"))))

    for i, p in enumerate(out, 1):
        print(f"\n[{p['priority']}] {i}. {p['transition']}  ({p['klass']})")
        print(f"   CLAIM       : {p['claim']}")
        print(f"   IMPLICATION : {p['implication']}")
        print(f"   ACTION      : {p['action']}")

    if not out:
        print("\n(no cross-dataset comparison possible yet - waiting on feedback files)")

    print("\n" + "=" * W)
    print("RESIDUAL RISK")
    print("=" * W)
    print("   * n=30 episodes/model: transition CIs are wide; DATA-ATTRIBUTABLE calls require")
    print("     a strictly positive difference CI, but UNRESOLVED calls are common at this n.")
    print("   * Demo statistics are correlational. The causal test is the closed loop:")
    print("     curate toward the target value, retrain, and confirm the predicted uplift.")

    if args.json_out:
        Path(args.json_out).write_text(json.dumps(
            {"classes": classes, "demo_stats": dstats, "prescriptions": out},
            indent=2, default=str))
        print(f"\nmachine-readable -> {args.json_out}")
    print("\nHARDEN_DONE")


if __name__ == "__main__":
    main()
