#!/usr/bin/env python3
"""Quantitative feedback layer: eval failures -> executable data-collection specs.

Everything here is estimated, not asserted. Five stages:

  A. FUNNEL DECOMPOSITION
     Model the task as a Markov chain over stages
         none -> reach -> grasp -> lift -> success
     Estimate conditional transition probabilities p_k = P(s_k | s_{k-1}) with
     Wilson intervals. Overall success = prod(p_k). The bottleneck is argmin p_k.
     COUNTERFACTUAL UPLIFT: repairing transition k to target t multiplies overall
     success by t/p_k, giving a *quantified* expected return per intervention.

  B. DISCRIMINATIVE STATISTICS
     For every behavioural metric, compare failing vs succeeding episodes with
     Cliff's delta (non-parametric, outlier-robust effect size) + Mann-Whitney U,
     bootstrap CI on delta, and Benjamini-Hochberg FDR control across the family.

  C. LOGISTIC ATTRIBUTION
     Ridge-penalised logistic regression (IRLS) of success on standardised
     features -> odds ratios with Wald CIs. Penalty keeps estimates finite under
     quasi-separation, which is common at small n.

  D. SPATIAL COVERAGE DEFICIT
     Gaussian KDE (Scott bandwidth) over training object positions f_tr and over
     failure positions f_fail. Deficit D = f_fail / (f_tr + eps) localises regions
     that fail often *and* are under-represented in training.
     A weighted logistic fit of success on log-density converts a density gap into
     a DEMO COUNT: how many demos are needed to raise that region to target.

  E. TEMPORAL HAZARD
     Discrete-time hazard of stage advancement; early plateau implies premature
     commitment rather than capability limits.

Only numpy is required; scipy is used when present for exact MWU p-values.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from collections import Counter
from pathlib import Path

import numpy as np

try:
    from scipy.stats import mannwhitneyu as _mwu
    _HAVE_SCIPY = True
except Exception:
    _HAVE_SCIPY = False

EPS = 1e-12
STAGE_ORDER = ["reach", "grasp", "lift", "success"]


# ============================================================ statistics
def wilson(k: int, n: int, z: float = 1.96):
    if n == 0:
        return 0.0, 0.0, 0.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(max(p * (1 - p) / n + z * z / (4 * n * n), 0.0)) / d
    return p, max(0.0, c - h), min(1.0, c + h)


def cliffs_delta(a, b):
    """Non-parametric effect size in [-1,1]. Robust to outliers and scale."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.size == 0 or b.size == 0:
        return None
    gt = (a[:, None] > b[None, :]).sum()
    lt = (a[:, None] < b[None, :]).sum()
    return float((gt - lt) / (a.size * b.size))


def cliff_magnitude(d):
    if d is None:
        return "n/a"
    a = abs(d)
    return ("negligible" if a < 0.147 else "small" if a < 0.33
            else "medium" if a < 0.474 else "large")


def mwu_p(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.size < 2 or b.size < 2:
        return None
    if _HAVE_SCIPY:
        try:
            return float(_mwu(a, b, alternative="two-sided").pvalue)
        except Exception:
            return None
    # normal approximation with tie correction
    x = np.concatenate([a, b])
    order = x.argsort()
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, x.size + 1)
    # average ranks for ties
    _, inv, cnt = np.unique(x, return_inverse=True, return_counts=True)
    for i, c in enumerate(cnt):
        if c > 1:
            ranks[inv == i] = ranks[inv == i].mean()
    n1, n2 = a.size, b.size
    R1 = ranks[:n1].sum()
    U = R1 - n1 * (n1 + 1) / 2
    mu = n1 * n2 / 2
    tie = np.sum(cnt ** 3 - cnt)
    N = n1 + n2
    sd = math.sqrt(max(n1 * n2 / 12 * ((N + 1) - tie / (N * (N - 1))), EPS))
    z = (U - mu) / sd
    return float(2 * (1 - 0.5 * (1 + math.erf(abs(z) / math.sqrt(2)))))


def bh_fdr(pvals, alpha=0.05):
    """Benjamini-Hochberg: returns boolean 'significant' mask."""
    idx = [i for i, p in enumerate(pvals) if p is not None]
    if not idx:
        return [False] * len(pvals)
    ps = sorted((pvals[i], i) for i in idx)
    m = len(ps)
    sig = [False] * len(pvals)
    kmax = -1
    for r, (p, _) in enumerate(ps, start=1):
        if p <= alpha * r / m:
            kmax = r
    for r, (_, i) in enumerate(ps, start=1):
        if r <= kmax:
            sig[i] = True
    return sig


def boot_ci_delta(a, b, n_boot=1500, seed=7):
    rng = np.random.default_rng(seed)
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.size < 2 or b.size < 2:
        return None, None
    ds = []
    for _ in range(n_boot):
        d = cliffs_delta(rng.choice(a, a.size, True), rng.choice(b, b.size, True))
        if d is not None:
            ds.append(d)
    if not ds:
        return None, None
    return float(np.percentile(ds, 2.5)), float(np.percentile(ds, 97.5))


def ridge_logistic(X, y, lam=1.0, iters=60):
    """IRLS with L2 penalty (intercept unpenalised). Returns beta, se."""
    X = np.asarray(X, float); y = np.asarray(y, float)
    n, d = X.shape
    Xd = np.hstack([np.ones((n, 1)), X])
    beta = np.zeros(d + 1)
    P = np.eye(d + 1) * lam
    P[0, 0] = 0.0
    for _ in range(iters):
        eta = np.clip(Xd @ beta, -30, 30)
        mu = 1 / (1 + np.exp(-eta))
        W = np.clip(mu * (1 - mu), 1e-6, None)
        z = eta + (y - mu) / W
        A = Xd.T @ (Xd * W[:, None]) + P
        try:
            new = np.linalg.solve(A, Xd.T @ (W * z))
        except np.linalg.LinAlgError:
            break
        if not np.all(np.isfinite(new)) or np.max(np.abs(new - beta)) < 1e-8:
            beta = new
            break
        beta = new
    eta = np.clip(Xd @ beta, -30, 30)
    mu = 1 / (1 + np.exp(-eta))
    W = np.clip(mu * (1 - mu), 1e-6, None)
    try:
        cov = np.linalg.inv(Xd.T @ (Xd * W[:, None]) + P)
        se = np.sqrt(np.clip(np.diag(cov), 0, None))
    except np.linalg.LinAlgError:
        se = np.full(d + 1, np.nan)
    return beta, se


def gauss_kde(points, grid, bw=None):
    """Isotropic Gaussian KDE with Scott's rule. points (n,2), grid (m,2)."""
    P = np.asarray(points, float)
    if P.size == 0:
        return np.zeros(len(grid))
    n = P.shape[0]
    sd = P.std(axis=0).mean()
    if bw is None:
        bw = max(sd * n ** (-1.0 / 6.0), 1e-3)   # Scott, d=2
    d2 = ((grid[:, None, :] - P[None, :, :]) ** 2).sum(-1)
    return np.exp(-0.5 * d2 / bw ** 2).sum(1) / (n * 2 * math.pi * bw ** 2)


# ============================================================ loading
def load(path: Path):
    d = json.load(open(path))
    tp = path.with_suffix(".traces.json")
    tr = json.load(open(tp))["episodes"] if tp.exists() else []
    return d, tr


def train_positions(h5: str) -> np.ndarray:
    try:
        import h5py
    except ImportError:
        return np.empty((0, 2))
    if not h5 or not os.path.exists(h5):
        return np.empty((0, 2))
    out = []
    with h5py.File(h5, "r") as f:
        for k in f["data"].keys():
            o = f["data"][k]["obs"]
            if "object" in o:
                out.append(np.asarray(o["object"][0][:2], float))
    return np.asarray(out) if out else np.empty((0, 2))


# ============================================================ A. funnel
def funnel(records):
    n = len(records)
    reached = sum(r["ever_reached"] for r in records)
    grasped = sum(r["ever_grasped"] for r in records)
    lifted = sum(r["ever_lifted"] for r in records)
    succ = sum(r["success"] for r in records)
    # a stage can be skipped in the flags; use monotone envelope
    grasped = min(grasped, reached) if reached else grasped
    lifted = min(lifted, max(grasped, succ))
    steps = [("P(reach)", reached, n),
             ("P(grasp|reach)", grasped, max(reached, 1)),
             ("P(lift|grasp)", max(lifted, succ), max(grasped, 1)),
             ("P(success|lift)", succ, max(max(lifted, succ), 1))]
    rows, probs = [], []
    for name, k, d in steps:
        p, lo, hi = wilson(min(k, d), d)
        rows.append((name, min(k, d), d, p, lo, hi))
        probs.append(p)
    overall = float(np.prod(probs))
    obs = succ / n if n else 0.0
    valid = [i for i, p in enumerate(probs) if steps[i][2] > 0]
    bidx = min(valid, key=lambda i: probs[i]) if valid else 0
    return {"rows": rows, "probs": probs, "overall_model": overall,
            "overall_obs": obs, "bottleneck": bidx, "n": n}


def counterfactual(f, targets=(0.5, 0.8, 0.95)):
    """Expected overall success if transition k were repaired to target t."""
    out = []
    for k, (name, _, _, p, _, _) in enumerate(f["rows"]):
        for t in targets:
            if p >= t or p <= 0:
                continue
            new = f["overall_model"] * (t / max(p, EPS))
            out.append((name, p, t, min(new, 1.0), min(new, 1.0) - f["overall_model"]))
    out.sort(key=lambda r: -r[4])
    return out


# ============================================================ B/C
# Metrics split by whether they can *drive* a data prescription.
# OUTCOME metrics are downstream of success itself (a failing episode necessarily
# runs to the horizon and never lifts the object), so conditioning prescriptions
# on them is circular. They are still reported, flagged as confirmatory.
OUTCOME_KEYS = ["steps", "max_can_height_gain", "max_can_displacement",
                "eef_path_length", "eef_net_displacement"]
# Duration-invariant behavioural metrics. Raw counts scale with episode length
# (failures run ~2x longer), which would manufacture spurious effects, so
# count-like quantities are converted to per-step rates.
BEHAV_KEYS = ["gripper_switch_rate", "gripper_closed_frac", "action_saturation_frac",
              "action_jerk_mean", "action_abs_mean", "path_efficiency",
              "path_per_step", "min_gripper_to_can", "reach_closure_frac"]


def derive_rates(records):
    """Add duration-normalised metrics; raw counts confound with episode length."""
    for r in records:
        n = max(int(r.get("steps") or 1), 1)
        if r.get("gripper_switches") is not None:
            r["gripper_switch_rate"] = r["gripper_switches"] / n
        if r.get("eef_path_length") is not None:
            r["path_per_step"] = r["eef_path_length"] / n
    return records


def discriminative(records, alpha=0.05, keys=None, tag="behavioural"):
    ok = [r for r in records if r["success"]]
    bad = [r for r in records if not r["success"]]
    res = []
    if not ok or not bad:
        return res, "need both successes and failures for contrast"
    for k in (keys if keys is not None else BEHAV_KEYS):
        a = [r[k] for r in ok if r.get(k) is not None]
        b = [r[k] for r in bad if r.get(k) is not None]
        if len(a) < 2 or len(b) < 2:
            continue
        d = cliffs_delta(b, a)                    # >0 => higher in FAILURES
        lo, hi = boot_ci_delta(b, a)
        res.append({"metric": k, "succ_mean": float(np.mean(a)),
                    "fail_mean": float(np.mean(b)), "delta": d,
                    "lo": lo, "hi": hi, "p": mwu_p(a, b), "kind": tag})
    sig = bh_fdr([r["p"] for r in res], alpha)
    for r, s in zip(res, sig):
        r["sig_fdr"] = bool(s)
    res.sort(key=lambda r: -abs(r["delta"] or 0))
    return res, None


def attribution(records, keys=None, lam=2.0):
    keys = keys or ["reach_closure_frac", "gripper_switch_rate", "path_efficiency",
                    "action_saturation_frac", "min_gripper_to_can"]
    rows = [r for r in records if all(r.get(k) is not None for k in keys)]
    y = np.array([1.0 if r["success"] else 0.0 for r in rows])
    if len(rows) < 8 or y.sum() == 0 or y.sum() == len(y):
        return [], "insufficient variation for logistic attribution"
    X = np.array([[float(r[k]) for k in keys] for r in rows])
    mu, sd = X.mean(0), X.std(0)
    sd[sd < EPS] = 1.0
    Z = (X - mu) / sd
    beta, se = ridge_logistic(Z, y, lam=lam)
    out = []
    for i, k in enumerate(keys, start=1):
        b, s = beta[i], se[i]
        out.append({"feature": k, "beta": float(b), "or": float(np.exp(b)),
                    "lo": float(np.exp(b - 1.96 * s)) if np.isfinite(s) else None,
                    "hi": float(np.exp(b + 1.96 * s)) if np.isfinite(s) else None})
    out.sort(key=lambda r: -abs(r["beta"]))
    return out, None


# ============================================================ D. spatial
def spatial(records, traces, train_xy, grid_n=24, target=0.8):
    by = {t["episode"]: t for t in traces}
    pts, succ = [], []
    for r in records:
        t = by.get(r["episode"])
        if not t or not t["steps"]:
            continue
        s0 = t["steps"][0]
        if s0.get("obj_x") is None:
            continue
        pts.append([s0["obj_x"], s0["obj_y"]])
        succ.append(1.0 if r["success"] else 0.0)
    if len(pts) < 5:
        return None, "no per-step traces (re-run eval with trace capture)"
    P = np.array(pts); S = np.array(succ)
    lo = np.minimum(P.min(0), train_xy.min(0) if train_xy.size else P.min(0))
    hi = np.maximum(P.max(0), train_xy.max(0) if train_xy.size else P.max(0))
    pad = 0.1 * (hi - lo + 1e-6)
    gx = np.linspace(lo[0] - pad[0], hi[0] + pad[0], grid_n)
    gy = np.linspace(lo[1] - pad[1], hi[1] + pad[1], grid_n)
    G = np.stack(np.meshgrid(gx, gy, indexing="ij"), -1).reshape(-1, 2)

    f_fail = gauss_kde(P[S == 0], G) if (S == 0).any() else np.zeros(len(G))
    f_tr = gauss_kde(train_xy, G) if train_xy.size else np.zeros(len(G))
    scale = f_tr.max() if f_tr.max() > 0 else 1.0
    deficit = f_fail / (f_tr / scale + 1e-3) if train_xy.size else f_fail
    if deficit.max() > 0:
        deficit = deficit / deficit.max()

    # density -> success relation (logistic on log train-density at each eval point)
    demo_est, slope_note = None, "training positions unavailable"
    if train_xy.size and len(P) >= 8 and 0 < S.sum() < len(S):
        dens_at = gauss_kde(train_xy, P)
        x = np.log(dens_at + 1e-9).reshape(-1, 1)
        b, _ = ridge_logistic((x - x.mean()) / (x.std() + EPS), S, lam=1.0)
        slope = float(b[1])
        slope_note = f"logistic slope on log-density = {slope:+.3f} (per SD)"
        if slope > 0.05:
            sd_log = float(x.std() + EPS)
            p_now = float(S.mean())
            need_logit = math.log(target / (1 - target)) - math.log(
                max(p_now, 1e-3) / (1 - min(p_now, 0.999)))
            sds = need_logit / slope
            demo_est = float(np.exp(sds * sd_log))   # multiplicative demo factor

    k = int(np.argmax(deficit))
    hot = [{"x": float(G[i, 0]), "y": float(G[i, 1]), "deficit": float(deficit[i])}
           for i in np.argsort(-deficit)[:200]]
    # thin to well-separated peaks
    peaks = []
    for h in hot:
        if all((h["x"] - q["x"]) ** 2 + (h["y"] - q["y"]) ** 2 > (0.25 * (hi - lo).mean()) ** 2
               for q in peaks):
            peaks.append(h)
        if len(peaks) >= 3:
            break
    return {"peaks": peaks, "argmax": {"x": float(G[k, 0]), "y": float(G[k, 1])},
            "n_eval": len(P), "n_train": int(len(train_xy)),
            "succ_rate": float(S.mean()), "demo_factor": demo_est,
            "slope_note": slope_note, "target": target}, None


# ============================================================ E. temporal
def temporal(records, traces):
    by = {t["episode"]: t for t in traces}
    first_adv, plateau, T = [], [], None
    for r in records:
        t = by.get(r["episode"])
        if not t or not t["steps"]:
            continue
        p = np.array([s["progress_pct"] for s in t["steps"]], float)
        T = len(p)
        if p.max() <= 0:
            continue
        idx = int(np.argmax(p >= 0.95 * p.max())) + 1
        plateau.append(idx / T)
        st = np.array([s["stage"] for s in t["steps"]])
        adv = np.where(np.diff(st) > 0)[0]
        first_adv.append((adv[0] + 1) / T if adv.size else 1.0)
    if not plateau:
        return None, "no per-step traces"
    return {"median_plateau_frac": float(np.median(plateau)),
            "frac_early": float(np.mean(np.array(plateau) < 0.25)),
            "median_first_advance_frac": float(np.median(first_adv)) if first_adv else None,
            "horizon": T}, None


# ============================================================ prescriptions
def prescriptions(f, cf, disc, attr, sp, tp, label):
    P = []
    names = ["reach", "grasp", "lift", "release/threshold"]
    bi = f["bottleneck"]
    bname, bk, bd, bp, blo, bhi = f["rows"][bi]
    stage_word = names[bi] if bi < len(names) else bname

    best = cf[0] if cf else None
    uplift_txt = ""
    if best:
        _, p0, t, newp, gain = best
        uplift_txt = (f"Raising it {p0:.2f}->{t:.2f} predicts overall success "
                      f"{f['overall_model']:.3f}->{newp:.3f} (+{100*gain:.1f} pts).")

    spec = {
        1: dict(what="APPROACH / VISUAL SERVOING",
                cond="object visible, arm far from object",
                demo="Vary initial arm pose and approach azimuth; keep motion monotone "
                     "toward the object; include partial occlusion and lighting variation.",
                accept="eef-to-object distance decreases monotonically in >=90% of frames; "
                       "final distance < 3 cm."),
        2: dict(what="GRASP INITIATION",
                cond="gripper within 5 cm of object, pre-contact",
                demo="Slow terminal approach (<=2 cm/s), single decisive open->closed "
                     "transition while in contact, then 0.3-0.5 s static hold before moving.",
                accept="exactly ONE gripper open->closed transition per episode; contact "
                       "maintained >=0.3 s; no re-open before lift."),
        3: dict(what="LIFT / LOAD TRANSFER",
                cond="object grasped, still on surface",
                demo="Firm grip then decisive vertical lift to >=2x the success threshold; "
                     "include slip-and-regrip recoveries.",
                accept="object height rises monotonically >= 8 cm; no drop within 1 s."),
        4: dict(what="TERMINAL PLACEMENT / HOLD",
                cond="object lifted",
                demo="Stable hold and controlled placement; avoid premature release.",
                accept="object remains above threshold for >=1 s."),
    }.get(bi + 1, None)

    if spec:
        P.append(dict(
            priority="P0",
            title=f"Bottleneck: {stage_word} — {spec['what']}",
            evidence=(f"{bname} = {bp:.2f} (95% CI {blo:.2f}-{bhi:.2f}, {bk}/{bd}); "
                      f"lowest transition in the funnel. {uplift_txt}"),
            regime=spec["cond"],
            protocol=spec["demo"],
            acceptance=spec["accept"],
            quantity=("Target >=200 demonstration SEGMENTS covering this transition "
                      "(segments, not whole episodes: upweighting the transition window "
                      "is more sample-efficient than adding full trajectories)."),
        ))

    for r in [r for r in disc if r.get("sig_fdr")][:3]:
        d, m = r["delta"], cliff_magnitude(r["delta"])
        direction = "higher" if d > 0 else "lower"
        hint = {
            "gripper_switch_rate": ("Gripper command chatters per unit time in failures (this is "
                                    "the duration-normalised rate, so it is not an artefact of "
                                    "failures simply running longer). Curate demos with CRISP "
                                    "binary gripper signals; reject demos with >2 transitions."),
            "path_per_step": ("Failures move more per step without progressing - large but "
                              "unproductive motion. Add demos with deliberate, slow terminal "
                              "alignment."),
            "path_efficiency": ("Failures wander. Prefer direct approach trajectories; filter "
                                "demos whose path/displacement ratio exceeds 2.0."),
            "action_saturation_frac": ("Failures saturate actuator limits. Add demos with fine, "
                                       "low-magnitude terminal corrections."),
            "min_gripper_to_can": ("Failures never close the final distance. Add demos that dwell "
                                   "in the terminal 0-3 cm band."),
            "reach_closure_frac": ("Failures close less of the initial gap. Emphasise complete "
                                   "approach-to-contact trajectories."),
            "action_jerk_mean": ("Failures are jerkier. Prefer smooth, low-jerk demonstrations."),
        }.get(r["metric"], "Curate demonstrations that shift this statistic toward the "
                           "successful regime.")
        P.append(dict(
            priority="P1",
            title=f"Behavioural driver: {r['metric']} ({m} effect)",
            evidence=(f"{direction} in failures: {r['fail_mean']:.4f} vs {r['succ_mean']:.4f}; "
                      f"Cliff's delta={d:+.3f} "
                      f"(95% CI {r['lo']:+.3f},{r['hi']:+.3f}), p={r['p']:.4f}, FDR-significant."),
            regime="whole episode",
            protocol=hint,
            acceptance="post-collection: re-measure this statistic on the new data and confirm "
                       "it matches the successful-episode distribution (Cliff's delta < 0.15 vs "
                       "the success reference).",
            quantity="Re-curate or replace the worst-decile demos on this statistic.",
        ))

    if sp:
        for pk in sp["peaks"][:2]:
            qty = ("unknown (need training positions)" if sp["demo_factor"] is None else
                   f"~{sp['demo_factor']:.1f}x the current local demo density to reach "
                   f"{100*sp['target']:.0f}% success in this region")
            P.append(dict(
                priority="P0" if pk["deficit"] > 0.6 else "P2",
                title=f"Spatial coverage gap at object (x={pk['x']:+.3f}, y={pk['y']:+.3f})",
                evidence=(f"normalised failure-density/training-density deficit = "
                          f"{pk['deficit']:.2f} (1.0 = worst). {sp['slope_note']}. "
                          f"Overall success {100*sp['succ_rate']:.0f}%."),
                regime=f"object initialised within ~2 cm of (x={pk['x']:+.3f}, y={pk['y']:+.3f})",
                protocol="Collect full task demonstrations with the object placed in this region, "
                         "sampling arm start poses uniformly.",
                acceptance="local training density in this cell reaches at least the global median "
                           "cell density; re-eval success in-cell within 10 pts of global mean.",
                quantity=qty,
            ))

    if tp and tp["frac_early"] > 0.5:
        P.append(dict(
            priority="P2",
            title="Premature commitment (early plateau)",
            evidence=(f"{100*tp['frac_early']:.0f}% of episodes plateau within the first quarter "
                      f"of the horizon (median plateau at "
                      f"{100*tp['median_plateau_frac']:.0f}% of horizon)."),
            regime="post-failure recovery",
            protocol="Include RECOVERY demonstrations: a failed or slipped attempt followed by "
                     "re-approach and success. Purely optimal demos never teach correction.",
            acceptance=">=15% of the corpus contains at least one recovery event.",
            quantity="~100 recovery episodes.",
        ))
    return P


# ============================================================ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", required=True)
    ap.add_argument("--train-hdf5", default=None)
    ap.add_argument("--target", type=float, default=0.8)
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    path = Path(os.path.expanduser(args.eval))
    d, traces = load(path)
    rec = d["records"]
    txy = train_positions(os.path.expanduser(args.train_hdf5) if args.train_hdf5 else "")

    W = 78
    print("=" * W)
    print(f"QUANTITATIVE FEEDBACK  |  {d['label']}  |  n={d['n_episodes']}")
    print(f"observed: progress {d['HEADLINE']['mean_progress_pct']}%  "
          f"success {100*d['rates']['success']['rate']:.1f}%  "
          f"grasp {100*d['rates']['grasped']['rate']:.1f}%"
          + (f"  |  training demos: {len(txy)}" if txy.size else ""))
    if not _HAVE_SCIPY:
        print("note: scipy absent - Mann-Whitney via normal approximation")
    print("=" * W)

    f = funnel(rec)
    print("\nA. FUNNEL DECOMPOSITION (Markov chain over stages)")
    print("-" * W)
    print(f"   {'transition':22s} {'k/n':>9s} {'p':>7s}  95% Wilson CI")
    for i, (nm, k, n, p, lo, hi) in enumerate(f["rows"]):
        mark = "  <-- BOTTLENECK" if i == f["bottleneck"] else ""
        print(f"   {nm:22s} {k:4d}/{n:<4d} {p:7.3f}  [{lo:.3f}, {hi:.3f}]{mark}")
    print(f"   model  prod(p_k) = {f['overall_model']:.4f}   observed = {f['overall_obs']:.4f}")

    cf = counterfactual(f)
    if cf:
        print("\n   COUNTERFACTUAL UPLIFT (repair one transition, hold others fixed)")
        print(f"   {'transition':22s} {'now':>6s} {'->':>3s} {'target':>7s} {'overall':>9s} {'gain':>8s}")
        for nm, p0, t, newp, gain in cf[:6]:
            print(f"   {nm:22s} {p0:6.3f} {'->':>3s} {t:7.2f} {newp:9.3f} {100*gain:+7.1f}pt")

    rec = derive_rates(rec)
    disc, err = discriminative(rec)
    outc, _ = discriminative(rec, keys=OUTCOME_KEYS, tag="outcome")
    print("\nB. DISCRIMINATIVE STATISTICS (failure vs success, BH-FDR q=0.05)")
    print("   [duration-invariant behavioural metrics only - these can drive prescriptions]")
    print("-" * W)
    if err:
        print(f"   {err}")
    else:
        print(f"   {'metric':24s} {'fail':>9s} {'succ':>9s} {'delta':>7s} {'95% CI':>17s} {'p':>8s} sig")
        for r in disc:
            ci = f"[{r['lo']:+.2f},{r['hi']:+.2f}]" if r["lo"] is not None else "n/a"
            print(f"   {r['metric']:24s} {r['fail_mean']:9.4f} {r['succ_mean']:9.4f} "
                  f"{r['delta']:+7.3f} {ci:>17s} {r['p']:8.4f} "
                  f"{'YES' if r['sig_fdr'] else '-'}  {cliff_magnitude(r['delta'])}")

    if outc:
        print("\n   CONFIRMATORY (outcome-contaminated - reported, never prescribed on):")
        print(f"   {'metric':24s} {'fail':>10s} {'succ':>10s} {'delta':>7s}   note")
        for r in outc:
            print(f"   {r['metric']:24s} {r['fail_mean']:10.4f} {r['succ_mean']:10.4f} "
                  f"{r['delta']:+7.3f}   downstream of success itself")

    attr, aerr = attribution(rec)
    print("\nC. LOGISTIC ATTRIBUTION (ridge IRLS, standardised, odds ratios)")
    print("-" * W)
    if aerr:
        print(f"   {aerr}")
    else:
        print(f"   {'feature':24s} {'beta':>8s} {'OR':>8s}  95% CI")
        for r in attr:
            ci = (f"[{r['lo']:.2f}, {r['hi']:.2f}]" if r["lo"] is not None else "n/a")
            print(f"   {r['feature']:24s} {r['beta']:+8.3f} {r['or']:8.3f}  {ci}")

    sp, serr = spatial(rec, traces, txy, target=args.target)
    print("\nD. SPATIAL COVERAGE DEFICIT (KDE: failure density / training density)")
    print("-" * W)
    if serr:
        print(f"   {serr}")
    else:
        print(f"   eval points {sp['n_eval']}  training demos {sp['n_train']}  "
              f"success {100*sp['succ_rate']:.0f}%")
        print(f"   {sp['slope_note']}")
        if sp["demo_factor"]:
            print(f"   demo-density multiplier to reach {100*sp['target']:.0f}% success: "
                  f"{sp['demo_factor']:.2f}x")
        for i, pk in enumerate(sp["peaks"], 1):
            print(f"   peak {i}: (x={pk['x']:+.4f}, y={pk['y']:+.4f})  "
                  f"normalised deficit {pk['deficit']:.3f}")

    tp, terr = temporal(rec, traces)
    print("\nE. TEMPORAL HAZARD")
    print("-" * W)
    if terr:
        print(f"   {terr}")
    else:
        print(f"   horizon {tp['horizon']} steps; median plateau at "
              f"{100*tp['median_plateau_frac']:.0f}% of horizon; "
              f"{100*tp['frac_early']:.0f}% plateau in first quarter")

    print("\n" + "=" * W)
    print("PRESCRIPTIONS FOR DATA GENERATORS")
    print("=" * W)
    presc = prescriptions(f, cf, disc, attr, sp, tp, d["label"])
    for i, p in enumerate(presc, 1):
        print(f"\n[{p['priority']}] {i}. {p['title']}")
        print(f"   EVIDENCE   : {p['evidence']}")
        print(f"   REGIME     : {p['regime']}")
        print(f"   PROTOCOL   : {p['protocol']}")
        print(f"   ACCEPTANCE : {p['acceptance']}")
        print(f"   QUANTITY   : {p['quantity']}")

    if args.json_out:
        Path(args.json_out).write_text(json.dumps({
            "label": d["label"], "funnel": f["rows"], "counterfactual": cf,
            "discriminative": disc, "attribution": attr, "spatial": sp,
            "temporal": tp, "prescriptions": presc}, indent=2, default=str))
        print(f"\nmachine-readable -> {args.json_out}")
    print("\nFEEDBACK_V2_DONE")


if __name__ == "__main__":
    main()
