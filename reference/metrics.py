"""Detailed per-episode metrics for robosuite PickPlaceCan rollouts.

Binary success is near-useless for short-trained VLAs (everything scores 0), so
this module tracks *how far along the task* a policy got, using robosuite's own
physics-grounded staged rewards plus behavioural diagnostics.

Progress scale (robosuite's own multipliers, so it is not an invented number):
    reach -> 0.10 | grasp -> 0.35 | lift -> 0.50 | hover -> 0.70 | success -> 1.00
"""
from __future__ import annotations

import math

import numpy as np

STAGE_NAMES = ["none", "reach", "grasp", "lift", "hover", "success"]
LIFT_STAGE_NAMES = ["none", "reach", "grasp", "lift", "success"]
# Lift progress scale: reaching contributes 0-30, a confirmed grasp 60,
# then height gain interpolates 60->100. Success (cube > table+0.04) == 100.
LIFT_REACH_W, LIFT_GRASP_V = 30.0, 60.0
LIFT_SUCCESS_HEIGHT = 0.04
# robosuite PickPlace multipliers
REACH_MULT, GRASP_MULT, LIFT_MULT, HOVER_MULT = 0.10, 0.35, 0.50, 0.70


def wilson(successes: int, n: int, z: float = 1.96):
    if n == 0:
        return 0.0, 0.0, 0.0
    p = successes / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(max(p * (1 - p) / n + z * z / (4 * n * n), 0.0)) / d
    return p, max(0.0, centre - half), min(1.0, centre + half)


def bootstrap_ci(values, n_boot: int = 2000, seed: int = 1729, alpha: float = 0.05):
    """Percentile bootstrap CI for a mean. Deterministic via fixed seed."""
    v = np.asarray(values, dtype=np.float64)
    if v.size == 0:
        return 0.0, 0.0, 0.0
    if v.size == 1:
        return float(v[0]), float(v[0]), float(v[0])
    rng = np.random.default_rng(seed)
    means = rng.choice(v, size=(n_boot, v.size), replace=True).mean(axis=1)
    return (float(v.mean()),
            float(np.percentile(means, 100 * alpha / 2)),
            float(np.percentile(means, 100 * (1 - alpha / 2))))


class EpisodeTracker:
    """Accumulates fine-grained signals over one rollout."""

    def __init__(self, env, task: str = "can"):
        self.env = env
        self.task = task
        self.r_lift_frac = 0.0
        self.r_reach = self.r_grasp = self.r_lift = self.r_hover = 0.0
        self.max_staged = 0.0
        self.stage = 0
        self.success = False

        self.ever_reached = self.ever_grasped = self.ever_lifted = self.ever_hovered = False
        self.step_first = {}                    # stage name -> step index
        self.steps_in_stage = [0] * len(STAGE_NAMES)

        self.min_grip_to_obj = float("inf")
        self.init_grip_to_obj = None
        self.min_obj_to_bin_xy = float("inf")
        self.init_obj_to_bin_xy = None
        self.max_obj_height_gain = 0.0
        self.init_obj_pos = None
        self.max_obj_disp = 0.0

        self.eef_path_len = 0.0
        self._prev_eef = None
        self.init_eef = None
        self.actions = []
        self._prev_action = None
        self.action_jerk = []
        self.grip_cmds = []
        self.n_steps = 0
        self.obj_fell = False
        self.trace = []          # per-step time series for plotting

    # ---- low-level probes (defensive: never let a missing attr kill a run) ----
    def _obj_index(self) -> int:
        """Index of the *active* object.

        PickPlaceCan still builds all four PickPlace objects (Milk/Bread/Cereal/Can)
        and parks the unused three off-world at [10,10,10]. objects[0] is Milk, so
        naive indexing measures distance to a dummy ~16.8 m away.
        """
        oid = getattr(self.env, "object_id", None)
        if isinstance(oid, (int, np.integer)):
            return int(oid)
        try:
            for i, o in enumerate(self.env.objects):
                if "can" in o.name.lower():
                    return i
        except Exception:
            pass
        return 0

    def _obj(self):
        try:
            if self.task == "lift":
                return self.env.cube
            return self.env.objects[self._obj_index()]
        except Exception:
            return None

    def _obj_pos(self):
        try:
            if self.task == "lift":
                return np.array(self.env.sim.data.body_xpos[self.env.cube_body_id], dtype=np.float64)
            o = self._obj()
            return np.array(self.env.sim.data.body_xpos[self.env.obj_body_id[o.name]], dtype=np.float64)
        except Exception:
            return None

    def _eef_pos(self):
        try:
            return np.array(self.env._eef_xpos, dtype=np.float64)
        except Exception:
            try:
                return np.array(self.env.sim.data.site_xpos[
                    self.env.sim.model.site_name2id("gripper0_grip_site")], dtype=np.float64)
            except Exception:
                return None

    def _grip_to_obj(self):
        try:
            o = self._obj()
            return float(self.env._gripper_to_target(
                gripper=self.env.robots[0].gripper, target=o.root_body,
                target_type="body", return_distance=True))
        except Exception:
            e, p = self._eef_pos(), self._obj_pos()
            return float(np.linalg.norm(e - p)) if e is not None and p is not None else None

    def _target_bin_xy(self):
        if self.task == "lift":
            return None      # Lift has no target bin
        try:
            # bin for the ACTIVE object, not bin 0
            return np.array(
                self.env.target_bin_placements[self._obj_index(), :2], dtype=np.float64)
        except Exception:
            return None

    def _lift_staged(self):
        """Lift has no staged_rewards(); derive equivalents from physics.

        reach: 1-tanh(10*d) in [0,1] (robosuite's own shaping)
        grasp: 0.25 when the gripper actually holds the cube
        lift : fraction of the 0.04 m success height achieved while grasped
        """
        d = self._grip_to_obj()
        rr = float(1.0 - np.tanh(10.0 * d)) if d is not None else 0.0
        try:
            grasped = bool(self.env._check_grasp(
                gripper=self.env.robots[0].gripper,
                object_geoms=self.env.cube.contact_geoms))
        except Exception:
            grasped = False
        rg = 0.25 if grasped else 0.0
        rl = 0.0
        if grasped:
            p = self._obj_pos()
            try:
                table_z = float(self.env.model.mujoco_arena.table_offset[2])
            except Exception:
                table_z = None
            if p is not None and table_z is not None:
                rl = float(np.clip((p[2] - table_z) / LIFT_SUCCESS_HEIGHT, 0.0, 1.0))
        self.r_lift_frac = max(self.r_lift_frac, rl)
        return rr, rg, rl, 0.0

    def reset(self):
        self.init_grip_to_obj = self._grip_to_obj()
        self.init_obj_pos = self._obj_pos()
        self.init_eef = self._prev_eef = self._eef_pos()
        p, b = self._obj_pos(), self._target_bin_xy()
        if p is not None and b is not None:
            self.init_obj_to_bin_xy = float(np.linalg.norm(p[:2] - b))

    def step(self, action, success: bool):
        self.n_steps += 1
        a = np.asarray(action, dtype=np.float64)
        self.actions.append(a)
        self.grip_cmds.append(float(a[6]))
        if self._prev_action is not None:
            self.action_jerk.append(float(np.mean(np.abs(a - self._prev_action))))
        self._prev_action = a

        if self.task == "lift":
            rr, rg, rl, rh = self._lift_staged()
        else:
            try:
                rr, rg, rl, rh = self.env.staged_rewards()
            except Exception:
                rr = rg = rl = rh = 0.0
        self.r_reach, self.r_grasp = max(self.r_reach, rr), max(self.r_grasp, rg)
        self.r_lift, self.r_hover = max(self.r_lift, rl), max(self.r_hover, rh)
        staged = max(rr, rg, rl, rh)
        self.max_staged = max(self.max_staged, staged)

        if success:
            self.success = True

        # stage ladder
        stage = 0
        if self.task == "lift":
            if rr > 0.30:
                stage = 1
            if rg > 0.0:
                stage = 2
            if rl > 0.25:
                stage = 3
            if success:
                stage = 4
        else:
            if rr > 0.02:
                stage = 1
            if rg >= GRASP_MULT - 1e-9:
                stage = 2
            if rl > GRASP_MULT + 1e-9:
                stage = 3
            if rh > LIFT_MULT + 1e-9:
                stage = 4
            if success:
                stage = 5
        self.stage = max(self.stage, stage)
        self.steps_in_stage[stage] += 1
        for idx, flag in ((1, "ever_reached"), (2, "ever_grasped"), (3, "ever_lifted"),
                          (4, "ever_hovered"), (5, "success")):
            if stage >= idx:
                setattr(self, flag, True) if idx < 5 else None
                self.step_first.setdefault(STAGE_NAMES[idx], self.n_steps)

        # ---- per-step trace row (time series for graphs) ----
        _p = self._obj_pos()
        _e = self._eef_pos()
        _d = self._grip_to_obj()
        _h = None
        if _p is not None and self.init_obj_pos is not None:
            _h = float(_p[2] - self.init_obj_pos[2])
        if self.task == "lift":
            _prog = (100.0 if success else
                     (LIFT_GRASP_V + (100.0 - LIFT_GRASP_V) * self.r_lift_frac
                      if self.ever_grasped else LIFT_REACH_W * min(self.r_reach, 1.0)))
        else:
            _prog = 100.0 if success else 100.0 * self.max_staged
        self.trace.append({
            "t": self.n_steps,
            "progress_pct": round(float(_prog), 3),
            "grip_to_obj": round(float(_d), 5) if _d is not None else None,
            "obj_height_gain": round(_h, 5) if _h is not None else None,
            "obj_x": round(float(_p[0]), 5) if _p is not None else None,
            "obj_y": round(float(_p[1]), 5) if _p is not None else None,
            "obj_z": round(float(_p[2]), 5) if _p is not None else None,
            "eef_x": round(float(_e[0]), 5) if _e is not None else None,
            "eef_y": round(float(_e[1]), 5) if _e is not None else None,
            "eef_z": round(float(_e[2]), 5) if _e is not None else None,
            "r_reach": round(float(rr), 4), "r_grasp": round(float(rg), 4),
            "r_lift": round(float(rl), 4), "r_hover": round(float(rh), 4),
            "stage": int(stage), "success": bool(success),
            "action_abs_mean": round(float(np.mean(np.abs(a))), 4),
            "gripper_cmd": round(float(a[6]), 4),
        })

        d = self._grip_to_obj()
        if d is not None:
            self.min_grip_to_obj = min(self.min_grip_to_obj, d)
        p = self._obj_pos()
        if p is not None:
            if self.init_obj_pos is not None:
                self.max_obj_height_gain = max(self.max_obj_height_gain,
                                               float(p[2] - self.init_obj_pos[2]))
                self.max_obj_disp = max(self.max_obj_disp,
                                        float(np.linalg.norm(p - self.init_obj_pos)))
                if p[2] < self.init_obj_pos[2] - 0.20:
                    self.obj_fell = True
            b = self._target_bin_xy()
            if b is not None:
                self.min_obj_to_bin_xy = min(self.min_obj_to_bin_xy,
                                             float(np.linalg.norm(p[:2] - b)))
        e = self._eef_pos()
        if e is not None and self._prev_eef is not None:
            self.eef_path_len += float(np.linalg.norm(e - self._prev_eef))
        if e is not None:
            self._prev_eef = e

    # ---- summary ----
    def result(self, episode: int, seed: int) -> dict:
        A = np.asarray(self.actions) if self.actions else np.zeros((1, 7))
        eef_disp = (float(np.linalg.norm(self._prev_eef - self.init_eef))
                    if self._prev_eef is not None and self.init_eef is not None else 0.0)
        reach_frac = None
        if self.init_grip_to_obj and self.min_grip_to_obj < float("inf") and self.init_grip_to_obj > 1e-9:
            reach_frac = float(np.clip(
                1.0 - self.min_grip_to_obj / self.init_grip_to_obj, 0.0, 1.0))
        bin_frac = None
        if self.init_obj_to_bin_xy and self.min_obj_to_bin_xy < float("inf") and self.init_obj_to_bin_xy > 1e-9:
            bin_frac = float(np.clip(
                1.0 - self.min_obj_to_bin_xy / self.init_obj_to_bin_xy, 0.0, 1.0))

        # headline: 0-100
        if self.task == "lift":
            # reach 0-30 | grasp 60 | grasp+height 60-100 | success 100
            if self.success:
                progress_pct = 100.0
            elif self.ever_grasped:
                progress_pct = round(LIFT_GRASP_V + (100.0 - LIFT_GRASP_V) * self.r_lift_frac, 2)
            else:
                progress_pct = round(LIFT_REACH_W * min(self.r_reach, 1.0), 2)
        else:
            # robosuite's own stage multipliers (success == 100)
            progress_pct = 100.0 if self.success else round(100.0 * self.max_staged, 2)

        grip = np.asarray(self.grip_cmds) if self.grip_cmds else np.zeros(1)
        grip_switches = int(np.sum(np.abs(np.diff(np.sign(grip))) > 0)) if grip.size > 1 else 0

        obj = "cube" if self.task == "lift" else "can"
        if self.success:
            failure = "none_success"
        elif self.max_obj_disp < 0.005 and eef_disp < 0.02:
            failure = "policy_frozen"
        elif not self.ever_reached:
            failure = f"never_reached_{obj}"
        elif not self.ever_grasped:
            failure = "reached_but_no_grasp"
        elif not self.ever_lifted:
            failure = "grasped_but_no_lift"
        elif self.task == "lift":
            failure = "lifted_but_below_threshold"
        elif not self.ever_hovered:
            failure = "lifted_but_no_hover"
        else:
            failure = "hovered_but_no_place"

        return {
            "episode": episode, "seed": seed,
            # headline
            "success": bool(self.success),
            "progress_pct": progress_pct,
            "stage_reached": (LIFT_STAGE_NAMES if self.task == "lift" else STAGE_NAMES)[self.stage],
            "stage_index": int(self.stage),
            "failure_mode": failure,
            # stage flags
            "ever_reached": bool(self.ever_reached), "ever_grasped": bool(self.ever_grasped),
            "ever_lifted": bool(self.ever_lifted), "ever_hovered": bool(self.ever_hovered),
            # staged reward components (max over episode)
            "r_reach_max": round(self.r_reach, 4), "r_grasp_max": round(self.r_grasp, 4),
            "r_lift_max": round(self.r_lift, 4), "r_hover_max": round(self.r_hover, 4),
            "max_staged_reward": round(self.max_staged, 4),
            # continuous geometry
            "init_gripper_to_can": round(self.init_grip_to_obj, 4) if self.init_grip_to_obj else None,
            "min_gripper_to_can": round(self.min_grip_to_obj, 4) if self.min_grip_to_obj < float("inf") else None,
            "reach_closure_frac": round(reach_frac, 4) if reach_frac is not None else None,
            "init_can_to_bin_xy": round(self.init_obj_to_bin_xy, 4) if self.init_obj_to_bin_xy else None,
            "min_can_to_bin_xy": round(self.min_obj_to_bin_xy, 4) if self.min_obj_to_bin_xy < float("inf") else None,
            "bin_closure_frac": round(bin_frac, 4) if bin_frac is not None else None,
            "max_can_height_gain": round(self.max_obj_height_gain, 4),
            "max_can_displacement": round(self.max_obj_disp, 4),
            "can_fell_off": bool(self.obj_fell),
            # timing
            "steps": self.n_steps,
            "step_first_reach": self.step_first.get("reach"),
            "step_first_grasp": self.step_first.get("grasp"),
            "step_first_lift": self.step_first.get("lift"),
            "step_first_hover": self.step_first.get("hover"),
            "step_first_success": self.step_first.get("success"),
            # behaviour diagnostics
            "eef_path_length": round(self.eef_path_len, 4),
            "eef_net_displacement": round(eef_disp, 4),
            "path_efficiency": round(eef_disp / self.eef_path_len, 4) if self.eef_path_len > 1e-9 else 0.0,
            "action_abs_mean": round(float(np.mean(np.abs(A))), 4),
            "action_abs_max": round(float(np.max(np.abs(A))), 4),
            "action_saturation_frac": round(float(np.mean(np.abs(A) > 0.99)), 4),
            "action_jerk_mean": round(float(np.mean(self.action_jerk)), 4) if self.action_jerk else 0.0,
            "gripper_cmd_mean": round(float(np.mean(grip)), 4),
            "gripper_closed_frac": round(float(np.mean(grip > 0)), 4),
            "gripper_switches": grip_switches,
        }


def aggregate(records: list[dict], label: str, extra: dict | None = None) -> dict:
    """Fleet-level summary: rates with Wilson CIs, means with bootstrap CIs."""
    n = len(records)
    out = {"label": label, "n_episodes": n}
    if extra:
        out.update(extra)
    if n == 0:
        return out

    def rate(flag):
        k = sum(bool(r[flag]) for r in records)
        p, lo, hi = wilson(k, n)
        return {"count": k, "rate": round(p, 4),
                "wilson95": [round(lo, 4), round(hi, 4)]}

    out["rates"] = {
        "success": rate("success"), "reached": rate("ever_reached"),
        "grasped": rate("ever_grasped"), "lifted": rate("ever_lifted"),
        "hovered": rate("ever_hovered"), "can_fell_off": rate("can_fell_off"),
    }

    cont = ["progress_pct", "max_staged_reward", "r_reach_max", "r_grasp_max",
            "r_lift_max", "r_hover_max", "min_gripper_to_can", "reach_closure_frac",
            "min_can_to_bin_xy", "bin_closure_frac", "max_can_height_gain",
            "max_can_displacement", "steps", "eef_path_length", "eef_net_displacement",
            "path_efficiency", "action_abs_mean", "action_saturation_frac",
            "action_jerk_mean", "gripper_cmd_mean", "gripper_closed_frac", "gripper_switches"]
    stats = {}
    for k in cont:
        vals = [r[k] for r in records if r.get(k) is not None]
        if not vals:
            continue
        m, lo, hi = bootstrap_ci(vals)
        stats[k] = {"mean": round(m, 4), "boot95": [round(lo, 4), round(hi, 4)],
                    "median": round(float(np.median(vals)), 4),
                    "std": round(float(np.std(vals)), 4),
                    "min": round(float(np.min(vals)), 4),
                    "max": round(float(np.max(vals)), 4)}
    out["continuous"] = stats

    names = LIFT_STAGE_NAMES if any(
        r["stage_reached"] in ("success",) or True for r in records[:0]) else STAGE_NAMES
    seen = {r["stage_reached"] for r in records}
    names = LIFT_STAGE_NAMES if seen <= set(LIFT_STAGE_NAMES) and "hover" not in seen else STAGE_NAMES
    out["stage_histogram"] = {s: sum(1 for r in records if r["stage_reached"] == s)
                              for s in names}
    modes = {}
    for r in records:
        modes[r["failure_mode"]] = modes.get(r["failure_mode"], 0) + 1
    out["failure_modes"] = dict(sorted(modes.items(), key=lambda kv: -kv[1]))
    # headline for quick scanning / ranking
    out["HEADLINE"] = {
        "mean_progress_pct": stats.get("progress_pct", {}).get("mean"),
        "progress_boot95": stats.get("progress_pct", {}).get("boot95"),
        "success_rate": out["rates"]["success"]["rate"],
        "grasp_rate": out["rates"]["grasped"]["rate"],
        "reach_closure": stats.get("reach_closure_frac", {}).get("mean"),
    }
    return out
