#!/usr/bin/env python3
"""Closed-loop evaluation of a GR00T policy on robosuite PickPlaceCan.

Scores a checkpoint by *task success* over N rollouts from fixed, paired seeds,
so every model faces byte-identical initial states. Reports success rate with a
Wilson 95% interval plus stage diagnostics.

Consistency contract with training data (scripts/convert_robomimic.py):
  * images rendered via env.sim.render(...)[::-1] -- same call, same flip
  * state = [eef_pos(3), quat2axisangle(3), gripper_qpos(2)]
  * actions passed through RAW (no gripper normalize/invert): the RoboMimic
    demos we trained on are recorded robosuite actions already.
Any drift here silently shifts the policy off-distribution, so keep it in sync.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import time

import numpy as np

from metrics import EpisodeTracker, aggregate

TASK_SPECS = {
    "can":  {"env": "PickPlaceCan", "text": "pick up the can and place it in the correct bin"},
    "lift": {"env": "Lift",         "text": "lift the cube"},
}
CAMERAS = ("agentview", "robot0_eye_in_hand")
STATE_KEYS = ("x", "y", "z", "roll", "pitch", "yaw", "gripper")
ACTION_ORDER = ("x", "y", "z", "roll", "pitch", "yaw", "gripper")


def quat2axisangle(q):
    q = np.asarray(q, dtype=np.float64)
    if q[3] > 1.0:
        q = q / np.linalg.norm(q)
    elif q[3] < -1.0:
        q = -q / np.linalg.norm(q)
        q[3] = min(q[3], 1.0)
    den = math.sqrt(max(1.0 - q[3] * q[3], 0.0))
    if math.isclose(den, 0.0):
        return np.zeros(3)
    return (q[:3] * 2.0 * math.acos(np.clip(q[3], -1.0, 1.0))) / den


def wilson(successes: int, n: int, z: float = 1.96):
    """Wilson score interval — correct at the 0%/100% extremes, unlike normal approx."""
    if n == 0:
        return 0.0, 0.0, 0.0
    p = successes / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(max(p * (1 - p) / n + z * z / (4 * n * n), 0.0)) / d
    return p, max(0.0, centre - half), min(1.0, centre + half)


class PickPlaceCanEnv:
    """robosuite PickPlaceCan exposing GR00T-format observations."""

    def __init__(self, res: int = 128, horizon: int = 400, task: str = "can"):
        import robosuite
        from robosuite.controllers import load_controller_config

        self.res = res
        self.task = task
        self.task_text = TASK_SPECS[task]["text"]
        self._env = robosuite.make(
            env_name=TASK_SPECS[task]["env"],
            robots="Panda",
            controller_configs=load_controller_config(default_controller="OSC_POSE"),
            has_renderer=False,
            has_offscreen_renderer=True,
            use_camera_obs=False,      # we render explicitly, matching the converter
            use_object_obs=True,
            ignore_done=True,
            horizon=horizon,
            reward_shaping=True,
            control_freq=20,           # matches dataset control_freq / fps
            camera_names=list(CAMERAS),
            camera_heights=res,
            camera_widths=res,
        )

    def _render(self, cam: str) -> np.ndarray:
        # identical to convert_robomimic.py: robosuite renders upside-down
        return self._env.sim.render(width=self.res, height=self.res, camera_name=cam)[::-1]

    def _obs(self, raw) -> dict:
        xyz = np.asarray(raw["robot0_eef_pos"], dtype=np.float32)
        rpy = quat2axisangle(raw["robot0_eef_quat"]).astype(np.float32)
        grip = np.asarray(raw["robot0_gripper_qpos"], dtype=np.float32)
        return {
            "video.image": self._render("agentview")[None],
            "video.wrist_image": self._render("robot0_eye_in_hand")[None],
            "state.x": xyz[0:1][None],
            "state.y": xyz[1:2][None],
            "state.z": xyz[2:3][None],
            "state.roll": rpy[0:1][None],
            "state.pitch": rpy[1:2][None],
            "state.yaw": rpy[2:3][None],
            "state.gripper": grip[None],
            "annotation.human.action.task_description": self.task_text,
        }

    def reset(self, seed: int) -> dict:
        """Deterministic initial state: same seed => same scene for every model.

        Raw robosuite envs have no .seed() (that lives on LIBERO's wrapper). The
        placement initialiser draws from the global numpy/random RNGs, so seeding
        those before reset() is what actually pins the scene. Verified by
        --check-determinism.
        """
        import random as _random

        np.random.seed(seed)
        _random.seed(seed)
        return self._obs(self._env.reset())

    def object_state(self) -> np.ndarray:
        """Object + eef pose, used to prove two resets with one seed match."""
        raw = self._env._get_observations()
        return np.concatenate([
            np.asarray(raw.get("Can_pos", raw.get("cube_pos", np.zeros(3))), dtype=np.float64).ravel(),
            np.asarray(raw.get("Can_quat", raw.get("cube_quat", np.zeros(4))), dtype=np.float64).ravel(),
            np.asarray(raw["robot0_eef_pos"], dtype=np.float64).ravel(),
        ])

    def step(self, action7: np.ndarray, render: bool = True):
        """Step the sim. ``render=False`` skips camera rendering.

        The policy only consumes images at action-chunk boundaries, so rendering
        every physics step wastes ~(n_action_steps-1)/n_action_steps of the work.
        Rendering dominates per-step cost, so this is the main eval speedup.
        """
        raw, reward, done, _info = self._env.step(np.asarray(action7, dtype=np.float64))
        obs = self._obs(raw) if render else None
        return obs, float(reward), bool(self._env._check_success())

    @property
    def raw_env(self):
        return self._env

    def close(self):
        self._env.close()


def build_policy(model_path: str, embodiment_tag: str, denoising_steps: int):
    import torch

    from gr00t.data.embodiment_tags import EmbodimentTag
    from gr00t.policy.gr00t_policy import Gr00tPolicy

    policy = Gr00tPolicy(
        embodiment_tag=EmbodimentTag.resolve(embodiment_tag),
        model_path=model_path,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    policy.model.action_head.num_inference_timesteps = denoising_steps
    return policy


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--label", default="")
    ap.add_argument("--n-episodes", type=int, default=50)
    ap.add_argument("--max-steps", type=int, default=400)
    ap.add_argument("--n-action-steps", type=int, default=8, help="chunk steps executed per inference")
    ap.add_argument("--res", type=int, default=128)
    ap.add_argument("--base-seed", type=int, default=100000)
    ap.add_argument("--embodiment-tag", default="libero_sim")
    ap.add_argument("--denoising-steps", type=int, default=4)
    ap.add_argument("--task", default="can", choices=["can", "lift"])
    ap.add_argument("--check-determinism", action="store_true",
                    help="verify same seed => same scene, then exit (no policy load)")
    ap.add_argument("--random-policy", action="store_true",
                    help="ignore the model and act uniformly at random (floor baseline)")
    args = ap.parse_args()

    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

    if args.check_determinism:
        env = PickPlaceCanEnv(res=args.res, horizon=args.max_steps + 10, task=args.task)
        ok = True
        for seed in (args.base_seed, args.base_seed + 1):
            env.reset(seed); a = env.object_state()
            env.reset(seed); b = env.object_state()
            env.reset(seed + 500); c = env.object_state()
            same = np.allclose(a, b, atol=1e-9)
            differs = not np.allclose(a, c, atol=1e-9)
            ok &= same and differs
            print(f"seed={seed}: repeat_identical={same} different_seed_differs={differs}")
            print(f"  scene={np.round(a, 5)}")
        env.close()
        print("DETERMINISM_OK" if ok else "DETERMINISM_FAILED")
        return

    if args.random_policy:
        policy = modality_cfg = parse_observation_gr00t = None
    else:
        from gr00t.data.utils import parse_observation_gr00t

        policy = build_policy(args.model_path, args.embodiment_tag, args.denoising_steps)
        modality_cfg = policy.get_modality_config()
    env = PickPlaceCanEnv(res=args.res, horizon=args.max_steps + 10, task=args.task)

    # paired seeds: identical across every model we score
    seeds = [args.base_seed + i for i in range(args.n_episodes)]
    records = []
    traces = []
    t_start = time.time()

    for ep, seed in enumerate(seeds):
        obs = env.reset(seed)
        tracker = EpisodeTracker(env.raw_env, task=args.task)
        tracker.reset()
        success, steps, best_reward = False, 0, 0.0

        while steps < args.max_steps and not success:
            if args.random_policy:
                chunk = None
            else:
                parsed = parse_observation_gr00t(obs, modality_cfg)
                chunk, _ = policy.get_action(parsed)
                chunk = {f"action.{k}": np.asarray(v)[0] for k, v in chunk.items()}

            for j in range(args.n_action_steps):
                if steps >= args.max_steps or success:
                    break
                if chunk is None:
                    act = np.random.uniform(-1.0, 1.0, size=7)
                else:
                    act = np.concatenate(
                        [np.atleast_1d(np.atleast_1d(chunk[f"action.{k}"])[j]) for k in ACTION_ORDER]
                    )
                # RAW pass-through: training actions were recorded robosuite actions.
                # Only render on the final chunk step -- that frame is the one the
                # policy actually sees next. Cuts rendering by ~n_action_steps x.
                last_of_chunk = (j == args.n_action_steps - 1) or (steps + 1 >= args.max_steps)
                new_obs, reward, success = env.step(act, render=last_of_chunk)
                if new_obs is not None:
                    obs = new_obs
                tracker.step(act, success)
                best_reward = max(best_reward, reward)
                steps += 1

        rec = tracker.result(ep, seed)
        rec["best_shaped_reward"] = round(best_reward, 4)
        records.append(rec)
        traces.append({"episode": ep, "seed": seed, "steps": tracker.trace})
        n_ok = sum(r["success"] for r in records)
        mp = float(np.mean([r["progress_pct"] for r in records]))
        print(f"  ep {ep+1}/{len(seeds)} seed={seed} progress={rec['progress_pct']:.1f}% "
              f"stage={rec['stage_reached']} fail={rec['failure_mode']} steps={steps} "
              f"| success {n_ok}/{len(records)} mean_progress {mp:.1f}%", flush=True)

    env.close()
    summary = aggregate(records, args.label or Path(args.model_path).name, extra={
        "model_path": args.model_path,
        "env": f"robosuite/{TASK_SPECS[args.task]['env']}",
        "task": args.task,
        "policy": "random" if args.random_policy else "gr00t",
        "max_steps": args.max_steps,
        "n_action_steps": args.n_action_steps,
        "base_seed": args.base_seed,
        "res": args.res,
        "wall_seconds": round(time.time() - t_start, 1),
        "progress_scale": "reach=10 grasp=35 lift=50 hover=70 success=100 (robosuite staged multipliers)",
    })
    summary["records"] = records
    tp = Path(args.out).with_suffix(".traces.json")
    tp.parent.mkdir(parents=True, exist_ok=True)
    tp.write_text(json.dumps({"label": summary["label"], "task": args.task,
                              "progress_scale": summary.get("progress_scale"),
                              "episodes": traces}))
    print(f"  traces -> {tp}")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2))
    h = summary["HEADLINE"]
    r = summary["rates"]
    print("\n" + "=" * 66)
    print(f"  {summary['label']}   n={summary['n_episodes']}")
    print(f"  MEAN PROGRESS : {h['mean_progress_pct']:.2f}%  boot95={h['progress_boot95']}")
    print(f"  success       : {r['success']['count']}/{summary['n_episodes']}  "
          f"({r['success']['rate']:.1%})  wilson95={r['success']['wilson95']}")
    for k in ("reached", "grasped", "lifted", "hovered"):
        print(f"  {k:14s}: {r[k]['count']}/{summary['n_episodes']} ({r[k]['rate']:.1%})")
    print(f"  stages        : {summary['stage_histogram']}")
    print(f"  failure modes : {summary['failure_modes']}")
    print("=" * 66)
    print(f"EVAL_DONE {summary['label']} mean_progress={h['mean_progress_pct']:.2f}% -> {out}")


if __name__ == "__main__":
    main()
