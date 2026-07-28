#!/usr/bin/env python3
"""Convert RoboMimic PickPlaceCan demos -> LeRobot v2.1 (GR00T libero_sim layout).

Renders images by replaying recorded MuJoCo sim states in robosuite, then writes
parquet + mp4 + meta mirroring demo_data/libero_demo exactly.

Run in the robomimic island venv (needs robosuite, h5py, pyarrow, imageio).
Stats (stats.json / relative_stats.json) are generated afterwards in the GR00T venv.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import h5py
import imageio.v2 as imageio
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

TASK_TEXTS = {
    "can":  "pick up the can and place it in the correct bin",
    "lift": "lift the cube",
}
CAM_MAP = {  # robosuite camera -> LeRobot video key
    "agentview": "observation.images.image",
    "robot0_eye_in_hand": "observation.images.wrist_image",
}


def postprocess_model_xml(xml_str: str) -> str:
    """Rewrite absolute asset paths baked into RoboMimic's stored MuJoCo XML.

    The published datasets embed the collector's machine paths
    (/home/soroushn/code/robosuite-dev/...). robosuite 1.4.1 dropped its own
    helper for this, so remap every mesh/texture file onto the local install.
    """
    import xml.etree.ElementTree as ET

    import robosuite

    local_root = os.path.split(robosuite.__file__)[0].split("/")
    root = ET.fromstring(xml_str)
    asset = root.find("asset")
    if asset is None:
        return xml_str
    for elem in list(asset.findall("mesh")) + list(asset.findall("texture")):
        old = elem.get("file")
        if not old:
            continue
        parts = old.split("/")
        if "robosuite" not in parts:
            continue
        idx = max(i for i, v in enumerate(parts) if v == "robosuite")
        elem.set("file", "/".join(local_root + parts[idx + 1:]))
    return ET.tostring(root, encoding="utf8").decode("utf8")


def quat2axisangle(q: np.ndarray) -> np.ndarray:
    """robosuite xyzw quaternion -> axis-angle (3,). Mirrors LIBERO's helper."""
    q = np.asarray(q, dtype=np.float64)
    if q[3] > 1.0:
        q = q / np.linalg.norm(q)
    elif q[3] < -1.0:
        q = -q / np.linalg.norm(q)
        q[3] = min(q[3], 1.0)
    den = np.sqrt(max(1.0 - q[3] * q[3], 0.0))
    if np.isclose(den, 0.0):
        return np.zeros(3)
    return (q[:3] * 2.0 * np.arccos(np.clip(q[3], -1.0, 1.0))) / den


def select_demos(f: h5py.File, target_frames: int, max_demos: int | None):
    """Deterministically pick demos (evenly spread) until target_frames is reached."""
    d = f["data"]
    names = sorted(d.keys(), key=lambda x: int(x.split("_")[1]))
    lens = np.array([d[n]["actions"].shape[0] for n in names])
    order = np.linspace(0, len(names) - 1, len(names)).astype(int)  # even spread == natural order
    chosen, total = [], 0
    for i in order:
        if target_frames and total >= target_frames:
            break
        if max_demos and len(chosen) >= max_demos:
            break
        chosen.append(names[i])
        total += int(lens[i])
    return chosen, total


def build_env(env_args: dict, res: int):
    import robosuite
    kw = dict(env_args["env_kwargs"])
    kw.update(
        has_renderer=False,
        has_offscreen_renderer=True,
        use_camera_obs=False,        # we call sim.render directly
        use_object_obs=True,
        ignore_done=True,
        camera_names=list(CAM_MAP.keys()),
        camera_heights=res,
        camera_widths=res,
        reward_shaping=False,
    )
    kw.pop("camera_depths", None)
    return robosuite.make(env_name=env_args["env_name"], **kw)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, choices=["ph", "mh", "mg"])
    ap.add_argument("--task", default="can", choices=["can", "lift"])
    ap.add_argument("--hdf5", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--target-frames", type=int, default=23207)
    ap.add_argument("--max-demos", type=int, default=None)
    ap.add_argument("--demo-list", default=None)
    ap.add_argument("--res", type=int, default=128)
    ap.add_argument("--fps", type=int, default=20)
    args = ap.parse_args()

    src_h5 = args.hdf5 or os.path.expanduser(f"~/robomimic_data/{args.task}/{args.src}/low_dim.hdf5")
    out = Path(os.path.expanduser(args.out))
    (out / "data" / "chunk-000").mkdir(parents=True, exist_ok=True)
    for vk in CAM_MAP.values():
        (out / "videos" / "chunk-000" / vk).mkdir(parents=True, exist_ok=True)
    (out / "meta").mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

    f = h5py.File(src_h5, "r")
    env_args = json.loads(f["data"].attrs["env_args"])
    if args.demo_list:
        names = [ln.strip() for ln in open(args.demo_list) if ln.strip()]
        demos = [n for n in names if n in f["data"]]
        planned = int(sum(f["data"][n]["actions"].shape[0] for n in demos))
        print(f"[{args.src}] explicit list: {len(demos)} demos, {planned} frames", flush=True)
    else:
        demos, planned = select_demos(f, args.target_frames, args.max_demos)
    print(f"[{args.src}] selected {len(demos)} demos ≈ {planned} frames (target {args.target_frames})",
          flush=True)

    env = build_env(env_args, args.res)
    episodes_meta, global_index, total_frames = [], 0, 0

    for ep_idx, name in enumerate(demos):
        ep = f["data"][name]
        states = ep["states"][:]
        actions = np.asarray(ep["actions"][:], dtype=np.float32)
        T = actions.shape[0]

        eef_pos = ep["obs"]["robot0_eef_pos"][:]
        eef_quat = ep["obs"]["robot0_eef_quat"][:]
        grip = ep["obs"]["robot0_gripper_qpos"][:]

        # restore this episode's scene, then replay recorded sim states
        env.reset()
        env.reset_from_xml_string(postprocess_model_xml(ep.attrs["model_file"]))
        env.sim.reset()

        frames = {c: [] for c in CAM_MAP}
        for t in range(T):
            env.sim.set_state_from_flattened(states[t])
            env.sim.forward()
            for cam in CAM_MAP:
                img = env.sim.render(width=args.res, height=args.res, camera_name=cam)
                frames[cam].append(img[::-1])  # robosuite renders upside-down

        for cam, vk in CAM_MAP.items():
            vpath = out / "videos" / "chunk-000" / vk / f"episode_{ep_idx:06d}.mp4"
            imageio.mimwrite(vpath, frames[cam], fps=args.fps, codec="libx264",
                             quality=8, macro_block_size=1, ffmpeg_log_level="error")

        state = np.concatenate(
            [eef_pos, np.stack([quat2axisangle(q) for q in eef_quat]), grip], axis=1
        ).astype(np.float32)
        assert state.shape == (T, 8), state.shape

        tbl = pa.table({
            "observation.state": pa.FixedSizeListArray.from_arrays(
                pa.array(state.reshape(-1), type=pa.float32()), 8),
            "action": pa.FixedSizeListArray.from_arrays(
                pa.array(actions.reshape(-1), type=pa.float32()), 7),
            "timestamp": pa.array(np.arange(T, dtype=np.float32) / args.fps, type=pa.float32()),
            "frame_index": pa.array(np.arange(T), type=pa.int64()),
            "episode_index": pa.array(np.full(T, ep_idx), type=pa.int64()),
            "index": pa.array(np.arange(global_index, global_index + T), type=pa.int64()),
            "task_index": pa.array(np.zeros(T), type=pa.int64()),
        })
        pq.write_table(tbl, out / "data" / "chunk-000" / f"episode_{ep_idx:06d}.parquet")

        episodes_meta.append({"episode_index": ep_idx, "tasks": [TASK_TEXTS[args.task]], "length": int(T)})
        global_index += T
        total_frames += T
        if ep_idx % 20 == 0 or ep_idx == len(demos) - 1:
            print(f"  ep {ep_idx+1}/{len(demos)} T={T} cum_frames={total_frames}", flush=True)

    env.close()
    f.close()

    # ---- meta ----
    def vinfo():
        return {"dtype": "video", "shape": [args.res, args.res, 3],
                "names": ["height", "width", "rgb"],
                "info": {"video.height": args.res, "video.width": args.res,
                         "video.codec": "h264", "video.pix_fmt": "yuv420p",
                         "video.is_depth_map": False, "video.fps": args.fps,
                         "video.channels": 3, "has_audio": False}}

    info = {
        "codebase_version": "v2.1", "robot_type": "franka",
        "total_episodes": len(demos), "total_frames": total_frames, "total_tasks": 1,
        "total_videos": len(demos) * 2, "total_chunks": 1, "chunks_size": 1000,
        "fps": args.fps, "splits": {"train": f"0:{len(demos)}"},
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": {
            "observation.images.wrist_image": vinfo(),
            "observation.images.image": vinfo(),
            "observation.state": {"dtype": "float32", "shape": [8], "names": {"motors": [
                "x", "y", "z", "axis_angle1", "axis_angle2", "axis_angle3", "gripper", "gripper"]}},
            "action": {"dtype": "float32", "shape": [7], "names": {"motors": [
                "x", "y", "z", "axis_angle1", "axis_angle2", "axis_angle3", "gripper"]}},
            "timestamp": {"dtype": "float32", "shape": [1], "names": None},
            "frame_index": {"dtype": "int64", "shape": [1], "names": None},
            "episode_index": {"dtype": "int64", "shape": [1], "names": None},
            "index": {"dtype": "int64", "shape": [1], "names": None},
            "task_index": {"dtype": "int64", "shape": [1], "names": None},
        },
    }
    (out / "meta" / "info.json").write_text(json.dumps(info, indent=4))

    modality = {
        "state": {"x": {"start": 0, "end": 1}, "y": {"start": 1, "end": 2},
                  "z": {"start": 2, "end": 3}, "roll": {"start": 3, "end": 4},
                  "pitch": {"start": 4, "end": 5}, "yaw": {"start": 5, "end": 6},
                  "gripper": {"start": 6, "end": 8}},
        "action": {"x": {"start": 0, "end": 1}, "y": {"start": 1, "end": 2},
                   "z": {"start": 2, "end": 3}, "roll": {"start": 3, "end": 4},
                   "pitch": {"start": 4, "end": 5}, "yaw": {"start": 5, "end": 6},
                   "gripper": {"start": 6, "end": 7}},
        "video": {"image": {"original_key": "observation.images.image"},
                  "wrist_image": {"original_key": "observation.images.wrist_image"}},
        "annotation": {"human.action.task_description": {"original_key": "task_index"}},
    }
    (out / "meta" / "modality.json").write_text(json.dumps(modality, indent=4))
    (out / "meta" / "tasks.jsonl").write_text(
        json.dumps({"task_index": 0, "task": TASK_TEXTS[args.task]}) + "\n")
    with open(out / "meta" / "episodes.jsonl", "w") as fh:
        for e in episodes_meta:
            fh.write(json.dumps(e) + "\n")

    print(f"[{args.src}] DONE episodes={len(demos)} frames={total_frames} -> {out}", flush=True)
    print(f"CONVERT_OK {args.src} {len(demos)} {total_frames}")


if __name__ == "__main__":
    main()
