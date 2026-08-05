"""Build a real LeRobot v2 dataset, with a real relationship between pixels and actions.

The bench's signal check asks one question: does the footage predict the hands?
A dataset of noise passes intake and the data report and then correctly fails
that gate, which makes it useless for demonstrating the happy path. So the
generator here paints a marker whose position *is* the commanded action: an
honest correlation a probe can actually find, and one that survives being
shuffled against the wrong episode, which is what the control arm tests.

Usage:
    python make_dataset.py <outdir> [--episodes N] [--frames N] [--scramble]

``--scramble`` writes the same footage with the actions rotated between
episodes, so the pixels and the hands no longer correspond. That is the dataset
the signal check is supposed to refuse, and it is worth having on hand to prove
the gate says no as readily as it says yes.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import av
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

DOF = 14
H, W = 96, 128
FPS = 30

#: The two arms' joint names, in the order an aloha-agilex writes them.
JOINTS = [f"{side}_{name}" for side in ("left", "right") for name in
          ("waist", "shoulder", "elbow", "forearm_roll", "wrist_angle", "wrist_rotate", "gripper")]


def trajectory(episode: int, frames: int, rng: np.random.Generator) -> np.ndarray:
    """A smooth reach, different every episode, in [0, 1] per joint."""
    t = np.linspace(0.0, 1.0, frames)[:, None]
    phase = rng.uniform(0, 2 * np.pi, size=(1, DOF))
    speed = rng.uniform(0.6, 1.4, size=(1, DOF))
    start = rng.uniform(0.2, 0.4, size=(1, DOF))
    span = rng.uniform(0.25, 0.5, size=(1, DOF))
    return np.clip(start + span * np.sin(speed * 2 * np.pi * t + phase) ** 2, 0.0, 1.0)


def frame_for(action: np.ndarray) -> np.ndarray:
    """Paint the action into the pixels.

    Two markers, one per arm, placed from that arm's first two joints, plus a
    brightness bar driven by the grippers. Nothing here is subtle: the point is
    a relationship that genuinely exists, not a hard problem.
    """
    img = np.full((H, W, 3), 24, dtype=np.uint8)
    img[:, :, 2] = 40
    for arm in (0, 1):
        cx = int(6 + action[arm * 7 + 0] * (W - 20))
        cy = int(6 + action[arm * 7 + 1] * (H - 20))
        colour = (235, 90, 60) if arm == 0 else (60, 200, 235)
        img[max(0, cy - 6):cy + 6, max(0, cx - 6):cx + 6] = colour
    grip = float((action[6] + action[13]) / 2.0)
    img[H - 8:, : int(grip * W)] = (245, 245, 120)
    return img


def write_video(path: Path, frames: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    container = av.open(str(path), mode="w")
    stream = container.add_stream("libx264", rate=FPS)
    stream.width, stream.height, stream.pix_fmt = W, H, "yuv420p"
    # A low CRF keeps the marker crisp; a probe should be screening the footage,
    # not the codec's opinion of it.
    stream.options = {"crf": "18"}
    for frame in frames:
        container.mux(stream.encode(av.VideoFrame.from_ndarray(frame, format="rgb24")))
    container.mux(stream.encode())
    container.close()


def build(out: Path, episodes: int, frames: int, scramble: bool) -> dict:
    if out.exists():
        shutil.rmtree(out)
    (out / "meta").mkdir(parents=True)

    rng = np.random.default_rng(7)
    actions = [trajectory(i, frames, rng) for i in range(episodes)]
    # The control: same footage, actions rotated by one episode.
    painted = actions[-1:] + actions[:-1] if scramble else actions

    rows, index = [], 0
    for ep in range(episodes):
        act, pix = actions[ep], painted[ep]
        write_video(out / f"videos/chunk-000/observation.images.top/episode_{ep:06d}.mp4",
                    np.stack([frame_for(a) for a in pix]))
        # State trails the action by one tick, which is what proprioception is.
        state = np.vstack([act[:1], act[:-1]])
        table = pa.table({
            "action": [a.astype("float32").tolist() for a in act],
            "observation.state": [s.astype("float32").tolist() for s in state],
            "timestamp": pa.array(np.arange(frames) / FPS, type=pa.float32()),
            "frame_index": pa.array(np.arange(frames), type=pa.int64()),
            "episode_index": pa.array(np.full(frames, ep), type=pa.int64()),
            "index": pa.array(np.arange(index, index + frames), type=pa.int64()),
            "task_index": pa.array(np.zeros(frames), type=pa.int64()),
        })
        path = out / f"data/chunk-000/episode_{ep:06d}.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(table, path)
        rows.append({"episode_index": ep, "tasks": ["pick up the two bottles"], "length": frames})
        index += frames

    (out / "meta/episodes.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    (out / "meta/tasks.jsonl").write_text(
        json.dumps({"task_index": 0, "task": "pick up the two bottles"}) + "\n")

    info = {
        "codebase_version": "v2.0",
        "robot_type": "aloha-agilex",
        "total_episodes": episodes,
        "total_frames": episodes * frames,
        "total_tasks": 1,
        "total_videos": episodes,
        "total_chunks": 1,
        "chunks_size": 1000,
        "fps": FPS,
        "splits": {"train": f"0:{episodes}"},
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": {
            "action": {"dtype": "float32", "shape": [DOF], "names": JOINTS},
            "observation.state": {"dtype": "float32", "shape": [DOF], "names": JOINTS},
            "observation.images.top": {
                "dtype": "video", "shape": [H, W, 3], "names": ["height", "width", "channel"],
                "info": {"video.fps": FPS, "video.codec": "h264", "video.pix_fmt": "yuv420p",
                         "video.height": H, "video.width": W, "video.is_depth_map": False,
                         "has_audio": False},
            },
            "timestamp": {"dtype": "float32", "shape": [1]},
            "frame_index": {"dtype": "int64", "shape": [1]},
            "episode_index": {"dtype": "int64", "shape": [1]},
            "index": {"dtype": "int64", "shape": [1]},
            "task_index": {"dtype": "int64", "shape": [1]},
        },
    }
    (out / "meta/info.json").write_text(json.dumps(info, indent=2))
    return info


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("out", type=Path)
    ap.add_argument("--episodes", type=int, default=14)
    ap.add_argument("--frames", type=int, default=48)
    ap.add_argument("--scramble", action="store_true")
    args = ap.parse_args()
    info = build(args.out, args.episodes, args.frames, args.scramble)
    videos = len(list(args.out.rglob("*.mp4")))
    print(f"  {args.out.name}: {info['total_episodes']} episodes, "
          f"{info['total_frames']} frames, {videos} videos"
          f"{' [SCRAMBLED control]' if args.scramble else ''}")


if __name__ == "__main__":
    main()
