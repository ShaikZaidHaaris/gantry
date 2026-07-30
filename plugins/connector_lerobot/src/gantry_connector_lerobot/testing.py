"""Building a LeRobot dataset on disk, in the layout the format actually ships.

Shipped with the plugin rather than kept beside its tests. It was a function one
test file imported from another by filename, which required the tests directory
to be on ``sys.path`` — an implicit dependency on how pytest happens to import,
and one that broke the moment two plugins named a test file the same thing.

Useful outside the tests too: this reader describes what a dataset declares and
refuses what it cannot read, and the cheapest way to see either is to write a
small dataset and point it at one.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


def build_dataset(root: Path, *, episodes: int = 3, steps: int = 12, version: str = "v2.1") -> Path:
    """Write a dataset in the layout the format actually ships."""
    (root / "meta").mkdir(parents=True, exist_ok=True)
    (root / "data" / "chunk-000").mkdir(parents=True, exist_ok=True)

    (root / "meta" / "info.json").write_text(
        json.dumps(
            {
                "codebase_version": version,
                "robot_type": "franka",
                "total_episodes": episodes,
                "fps": 20,
                "chunks_size": 1000,
                "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
                "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
                "features": {
                    "observation.images.top": {
                        "dtype": "video",
                        "shape": [128, 128, 3],
                        "names": ["height", "width", "rgb"],
                    },
                    "observation.state": {
                        "dtype": "float32",
                        "shape": [4],
                        "names": {"motors": ["x", "y", "z", "gripper"]},
                    },
                    "action": {
                        "dtype": "float32",
                        "shape": [7],
                        "names": {
                            "motors": [
                                "x",
                                "y",
                                "z",
                                "axis_angle1",
                                "axis_angle2",
                                "axis_angle3",
                                "gripper",
                            ]
                        },
                    },
                    "timestamp": {"dtype": "float32", "shape": [1], "names": None},
                    "frame_index": {"dtype": "int64", "shape": [1], "names": None},
                    "episode_index": {"dtype": "int64", "shape": [1], "names": None},
                    "index": {"dtype": "int64", "shape": [1], "names": None},
                    "task_index": {"dtype": "int64", "shape": [1], "names": None},
                },
            }
        )
    )
    (root / "meta" / "tasks.jsonl").write_text(
        json.dumps({"task_index": 0, "task": "lift the cube"}) + "\n"
    )
    (root / "meta" / "episodes.jsonl").write_text(
        "".join(
            json.dumps({"episode_index": i, "tasks": ["lift the cube"], "length": steps}) + "\n"
            for i in range(episodes)
        )
    )
    rng = np.random.default_rng(0)
    for index in range(episodes):
        table = pa.table(
            {
                "observation.state": pa.array(
                    rng.normal(size=(steps, 4)).tolist(), pa.list_(pa.float32(), 4)
                ),
                "action": pa.array(rng.normal(size=(steps, 7)).tolist(), pa.list_(pa.float32(), 7)),
                "timestamp": pa.array(np.arange(steps) / 20.0, pa.float32()),
                "frame_index": pa.array(np.arange(steps), pa.int64()),
                "episode_index": pa.array([index] * steps, pa.int64()),
                "index": pa.array(np.arange(steps), pa.int64()),
                "task_index": pa.array([0] * steps, pa.int64()),
            }
        )
        pq.write_table(table, root / "data" / "chunk-000" / f"episode_{index:06d}.parquet")
    return root
