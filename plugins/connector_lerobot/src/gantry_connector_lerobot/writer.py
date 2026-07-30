"""Write episodes out as a LeRobot v2.1 dataset.

The plugin owns both directions of its own format, so this is where "convert
anything to LeRobot" lives. There is no robomimic-to-lerobot converter and there
should not be one: read with any connector, write with this, and every pair is
covered — csv, evallog, robomimic, or whatever gets written next.

Loss is declared, not discovered
--------------------------------
This is the whole reason the writer exists rather than a script.

LeRobot v2.1 has nowhere to put a per-episode outcome and nowhere to put a
milestone. A dataset that has them — robomimic's ``dones``, an evaluation log's
success column — comes out the other side without them, and nothing about the
result looks wrong. That is not hypothetical: the lift conversion in this
project's own history lost every success label and 96% of one split, and the
single most useful fact about that data (244 of 1500 demonstrations succeed)
became unavailable from the copy everybody then used.

So :func:`write_episodes` computes what it cannot carry *before* writing and
refuses unless the caller says, in the call, that the loss is acceptable. The
refusal names each thing being dropped. Somebody may still choose to drop it —
that is a legitimate choice and it should be a choice.

What is carried
---------------
Numeric channels, per-dimension names (as ``modality.json`` spans, which is the
unambiguous form the reader here prefers), the frame rate, the task strings, and
episode lengths. Images are not encoded: writing mp4s is a different job needing
an encoder, so image channels are reported as dropped rather than half-written.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from gantry.errors import ConfigError
from gantry.spine import ChannelSpec, EpisodeRecord

#: The version this writer emits.
CODEBASE_VERSION = "v2.1"

#: Where the pieces go. Matches what the reader in this plugin expects.
DATA_PATH = "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"
VIDEO_PATH = "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"
CHUNK_SIZE = 1000

#: Columns LeRobot always carries, written so the result is a real dataset
#: rather than one that only this reader can open.
BOOKKEEPING = ("timestamp", "frame_index", "episode_index", "index", "task_index")

#: Kinds this writer can put in a parquet column.
NUMERIC_KINDS = frozenset({"scalar", "vector", "boolean", "categorical", "timestamp"})


@dataclass(frozen=True)
class Loss:
    """One thing the format cannot carry, named so it can be argued with."""

    what: str
    detail: str

    def __str__(self) -> str:
        return f"{self.what}: {self.detail}"


@dataclass(frozen=True)
class WriteReport:
    """What was written, and what did not fit."""

    path: Path
    episodes: int
    frames: int
    channels: tuple[str, ...] = ()
    losses: tuple[Loss, ...] = ()

    @property
    def lossless(self) -> bool:
        return not self.losses

    def as_dict(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "episodes": self.episodes,
            "frames": self.frames,
            "channels": list(self.channels),
            "losses": [{"what": loss.what, "detail": loss.detail} for loss in self.losses],
        }

    def explain(self) -> str:
        head = f"{self.episodes} episode(s), {self.frames} frame(s) -> {self.path}"
        if self.lossless:
            return f"{head}\n  nothing was dropped"
        return "\n".join([head, *(f"  dropped {loss}" for loss in self.losses)])


def survey(episodes: Sequence[EpisodeRecord], *, videos: bool = False) -> tuple[Loss, ...]:
    """What this format cannot carry from these episodes. Cheap; reads nothing.

    ``videos`` says image channels will be encoded rather than dropped, so they
    stop counting as a loss. Anything else non-numeric still does.
    """
    losses: list[Loss] = []
    if not episodes:
        return ()

    outcomes = sum(1 for e in episodes if e.labels.success is not None)
    if outcomes:
        losses.append(
            Loss(
                "outcomes",
                f"{outcomes} of {len(episodes)} episode(s) carry a success label and "
                "LeRobot v2.1 has nowhere to put one",
            )
        )
    staged = sum(1 for e in episodes if e.labels.stage_events)
    if staged:
        names = sorted({s for e in episodes for s in e.labels.stages})
        losses.append(Loss("stage_events", f"{staged} episode(s) reach milestones {names}"))
    kinds = {"image"} if videos else set()
    unwritable = sorted(
        {
            spec.name
            for e in episodes
            for spec in e.schema
            if spec.kind not in NUMERIC_KINDS and spec.kind not in kinds
        }
    )
    if unwritable:
        losses.append(
            Loss(
                "channels",
                f"{unwritable} are not numeric columns"
                + ("" if videos else "; pass videos=True to encode image channels"),
            )
        )
    annotations = sorted({key for e in episodes for key in e.labels.annotations})
    if annotations:
        losses.append(Loss("annotations", f"per-episode notes {annotations}"))
    return tuple(losses)


def encode_video(
    frames: np.ndarray, target: "os.PathLike[str] | str", *, fps: float, crf: int = 23
) -> int:
    """One episode's frames as an h264 file, in the layout LeRobot expects.

    Separated from the column writing because it is a genuinely different
    operation with a genuinely different dependency, and because a half-written
    video is worse than an absent one: a dataset whose parquet says 300 steps and
    whose video holds 40 trains a policy on frames misaligned with their actions,
    which is unrecoverable and looks like a hard task.
    """
    try:
        import av
    except ImportError as error:  # pragma: no cover - needs the extra
        raise ConfigError(
            "writing video needs an encoder: pip install 'gantry-connector-lerobot[video]'"
        ) from error
    array = np.asarray(frames)
    if array.ndim != 4 or array.shape[-1] not in (1, 3):
        raise ConfigError(f"expected (steps, h, w, 3) frames to encode, got {array.shape}")
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    height, width = int(array.shape[1]), int(array.shape[2])
    with av.open(str(target), "w") as container:
        stream = container.add_stream("libx264", rate=int(round(fps)) or 1)
        stream.width, stream.height = width, height
        stream.pix_fmt = "yuv420p"
        stream.options = {"crf": str(crf)}
        for frame in array:
            image = np.repeat(frame, 3, axis=-1) if frame.shape[-1] == 1 else frame
            container.mux(
                stream.encode(
                    av.VideoFrame.from_ndarray(
                        np.ascontiguousarray(image.astype("uint8")), format="rgb24"
                    )
                )
            )
        container.mux(stream.encode())
    return len(array)


def write_episodes(
    episodes: Iterable[EpisodeRecord],
    path: str | os.PathLike[str],
    *,
    fps: float | None = None,
    robot_type: str | None = None,
    accept_loss: bool = False,
    videos: bool = False,
) -> WriteReport:
    """Write ``episodes`` as a LeRobot v2.1 dataset at ``path``.

    Refuses if anything would be silently dropped, unless ``accept_loss=True``.
    The refusal lists what would go, so the decision is made with the list in
    front of you rather than discovered by a reader six weeks later.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    episodes = list(episodes)
    if not episodes:
        raise ConfigError("nothing to write")

    losses = survey(episodes, videos=videos)
    if losses and not accept_loss:
        raise ConfigError(
            "writing these episodes as LeRobot would drop:\n"
            + "\n".join(f"  - {loss}" for loss in losses)
            + "\n\nPass accept_loss=True to write anyway. This is the failure that made "
            "this check exist: a conversion that quietly discards outcomes produces a "
            "dataset nobody can screen, and it looks exactly like one they can."
        )

    root = Path(path)
    (root / "meta").mkdir(parents=True, exist_ok=True)
    (root / "data" / "chunk-000").mkdir(parents=True, exist_ok=True)

    written = [spec for spec in episodes[0].schema if spec.kind in NUMERIC_KINDS]
    if not written:
        raise ConfigError("no numeric channels to write; a LeRobot dataset needs columns")
    rate = float(fps if fps is not None else _rate_of(written) or 0.0)

    cameras = [spec for spec in episodes[0].schema if spec.kind == "image"] if videos else []
    tasks = _tasks(episodes)
    total_frames = 0
    total_videos = 0
    lines: list[str] = []
    stat_lines: list[str] = []
    for index, episode in enumerate(episodes):
        steps = len(episode)
        table = _table(pa, episode, written, index, total_frames, rate, tasks)
        pq.write_table(table, root / "data" / "chunk-000" / f"episode_{index:06d}.parquet")
        for camera in cameras:
            frames = episode.array(camera.name)
            if len(frames) != steps:
                raise ConfigError(
                    f"{episode.meta.id}: {camera.name} has {len(frames)} frames for "
                    f"{steps} rows. A video misaligned with its actions trains a "
                    "policy on the wrong frames, which is unrecoverable and looks "
                    "like a hard task"
                )
            encode_video(
                frames,
                root
                / "videos"
                / "chunk-000"
                / _camera_key(camera.name)
                / f"episode_{index:06d}.mp4",
                fps=rate or 1.0,
            )
            total_videos += 1
        stat_lines.append(
            json.dumps({"episode_index": index, "stats": _stats(episode, written, cameras)})
        )
        lines.append(
            json.dumps(
                {
                    "episode_index": index,
                    "tasks": list(episode.labels.annotations.get("tasks", ()))
                    or ([episode.meta.task] if episode.meta.task else []),
                    "length": steps,
                    # What this episode was called before the conversion. An
                    # extra key here is ignored by any other reader of this
                    # format, and without it a demonstration renumbered by a
                    # format change can never again be matched to anything
                    # measured about it in the format it came from.
                    "derived_from": list(episode.meta.lineage),
                }
            )
        )
        total_frames += steps

    (root / "meta" / "episodes.jsonl").write_text("\n".join(lines) + "\n")
    (root / "meta" / "episodes_stats.jsonl").write_text("\n".join(stat_lines) + "\n")
    (root / "meta" / "tasks.jsonl").write_text(
        "".join(json.dumps({"task_index": i, "task": t}) + "\n" for t, i in tasks.items())
    )
    (root / "meta" / "info.json").write_text(
        json.dumps(
            _info(written, episodes, total_frames, rate, robot_type, tasks, cameras), indent=2
        )
    )
    modality = _modality(written)
    if modality:
        (root / "meta" / "modality.json").write_text(json.dumps(modality, indent=2))

    return WriteReport(
        path=root,
        episodes=len(episodes),
        frames=total_frames,
        channels=tuple(spec.name for spec in written),
        losses=losses,
    )


# -- the pieces -------------------------------------------------------------


def _rate_of(specs: Sequence[ChannelSpec]) -> float | None:
    rates = {spec.rate_hz for spec in specs if spec.rate_hz}
    return next(iter(rates)) if len(rates) == 1 else None


def _tasks(episodes: Sequence[EpisodeRecord]) -> dict[str, int]:
    seen: dict[str, int] = {}
    for episode in episodes:
        for task in episode.labels.annotations.get("tasks", ()) or (
            [episode.meta.task] if episode.meta.task else []
        ):
            seen.setdefault(str(task), len(seen))
    return seen


def _table(pa: Any, episode: EpisodeRecord, specs, index: int, offset: int, rate: float, tasks):
    steps = len(episode)
    arrays = episode.read([spec.name for spec in specs])
    columns: dict[str, Any] = {}
    for spec in specs:
        values = np.asarray(arrays[spec.name])
        if spec.shape in ((), (1,)):
            columns[spec.name] = pa.array(values.reshape(-1).tolist(), _arrow(pa, spec.dtype))
        else:
            width = int(np.prod(spec.shape))
            columns[spec.name] = pa.array(
                values.reshape(steps, width).tolist(),
                pa.list_(_arrow(pa, spec.dtype), width),
            )
    names = list(episode.labels.annotations.get("tasks", ())) or (
        [episode.meta.task] if episode.meta.task else []
    )
    task_index = tasks.get(str(names[0]), 0) if names else 0
    columns.setdefault(
        "timestamp",
        pa.array((np.arange(steps) / rate if rate else np.zeros(steps)), pa.float32()),
    )
    columns.setdefault("frame_index", pa.array(np.arange(steps), pa.int64()))
    columns.setdefault("episode_index", pa.array([index] * steps, pa.int64()))
    columns.setdefault("index", pa.array(np.arange(offset, offset + steps), pa.int64()))
    columns.setdefault("task_index", pa.array([task_index] * steps, pa.int64()))
    return pa.table(columns)


def _arrow(pa: Any, dtype: str) -> Any:
    kind = np.dtype(dtype)
    if kind == np.float64:
        return pa.float64()
    if kind in (np.int64, np.int32):
        return pa.int64()
    if kind == np.bool_:
        return pa.bool_()
    return pa.float32()


def _stats(episode, specs, cameras) -> dict[str, Any]:
    """Per-episode min/max/mean/std/count for every feature.

    v2.1 readers require these and will go to the Hub for them if they are
    absent, which for a local dataset means a 404 rather than a useful error.
    Written here because this is the only place that has the arrays.

    Image statistics are per channel over pixels, normalised to [0, 1], which is
    the convention the format uses and not an obvious one — a reader that
    expected 0-255 would silently mis-normalise every frame.
    """
    out: dict[str, Any] = {}
    for spec in specs:
        values = np.asarray(episode.array(spec.name), dtype="float64").reshape(len(episode), -1)
        out[spec.name] = {
            "min": values.min(axis=0).tolist(),
            "max": values.max(axis=0).tolist(),
            "mean": values.mean(axis=0).tolist(),
            "std": values.std(axis=0).tolist(),
            "count": [int(len(values))],
        }
    for spec in cameras:
        frames = np.asarray(episode.array(spec.name), dtype="float64") / 255.0
        flat = frames.reshape(-1, frames.shape[-1])
        out[_camera_key(spec.name)] = {
            "min": [[[v]] for v in flat.min(axis=0)],
            "max": [[[v]] for v in flat.max(axis=0)],
            "mean": [[[v]] for v in flat.mean(axis=0)],
            "std": [[[v]] for v in flat.std(axis=0)],
            "count": [int(len(frames))],
        }
    steps = len(episode)
    for name in BOOKKEEPING:
        column = {
            "timestamp": np.arange(steps, dtype="float64"),
            "frame_index": np.arange(steps, dtype="float64"),
            "episode_index": np.zeros(steps),
            "index": np.arange(steps, dtype="float64"),
            "task_index": np.zeros(steps),
        }[name]
        out[name] = {
            "min": [float(column.min())],
            "max": [float(column.max())],
            "mean": [float(column.mean())],
            "std": [float(column.std())],
            "count": [steps],
        }
    return out


def _camera_key(name: str) -> str:
    """LeRobot names camera features observation.images.<x>; ours may already."""
    return name if name.startswith("observation.images.") else f"observation.images.{name}"


def _info(
    specs, episodes, frames: int, rate: float, robot_type, tasks, cameras=()
) -> dict[str, Any]:
    features: dict[str, Any] = {}
    for spec in cameras:
        features[_camera_key(spec.name)] = {
            "dtype": "video",
            "shape": list(spec.shape),
            "names": ["height", "width", "channel"],
            "info": {
                "video.fps": rate,
                "video.height": int(spec.shape[0]),
                "video.width": int(spec.shape[1]),
                "video.channels": int(spec.shape[2]),
                "video.codec": "h264",
                "video.pix_fmt": "yuv420p",
                "video.is_depth_map": False,
                "has_audio": False,
            },
        }
    for spec in specs:
        features[spec.name] = {
            "dtype": str(np.dtype(spec.dtype)),
            "shape": list(spec.shape) if spec.shape else [1],
            "names": {"motors": list(spec.dim_labels)} if spec.dim_labels else None,
        }
    for name in BOOKKEEPING:
        features.setdefault(
            name,
            {"dtype": "float32" if name == "timestamp" else "int64", "shape": [1], "names": None},
        )
    return {
        "codebase_version": CODEBASE_VERSION,
        "robot_type": robot_type or episodes[0].meta.embodiment,
        "total_episodes": len(episodes),
        "total_frames": frames,
        "total_tasks": len(tasks),
        "total_videos": len(cameras) * len(episodes),
        "total_chunks": 1,
        "chunks_size": CHUNK_SIZE,
        "fps": rate,
        "splits": {"train": f"0:{len(episodes)}"},
        "data_path": DATA_PATH,
        "video_path": VIDEO_PATH,
        "features": features,
    }


def _modality(specs) -> dict[str, Any]:
    """Spans for the channels this format names, from their dimension labels.

    Written because the reader prefers it: ``info.json``'s ``names`` list cannot
    express a field spanning two columns without repeating a label, and a
    repeated label addresses nothing. The spans always can.
    """
    from .connector import MODALITY_CHANNELS

    out: dict[str, Any] = {}
    for modality, channel in MODALITY_CHANNELS.items():
        spec = next((s for s in specs if s.name == channel), None)
        if spec is None or not spec.dim_labels:
            continue
        spans: dict[str, Any] = {}
        for position, label in enumerate(spec.dim_labels):
            base = label.rsplit(".", 1)[0] if _numbered(label) else label
            if base in spans:
                spans[base]["end"] = position + 1
            else:
                spans[base] = {"start": position, "end": position + 1}
        out[modality] = spans
    return out


def _numbered(label: str) -> bool:
    head, _, tail = label.rpartition(".")
    return bool(head) and tail.isdigit()
