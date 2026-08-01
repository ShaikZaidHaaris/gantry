"""Read a LeRobot v2.x dataset as Gantry episodes.

Layout, as the format actually ships it::

    root/
      meta/info.json       fps, feature schema, path templates
      meta/episodes.jsonl  one line per episode: index, tasks, length
      meta/tasks.jsonl     task index -> instruction
      data/chunk-000/episode_000000.parquet
      videos/chunk-000/<video key>/episode_000000.mp4

The parquet holds the numeric channels. Features whose dtype is ``video`` are
*not* in it — they are mp4 side-cars, decoded on demand by :mod:`.video` and
only when a decoder is installed and the files are actually there.

A camera is therefore in the schema when its frames can be read and out of it
when they cannot, rather than declared unconditionally and failing later. The
descriptor says which of the two happened and why, because "this dataset has no
images" and "this install cannot open them" are different problems with
different fixes, and a schema alone cannot tell them apart.

What is declared, and what is not
---------------------------------
LeRobot declares dtype, shape, per-dimension names, and a frame rate. Those are
carried through exactly: ``names`` becomes ``dim_labels``, ``fps`` becomes
``rate_hz``.

It does not declare units, reference frames, or what the numbers mean. This
connector therefore declares none of those either. That is not a gap to be
filled with a plausible guess — ``observation.state`` being eight wide with a
gripper at the end is a strong hint and still a hint, and a connector that
promotes hints to declarations is how a millimetre reads as a metre. Pass
``schema_overrides`` to say what you actually know; :data:`SUGGESTED_SEMANTICS`
is a starting point that is deliberately not applied by default.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pyarrow.parquet as pq

from gantry.contracts.connector import Connector, connector_descriptor
from gantry.errors import ConfigError
from gantry.spine import (
    ChannelSpec,
    Descriptor,
    EpisodeLabels,
    EpisodeMeta,
    EpisodeRecord,
    StageEvent,
)

from .video import MultiSource, VideoSource, available

VERSION = "0.1.0.dev0"

#: Default ``video=`` — read the cameras when they can be read, say so when they
#: cannot. The two other settings are ``True`` (insist, and refuse at
#: construction if anything is missing) and ``False`` (parquet only).
AUTO = "auto"

#: Which flat column each ``modality.json`` section describes. The mapping is
#: the format's own convention, not an inference: a dataset shipping the sidecar
#: uses these names for these columns.
MODALITY_CHANNELS: Mapping[str, str] = {
    "state": "observation.state",
    "action": "action",
}

#: Codebase versions this reader understands.
SUPPORTED_VERSIONS = ("v2.0", "v2.1")

#: Feature dtypes that live in mp4 side-cars rather than the parquet.
MEDIA_DTYPES = frozenset({"video", "image"})

#: Columns LeRobot adds for its own bookkeeping. Real channels, but not the
#: recording — kept out of the default schema so a screen does not treat a row
#: counter as a behaviour.
BOOKKEEPING = ("index", "episode_index", "frame_index", "task_index")

#: A starting point for ``schema_overrides``, offered and never applied. The
#: names come from the format's own conventions; whether they are true of *your*
#: dataset is something only you know.
SUGGESTED_SEMANTICS: Mapping[str, Mapping[str, Any]] = {
    "observation.state": {"semantics": "state.eef_pose"},
    "action": {"semantics": "action.eef_delta_pose", "metadata": {"rotation_repr": "axis_angle"}},
    "timestamp": {"semantics": "time", "units": "s"},
}


@dataclass(frozen=True)
class _Episode:
    """Where one episode lives and how long it is."""

    index: int
    path: Path
    length: int
    tasks: tuple[str, ...]
    #: Names it had before this format, when whoever wrote it recorded them.
    derived_from: tuple[str, ...] = ()


class _ParquetSource:
    """Reads one episode's parquet, projecting columns and slicing rows."""

    def __init__(self, path: Path, columns: Mapping[str, ChannelSpec], length: int):
        self._path = path
        self._columns = columns
        self._length = length

    @property
    def num_steps(self) -> int:
        return self._length

    @property
    def channels(self) -> tuple[str, ...]:
        return tuple(self._columns)

    def read(
        self,
        channels: Sequence[str] | None = None,
        start: int = 0,
        stop: int | None = None,
    ) -> dict[str, np.ndarray]:
        wanted = tuple(channels) if channels is not None else self.channels
        missing = [name for name in wanted if name not in self._columns]
        if missing:
            raise KeyError(f"no such channel(s): {missing}")

        table = pq.read_table(self._path, columns=list(wanted))
        stop = self._length if stop is None else min(stop, self._length)
        start = max(0, start)
        if stop > start:
            table = table.slice(start, stop - start)
        else:
            table = table.slice(0, 0)

        out: dict[str, np.ndarray] = {}
        for name in wanted:
            spec = self._columns[name]
            values = table.column(name).to_pylist()
            array = np.asarray(values, dtype=spec.dtype)
            # A one-wide LeRobot feature is stored as a bare scalar column, and
            # is described here as a scalar, so the shapes already agree.
            out[name] = array
        return out


class LeRobotConnector(Connector):
    """One LeRobot dataset directory."""

    def __init__(
        self,
        path: str | os.PathLike[str],
        *,
        source: str | None = None,
        schema_overrides: Mapping[str, Mapping[str, Any]] | None = None,
        include_bookkeeping: bool = False,
        video: bool | str = AUTO,
    ):
        self._root = Path(path)
        if not (self._root / "meta" / "info.json").exists():
            raise ConfigError(
                f"{self._root}: no meta/info.json, so this is not a LeRobot dataset root"
            )
        self._info = json.loads((self._root / "meta" / "info.json").read_text())
        version = str(self._info.get("codebase_version", "unknown"))
        if version not in SUPPORTED_VERSIONS:
            raise ConfigError(
                f"{self._root}: codebase_version {version!r} is not supported; "
                f"this reader understands {list(SUPPORTED_VERSIONS)}"
            )
        self._source = source or self._root.name
        self._overrides = dict(schema_overrides or {})
        self._include_bookkeeping = include_bookkeeping
        self._modality = self._modality_labels()
        self._schema, self._media = self._build_schema()
        self._episodes = self._index_episodes()
        self._tasks = self._read_tasks()
        self._cameras, self._why_not = self._resolve_video(video)
        self._schema = self._schema + tuple(self._cameras.values())

    # -- reading the metadata ---------------------------------------------

    def _jsonl(self, name: str) -> list[dict[str, Any]]:
        path = self._root / "meta" / name
        if not path.exists():
            return []
        return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]

    def _read_tasks(self) -> dict[int, str]:
        return {
            int(entry["task_index"]): str(entry.get("task", ""))
            for entry in self._jsonl("tasks.jsonl")
        }

    def _index_episodes(self) -> dict[str, _Episode]:
        template = self._info.get(
            "data_path", "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"
        )
        chunk_size = int(self._info.get("chunks_size", 1000))
        episodes: dict[str, _Episode] = {}
        for entry in self._jsonl("episodes.jsonl"):
            index = int(entry["episode_index"])
            relative = template.format(episode_chunk=index // chunk_size, episode_index=index)
            path = self._root / relative
            if not path.exists():
                continue
            episodes[f"episode_{index:06d}"] = _Episode(
                index=index,
                path=path,
                length=int(entry.get("length", 0)),
                tasks=tuple(entry.get("tasks", ()) or ()),
                derived_from=tuple(entry.get("derived_from", ()) or ()),
            )
        if not episodes:
            raise ConfigError(
                f"{self._root}: meta/episodes.jsonl named no episode whose parquet exists"
            )
        return episodes

    def _build_schema(self) -> tuple[tuple[ChannelSpec, ...], tuple[str, ...]]:
        specs: list[ChannelSpec] = []
        media: list[str] = []
        for name, feature in self._info.get("features", {}).items():
            dtype = str(feature.get("dtype", "float32"))
            if dtype in MEDIA_DTYPES:
                media.append(name)
                continue
            if name in BOOKKEEPING and not self._include_bookkeeping:
                continue
            specs.append(self._spec_for(name, feature, dtype))
        return tuple(specs), tuple(media)

    def _modality_labels(self) -> dict[str, tuple[str, ...]]:
        """Per-element names from ``meta/modality.json``, where a dataset ships one.

        ``info.json`` gives a ``names`` list that is often unusable — the lift
        conversion here writes ``gripper`` twice, and a label that does not
        address exactly one dimension addresses nothing. ``modality.json`` gives
        the same information as non-overlapping ``start``/``end`` spans, which
        cannot be ambiguous.

        Reading it is describing, not guessing: it is a file the dataset's own
        author wrote to say what the columns are. It is used only when the spans
        tile the declared width exactly, so a sidecar belonging to a different
        version of the data is ignored rather than believed.
        """
        path = self._root / "meta" / "modality.json"
        if not path.exists():
            return {}
        try:
            payload = json.loads(path.read_text())
        except json.JSONDecodeError:
            return {}

        labels: dict[str, tuple[str, ...]] = {}
        for modality, channel in MODALITY_CHANNELS.items():
            spans = payload.get(modality)
            feature = self._info.get("features", {}).get(channel)
            if not isinstance(spans, Mapping) or not feature:
                continue
            width = next(iter(feature.get("shape") or ()), None)
            names = _spans_to_labels(spans, width)
            if names:
                labels[channel] = names
        return labels

    def _spec_for(self, name: str, feature: Mapping[str, Any], dtype: str) -> ChannelSpec:
        shape = tuple(int(value) for value in feature.get("shape", []) or ())
        # LeRobot writes a one-wide feature as a bare column, so it is a scalar
        # here rather than a vector of length one. Describing it as a vector
        # would make every consumer index into something that has no axis.
        scalar = shape in ((), (1,))
        override = dict(self._overrides.get(name, {}))
        metadata = dict(override.pop("metadata", {}))
        return ChannelSpec(
            name=name,
            kind=override.pop("kind", "scalar" if scalar else "vector"),
            shape=() if scalar else shape,
            dtype=override.pop("dtype", dtype),
            units=override.pop("units", None),
            frame=override.pop("frame", None),
            rate_hz=override.pop("rate_hz", float(self._info.get("fps", 0)) or None),
            semantics=override.pop("semantics", None),
            # modality.json first: it is unambiguous by construction, where
            # info.json's names are a list somebody typed.
            dim_labels=override.pop(
                "dim_labels",
                self._modality.get(name) or _labels(feature, () if scalar else shape),
            ),
            optional=override.pop("optional", False),
            metadata={**metadata, **override},
        )

    # -- the cameras -------------------------------------------------------

    def _resolve_video(self, video: bool | str) -> tuple[dict[str, ChannelSpec], str | None]:
        """Which cameras can actually be read, and why the rest cannot.

        Resolved once, here, rather than at each read: whether frames are
        available changes what the schema says, and a schema that changes
        depending on which file you touch first is not a description.
        """
        if video is False:
            return {}, "video=False"
        insist = video is True
        if not self._media:
            reason = "this dataset declares no video features"
            if insist:
                raise ConfigError(f"{self._root}: {reason}")
            return {}, reason
        if not available():
            reason = (
                "no decoder installed; pip install 'gantry-connector-lerobot[video]' "
                "to read the cameras"
            )
            if insist:
                raise ConfigError(f"{self._root}: {reason}")
            return {}, reason

        first = next(iter(self._episodes.values()))
        cameras: dict[str, ChannelSpec] = {}
        absent: list[str] = []
        for name in self._media:
            if self._video_path(name, first.index).exists():
                cameras[name] = self._video_spec(name)
            else:
                absent.append(name)
        if absent and insist:
            raise ConfigError(
                f"{self._root}: no mp4 for {absent} at "
                f"{self._video_path(absent[0], first.index)}; the metadata declares the "
                "camera(s) but the files were not downloaded with the dataset"
            )
        reason = None if not absent else f"no mp4 files found for {absent}"
        return cameras, reason

    def _video_path(self, key: str, index: int) -> Path:
        template = self._info.get(
            "video_path",
            "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        )
        chunk_size = int(self._info.get("chunks_size", 1000))
        return self._root / template.format(
            episode_chunk=index // chunk_size, episode_index=index, video_key=key
        )

    def _video_spec(self, name: str) -> ChannelSpec:
        feature = self._info["features"][name]
        shape = tuple(int(value) for value in feature.get("shape", []) or ())
        override = dict(self._overrides.get(name, {}))
        metadata = dict(override.pop("metadata", {}))
        info = dict(feature.get("info") or {})
        return ChannelSpec(
            name=name,
            kind=override.pop("kind", "image"),
            shape=override.pop("shape", shape),
            # Decoded to rgb24 whatever the file holds, so the dtype is a fact
            # about this reader rather than a guess about the encoding.
            dtype=override.pop("dtype", "uint8"),
            units=override.pop("units", None),
            frame=override.pop("frame", None),
            rate_hz=override.pop("rate_hz", float(info.get("video.fps", 0)) or None),
            semantics=override.pop("semantics", None),
            # ``names`` here is ["height", "width", "rgb"] — the axes, not one
            # label per element, so it is not a dimension label and is not used
            # as one.
            optional=override.pop("optional", False),
            metadata={"codec": info.get("video.codec"), **metadata, **override},
        )

    def _sources(self, episode: _Episode) -> dict[str, VideoSource]:
        fps = float(self._info.get("fps", 0)) or 1.0
        return {
            name: VideoSource(
                self._video_path(name, episode.index), name, spec.shape, episode.length, fps
            )
            for name, spec in self._cameras.items()
        }

    # -- contract ----------------------------------------------------------

    @classmethod
    def write(cls, episodes, path, **options):
        """Produce a dataset in this format, through this package's own writer.

        The contract's inverse of reading. Delegates rather than duplicating:
        the writer already refuses to drop anything silently, and that refusal
        is the reason to go through it.
        """
        from .writer import write_episodes

        return write_episodes(episodes, path, **options)

    def descriptor(self) -> Descriptor:
        return connector_descriptor(
            name="lerobot",
            version=VERSION,
            lazy=True,
            writes=True,
            # v2.x records no per-episode outcome and no milestones. Saying so
            # is what lets the resolver refuse a funnel here rather than produce
            # an empty one.
            stage_events=False,
            outcomes=False,
            media=bool(self._cameras),
            path=str(self._root),
            codebase_version=self._info.get("codebase_version"),
            robot_type=self._info.get("robot_type"),
            fps=self._info.get("fps"),
            episodes=len(self._episodes),
            video_features=list(self._media),
            cameras=list(self._cameras),
            # Present only when something is missing, and then it says which of
            # "this dataset has none" and "this install cannot read them" it is.
            **({"video_unavailable": self._why_not} if self._why_not else {}),
        )

    def episode_ids(self) -> tuple[str, ...]:
        return tuple(self._episodes)

    def schema(self, episode_id: str) -> tuple[ChannelSpec, ...]:
        self._require(episode_id)
        return self._schema

    def open(self, episode_id: str) -> EpisodeRecord:
        episode = self._require(episode_id)
        instruction = episode.tasks[0] if episode.tasks else None
        kept = self._sidecar.get(self._index_of(episode_id), {})
        return EpisodeRecord(
            meta=EpisodeMeta(
                id=episode_id,
                source=self._source,
                embodiment=self._info.get("robot_type"),
                task=instruction,
                # Names this episode had before it was written here, when
                # whoever wrote it recorded them. Absent for a dataset from
                # any other producer, which is why it defaults to empty rather
                # than to a guess.
                derived_from=tuple(getattr(episode, "derived_from", ()) or ()),
                extra={
                    "path": str(episode.path),
                    "tasks": list(episode.tasks),
                    **dict(kept.get("extra", {})),
                },
                license=kept.get("license"),
            ),
            schema=self._schema,
            source=MultiSource(
                _ParquetSource(
                    episode.path,
                    {spec.name: spec for spec in self._schema if spec.name not in self._cameras},
                    episode.length,
                ),
                self._sources(episode),
            ),
            labels=EpisodeLabels(
                # An outcome and per-episode notes that v2.1 has nowhere to put,
                # read back from the sidecar this package writes. Absent for a
                # dataset from any other producer, which is the honest default.
                success=kept.get("success"),
                stage_events=tuple(
                    StageEvent(name=str(e["name"]), step=int(e["step"]))
                    for e in kept.get("stage_events", ())
                ),
                annotations={
                    "tasks": list(episode.tasks),
                    # Under the name the vocabulary uses, not only the plural
                    # list LeRobot stores. A round trip that keeps the sentence
                    # but renames it loses it for every module that looks for
                    # "instruction", which is all of them.
                    **({"instruction": instruction} if instruction else {}),
                    **dict(kept.get("annotations", {})),
                },
            ),
        )

    def _index_of(self, episode_id: str) -> int:
        try:
            return list(self.episode_ids()).index(episode_id)
        except ValueError:  # pragma: no cover - guarded by _require
            return -1

    @property
    def _sidecar(self) -> dict[int, dict]:
        """What the writer kept beside the dataset, if this package wrote it."""
        if self.__dict__.get("_sidecar_cache") is None:
            rows: dict[int, dict] = {}
            path = self._root / "meta" / "gantry.jsonl"
            if path.exists():
                for line in path.read_text().splitlines():
                    if line.strip():
                        row = json.loads(line)
                        rows[int(row.get("episode_index", -1))] = row
            self.__dict__["_sidecar_cache"] = rows
        return self.__dict__["_sidecar_cache"]

    def _require(self, episode_id: str) -> _Episode:
        try:
            return self._episodes[episode_id]
        except KeyError:
            raise KeyError(f"{self._source}: no episode {episode_id!r}") from None

    # -- extras ------------------------------------------------------------

    @property
    def info(self) -> Mapping[str, Any]:
        return self._info

    @property
    def video_features(self) -> tuple[str, ...]:
        """Features the metadata declares as mp4 side-cars, readable or not."""
        return self._media

    @property
    def cameras(self) -> tuple[str, ...]:
        """The side-cars this connector can actually decode."""
        return tuple(self._cameras)

    @property
    def video_unavailable(self) -> str | None:
        """Why the declared cameras are not in the schema, when they are not."""
        return self._why_not

    @property
    def tasks(self) -> Mapping[int, str]:
        return self._tasks


def _spans_to_labels(spans: Mapping[str, Any], width: int | None) -> tuple[str, ...] | None:
    """``{"x": {"start": 0, "end": 1}, ...}`` to one label per element.

    A span wider than one element is numbered, because a gripper occupying two
    columns is two addressable dimensions and calling them both ``gripper``
    is the very ambiguity this is here to remove.

    Returns nothing unless the spans tile ``width`` exactly with no gap and no
    overlap. A partial description is worse than none: it would silently
    mislabel every dimension after the first hole.
    """
    try:
        fields = sorted(
            ((str(key), int(span["start"]), int(span["end"])) for key, span in spans.items()),
            key=lambda field: field[1],
        )
    except (KeyError, TypeError, ValueError):
        return None
    if not fields or fields[0][1] != 0:
        return None
    labels: list[str] = []
    cursor = 0
    for key, start, end in fields:
        if start != cursor or end <= start:
            return None
        labels.extend([key] if end - start == 1 else [f"{key}.{i}" for i in range(end - start)])
        cursor = end
    if width is not None and cursor != int(width):
        return None
    return tuple(labels) if len(set(labels)) == len(labels) else None


def _labels(feature: Mapping[str, Any], shape: tuple[int, ...]) -> tuple[str, ...] | None:
    """Per-dimension names, when the dataset gave them.

    LeRobot writes ``names`` either as a flat list or as a single-key mapping
    such as ``{"motors": [...]}``. Both are the author telling you what the
    columns are, which is exactly what a dimension label is for.
    """
    names = feature.get("names")
    if isinstance(names, Mapping):
        values = next(iter(names.values()), None)
    else:
        values = names
    if not isinstance(values, (list, tuple)) or not values:
        return None
    labels = tuple(str(value) for value in values)
    width = shape[0] if shape else 1
    if len(labels) != width or len(set(labels)) != len(labels):
        # A duplicated or mis-counted name list cannot address anything, and a
        # label that does not address a dimension is worse than none.
        return None
    return labels
