"""Read long-format CSV trajectories as Gantry episodes.

Format
------
One row per timestep, one header line, an ``episode`` column grouping rows::

    episode,step,position.0,position.1,position.2,engagement,success,stage.engage
    ep_0000,0,0.11,-0.04,0.31,0.0,1,0
    ep_0000,1,0.12,-0.03,0.30,0.0,1,0

Dotted suffixes make a vector: ``position.0..2`` becomes one 3-wide channel.
``success`` is an episode-level label. ``stage.<name>`` columns mark milestones
at their first true row.

Describing, not guessing
------------------------
Inference gives shape and dtype, which are readable off the numbers. It never
invents units, frames or semantics, because those are not in a CSV and a
plausible guess is worse than an honest absence. Supply them in a sidecar
``<file>.schema.json`` and they are used verbatim::

    {"position": {"units": "m", "frame": "world", "semantics": "position",
                  "rate_hz": 20.0}}

Without the sidecar the connector still works, and conformance will note the
missing units rather than pretend.

Laziness
--------
Opening the file indexes each episode's byte offset and row count, so
enumerating costs one scan and no step data. Reads seek to the episode and
parse only the requested window, so a file far larger than memory is fine.
"""

from __future__ import annotations

import csv
import json
import os
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from gantry.contracts.connector import Connector, connector_descriptor
from gantry.spine import (
    ChannelSpec,
    Descriptor,
    EpisodeLabels,
    EpisodeMeta,
    EpisodeRecord,
    StageEvent,
)

VERSION = "0.1.0.dev0"

EPISODE_COLUMN = "episode"
STEP_COLUMN = "step"
SUCCESS_COLUMN = "success"
STAGE_PREFIX = "stage."
RESERVED = {EPISODE_COLUMN, STEP_COLUMN, SUCCESS_COLUMN}


@dataclass(frozen=True)
class _Index:
    """Where one episode's rows are, so they can be read without a full scan."""

    offset: int
    rows: int


class _CsvSource:
    """A lazy :class:`~gantry.spine.episode.StepSource` over a slice of a file."""

    def __init__(
        self,
        path: Path,
        header: tuple[str, ...],
        index: _Index,
        columns: Mapping[str, tuple[int, ...]],
        dtypes: Mapping[str, str],
    ):
        self._path = path
        self._header = header
        self._index = index
        self._columns = columns
        self._dtypes = dtypes

    @property
    def num_steps(self) -> int:
        return self._index.rows

    @property
    def channels(self) -> tuple[str, ...]:
        return tuple(self._columns)

    def _rows(self, start: int, stop: int) -> list[list[str]]:
        with self._path.open("r", newline="") as handle:
            handle.seek(self._index.offset)
            reader = csv.reader(handle)
            out = []
            for position, row in enumerate(reader):
                if position >= stop:
                    break
                if position >= start:
                    out.append(row)
            return out

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

        start = max(0, start)
        stop = self._index.rows if stop is None else min(stop, self._index.rows)
        rows = self._rows(start, stop) if stop > start else []

        out: dict[str, np.ndarray] = {}
        for name in wanted:
            indices = self._columns[name]
            values = [[row[i] for i in indices] for row in rows]
            array = np.array(values, dtype=self._dtypes[name])
            out[name] = array.reshape(len(rows)) if len(indices) == 1 else array
        return out


def _group_columns(header: Sequence[str]) -> dict[str, tuple[int, ...]]:
    """Map channel name -> column indices, collapsing ``name.0``, ``name.1``…"""
    grouped: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for position, column in enumerate(header):
        if column in RESERVED or column.startswith(STAGE_PREFIX):
            continue
        name, _, suffix = column.rpartition(".")
        if name and suffix.isdigit():
            grouped[name].append((int(suffix), position))
        else:
            grouped[column].append((0, position))
    return {
        name: tuple(position for _, position in sorted(entries))
        for name, entries in grouped.items()
    }


class CsvConnector(Connector):
    """A connector over one long-format CSV file."""

    def __init__(
        self,
        path: str | os.PathLike[str],
        *,
        source: str | None = None,
        schema_overrides: Mapping[str, Mapping[str, Any]] | None = None,
        embodiment: str | None = None,
        task: str | None = None,
    ):
        self._path = Path(path)
        if not self._path.exists():
            raise FileNotFoundError(self._path)
        self._source = source or self._path.stem
        self._embodiment = embodiment
        self._task = task
        self._overrides = dict(schema_overrides or self._load_sidecar())
        self._header, self._index, self._labels = self._scan()
        self._columns = _group_columns(self._header)
        self._dtypes = self._infer_dtypes()
        self._schema = self._build_schema()
        # The schema is authoritative once built, so a sidecar dtype reaches
        # the reader too rather than only the description.
        self._dtypes = {spec.name: spec.dtype for spec in self._schema}

    # -- construction ------------------------------------------------------

    def _load_sidecar(self) -> Mapping[str, Mapping[str, Any]]:
        sidecar = self._path.with_suffix(self._path.suffix + ".schema.json")
        if not sidecar.exists():
            return {}
        return json.loads(sidecar.read_text())

    def _scan(self) -> tuple[tuple[str, ...], dict[str, _Index], dict[str, EpisodeLabels]]:
        """One pass: record where each episode starts and what it is labelled."""
        offsets: dict[str, _Index] = {}
        rows: dict[str, int] = defaultdict(int)
        success: dict[str, bool] = {}
        stages: dict[str, dict[str, int]] = defaultdict(dict)

        with self._path.open("r", newline="") as handle:
            header_line = handle.readline()
            if not header_line.strip():
                raise ValueError(f"{self._path}: file is empty")
            header = tuple(next(csv.reader([header_line])))
            if EPISODE_COLUMN not in header:
                raise ValueError(
                    f"{self._path}: no {EPISODE_COLUMN!r} column; found {list(header)[:8]}"
                )
            episode_at = header.index(EPISODE_COLUMN)
            success_at = header.index(SUCCESS_COLUMN) if SUCCESS_COLUMN in header else None
            stage_at = {
                column[len(STAGE_PREFIX) :]: position
                for position, column in enumerate(header)
                if column.startswith(STAGE_PREFIX)
            }

            offset = handle.tell()
            for line in iter(handle.readline, ""):
                if not line.strip():
                    offset = handle.tell()
                    continue
                row = next(csv.reader([line]))
                episode_id = row[episode_at]
                if episode_id not in offsets:
                    offsets[episode_id] = _Index(offset, 0)
                step = rows[episode_id]
                rows[episode_id] += 1
                if success_at is not None:
                    success[episode_id] = _truthy(row[success_at])
                for name, position in stage_at.items():
                    if name not in stages[episode_id] and _truthy(row[position]):
                        stages[episode_id][name] = step
                offset = handle.tell()

        index = {name: _Index(entry.offset, rows[name]) for name, entry in offsets.items()}
        labels = {
            name: EpisodeLabels(
                success=success.get(name),
                stage_events=tuple(
                    StageEvent(stage, step)
                    for stage, step in sorted(stages[name].items(), key=lambda kv: kv[1])
                ),
            )
            for name in index
        }
        return header, index, labels

    def _infer_dtypes(self, episodes: int = 3, rows: int = 512) -> dict[str, str]:
        """Decide float vs int by sampling, never by guessing meaning.

        Sampling widely matters more than it looks: a channel whose first row
        happens to be exactly zero reads as an integer, and every later row
        then fails to parse. Integer is claimed only when every sampled value
        is one, so the failure mode is a wider type rather than a crash.

        A sidecar dtype always wins over anything inferred here.
        """
        if not self._index:
            return {name: "float32" for name in self._columns}

        seen: dict[str, bool] = {name: True for name in self._columns}
        for episode_id in list(self._index)[:episodes]:
            source = _CsvSource(
                self._path,
                self._header,
                self._index[episode_id],
                self._columns,
                {name: "U64" for name in self._columns},
            )
            for name, values in source.read(start=0, stop=rows).items():
                if not seen[name]:
                    continue
                seen[name] = all(_is_integral(v) for v in np.atleast_1d(values).ravel())
        return {name: ("int64" if integral else "float32") for name, integral in seen.items()}

    def _build_schema(self) -> tuple[ChannelSpec, ...]:
        specs = []
        for name, indices in self._columns.items():
            override = dict(self._overrides.get(name, {}))
            width = len(indices)
            specs.append(
                ChannelSpec(
                    name=name,
                    kind=override.pop("kind", "scalar" if width == 1 else "vector"),
                    shape=tuple(override.pop("shape", () if width == 1 else (width,))),
                    dtype=override.pop("dtype", self._dtypes[name]),
                    units=override.pop("units", None),
                    frame=override.pop("frame", None),
                    rate_hz=override.pop("rate_hz", None),
                    semantics=override.pop("semantics", None),
                    optional=override.pop("optional", False),
                    metadata=override,
                )
            )
        return tuple(specs)

    # -- contract ----------------------------------------------------------

    def descriptor(self) -> Descriptor:
        labels = self._labels.values()
        return connector_descriptor(
            name="csv",
            version=VERSION,
            lazy=True,
            stage_events=any(label.stage_events for label in labels),
            outcomes=any(label.success is not None for label in labels),
            media=False,
            path=str(self._path),
            sidecar=bool(self._overrides),
        )

    def episode_ids(self) -> tuple[str, ...]:
        return tuple(self._index)

    def schema(self, episode_id: str) -> tuple[ChannelSpec, ...]:
        if episode_id not in self._index:
            raise KeyError(f"{self._source}: no episode {episode_id!r}")
        return self._schema

    def open(self, episode_id: str) -> EpisodeRecord:
        if episode_id not in self._index:
            raise KeyError(f"{self._source}: no episode {episode_id!r}")
        return EpisodeRecord(
            meta=EpisodeMeta(
                id=episode_id,
                source=self._source,
                embodiment=self._embodiment,
                task=self._task,
                extra={"path": str(self._path)},
            ),
            schema=self._schema,
            source=_CsvSource(
                self._path, self._header, self._index[episode_id], self._columns, self._dtypes
            ),
            labels=self._labels[episode_id],
        )


def _truthy(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "y", "t"}


def _is_integral(value: Any) -> bool:
    try:
        int(str(value))
    except ValueError:
        return False
    return True
