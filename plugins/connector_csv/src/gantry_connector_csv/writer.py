"""Write episodes back out as long-format CSV.

The plugin owns both directions of its own format. Core has no idea CSV exists
and must not acquire one, so a round-trip test belongs here rather than in the
conformance kit.

Also the practical way to hand someone a fixture: generate a suite, write it,
and the file is readable by anything that reads CSV.
"""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from gantry.spine import EpisodeRecord

from .connector import EPISODE_COLUMN, STAGE_PREFIX, STEP_COLUMN, SUCCESS_COLUMN


def write_episodes(
    episodes: Iterable[EpisodeRecord],
    path: str | os.PathLike[str],
    *,
    sidecar: bool = True,
) -> Path:
    """Write episodes to ``path``, optionally with a schema sidecar.

    The sidecar is how units, frames and semantics survive the trip. Without
    it the numbers round-trip but their meaning does not, which is exactly the
    silent-coercion failure the whole design is built against.
    """
    episodes = list(episodes)
    if not episodes:
        raise ValueError("nothing to write")

    schema = episodes[0].schema
    stages = sorted({stage for e in episodes for stage in e.labels.stages})
    header = [EPISODE_COLUMN, STEP_COLUMN]
    for spec in schema:
        header.extend(_columns_for(spec.name, spec.shape))
    if any(e.labels.success is not None for e in episodes):
        header.append(SUCCESS_COLUMN)
    header.extend(f"{STAGE_PREFIX}{stage}" for stage in stages)

    target = Path(path)
    with target.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        for episode in episodes:
            data = episode.read()
            for step in range(len(episode)):
                row: list[object] = [episode.meta.id, step]
                for spec in schema:
                    value = np.atleast_1d(data[spec.name][step]).ravel()
                    row.extend(_format(v) for v in value)
                if SUCCESS_COLUMN in header:
                    row.append(int(bool(episode.labels.success)))
                for stage in stages:
                    at = episode.labels.step_of(stage)
                    row.append(1 if at is not None and step >= at else 0)
                writer.writerow(row)

    if sidecar:
        target.with_suffix(target.suffix + ".schema.json").write_text(
            json.dumps(
                {
                    spec.name: {
                        key: value
                        for key, value in [
                            ("kind", spec.kind),
                            ("shape", list(spec.shape)),
                            ("dtype", spec.dtype),
                            ("units", spec.units),
                            ("frame", spec.frame),
                            ("rate_hz", spec.rate_hz),
                            ("semantics", spec.semantics),
                        ]
                        if value is not None
                    }
                    for spec in schema
                },
                indent=2,
                sort_keys=True,
            )
        )
    return target


def _columns_for(name: str, shape: Sequence[int | None]) -> list[str]:
    if not shape:
        return [name]
    width = int(np.prod([dimension or 1 for dimension in shape]))
    return [f"{name}.{i}" for i in range(width)]


def _format(value: object) -> object:
    if isinstance(value, (np.floating, float)):
        return f"{float(value):.9g}"
    if isinstance(value, (np.integer, int)):
        return int(value)
    return value
