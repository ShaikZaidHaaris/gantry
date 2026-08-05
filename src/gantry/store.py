"""Write records to disk, and read them back as the same thing.

A run that cannot be reread is a run that cannot be re-scored, and a benchmark
whose results only exist inside the process that produced them is a benchmark
you have to trust rather than check.

Two guarantees:

**Versioned.** Every file carries a schema version and is refused by number if
this reader does not know it. A newer file failing with a sentence beats a newer
file being half-understood.

**Round-trip.** What comes back equals what went out -- provenance, labels,
milestones, measurements, and the arrays. :func:`same_run` is what the golden
test compares with, because "we serialise it" and "we can read it back" are
different claims and only the second one matters.

Layout: one JSON file for everything describable, and -- only when the episodes
carry channels -- a sibling ``.npz`` for the arrays. Step data is bulky and
belongs in a format built for it; the JSON stays readable, diffable and
greppable, which is what a record is for. Label-only records write no sidecar at
all, so an evaluation log stays a single small file.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .errors import ConfigError
from .serial import (
    labels_from_dict,
    labels_to_dict,
    measurement_from_dict,
    meta_from_dict,
    meta_to_dict,
    provenance_from_dict,
    spec_from_dict,
    spec_to_dict,
)
from .spine import (
    ArraySource,
    EmptySource,
    EpisodeRecord,
    Provenance,
    RunRecord,
)

STORE_VERSION = 1
SUPPORTED_VERSIONS = (1,)


# --------------------------------------------------------------------------
# writing
# --------------------------------------------------------------------------


def _episode_to_dict(
    episode: EpisodeRecord, key: str, arrays: dict[str, np.ndarray]
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "meta": meta_to_dict(episode.meta),
        "schema": [spec_to_dict(spec) for spec in episode.schema],
        "labels": labels_to_dict(episode.labels),
        "steps": len(episode),
    }
    if episode.schema:
        for name, values in episode.read().items():
            arrays[f"{key}::{name}"] = np.asarray(values)
        payload["arrays"] = key
    return payload


def write_run(record: RunRecord, path: str | os.PathLike[str]) -> Path:
    """Write one run. Returns the JSON path; a sidecar sits beside it if needed."""
    target = Path(path).with_suffix(".json")
    arrays: dict[str, np.ndarray] = {}
    payload: dict[str, Any] = {
        "version": STORE_VERSION,
        "kind": "run",
        "digest": record.digest,
        "provenance": record.provenance.as_dict(),
        "metrics": {name: value.as_dict() for name, value in record.metrics.items()},
        "episodes": [
            _episode_to_dict(episode, f"e{index:06d}", arrays)
            for index, episode in enumerate(record.episodes)
        ],
    }
    if arrays:
        sidecar = target.with_suffix(".npz")
        np.savez_compressed(sidecar, **arrays)
        payload["sidecar"] = sidecar.name
    target.write_text(json.dumps(payload, indent=2, default=str) + "\n")
    return target


def write_episodes(
    episodes: Sequence[EpisodeRecord], path: str | os.PathLike[str], **provenance: Any
) -> Path:
    """Write a bare set of episodes as a run with no metrics."""
    return write_run(RunRecord(Provenance(**provenance), tuple(episodes)), path)


# --------------------------------------------------------------------------
# reading
# --------------------------------------------------------------------------


def read_run(path: str | os.PathLike[str]) -> RunRecord:
    """Read one run back. Refuses a schema version it does not know, by number."""
    target = Path(path)
    if not target.exists():
        raise ConfigError(f"no record at {target}")
    try:
        payload = json.loads(target.read_text())
    except json.JSONDecodeError as error:
        raise ConfigError(f"{target}: not valid JSON ({error})") from error

    version = payload.get("version")
    if version not in SUPPORTED_VERSIONS:
        raise ConfigError(
            f"{target}: record schema version {version!r} is not supported; "
            f"this Gantry reads {list(SUPPORTED_VERSIONS)}"
        )
    if payload.get("kind") != "run":
        raise ConfigError(f"{target}: expected a run record, found {payload.get('kind')!r}")

    stored: dict[str, np.ndarray] = {}
    if payload.get("sidecar"):
        sidecar = target.parent / payload["sidecar"]
        if not sidecar.exists():
            raise ConfigError(
                f"{target}: names sidecar {payload['sidecar']!r}, which is missing. "
                "The arrays and the description travel together."
            )
        with np.load(sidecar, allow_pickle=False) as handle:
            stored = {name: handle[name] for name in handle.files}

    episodes = tuple(_episode_from_dict(entry, stored) for entry in payload.get("episodes") or ())
    metrics = {
        name: measurement_from_dict(value) for name, value in (payload.get("metrics") or {}).items()
    }
    return RunRecord(provenance_from_dict(payload.get("provenance") or {}), episodes, metrics)


def _episode_from_dict(
    payload: Mapping[str, Any], stored: Mapping[str, np.ndarray]
) -> EpisodeRecord:
    meta = meta_from_dict(payload["meta"])
    schema = tuple(spec_from_dict(spec) for spec in payload.get("schema") or ())
    key = payload.get("arrays")
    if key:
        prefix = f"{key}::"
        arrays = {
            name[len(prefix) :]: values
            for name, values in stored.items()
            if name.startswith(prefix)
        }
        if not arrays:
            raise ConfigError(
                f"episode {meta.id!r} declares channels but its arrays are not in the sidecar"
            )
        source: Any = ArraySource(arrays)
    else:
        source = EmptySource(int(payload.get("steps", 0)))

    return EpisodeRecord(
        meta=meta,
        schema=schema,
        source=source,
        labels=labels_from_dict(payload.get("labels") or {}),
    )


# --------------------------------------------------------------------------
# comparison
# --------------------------------------------------------------------------


def same_run(left: RunRecord, right: RunRecord) -> bool:
    """Structural equality, including array contents. Used by the round-trip test."""
    if left.provenance != right.provenance or len(left.episodes) != len(right.episodes):
        return False
    if set(left.metrics) != set(right.metrics):
        return False
    if any(left.metrics[name] != right.metrics[name] for name in left.metrics):
        return False
    for mine, theirs in zip(left.episodes, right.episodes):
        if (mine.meta, mine.schema, mine.labels) != (theirs.meta, theirs.schema, theirs.labels):
            return False
        if len(mine) != len(theirs):
            return False
        ours, yours = mine.read(), theirs.read()
        if set(ours) != set(yours):
            return False
        if any(not np.array_equal(ours[name], yours[name]) for name in ours):
            return False
    return True
