"""Read a RoboMimic HDF5 demonstration file as Gantry episodes.

Layout, as the format ships it::

    data/                       attrs: env_args (json), total
      demo_0/                   attrs: model_file, num_samples
        actions   (T, A)
        rewards   (T,)
        dones     (T,)
        states    (T, S)        simulator state, for exact replay
        obs/
          robot0_eef_pos    (T, 3)
          robot0_eef_quat   (T, 4)
          robot0_joint_pos  (T, 7)
          ...

Why this connector declares meanings and the LeRobot one does not
----------------------------------------------------------------
It looks inconsistent and is not. LeRobot feature names are chosen by whoever
built the dataset, so a reader that assigned meanings to them would be guessing
about somebody else's naming. RoboMimic observation keys are a *fixed
vocabulary* defined by robosuite: ``robot0_eef_pos`` is the end-effector
position in metres in every file that uses the name, because the simulator wrote
it. Carrying that through is reporting the format's own declaration, not
inventing one — and :data:`CONVENTIONS` is the whole of what is claimed, so it
can be read, argued with, and overridden.

Anything not in that table is left undescribed rather than guessed at.

Outcomes
--------
``dones`` marks the end of a successful demonstration, so unlike most recorded
formats this one carries a real per-episode outcome. That is what lets
attribution and hardening run here at all.

Episode order
-------------
Demos are named ``demo_0 … demo_199`` and HDF5 iterates them lexicographically,
which puts ``demo_100`` immediately after ``demo_10``. They are sorted
numerically here, because a "first fifty episodes" that silently means a
different fifty than anyone expects is the kind of thing that quietly changes a
benchmark.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

import h5py
import numpy as np

from gantry.contracts.connector import Connector, connector_descriptor
from gantry.errors import ConfigError
from gantry.spine import (
    ChannelSpec,
    Descriptor,
    EpisodeLabels,
    EpisodeMeta,
    EpisodeRecord,
)

VERSION = "0.1.0.dev0"

#: What robosuite means by each observation key. Facts about the simulator that
#: wrote the file, not inferences about it.
CONVENTIONS: Mapping[str, Mapping[str, Any]] = {
    "robot0_eef_pos": {"units": "m", "semantics": "state.eef_pos", "frame": "world"},
    "robot0_eef_quat": {
        "units": "1",
        "semantics": "state.eef_quat",
        "frame": "world",
        # robosuite stores quaternions scalar-last.
        "metadata": {"rotation_repr": "quat_xyzw"},
        "discriminators": ("rotation_repr",),
    },
    "robot0_eef_vel_lin": {"units": "m/s", "semantics": "linear_velocity", "frame": "world"},
    "robot0_eef_vel_ang": {"units": "rad/s", "semantics": "angular_velocity", "frame": "world"},
    "robot0_joint_pos": {"units": "rad", "semantics": "state.joint_pos"},
    "robot0_joint_vel": {"units": "rad/s", "semantics": "state.joint_vel"},
    "robot0_gripper_qpos": {"units": "m", "semantics": "state.gripper_width"},
    "robot0_gripper_qvel": {"units": "m/s", "semantics": "joint_velocity"},
    "actions": {"semantics": "actuation"},
}

#: Bookkeeping arrays that are not the recording. ``states`` is the simulator's
#: internal state for exact replay, not a behaviour anyone should screen.
NOT_BEHAVIOUR = ("states", "dones", "rewards")

_DEMO = re.compile(r"^demo_(\d+)$")


class _Hdf5Source:
    """Reads one demo, slicing rows in the file rather than in memory."""

    def __init__(self, path: Path, demo: str, columns: Mapping[str, str], length: int):
        self._path = path
        self._demo = demo
        self._columns = columns  # channel name -> path within the demo group
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
        start = max(0, start)
        stop = self._length if stop is None else min(stop, self._length)
        with h5py.File(self._path, "r") as handle:
            group = handle["data"][self._demo]
            return {name: np.asarray(group[self._columns[name]][start:stop]) for name in wanted}


class RoboMimicConnector(Connector):
    """One RoboMimic HDF5 file."""

    def __init__(
        self,
        path: str | os.PathLike[str],
        *,
        source: str | None = None,
        observations: Sequence[str] | None = None,
        include_actions: bool = True,
        schema_overrides: Mapping[str, Mapping[str, Any]] | None = None,
    ):
        self._path = Path(path)
        if not self._path.exists():
            raise FileNotFoundError(self._path)
        self._source = source or self._path.parent.name or self._path.stem
        self._overrides = dict(schema_overrides or {})
        self._wanted = tuple(observations) if observations else None
        self._include_actions = include_actions

        with h5py.File(self._path, "r") as handle:
            if "data" not in handle:
                raise ConfigError(f"{self._path}: no 'data' group, so this is not a RoboMimic file")
            data = handle["data"]
            self._env = self._read_env(data)
            self._demos = self._order(list(data.keys()))
            if not self._demos:
                raise ConfigError(f"{self._path}: contains no demo_N groups")
            self._lengths = {
                demo: int(data[demo].attrs.get("num_samples", len(data[demo]["actions"])))
                for demo in self._demos
            }
            self._columns, self._schema = self._build_schema(data[self._demos[0]])
            self._outcomes = {demo: self._outcome(data[demo]) for demo in self._demos}

    # -- reading the file --------------------------------------------------

    @staticmethod
    def _read_env(data: Any) -> Mapping[str, Any]:
        raw = data.attrs.get("env_args")
        if not raw:
            return {}
        try:
            return json.loads(raw)
        except (TypeError, ValueError):
            return {}

    @staticmethod
    def _order(names: Sequence[str]) -> tuple[str, ...]:
        """Numerically, so demo_100 does not land next to demo_10."""
        numbered = [(int(m.group(1)), name) for name in names if (m := _DEMO.match(name))]
        return tuple(name for _, name in sorted(numbered))

    @staticmethod
    def _outcome(demo: Any) -> bool | None:
        """``dones`` marks the end of a successful demonstration."""
        if "dones" in demo:
            dones = np.asarray(demo["dones"])
            return bool(dones.size and dones[-1])
        if "rewards" in demo:
            return bool(np.asarray(demo["rewards"]).max() > 0)
        return None

    def _build_schema(self, demo: Any) -> tuple[dict[str, str], tuple[ChannelSpec, ...]]:
        columns: dict[str, str] = {}
        specs: list[ChannelSpec] = []

        keys = list(demo["obs"].keys()) if "obs" in demo else []
        if self._wanted is not None:
            unknown = [name for name in self._wanted if name not in keys]
            if unknown:
                raise ConfigError(
                    f"{self._path}: no observation(s) {unknown}; it has {sorted(keys)}"
                )
            keys = [name for name in self._wanted]

        for key in keys:
            columns[key] = f"obs/{key}"
            specs.append(self._spec_for(key, demo["obs"][key]))
        if self._include_actions and "actions" in demo:
            columns["actions"] = "actions"
            specs.append(self._spec_for("actions", demo["actions"]))
        return columns, tuple(specs)

    def _spec_for(self, name: str, dataset: Any) -> ChannelSpec:
        width = int(dataset.shape[1]) if dataset.ndim > 1 else 1
        convention = dict(CONVENTIONS.get(name, {}))
        override = dict(self._overrides.get(name, {}))
        merged = {**convention, **override}
        metadata = dict(merged.pop("metadata", {}))
        discriminators = tuple(merged.pop("discriminators", ()))
        return ChannelSpec(
            name=name,
            kind=merged.pop("kind", "vector" if width > 1 else "scalar"),
            shape=(width,) if width > 1 else (),
            dtype=merged.pop("dtype", str(dataset.dtype)),
            units=merged.pop("units", None),
            frame=merged.pop("frame", None),
            rate_hz=merged.pop("rate_hz", None),
            semantics=merged.pop("semantics", None),
            optional=merged.pop("optional", False),
            discriminators=discriminators,
            metadata={**metadata, **merged},
        )

    # -- contract ----------------------------------------------------------

    def descriptor(self) -> Descriptor:
        return connector_descriptor(
            name="robomimic",
            version=VERSION,
            lazy=True,
            # No milestones in the format. Outcomes yes, from ``dones``, which
            # is what lets attribution and hardening run on this at all.
            stage_events=False,
            outcomes=any(value is not None for value in self._outcomes.values()),
            media=False,
            path=str(self._path),
            env=self._env.get("env_name"),
            demos=len(self._demos),
            observations=[spec.name for spec in self._schema],
        )

    def episode_ids(self) -> tuple[str, ...]:
        return self._demos

    def schema(self, episode_id: str) -> tuple[ChannelSpec, ...]:
        self._require(episode_id)
        return self._schema

    def open(self, episode_id: str) -> EpisodeRecord:
        self._require(episode_id)
        return EpisodeRecord(
            meta=EpisodeMeta(
                id=episode_id,
                source=self._source,
                embodiment=self._env.get("env_kwargs", {}).get("robots"),
                task=self._env.get("env_name"),
                extra={"path": str(self._path)},
            ),
            schema=self._schema,
            source=_Hdf5Source(self._path, episode_id, self._columns, self._lengths[episode_id]),
            labels=EpisodeLabels(success=self._outcomes[episode_id]),
        )

    def _require(self, episode_id: str) -> None:
        if episode_id not in self._lengths:
            raise KeyError(f"{self._source}: no demo {episode_id!r}")

    # -- extras ------------------------------------------------------------

    @property
    def env(self) -> Mapping[str, Any]:
        """The simulator configuration the file was recorded with."""
        return self._env

    def initial_states(self, episode_ids: Sequence[str] | None = None) -> tuple[np.ndarray, ...]:
        """The simulator state each demonstration started from, in order.

        Kept out of the schema on purpose — a simulator's internal state vector
        is not a behaviour anyone should screen, and :data:`NOT_BEHAVIOUR` says
        so. It is exposed here because it is the one thing needed to *re-enter*
        the world these demonstrations were recorded in: reconstruct the
        environment from :attr:`env`, reset to one of these, and a policy faces
        exactly the scene a human faced.

        That is what turns "how close were the predicted actions" into "did the
        arm actually pick up the cube".
        """
        wanted = tuple(episode_ids) if episode_ids is not None else self._demos
        with h5py.File(self._path, "r") as handle:
            states = []
            for episode_id in wanted:
                if episode_id not in self._lengths:
                    raise KeyError(f"{self._source}: no demo {episode_id!r}")
                demo = handle["data"][episode_id]
                if "states" not in demo:
                    raise ConfigError(
                        f"{self._path}: demo {episode_id!r} stores no 'states', so the "
                        "scene it began in cannot be restored"
                    )
                states.append(np.asarray(demo["states"][0]))
        return tuple(states)

    def available_observations(self) -> tuple[str, ...]:
        with h5py.File(self._path, "r") as handle:
            demo = handle["data"][self._demos[0]]
            return tuple(sorted(demo["obs"].keys())) if "obs" in demo else ()
