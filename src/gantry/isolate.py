"""Running a component in a different interpreter than the one asking.

The problem this exists for is boring and unavoidable: the stacks these plugins
wrap conflict. Two simulators want incompatible builds of the same library, a
policy runtime pins a CUDA that a dataset reader cannot live with, and no amount
of interface design makes them share an interpreter.

So a descriptor declares ``isolation``, and this module honours it. A component
marked ``venv`` or ``container`` is built in its own process, which discovers its
*own* installed plugins and imports its *own* dependencies. The host never
imports the stack it was avoiding, which is the whole point -- an isolation
mechanism that still imports the plugin has isolated nothing.

The proxy speaks the connector contract, so nothing downstream can tell the
difference. Episodes come back as ordinary records whose source reads over the
pipe; a step window is forwarded rather than materialised, so laziness survives
the boundary too.

What is deliberately *not* here: a general RPC for every plane. Connectors have a
small, data-shaped interface that proxies faithfully. A policy called at control
rate, or an evaluator holding a simulator, would need a different design, and a
half-working version of that is worse than a clear refusal -- so
:func:`isolated_or_refuse` refuses those by name instead.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .contracts.connector import Connector
from .errors import ComponentError, ConfigError
from .serial import labels_from_dict, meta_from_dict, spec_from_dict
from .spine import ChannelSpec, Descriptor, EpisodeRecord, Verdict

#: Isolation levels that mean "not in this interpreter".
ISOLATED = ("venv", "container")


def PROXYABLE() -> tuple[str, ...]:
    """Planes that declared themselves proxyable. Others are refused."""
    from .spine import proxyable_planes

    return proxyable_planes()


class Worker:
    """A subprocess holding one built component."""

    def __init__(self, python: str | os.PathLike[str] | None = None, timeout: float = 120.0):
        self.python = str(python or sys.executable)
        self.timeout = timeout
        self._process: subprocess.Popen | None = None

    def start(self) -> None:
        if self._process is not None:
            return
        try:
            self._process = subprocess.Popen(
                [self.python, "-m", "gantry._worker"],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
            )
        except OSError as error:
            raise ConfigError(f"cannot start a worker with {self.python!r}: {error}") from error

    def ask(self, **request: Any) -> dict[str, Any]:
        self.start()
        assert self._process is not None
        if self._process.poll() is not None:
            stderr = self._process.stderr.read() if self._process.stderr else ""
            raise ComponentError(f"the worker exited before answering. {stderr.strip()}")
        self._process.stdin.write(json.dumps(request, default=str) + "\n")
        self._process.stdin.flush()
        line = self._process.stdout.readline()
        if not line:
            stderr = self._process.stderr.read() if self._process.stderr else ""
            raise ComponentError(
                f"the worker closed the pipe without answering {request.get('op')!r}. "
                f"{stderr.strip()}"
            )
        reply = json.loads(line)
        if "error" in reply:
            raise ComponentError(f"{self.python}: {reply['error']}")
        return reply

    def close(self) -> None:
        if self._process is None:
            return
        try:
            self._process.stdin.write(json.dumps({"op": "close"}) + "\n")
            self._process.stdin.flush()
            self._process.wait(timeout=5)
        except Exception:  # noqa: BLE001 - shutting down, never raise
            self._process.kill()
        finally:
            self._process = None

    def __enter__(self) -> "Worker":
        self.start()
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


class RemoteSource:
    """A step source that reads over the pipe, one window at a time."""

    def __init__(self, worker: Worker, episode_id: str, channels: Sequence[str], steps: int):
        self._worker = worker
        self._episode_id = episode_id
        self._channels = tuple(channels)
        self._steps = steps

    @property
    def num_steps(self) -> int:
        return self._steps

    @property
    def channels(self) -> tuple[str, ...]:
        return self._channels

    def read(
        self,
        channels: Sequence[str] | None = None,
        start: int = 0,
        stop: int | None = None,
    ) -> dict[str, np.ndarray]:
        wanted = list(channels) if channels is not None else list(self._channels)
        missing = [name for name in wanted if name not in self._channels]
        if missing:
            raise KeyError(f"no such channel(s): {missing}")
        reply = self._worker.ask(
            op="read",
            episode_id=self._episode_id,
            channels=wanted,
            start=start,
            stop=stop,
        )
        path = Path(reply["arrays"])
        try:
            with np.load(path, allow_pickle=False) as handle:
                return {name: handle[name] for name in handle.files}
        finally:
            path.unlink(missing_ok=True)


class RemoteConnector(Connector):
    """A connector living in another interpreter, indistinguishable from here."""

    def __init__(
        self,
        name: str,
        config: Mapping[str, Any] | None = None,
        *,
        python: str | os.PathLike[str] | None = None,
    ):
        self._worker = Worker(python)
        reply = self._worker.ask(op="build", plane="dataset", name=name, config=dict(config or {}))
        self._descriptor = Descriptor.from_dict(reply["descriptor"])
        self._ids: tuple[str, ...] | None = None

    def descriptor(self) -> Descriptor:
        return self._descriptor

    def episode_ids(self) -> tuple[str, ...]:
        if self._ids is None:
            self._ids = tuple(self._worker.ask(op="episode_ids")["ids"])
        return self._ids

    def schema(self, episode_id: str) -> tuple[ChannelSpec, ...]:
        if episode_id not in self.episode_ids():
            raise KeyError(f"no episode {episode_id!r} in {self._descriptor.ref}")
        reply = self._worker.ask(op="schema", episode_id=episode_id)
        return tuple(spec_from_dict(spec) for spec in reply["schema"])

    def open(self, episode_id: str) -> EpisodeRecord:
        if episode_id not in self.episode_ids():
            raise KeyError(f"no episode {episode_id!r} in {self._descriptor.ref}")
        reply = self._worker.ask(op="open", episode_id=episode_id)
        schema = tuple(spec_from_dict(spec) for spec in reply["schema"])
        return EpisodeRecord(
            meta=meta_from_dict(reply["meta"]),
            schema=schema,
            source=RemoteSource(
                self._worker, episode_id, [spec.name for spec in schema], int(reply["steps"])
            ),
            labels=labels_from_dict(reply["labels"]),
        )

    def close(self) -> None:
        self._worker.close()


# --------------------------------------------------------------------------
# policy
# --------------------------------------------------------------------------


def wants_isolation(descriptor: Descriptor) -> bool:
    return descriptor.isolation in ISOLATED


def isolated_or_refuse(
    descriptor: Descriptor,
    plane: str,
    name: str,
    config: Mapping[str, Any] | None = None,
    *,
    python: str | os.PathLike[str] | None = None,
) -> tuple[Any | None, Verdict]:
    """Build an isolated component, or say why it cannot be.

    A plane that cannot be proxied is refused rather than quietly run in-process.
    That refusal is the honest answer: the component said it needs its own
    environment, and running it here anyway is how a conflicting stack gets
    imported into someone's interpreter halfway through a run.
    """
    if plane not in PROXYABLE():
        return None, Verdict.no(
            "isolate.unsupported_plane",
            f"{plane}:{name} declares isolation {descriptor.isolation!r}, which this "
            f"host can only honour for {', '.join(PROXYABLE())}",
            hint="run it in-process by declaring isolation 'in-process', or run the "
            "whole manifest inside the environment it needs",
            plane=plane,
            name=name,
        )
    try:
        return RemoteConnector(name, config, python=python), Verdict.note(
            "isolate.spawned",
            f"{plane}:{name} is running in a separate interpreter as it declared",
        )
    except (ComponentError, ConfigError) as error:
        return None, Verdict.no(
            "isolate.failed",
            f"could not start {plane}:{name} in its own interpreter: {error}",
            plane=plane,
            name=name,
        )
