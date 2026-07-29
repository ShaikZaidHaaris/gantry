"""The other side of an isolated component. Run as ``python -m gantry._worker``.

Speaks one JSON object per line on stdin and answers one per line on stdout.
Arrays are the exception: they go through a temporary ``.npz`` whose path is
named in the reply, because base64 in a JSON line is both slower and a good way
to exhaust memory on a large window.

Nothing here imports the host's plugins — that is the entire point. This process
discovers what *it* has installed, in *its* environment, and the host never
imports the dependency stack it was avoiding.

Errors come back as ``{"error": ...}`` rather than a traceback on stderr, so a
plugin that fails to build produces a refusal the host can report rather than a
dead pipe.
"""

from __future__ import annotations

import json
import sys
import tempfile
from typing import Any

import numpy as np

from .serial import labels_to_dict, meta_to_dict, spec_to_dict


def _fail(message: str) -> dict[str, Any]:
    return {"error": message}


class _Session:
    """One built component, answered against for the life of the process."""

    def __init__(self) -> None:
        self.component: Any = None

    def build(self, plane: str, name: str, config: dict[str, Any]) -> dict[str, Any]:
        from .resolve import Registry

        registry = Registry()
        registry.discover()
        if not registry.has(plane, name):
            return _fail(
                f"{plane}:{name} is not installed in {sys.executable}; "
                f"this environment has {list(registry.names(plane))}"
            )
        self.component = registry.get(plane, name).build(config)
        return {"descriptor": self.component.descriptor().as_dict()}

    def episode_ids(self) -> dict[str, Any]:
        return {"ids": list(self.component.episode_ids())}

    def schema(self, episode_id: str) -> dict[str, Any]:
        return {"schema": [spec_to_dict(spec) for spec in self.component.schema(episode_id)]}

    def open(self, episode_id: str) -> dict[str, Any]:
        episode = self.component.open(episode_id)
        return {
            "meta": meta_to_dict(episode.meta),
            "schema": [spec_to_dict(spec) for spec in episode.schema],
            "labels": labels_to_dict(episode.labels),
            "steps": len(episode),
        }

    def read(
        self, episode_id: str, channels: list[str] | None, start: int, stop: int | None
    ) -> dict[str, Any]:
        episode = self.component.open(episode_id)
        arrays = episode.read(channels, start, stop)
        handle = tempfile.NamedTemporaryFile(suffix=".npz", delete=False)
        handle.close()
        np.savez(handle.name, **{name: np.asarray(v) for name, v in arrays.items()})
        return {"arrays": handle.name, "channels": list(arrays)}


def serve(stdin=None, stdout=None) -> None:
    """Read requests until the pipe closes."""
    stdin = stdin or sys.stdin
    stdout = stdout or sys.stdout
    session = _Session()

    for line in stdin:
        line = line.strip()
        if not line:
            continue
        try:
            request = json.loads(line)
            operation = request.pop("op")
        except Exception as error:  # noqa: BLE001 - a malformed request is data
            reply = _fail(f"malformed request: {error}")
        else:
            try:
                if operation == "build":
                    reply = session.build(request["plane"], request["name"], request.get("config", {}))
                elif operation == "close":
                    stdout.write(json.dumps({"ok": True}) + "\n")
                    stdout.flush()
                    return
                elif session.component is None:
                    reply = _fail(f"{operation!r} was asked before anything was built")
                else:
                    reply = getattr(session, operation)(**request)
            except AttributeError:
                reply = _fail(f"unknown operation {operation!r}")
            except Exception as error:  # noqa: BLE001 - report, never die
                reply = _fail(f"{type(error).__name__}: {error}")

        stdout.write(json.dumps(reply, default=str) + "\n")
        stdout.flush()


if __name__ == "__main__":  # pragma: no cover - process entry point
    serve()
