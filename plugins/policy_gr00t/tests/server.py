"""A stand-in for the GR00T inference server, speaking the same wire.

Dispatch, envelope and error shape are copied from ``PolicyServer.run`` in the
GR00T package — deliberately re-implemented rather than imported, because a test
that imported the real server would be testing that GR00T is installed, which is
the one thing this plugin promises is unnecessary.

It answers in milliseconds and needs no GPU, so the protocol can be exercised on
a laptop. What it cannot check is whether a real checkpoint's numbers are good;
that is what the marked live test against a running server is for.
"""

from __future__ import annotations

import threading
from typing import Any, Callable

import numpy as np
import zmq
from gantry_policy_gr00t.wire import Codec


def modality_payload(delta_indices: list[int], modality_keys: list[str]) -> dict[str, Any]:
    """A ModalityConfig as GR00T puts it on the wire, marker and all."""
    return {
        "__ModalityConfig__": True,
        "as_json": {"delta_indices": delta_indices, "modality_keys": modality_keys},
    }


class FakeServer:
    """A REP socket with GR00T's endpoint table and a scripted policy."""

    def __init__(
        self,
        modality: dict[str, Any],
        action: Callable[[dict, Any], dict[str, np.ndarray]],
        *,
        identity: dict[str, Any] | None = None,
    ):
        self.modality = modality
        self._action = action
        self.identity_payload = identity or {}
        self.seen: list[dict[str, Any]] = []
        self.resets: list[Any] = []
        self._context = zmq.Context()
        self._socket = self._context.socket(zmq.REP)
        self._socket.bind("tcp://127.0.0.1:0")
        self.address = self._socket.getsockopt_string(zmq.LAST_ENDPOINT)
        self.port = int(self.address.rsplit(":", 1)[1])
        self._running = True
        self._thread = threading.Thread(target=self._serve, daemon=True)
        self._thread.start()

    def _serve(self) -> None:
        poller = zmq.Poller()
        poller.register(self._socket, zmq.POLLIN)
        while self._running:
            if not dict(poller.poll(timeout=50)):
                continue
            try:
                request = Codec.from_bytes(self._socket.recv())
                result = self._dispatch(request)
                self._socket.send(Codec.to_bytes(result))
            except Exception as error:  # noqa: BLE001 - the server's own error shape
                self._socket.send(Codec.to_bytes({"error": str(error)}))

    def _dispatch(self, request: dict[str, Any]) -> Any:
        endpoint = request.get("endpoint", "get_action")
        data = request.get("data") or {}
        if endpoint == "ping":
            return {"status": "ok", "message": "Server is running"}
        if endpoint == "get_modality_config":
            return self.modality
        if endpoint == "get_policy_identity":
            return self.identity_payload
        if endpoint == "describe_backend":
            return {}
        if endpoint == "reset":
            self.resets.append(data.get("options"))
            return {}
        if endpoint == "get_action":
            observation = data.get("observation")
            self.seen.append(observation)
            return [self._action(observation, data.get("options")), {}]
        raise ValueError(f"Unknown endpoint: {endpoint}")

    def stop(self) -> None:
        self._running = False
        self._thread.join(timeout=5)
        self._socket.close(linger=0)
        self._context.term()
