"""The GR00T inference protocol, spoken from outside.

This is the whole reason the plugin exists. GR00T's own client lives inside the
GR00T package, and importing it would drag torch, transformers, a CUDA build and
a pinned numpy into whatever process wants to *measure* the model. Speaking the
protocol instead costs about a hundred lines and buys complete independence: the
model can sit on an H100 in another building, on another Python, on another
numpy, and this plugin does not care.

The protocol
------------
ZeroMQ REQ/REP, msgpack with numpy support. One request per call::

    {"endpoint": "get_action", "data": {...}, "api_token": optional}

The server dispatches by endpoint name and replies with the handler's return
value, or with ``{"error": "..."}`` when it raised.

Decoding stays plain
--------------------
GR00T encodes its ``ModalityConfig`` objects with a marker so its own client can
rebuild them. Rebuilding them here would mean importing the class, so the marker
is unwrapped to the plain mapping underneath. Nothing in this plugin needs the
type, only the two fields inside it.

Pickle stays off
----------------
``msgpack_numpy`` serialises an object-dtype array by pickling it, and decodes
one by calling ``pickle.loads`` on bytes that arrived over a socket. Both
directions are refused here, before the library sees them. This mirrors the
guarantee GR00T's own serialiser makes, and it has to be re-made rather than
inherited, because the whole point is that we do not import theirs.
"""

from __future__ import annotations

import functools
import io
from dataclasses import dataclass, field
from typing import Any, Mapping

import msgpack
import msgpack_numpy as mnp
import numpy as np
import zmq

from gantry.errors import ComponentError, ConfigError

#: The port ``run_gr00t_server.py`` binds unless told otherwise.
DEFAULT_PORT = 5555

#: Endpoints the server registers unconditionally.
PING = "ping"
GET_ACTION = "get_action"
RESET = "reset"
MODALITY_CONFIG = "get_modality_config"
DESCRIBE = "describe_backend"
IDENTITY = "get_policy_identity"

#: The marker GR00T wraps a ModalityConfig in.
MODALITY_MARKERS = ("__ModalityConfig__", "__ModalityConfig_class__")


class Codec:
    """msgpack with numpy, and no path to pickle in either direction."""

    @staticmethod
    def to_bytes(payload: Any) -> bytes:
        return msgpack.packb(payload, default=functools.partial(_encode, chain=None))

    @staticmethod
    def from_bytes(raw: bytes) -> Any:
        return msgpack.unpackb(
            raw, object_hook=functools.partial(_decode, chain=_plain_modality), raw=False
        )


def _encode(obj: Any, chain=None) -> Any:
    if isinstance(obj, np.ndarray) and obj.dtype.kind == "O":
        raise TypeError(
            f"refusing to send an object-dtype array (shape={obj.shape}); it would be "
            "pickled on the way out. Convert it to a numeric dtype first"
        )
    return mnp.encode(obj, chain=chain)


def _decode(obj: Any, chain=None) -> Any:
    if isinstance(obj, dict):
        marker = obj.get("__ndarray_class__", obj.get(b"__ndarray_class__"))
        if marker:
            payload = obj.get("as_npy", obj.get(b"as_npy"))
            if payload is None:
                raise ValueError("malformed array payload: marker present but 'as_npy' missing")
            return np.load(io.BytesIO(payload), allow_pickle=False)
        # Accept any truthy ``nd``, not just True, so an int-coded forgery cannot
        # slip past on the way to pickle.loads.
        if obj.get(b"nd", obj.get("nd")) and obj.get(b"kind", obj.get("kind")) in (b"O", "O"):
            raise ValueError(
                "refusing to decode an object-dtype array from the wire; it would be unpickled"
            )
    return mnp.decode(obj, chain=chain)


def _plain_modality(obj: Any) -> Any:
    """Unwrap a ModalityConfig marker into the mapping it carries.

    GR00T's client rebuilds the dataclass here. We deliberately do not: the type
    lives in the package this plugin exists to avoid importing, and the two
    fields inside are all anyone needs.
    """
    if not isinstance(obj, dict):
        return obj
    if not any(marker in obj or marker.encode() in obj for marker in MODALITY_MARKERS):
        return obj
    key = next((k for k in ("as_json", b"as_json") if k in obj), None)
    if key is None:
        raise ValueError("malformed modality payload: marker present but 'as_json' missing")
    payload = obj[key]
    if isinstance(payload, bytes):
        payload = payload.decode()
    if isinstance(payload, str):
        import json

        payload = json.loads(payload)
    return payload


@dataclass(frozen=True)
class Endpoint:
    """Where the model is listening, and how long to wait for it."""

    host: str = "localhost"
    port: int = DEFAULT_PORT
    timeout_ms: int = 15000
    api_token: str | None = None

    def __post_init__(self) -> None:
        if self.timeout_ms <= 0:
            raise ValueError(f"timeout_ms must be positive, got {self.timeout_ms}")

    @property
    def address(self) -> str:
        return f"tcp://{self.host}:{self.port}"


@dataclass
class Client:
    """One REQ socket, with the failure taxonomy the run actually needs.

    A REQ socket that times out is left waiting for a reply that will never
    come, and every later call on it fails for a reason that has nothing to do
    with the model. So a timeout rebuilds the socket before re-raising, which is
    what makes "one slow inference" cost one trial instead of the run.
    """

    endpoint: Endpoint = field(default_factory=Endpoint)
    _context: Any = None
    _socket: Any = None
    #: Whether the server has ever replied. Decides whether a failure is a lost
    #: trial or a wrong address -- the same split the served-policy client makes,
    #: for the same reason: a typo'd port otherwise produces a full run of
    #: uniform failures that reads exactly like a bad model.
    answered: bool = False

    def __post_init__(self) -> None:
        self._context = zmq.Context.instance()
        self._connect()

    def _connect(self) -> None:
        if self._socket is not None:
            self._socket.close(linger=0)
        self._socket = self._context.socket(zmq.REQ)
        self._socket.setsockopt(zmq.RCVTIMEO, self.endpoint.timeout_ms)
        self._socket.setsockopt(zmq.SNDTIMEO, self.endpoint.timeout_ms)
        self._socket.connect(self.endpoint.address)

    def call(self, endpoint: str, data: Mapping[str, Any] | None = None) -> Any:
        request: dict[str, Any] = {"endpoint": endpoint}
        if data is not None:
            request["data"] = dict(data)
        if self.endpoint.api_token:
            request["api_token"] = self.endpoint.api_token

        try:
            self._socket.send(Codec.to_bytes(request))
            raw = self._socket.recv()
        except zmq.error.Again as error:
            self._connect()
            raise self._silent(endpoint, error) from error
        except zmq.error.ZMQError as error:
            self._connect()
            raise self._silent(endpoint, error) from error

        self.answered = True
        if raw == b"ERROR":
            raise ComponentError(f"{self.endpoint.address}: server refused {endpoint!r}")
        reply = Codec.from_bytes(raw)
        if isinstance(reply, Mapping) and "error" in reply:
            raise ComponentError(
                f"{self.endpoint.address} raised on {endpoint!r}: {reply['error']}"
            )
        return reply

    def _silent(self, endpoint: str, error: Exception) -> Exception:
        detail = f"{type(error).__name__}: {error}"
        if not self.answered:
            return ConfigError(
                f"nothing is answering at {self.endpoint.address} "
                f"({detail}); is the GR00T server running?",
            )
        return ComponentError(
            f"{self.endpoint.address} did not answer {endpoint!r} within "
            f"{self.endpoint.timeout_ms} ms ({detail})"
        )

    # -- the endpoints -----------------------------------------------------

    def ping(self) -> bool:
        try:
            self.call(PING)
        except (ConfigError, ComponentError):
            return False
        return True

    def modality_config(self) -> dict[str, Any]:
        return dict(self.call(MODALITY_CONFIG))

    def describe(self) -> dict[str, Any]:
        """The backend descriptor, where the server publishes one."""
        try:
            return dict(self.call(DESCRIBE) or {})
        except ComponentError:
            return {}

    def identity(self) -> dict[str, Any]:
        try:
            return dict(self.call(IDENTITY) or {})
        except ComponentError:
            return {}

    def reset(self, options: Mapping[str, Any] | None = None) -> Any:
        return self.call(RESET, {"options": dict(options) if options else None})

    def get_action(
        self, observation: Mapping[str, Any], options: Mapping[str, Any] | None = None
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        reply = self.call(
            GET_ACTION,
            {"observation": dict(observation), "options": dict(options) if options else None},
        )
        # The handler returns a tuple; msgpack has no tuple, so it arrives as a
        # two-element sequence.
        if isinstance(reply, Mapping):
            return dict(reply), {}
        if not isinstance(reply, (list, tuple)) or len(reply) != 2:
            raise ComponentError(
                f"expected (action, info) from {GET_ACTION!r}, got "
                f"{type(reply).__name__} of length {len(reply) if hasattr(reply, '__len__') else '?'}"
            )
        action, info = reply
        return dict(action), dict(info or {})

    def close(self) -> None:
        if self._socket is not None:
            self._socket.close(linger=0)
            self._socket = None
