"""A policy that lives somewhere else.

The model runs in its own process, its own container, usually its own machine
with the accelerator in it, and answers over a socket. That is how most real
inference is served, and it is the one plugin that lets Gantry evaluate a system
it cannot import -- a model whose dependencies conflict with everything here, a
proprietary endpoint, a colleague's checkpoint behind a URL.

The caller declares the contract
--------------------------------
There is no reliable way to ask a server what it emits. Some expose a schema
endpoint, most do not, and the ones that do disagree about its shape. So the
action spec and the observation requirement are constructor arguments, declared
by whoever wires the thing up. That is not a gap: the alternative is inferring a
contract from a JSON array's length, which is exactly the class of guess that
puts a scalar-last quaternion into a scalar-first slot.

Transport is a seam
-------------------
The default speaks JSON over HTTP with the standard library and no dependency at
all. Anything else -- gRPC, a unix socket, a message queue, an in-process stub -- is a ``transport`` callable. Likewise ``encode`` and ``decode``, because every
serving stack has invented its own envelope and none of them is the right one to
privilege.

Failure has two meanings
------------------------
A server that answers and then stops answering has lost one trial:
:class:`~gantry.errors.ComponentError`, recoverable, the run continues and the
episode is recorded as failed. A server that has *never* answered is a wrong
URL, a dead container, or a typo in a port number: :class:`ConfigError`, which
halts. The distinction matters because the second one otherwise produces a
complete run of uniform failures that looks exactly like a bad model.

A rejection is read the same way. A 4xx means the request is wrong and will be
wrong every time, so it halts; a 5xx means the server is having a bad minute and
costs one trial. Both are answers, which is more than a dead socket gives you.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from gantry.contracts.policy import (
    EpisodeContext,
    Observation,
    Policy,
    policy_descriptor,
)
from gantry.errors import ComponentError, ConfigError
from gantry.resolve import Requirement, requires_channels
from gantry.spine import ChannelSpec, Descriptor

VERSION = "0.1.0.dev0"

#: Signature of a transport: url, body, headers, timeout -> response bytes.
Transport = Callable[[str, bytes, Mapping[str, str], float], bytes]

#: What the default envelope calls the returned actions.
ACTION_KEY = "action"


def channel(spec: ChannelSpec | Mapping[str, Any]) -> ChannelSpec:
    """A channel spec, or the JSON object a manifest can carry one as.

    Without this the only way to point at a served model is from Python, and a
    run that cannot be described in a file cannot be committed, diffed, or
    re-run by somebody else.
    """
    if isinstance(spec, ChannelSpec):
        return spec
    from gantry.serial import spec_from_dict

    return spec_from_dict(spec)


def post_json(url: str, body: bytes, headers: Mapping[str, str], timeout: float) -> bytes:
    """The default transport: one POST, standard library only."""
    request = urllib.request.Request(
        url, data=body, headers={"Content-Type": "application/json", **dict(headers)}
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310 - caller's URL
        return response.read()


def encode_observation(observation: Observation, context: EpisodeContext | None) -> dict[str, Any]:
    """The default request envelope.

    Arrays go out as nested lists rather than anything packed, because the point
    of the default is that it can be answered by twelve lines of Flask somebody
    writes in an afternoon. A server that wants something denser supplies its own
    ``encode``.
    """
    payload: dict[str, Any] = {
        "step": observation.step,
        "observation": {
            name: np.asarray(value).tolist() for name, value in observation.channels.items()
        },
    }
    if context is not None:
        payload["episode_id"] = context.episode_id
        payload["instruction"] = context.instruction
        payload["seed"] = context.seed
    return payload


def decode_action(payload: Any, key: str = ACTION_KEY) -> Any:
    """The default response envelope: ``{"action": [...]}`` or a bare array."""
    if isinstance(payload, Mapping):
        if key not in payload:
            raise KeyError(f"response has no {key!r}; keys are {sorted(payload)}")
        return payload[key]
    return payload


@dataclass(frozen=True)
class Endpoint:
    """Where the server is and how to address it.

    Separate from the policy so the same endpoint can be pointed at by more than
    one declared contract -- the same served model driving two embodiments with
    different action specs is an ordinary thing to want.
    """

    url: str
    reset_url: str | None = None
    headers: Mapping[str, str] = field(default_factory=dict)
    timeout: float = 30.0
    attempts: int = 1
    retry_wait: float = 0.0

    def __post_init__(self) -> None:
        if self.attempts < 1:
            raise ValueError(f"attempts must be at least 1, got {self.attempts}")
        if self.timeout <= 0:
            raise ValueError(f"timeout must be positive, got {self.timeout}")


class ServedPolicy(Policy):
    """Observations out over the wire, action chunks back.

    ``chunk`` is the largest chunk the server will ever return. It is declared
    rather than discovered because a runner sizes its buffers from the
    declaration, and a chunk longer than declared is refused rather than
    truncated -- silently executing part of a prediction is how a protocol
    difference becomes an unexplained regression.
    """

    def __init__(
        self,
        endpoint: Endpoint | str | Mapping[str, Any],
        action: ChannelSpec | Mapping[str, Any],
        observes: Sequence[ChannelSpec | Mapping[str, Any]] = (),
        *,
        name: str = "served",
        chunk: int = 1,
        deterministic: bool = False,
        transport: Transport = post_json,
        encode: Callable[[Observation, EpisodeContext | None], Any] = encode_observation,
        decode: Callable[[Any], Any] = decode_action,
        aliases: Mapping[str, Sequence[str]] | None = None,
    ):
        """Every argument that has to cross a manifest accepts plain data:
        the endpoint as a string or an object, the channels as JSON."""
        if chunk < 1:
            raise ValueError(f"chunk must be at least 1, got {chunk}")
        action = channel(action)
        verdict = action.validate()
        if not verdict.ok:
            raise ConfigError(f"the declared action spec is not usable\n{verdict.explain()}")
        if isinstance(endpoint, str):
            self.endpoint = Endpoint(endpoint)
        elif isinstance(endpoint, Mapping):
            self.endpoint = Endpoint(**endpoint)
        else:
            self.endpoint = endpoint
        self._action = action
        self._observes = tuple(channel(spec) for spec in observes)
        self._aliases = {key: tuple(value) for key, value in (aliases or {}).items()}
        self._name = name
        self._chunk = chunk
        self._deterministic = deterministic
        self._transport = transport
        self._encode = encode
        self._decode = decode
        self._context: EpisodeContext | None = None
        #: Whether the server has ever answered. Decides whether a failure is
        #: one lost trial or a wrong address.
        self._answered = False

    # -- contract ----------------------------------------------------------

    def descriptor(self) -> Descriptor:
        return policy_descriptor(
            self._name,
            VERSION,
            chunk=self._chunk,
            deterministic=self._deterministic,
            # The server may well keep per-episode state, and from here there is
            # no way to know that it does not. Claiming statelessness that
            # cannot be checked is the more expensive mistake.
            stateful=True,
            # In-process is right, and not a hedge: isolation says how the host
            # should run *this object*, and this object is a thin client. The
            # heavy thing is already somewhere else, which is why the endpoint
            # is metadata rather than a fourth isolation mode.
            isolation="in-process",
            endpoint=self.endpoint.url,
        )

    def action_spec(self) -> ChannelSpec:
        return self._action

    def observes(self) -> Requirement:
        return requires_channels(
            self._name,
            "policy",
            *self._observes,
            aliases=self._aliases,
            description=f"whatever {self.endpoint.url} was told it would be sent",
        )

    def reset(self, context: EpisodeContext) -> None:
        self._context = context
        if self.endpoint.reset_url:
            self._call(
                self.endpoint.reset_url,
                {
                    "episode_id": context.episode_id,
                    "instruction": context.instruction,
                    "seed": context.seed,
                },
            )

    def act(self, observation: Observation) -> np.ndarray:
        payload = self._call(self.endpoint.url, self._encode(observation, self._context))
        try:
            raw = self._decode(payload)
        except Exception as error:  # noqa: BLE001 - the server's envelope, not ours
            raise ComponentError(
                f"{self._name}: could not read the response from {self.endpoint.url}: "
                f"{type(error).__name__}: {error}"
            ) from error
        return self._as_chunk(raw)

    # -- the wire ----------------------------------------------------------

    def _call(self, url: str, payload: Any) -> Any:
        body = json.dumps(payload).encode()
        last: Exception | None = None
        for attempt in range(self.endpoint.attempts):
            try:
                raw = self._transport(url, body, self.endpoint.headers, self.endpoint.timeout)
            except urllib.error.HTTPError as error:
                # A rejection is still an answer, so the endpoint is not in
                # question -- only this request is.
                self._answered = True
                if 400 <= error.code < 500:
                    raise ConfigError(
                        f"{self._name}: {url} rejected the request with {error.code} "
                        f"{error.reason}; a malformed request will be malformed every time"
                    ) from error
                last = error
            except (urllib.error.URLError, OSError) as error:
                last = error
            else:
                self._answered = True
                try:
                    return json.loads(raw)
                except json.JSONDecodeError as error:
                    raise ComponentError(
                        f"{self._name}: {url} returned something that is not JSON: {error}"
                    ) from error
            if attempt + 1 < self.endpoint.attempts and self.endpoint.retry_wait:
                time.sleep(self.endpoint.retry_wait)
        raise self._gave_up(url, last)

    def _gave_up(self, url: str, error: Exception | None) -> Exception:
        detail = f"{type(error).__name__}: {error}" if error else "no response"
        attempts = (
            f" after {self.endpoint.attempts} attempt(s)" if self.endpoint.attempts > 1 else ""
        )
        if not self._answered:
            return ConfigError(f"{self._name}: {url} has not answered once{attempts} ({detail})")
        return ComponentError(f"{self._name}: {url} did not answer{attempts} ({detail})")

    def _as_chunk(self, raw: Any) -> np.ndarray:
        """Shape the response into ``(horizon, *action shape)`` or refuse.

        A response whose rank is the action's own rank is a chunk of one. That
        is a deduction, not a convention: the spec fixes the action's rank, so
        one more axis is time and no more axes is a single action. Anything else
        is refused with what was expected, because a wrongly-shaped action
        executed anyway is motion nobody predicted.
        """
        try:
            chunk = np.asarray(raw, dtype=self._action.dtype)
        except (TypeError, ValueError) as error:
            raise ComponentError(
                f"{self._name}: the response is not an array of {self._action.dtype}: {error}"
            ) from error

        # Emptiness is settled before rank, because an empty response has no
        # unambiguous rank: ``[]`` reads equally as no actions and as one action
        # of width nothing, and "the server sent no actions" is the useful half
        # of that either way.
        if chunk.size == 0:
            raise ComponentError(f"{self._name}: the response contained no actions")

        rank = len(self._action.shape)
        if chunk.ndim == rank:
            chunk = chunk[None, ...]
        if chunk.ndim != rank + 1:
            raise ComponentError(
                f"{self._name}: got rank {chunk.ndim}, expected {rank + 1} "
                f"(horizon + action {self._action.shape}) or {rank} for a single action"
            )
        if len(chunk) > self._chunk:
            raise ComponentError(
                f"{self._name}: declares a chunk of {self._chunk} and the server returned "
                f"{len(chunk)}; a runner sizes its buffers from the declaration"
            )
        verdict = self._action.accepts(chunk)
        if not verdict.ok:
            raise ComponentError(
                f"{self._name}: the response does not match the declared action spec\n"
                f"{verdict.explain()}"
            )
        return chunk
