"""π₀.₅ over openpi's websocket, as one implementation of the policy plane.

Nothing in Gantry knows this exists. The policy plane is a socket that takes an
observation and returns an action chunk, and this is one thing that satisfies it —
the same standing as the GR00T client beside it, the same standing as whatever
comes next. The reason to write it now is that it is the model this project's ego
pipeline is actually pointed at, not that it is privileged.

What is genuinely different about serving π₀.₅
----------------------------------------------
**The prompt goes in every call.** The server is stateless per inference: it does
not remember what task it was told about on the previous step. So the language
instruction is sent with every observation, and a policy that forgets it does not
error — it quietly becomes an unconditioned policy, scores badly, and looks like a
checkpoint that did not train. That failure is invisible in every log, which is
why ``reset`` refusing an episode with no instruction is worth more than it looks.

**The chunk horizon belongs to the server.** π₀ and π₀.₅ emit an action chunk
whose length is a property of the served checkpoint, not of this client. Guessing
it would put a wrong number in the descriptor, which is then in the provenance of
every run — and chunk size moved a measured success rate by fourteen points in
this project's own benchmark, so it is not a cosmetic field. It is read from the
first response and pinned; a later response of a different length is a refusal,
because a server whose horizon changed mid-run is a server that was swapped.

**Bimanual is where this gets dangerous.** Fourteen numbers, left arm then right,
and nothing about the array says which is which. The layout carries the labels,
the action spec carries the layout's labels, and the resolver compares them — so
a retargeter that produced right-then-left is caught by name rather than by an
arm moving somewhere unexpected.

Determinism
-----------
Declared ``False`` by default and that is not caution. π₀.₅ decodes with a flow
matching sampler; identical observations do not give identical actions unless the
server was configured to pin its seed. A policy that claimed determinism it did
not have would let a paired comparison attribute sampling noise to whatever else
changed.
"""

from __future__ import annotations

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

from .layouts import Layout, layout_for

VERSION = "0.1.0.dev0"

#: What openpi calls the array it returns.
ACTIONS = "actions"

#: Checkpoint families served over this wire. Recorded rather than checked — a
#: new one should not need this file edited — but listed so a report can say
#: which was measured.
VARIANTS = ("pi05", "pi0", "pi0_fast")

#: openpi resizes to this internally; sending anything else works and wastes
#: bandwidth on a socket that is already the slow part of a rollout.
IMAGE = 224


def websocket(host: str = "localhost", port: int = 8000, *, timeout: float = 60.0) -> Any:
    """openpi's own websocket client. Imported here, not at module load.

    Lazily so that this plugin installs, declares itself and plans a run on a
    laptop with no JAX and no model weights present.
    """
    try:
        from openpi_client import websocket_client_policy
    except ImportError as error:  # pragma: no cover - needs the client
        raise ConfigError(
            "talking to a pi0 server needs openpi's client: pip install "
            "'gantry-policy-pi0[client]' (openpi-client is the small half — the "
            "model and JAX live on the server)"
        ) from error
    return websocket_client_policy.WebsocketClientPolicy(host=host, port=port)


class Pi0Policy(Policy):
    """A π₀.₅ server, as a policy."""

    def __init__(
        self,
        *,
        layout: Layout | Mapping[str, Any] | str = "aloha",
        host: str = "localhost",
        port: int = 8000,
        client: Any = None,
        connect: Callable[..., Any] = websocket,
        name: str = "pi0",
        variant: str = "pi05",
        instruction: str | None = None,
        deterministic: bool = False,
        chunk: int | None = None,
        image_size: int = IMAGE,
    ):
        """``instruction`` is a fallback, not a default.

        It is used only when a scene carries no instruction of its own, and it is
        recorded when used. A single prompt hard-wired across a whole run is a
        legitimate thing to do for a single-task benchmark and a silently wrong
        thing to do for anything language-conditioned, so it has to be visible in
        the record either way.
        """
        self._layout = layout_for(layout)
        self._name = name
        self._variant = str(variant)
        self._host = host
        self._port = int(port)
        self._connect = connect
        self._client = client
        self._fallback = instruction
        self._deterministic = bool(deterministic)
        self._image_size = int(image_size)
        # None until the server has answered once. Pinned then, and a later
        # disagreement is a refusal.
        self._chunk = int(chunk) if chunk else None
        self._context: EpisodeContext | None = None
        self._prompt: str | None = None

    # -- the wire ----------------------------------------------------------

    @property
    def client(self) -> Any:
        if self._client is None:
            self._client = self._connect(host=self._host, port=self._port)
        return self._client

    @property
    def address(self) -> str:
        return f"{self._host}:{self._port}"

    # -- contract ----------------------------------------------------------

    def descriptor(self) -> Descriptor:
        return policy_descriptor(
            self._name,
            VERSION,
            # Before the first call this is what the caller declared, or the
            # layout's action width as a stand-in. It is replaced by the
            # server's own answer the moment there is one.
            chunk=self._chunk or 1,
            # Flow matching samples. Claiming otherwise would let a paired
            # comparison attribute sampling noise to whatever else changed.
            deterministic=self._deterministic,
            # The server holds the model; whether it carries anything between
            # calls is its business and not visible from here.
            stateful=True,
            variant=self._variant,
            endpoint=self.address,
            chunk_source="server" if self._chunk is None else "declared",
            **self._layout.as_dict(),
        )

    def action_spec(self) -> ChannelSpec:
        """What comes back, with the arm order written down.

        The labels are the load-bearing part for anything bimanual: a fourteen-
        wide float32 vector is the same object whichever arm is first, and the
        resolver can only catch a swap if both sides said.
        """
        return ChannelSpec(
            "action",
            "vector",
            (self._layout.action,),
            "float32",
            semantics="actuation",
            dim_labels=self._layout.labels or None,
            discriminators=("arms",),
            metadata={"arms": self._layout.arms, "layout": self._layout.name},
        )

    def observes(self) -> Requirement:
        """Cameras and state, at the widths the layout declares.

        Built from the layout rather than from anything the server said, on
        purpose: this is what the *run* must supply, and it should be checkable
        before a socket is opened.
        """
        channels = [
            ChannelSpec(
                name,
                "image",
                (None, None, 3),
                "uint8",
                metadata={"openpi_key": key},
            )
            for name, key in self._layout.images.items()
        ]
        channels.append(
            ChannelSpec(
                self._layout.state_key,
                "vector",
                (self._layout.state,),
                "float32",
                dim_labels=self._layout.labels[: self._layout.state] or None,
                metadata={"arms": self._layout.arms},
            )
        )
        return requires_channels(
            self._name,
            "policy",
            *channels,
            description=f"what a {self._variant} server under the "
            f"{self._layout.name!r} config reads",
        )

    # -- running -----------------------------------------------------------

    def reset(self, context: EpisodeContext) -> None:
        """Fix the prompt for this episode, and refuse to run without one.

        The refusal is the point. A language-conditioned model handed no prompt
        does not fail — it becomes unconditioned, scores badly, and is
        indistinguishable in every log from a checkpoint that did not train. On
        an ego pipeline whose entire premise is that the language mattered, that
        is the single most expensive silent failure available.
        """
        prompt = context.instruction or self._fallback
        if self._layout.prompt_key and not (prompt or "").strip():
            raise ConfigError(
                f"{self._name}: episode {context.episode_id!r} carries no instruction "
                f"and no fallback was configured. A {self._variant} server takes the "
                "prompt with every observation; without one the model runs "
                "unconditioned, which scores badly and looks exactly like a "
                "checkpoint that did not train"
            )
        self._context = context
        self._prompt = (prompt or "").strip() or None
        if callable(getattr(self.client, "reset", None)):
            self.client.reset()

    def act(self, observation: Observation) -> np.ndarray:
        payload = self._payload(observation)
        try:
            reply = self.client.infer(payload)
        except Exception as error:  # noqa: BLE001 - the far end, whatever it is
            raise ComponentError(
                f"{self._name}: {self.address} failed on step {observation.step}: "
                f"{type(error).__name__}: {error}"
            ) from error
        return self._chunk_from(reply)

    def _payload(self, observation: Observation) -> dict[str, Any]:
        """One openpi observation dict, with every key the layout named.

        A missing camera is a refusal rather than a zero image. Sending zeros
        would produce a plausible action chunk from a model that saw a black
        frame, and nothing anywhere would say so.
        """
        channels = observation.channels
        payload: dict[str, Any] = {}
        missing: list[str] = []
        for name, key in self._layout.images.items():
            if name not in channels:
                missing.append(name)
                continue
            payload[key] = _image(channels[name])
        if self._layout.state_key not in channels:
            missing.append(self._layout.state_key)
        else:
            state = np.asarray(channels[self._layout.state_key], dtype="float32").reshape(-1)
            if state.size != self._layout.state:
                raise ComponentError(
                    f"{self._name}: state is {state.size} wide and the "
                    f"{self._layout.name!r} layout declares {self._layout.state}. "
                    + (
                        "For a two-armed layout this is usually one arm's worth of "
                        "numbers where two were expected."
                        if self._layout.arms > 1
                        else ""
                    )
                )
            payload[self._layout.state_key] = state
        if missing:
            raise ComponentError(
                f"{self._name}: the observation is missing {missing}; the "
                f"{self._layout.name!r} layout reads {list(self._layout.channels)}. "
                "Substituting blank frames would produce a confident action chunk "
                "from a model that saw nothing"
            )
        if self._layout.prompt_key:
            payload[self._layout.prompt_key] = self._prompt
        return payload

    def _chunk_from(self, reply: Any) -> np.ndarray:
        if not isinstance(reply, Mapping) or ACTIONS not in reply:
            raise ComponentError(
                f"{self._name}: {self.address} returned "
                f"{type(reply).__name__} with no {ACTIONS!r} key"
            )
        chunk = np.asarray(reply[ACTIONS], dtype="float32")
        if chunk.ndim == 1:
            chunk = chunk[None, :]
        if chunk.ndim != 2 or chunk.shape[1] != self._layout.action:
            raise ComponentError(
                f"{self._name}: {self.address} returned a chunk of shape "
                f"{chunk.shape}; the {self._layout.name!r} layout expects "
                f"(horizon, {self._layout.action})"
            )
        horizon = int(chunk.shape[0])
        if self._chunk is None:
            # Pinned from the first answer. It goes into the provenance of every
            # run, and chunk size moved a measured rate by fourteen points here.
            self._chunk = horizon
        elif horizon != self._chunk:
            raise ComponentError(
                f"{self._name}: {self.address} returned a chunk of {horizon} having "
                f"previously returned {self._chunk}. A horizon that changes mid-run "
                "means the served checkpoint changed, and the trials either side of "
                "it are not one measurement"
            )
        return chunk

    # -- what a report reads -----------------------------------------------

    @property
    def prompt(self) -> str | None:
        """The instruction this episode was actually given."""
        return self._prompt

    @property
    def layout(self) -> Layout:
        return self._layout

    def used_fallback(self) -> bool:
        """Whether the prompt came from configuration rather than from the scene.

        Recorded because a single hard-wired prompt across a whole run is fine
        for a single-task benchmark and quietly wrong for anything varying the
        task, and the two cases look identical afterwards otherwise.
        """
        return (
            self._context is not None
            and not (self._context.instruction or "").strip()
            and bool(self._fallback)
        )


def bimanual(host: str = "localhost", port: int = 8000, **kwargs: Any) -> Pi0Policy:
    """A π₀.₅ server under the bimanual ALOHA config.

    A named constructor rather than a default, so that "two arms" is something
    the caller said out loud. The single-arm and two-arm configurations differ
    only in a width, and a run that meant one and got the other reports as poor
    performance.
    """
    return Pi0Policy(layout="aloha", host=host, port=port, **kwargs)


def _image(value: Any) -> np.ndarray:
    """One camera, as uint8 HWC.

    Float images in [0, 1] are common from a pipeline that normalised early, and
    handing one to a server expecting bytes gives a near-black frame rather than
    an error. So the conversion is explicit and the range is checked.
    """
    array = np.asarray(value)
    if array.dtype == np.uint8:
        return array
    if np.issubdtype(array.dtype, np.floating):
        peak = float(np.nanmax(array)) if array.size else 0.0
        scaled = array * 255.0 if peak <= 1.0 else array
        return np.clip(scaled, 0, 255).astype("uint8")
    return array.astype("uint8")


def prompts_of(episodes: Sequence[Any]) -> dict[str, int]:
    """How many episodes were run under each instruction.

    A one-line audit for the failure this module worries about most: a run whose
    prompts are all one string was not language-conditioned, whatever the
    manifest said.
    """
    counts: dict[str, int] = {}
    for episode in episodes:
        prompt = str(
            (getattr(episode, "labels", None) and episode.labels.annotations or {}).get(
                "instruction", ""
            )
        )
        counts[prompt] = counts.get(prompt, 0) + 1
    return counts
