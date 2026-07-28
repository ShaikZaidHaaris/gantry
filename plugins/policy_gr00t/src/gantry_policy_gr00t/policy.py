"""GR00T N1.7 as a Gantry policy.

Start the model the way its own documentation says::

    python gr00t/eval/run_gr00t_server.py --model-path <ckpt> --embodiment-tag <tag>

then point this at it. Nothing about GR00T is imported: the model keeps its
torch, its CUDA build and its pinned numpy, and Gantry keeps a socket.

What is discovered and what is declared
---------------------------------------
The server is asked what it reads — which keys, and how many timesteps of each —
because it knows and it will answer. The dataset's ``modality.json`` supplies the
widths, because the server does not know them. The two are checked against each
other before an episode is opened, so a checkpoint pointed at the wrong dataset
refuses in a second instead of producing a night of plausible numbers.

Everything else is declared: the action channel that comes out is described with
one dimension label per element, in layout order, which is what lets core refuse
a binding where two seven-wide vectors mean different sevens.

History
-------
Some embodiments read a frame from fifteen steps ago as well as the current one.
Gantry hands a policy one timestep at a time, so the frames are buffered here and
indexed by the deltas the server declared. Before enough steps have happened the
earliest available frame is repeated — the same padding GR00T's own loader
applies at an episode boundary. It is stated in the descriptor rather than done
quietly, because a repeated frame is not a real observation and a reader is
entitled to know which steps have one.
"""

from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from gantry.contracts.policy import EpisodeContext, Observation, Policy, policy_descriptor
from gantry.errors import ComponentError, ConfigError
from gantry.resolve import Requirement, requires_channels
from gantry.spine import ChannelSpec, Descriptor

from .modality import ACTION, LANGUAGE, STATE, VIDEO, Layout, Wants, check
from .wire import Client, Endpoint

VERSION = "0.1.0.dev0"

#: What a LeRobot dataset calls its flat state vector.
STATE_CHANNEL = "observation.state"

#: Sent when a task carries no instruction of its own and none was configured.
NO_INSTRUCTION = ""


class Gr00tPolicy(Policy):
    """A GR00T inference server, driven through the policy contract."""

    def __init__(
        self,
        layout: Layout | str | Path,
        endpoint: Endpoint | Mapping[str, Any] | None = None,
        *,
        name: str = "gr00t",
        instruction: str | None = None,
        state_channel: str = STATE_CHANNEL,
        video_channels: Mapping[str, str] | None = None,
        deterministic: bool = False,
        client: Client | None = None,
    ):
        """``layout`` is a dataset directory, a ``modality.json``, or a built
        :class:`~gantry_policy_gr00t.modality.Layout`.

        ``video_channels`` overrides where each of the model's video keys is
        read from. The default is what ``modality.json`` already says — the
        model's ``image`` comes from the dataset's ``observation.images.image``
        — so the usual case needs no mapping at all.
        """
        self.layout = layout if isinstance(layout, Layout) else Layout.from_json(layout)
        if not self.layout.action:
            raise ConfigError(
                "this modality.json describes no action fields, so there is no way to "
                "know what the model's output means"
            )
        self._name = name
        self._instruction = instruction
        self._state_channel = state_channel
        self._video_channels = dict(video_channels or self.layout.video)
        self._deterministic = deterministic

        if client is not None:
            self.client = client
        else:
            if endpoint is None:
                endpoint = Endpoint()
            self.client = Client(Endpoint(**endpoint) if isinstance(endpoint, Mapping) else endpoint)

        self.wants = Wants.from_server(self.client.modality_config())
        check(self.layout, self.wants).raise_if_refused(
            f"{self._name} cannot be wired to this dataset"
        )
        # Asked once. A descriptor is read constantly — for provenance, by the
        # conformance kit, by anything planning a run — and a descriptor that
        # goes to the network each time would make identity depend on whether
        # the server happened to be up when somebody printed it.
        self.identity = self.client.identity()
        self._history: deque[dict[str, np.ndarray]] = deque(maxlen=self._depth())
        self._context: EpisodeContext | None = None

    def _depth(self) -> int:
        """How far back anything the model reads reaches."""
        reach = [
            -min(self.wants.deltas.get(modality, (0,))) for modality in (STATE, VIDEO)
        ]
        return max(1, max(reach) + 1)

    # -- contract ----------------------------------------------------------

    def descriptor(self) -> Descriptor:
        return policy_descriptor(
            self._name,
            VERSION,
            chunk=self.wants.horizon(ACTION),
            deterministic=self._deterministic,
            # The server holds the model; whether it carries state between calls
            # is its business and not visible from here.
            stateful=True,
            endpoint=self.client.endpoint.address,
            action_keys=[field.key for field in self.layout.action],
            pads_history=self._depth() > 1,
            # Kept under one key rather than splatted: whatever a server chooses
            # to call its fields, it cannot collide with a capability here.
            identity=dict(self.identity),
        )

    def action_spec(self) -> ChannelSpec:
        return ChannelSpec(
            "action",
            "vector",
            (self.layout.width(ACTION),),
            "float32",
            semantics="actuation",
            dim_labels=self.layout.labels(ACTION),
            metadata={"gr00t_keys": [field.key for field in self.layout.action]},
        )

    def observes(self) -> Requirement:
        channels = [
            ChannelSpec(
                self._state_channel,
                "vector",
                (self.layout.width(STATE),),
                "float32",
                dim_labels=self.layout.labels(STATE),
            )
        ]
        for key in self.wants.keys.get(VIDEO, ()):
            channels.append(
                ChannelSpec(
                    self._video_channels.get(key, key),
                    "image",
                    (None, None, 3),
                    "uint8",
                    metadata={"gr00t_key": key},
                )
            )
        return requires_channels(
            self._name,
            "policy",
            *channels,
            description=f"what {self.client.endpoint.address} said it reads",
        )

    # -- running -----------------------------------------------------------

    def reset(self, context: EpisodeContext) -> None:
        self._context = context
        self._history.clear()
        self.client.reset({"episode_id": context.episode_id, "seed": context.seed})

    def act(self, observation: Observation) -> np.ndarray:
        self._history.append(self._frame(observation))
        actions, _info = self.client.get_action(self._observation())
        return self._chunk(actions)

    def _frame(self, observation: Observation) -> dict[str, np.ndarray]:
        """One timestep, split the way the model reads it."""
        frame = {
            f"{STATE}.{key}": value
            for key, value in self.layout.split(
                STATE, np.asarray(observation[self._state_channel], dtype="float32")
            ).items()
        }
        for key in self.wants.keys.get(VIDEO, ()):
            channel = self._video_channels.get(key, key)
            image = np.asarray(observation[channel])
            if image.dtype != np.uint8:
                raise ComponentError(
                    f"{self._name}: video {channel!r} is {image.dtype}, and the model reads "
                    "uint8; a float image here is a normalisation nobody agreed on"
                )
            frame[f"{VIDEO}.{key}"] = image
        return frame

    def _at(self, delta: int, key: str) -> np.ndarray:
        """The frame ``delta`` steps back, padded with the earliest we have."""
        index = len(self._history) - 1 + delta
        return self._history[max(0, index)][key]

    def _observation(self) -> dict[str, Any]:
        """The nested, batched observation GR00T's server expects."""
        state = {
            key: np.stack(
                [self._at(delta, f"{STATE}.{key}") for delta in self.wants.deltas.get(STATE, (0,))]
            )[None, ...].astype("float32")
            for key in self.wants.keys.get(STATE, ())
        }
        video = {
            key: np.stack(
                [self._at(delta, f"{VIDEO}.{key}") for delta in self.wants.deltas.get(VIDEO, (0,))]
            )[None, ...].astype(np.uint8)
            for key in self.wants.keys.get(VIDEO, ())
        }
        payload: dict[str, Any] = {STATE: state, VIDEO: video}
        language = self.wants.language_key
        if language is not None:
            payload[LANGUAGE] = {language: [[self._instruction_now()]]}
        return payload

    def _instruction_now(self) -> str:
        if self._instruction is not None:
            return self._instruction
        if self._context is not None and self._context.instruction:
            return self._context.instruction
        return NO_INSTRUCTION

    def _chunk(self, actions: Mapping[str, np.ndarray]) -> np.ndarray:
        """``{key: (1, horizon, width)}`` to one ``(horizon, width)`` chunk."""
        parts: dict[str, np.ndarray] = {}
        for field in self.layout.action:
            if field.key not in actions:
                raise ComponentError(
                    f"{self._name}: the server returned {sorted(actions)} and this layout "
                    f"needs {field.key!r}"
                )
            array = np.asarray(actions[field.key], dtype="float32")
            if array.ndim == 3:  # (batch, horizon, width); one episode at a time
                array = array[0]
            if array.ndim != 2:
                raise ComponentError(
                    f"{self._name}: action {field.key!r} came back with shape {array.shape}; "
                    "expected (horizon, width) or (batch, horizon, width)"
                )
            if array.shape[-1] != field.width:
                raise ComponentError(
                    f"{self._name}: action {field.key!r} is {array.shape[-1]} wide and the "
                    f"layout says {field.width}; the checkpoint and the dataset disagree"
                )
            parts[field.key] = array
        widths = {name: part.shape[0] for name, part in parts.items()}
        if len(set(widths.values())) > 1:
            raise ComponentError(
                f"{self._name}: action keys came back with different horizons: {widths}"
            )
        return self.layout.join(ACTION, parts).astype("float32")

    def close(self) -> None:
        self.client.close()


def observation_specs(layout: Layout, wants: Wants, state_channel: str = STATE_CHANNEL) -> Sequence[ChannelSpec]:
    """What a dataset must provide to drive this model.

    Useful before anything is run: hand it to a connector's schema and see what
    is missing, without starting a server or opening an episode.
    """
    specs = [
        ChannelSpec(
            state_channel, "vector", (layout.width(STATE),), "float32",
            dim_labels=layout.labels(STATE),
        )
    ]
    specs.extend(
        ChannelSpec(layout.video.get(key, key), "image", (None, None, 3), "uint8")
        for key in wants.keys.get(VIDEO, ())
    )
    return specs
