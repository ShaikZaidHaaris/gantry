"""Two half-descriptions of the same interface, and the check that they meet.

Neither side knows the whole thing:

**The server** knows what the model wants — which state, video and language keys
it reads, and how many timesteps of each. It does not know how wide any of them
are, because that is a property of the data it was trained on.

**The dataset** knows the widths. ``meta/modality.json`` is where a LeRobot
dataset records how its flat ``observation.state`` and ``action`` vectors slice
into named fields, and it is the only place that mapping is written down.

Put together they are a complete wiring. Left apart, the failure is silent: a
seven-wide action vector split into the wrong fields is still seven numbers, and
a run built on it produces a full set of plausible results.

So :func:`check` compares them and refuses on disagreement, before an episode is
opened. A model asking for a key the dataset has never heard of is a wiring
mistake, and it is worth exactly one second of somebody's time to hear about it
at that point rather than after a night of inference.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from gantry.errors import ConfigError
from gantry.spine import Verdict

#: Modalities a GR00T server describes.
VIDEO, STATE, ACTION, LANGUAGE = "video", "state", "action", "language"

#: The prefix GR00T's language keys carry and ``modality.json`` does not.
ANNOTATION = "annotation"


@dataclass
class Field:
    """One named slice of a flat vector."""

    key: str
    start: int
    end: int

    @property
    def width(self) -> int:
        return self.end - self.start

    def __post_init__(self) -> None:
        if self.end <= self.start:
            raise ConfigError(f"field {self.key!r} spans {self.start}:{self.end}, which is empty")


@dataclass(frozen=True)
class Layout:
    """How a dataset's flat vectors divide into named fields.

    This is ``meta/modality.json``, read and checked. Fields are ordered by
    where they sit in the vector rather than by the order the file lists them,
    because position is the fact and file order is a formatting detail.
    """

    state: tuple[Field, ...] = ()
    action: tuple[Field, ...] = ()
    #: The model's key -> the column it is stored under. Both halves matter: the
    #: model asks for ``image``, the dataset stores ``observation.images.image``,
    #: and this file is where the two names are written down together.
    video: Mapping[str, str] = field(default_factory=dict)
    annotation: Mapping[str, str] = field(default_factory=dict)

    @classmethod
    def from_schema(
        cls,
        schema: Sequence[Any],
        *,
        state: str = "observation.state",
        action: str = "action",
        video: Sequence[str] = (),
    ) -> "Layout":
        """Build from channel specs, which any connector already provides.

        The preferred path, and the one that keeps this plugin on its own plane.
        Reading ``modality.json`` directly means a *policy* knows a *dataset*
        format — it works, and it is a leak: two plugins on two planes end up
        parsing the same file, and the third format to arrive has to teach both.

        A connector's job is to describe its data, and a described channel
        already carries ``dim_labels`` — one name per element. That is the same
        information the sidecar holds, in the vocabulary the framework already
        speaks, so it is what this reads.
        """
        by_name = {spec.name: spec for spec in schema}
        return cls(
            state=_from_labels(by_name.get(state), STATE),
            action=_from_labels(by_name.get(action), ACTION),
            video={key: key for key in video},
        )

    @classmethod
    def from_json(cls, path: str | Path) -> "Layout":
        path = Path(path)
        if path.is_dir():
            path = path / "meta" / "modality.json"
        if not path.exists():
            raise ConfigError(
                f"no modality.json at {path}; a GR00T policy needs one to know how the "
                "dataset's flat vectors divide into the fields the model reads"
            )
        payload = json.loads(path.read_text())
        return cls(
            state=_fields(payload.get(STATE, {}), STATE),
            action=_fields(payload.get(ACTION, {}), ACTION),
            video=_originals(payload.get(VIDEO, {})),
            annotation=_originals(payload.get(ANNOTATION, {})),
        )

    # -- the flat vector ---------------------------------------------------

    def width(self, modality: str) -> int:
        fields = self.fields(modality)
        return max((field.end for field in fields), default=0)

    def fields(self, modality: str) -> tuple[Field, ...]:
        if modality == STATE:
            return self.state
        if modality == ACTION:
            return self.action
        raise ConfigError(f"{modality!r} has no fields; only {STATE!r} and {ACTION!r} do")

    def labels(self, modality: str) -> tuple[str, ...]:
        """One name per element, so a dimension can be addressed by what it is.

        A field wider than one gets numbered suffixes. Core checks label order
        when two channels are bound, which is what turns "these are both seven
        wide" into "these are the same seven things in the same order".
        """
        names: list[str] = []
        for entry in self.fields(modality):
            if entry.width == 1:
                names.append(entry.key)
            else:
                names.extend(f"{entry.key}.{index}" for index in range(entry.width))
        return tuple(names)

    def split(self, modality: str, vector: np.ndarray) -> dict[str, np.ndarray]:
        """A flat vector to ``{key: slice}``, along the last axis."""
        array = np.asarray(vector)
        expected = self.width(modality)
        if array.shape[-1] != expected:
            raise ConfigError(
                f"{modality} is {array.shape[-1]} wide but this layout describes "
                f"{expected}: {[f.key for f in self.fields(modality)]}"
            )
        return {field.key: array[..., field.start : field.end] for field in self.fields(modality)}

    def join(self, modality: str, parts: Mapping[str, np.ndarray]) -> np.ndarray:
        """``{key: slice}`` back to one flat vector, in layout order."""
        fields = self.fields(modality)
        missing = [field.key for field in fields if field.key not in parts]
        if missing:
            raise ConfigError(f"{modality}: nothing returned for {missing}")
        return np.concatenate([np.asarray(parts[field.key]) for field in fields], axis=-1)


def _fields(payload: Mapping[str, Any], modality: str) -> tuple[Field, ...]:
    fields = tuple(
        sorted(
            (Field(key, int(span["start"]), int(span["end"])) for key, span in payload.items()),
            key=lambda field: field.start,
        )
    )
    for earlier, later in zip(fields, fields[1:]):
        if earlier.end != later.start:
            gap = "overlaps" if later.start < earlier.end else "leaves a gap before"
            raise ConfigError(
                f"{modality} layout: {later.key!r} {gap} {earlier.key!r} "
                f"({earlier.key}={earlier.start}:{earlier.end}, "
                f"{later.key}={later.start}:{later.end})"
            )
    return fields


def _from_labels(spec: Any, modality: str) -> tuple[Field, ...]:
    """Dimension labels back to spans, collapsing ``gripper.0``/``gripper.1``.

    The inverse of what a connector does when it reads a sidecar, so a channel
    labelled by any reader — from a sidecar, from a feature list, or declared by
    hand — describes the same fields to the model.
    """
    if spec is None:
        return ()
    labels = getattr(spec, "dim_labels", None)
    if not labels:
        raise ConfigError(
            f"channel {getattr(spec, 'name', modality)!r} has no dimension labels, so "
            f"there is no way to know which of its numbers are which {modality} field. "
            "Use a connector that labels its channels, or pass a Layout built by hand"
        )
    fields: list[Field] = []
    for position, label in enumerate(labels):
        head, _, tail = label.rpartition(".")
        base = head if head and tail.isdigit() else label
        if fields and fields[-1].key == base:
            fields[-1] = Field(base, fields[-1].start, position + 1)
        else:
            fields.append(Field(base, position, position + 1))
    return tuple(fields)


def _originals(payload: Mapping[str, Any]) -> dict[str, str]:
    """``{model key: dataset column}``, falling back to the key itself."""
    return {
        key: str((entry or {}).get("original_key", key)) if isinstance(entry, Mapping) else key
        for key, entry in payload.items()
    }


@dataclass(frozen=True)
class Wants:
    """What the model asked for: keys per modality, and how many timesteps.

    ``deltas`` are offsets from now, so ``[0]`` is the current frame alone and
    ``[-15, 0]`` is the current frame plus one from fifteen steps ago. The
    number of them is the temporal depth the model expects, and getting it wrong
    is not a shape error the model will catch politely.
    """

    keys: Mapping[str, tuple[str, ...]]
    deltas: Mapping[str, tuple[int, ...]]

    @classmethod
    def from_server(cls, config: Mapping[str, Any]) -> "Wants":
        keys, deltas = {}, {}
        for modality, entry in config.items():
            entry = _as_mapping(entry, modality)
            keys[modality] = tuple(entry.get("modality_keys") or ())
            deltas[modality] = tuple(int(d) for d in entry.get("delta_indices") or (0,))
        return cls(keys, deltas)

    def horizon(self, modality: str) -> int:
        return len(self.deltas.get(modality, (0,)))

    def needs_history(self, modality: str) -> bool:
        return any(delta != 0 for delta in self.deltas.get(modality, (0,)))

    @property
    def language_key(self) -> str | None:
        """The first language key. GR00T uses one at inference even when the
        embodiment declares several paraphrases for training."""
        keys = self.keys.get(LANGUAGE, ())
        return keys[0] if keys else None


def _as_mapping(entry: Any, modality: str) -> Mapping[str, Any]:
    if isinstance(entry, Mapping):
        return {
            (key.decode() if isinstance(key, bytes) else key): value for key, value in entry.items()
        }
    raise ConfigError(
        f"the server described {modality!r} as {type(entry).__name__}, expected an object "
        "with 'modality_keys' and 'delta_indices'"
    )


def check(layout: Layout, wants: Wants) -> Verdict:
    """Does this dataset carry what this model asked for?

    Named after what it answers rather than what it inspects: every refusal here
    is a wiring mistake that would otherwise surface as a number.
    """
    checks = [
        _keys_present(STATE, wants.keys.get(STATE, ()), {f.key for f in layout.state}, layout),
        _keys_present(ACTION, wants.keys.get(ACTION, ()), {f.key for f in layout.action}, layout),
        _keys_present(VIDEO, wants.keys.get(VIDEO, ()), set(layout.video), layout),
    ]
    language = wants.language_key
    if language is not None and layout.annotation:
        available = {f"{ANNOTATION}.{key}" for key in layout.annotation}

        if language not in available:
            checks.append(
                Verdict.no(
                    "gr00t.language_key",
                    f"the model reads {language!r} and the dataset annotates {sorted(available)}",
                    hint="pass instruction= to send the text yourself",
                )
            )
    return Verdict.all(checks)


def _keys_present(
    modality: str, wanted: Sequence[str], available: set[str], layout: Layout
) -> Verdict:
    missing = [key for key in wanted if key not in available]
    if not missing:
        return Verdict.yes()
    return Verdict.no(
        "gr00t.missing_key",
        f"the model reads {modality} {missing} and this dataset does not have "
        f"{'it' if len(missing) == 1 else 'them'}; it has {sorted(available)}",
        hint="a modality.json from the dataset the checkpoint was trained on will match; "
        "one from a different embodiment will not",
        modality=modality,
        missing=missing,
    )
