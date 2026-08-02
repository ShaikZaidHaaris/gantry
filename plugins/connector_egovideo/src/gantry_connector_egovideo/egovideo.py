"""Reading a folder of egocentric clips, and refusing the ones that describe nothing.

This is the front door. Somebody uploads video through a GUI and it lands here,
and everything downstream -- the estimator, the retargeter, the training run, the
ranking against public datasets -- inherits whatever this reader decided was true.
So it decides as little as possible.

The manifest is the contract with the user
------------------------------------------
A folder of MP4s is not a dataset. What makes it one is a sidecar file saying,
per clip, what the person was doing and where. That file is the entire Tier-1
input contract, and it exists because two of its fields are load-bearing in ways
that are not obvious to whoever is uploading:

``instruction`` -- required, refused if absent. Not for politeness: a
language-conditioned policy trains on the sentence, so a clip without one is a
clip that cannot be used for the thing being measured. Silently dropping it would
shrink somebody's dataset without telling them; silently substituting the folder
name would train on a label nobody wrote.

``scene`` -- required, and the reason is the ranking. "Forty clips" and "forty
clips in one kitchen" are wildly different datasets, and only the second sentence
is checkable. Without a scene id there is no way to tell them apart afterwards,
and location diversity turns out to be the single biggest separator between the
ego datasets that help and the ones that do not.

What it does not do
-------------------
No hand estimation, no segmentation, no captioning, no resampling. Those are
curation ops and they run *after* this, with lineage, so the enrichment can be
argued with and re-run without re-uploading. A connector that quietly estimated
hands would make the estimate indistinguishable from a measurement, which for ego
data is precisely the distinction that matters.

Frames are read lazily
----------------------
An hour of video is tens of gigabytes decoded. ``episode_ids`` and ``schema``
answer from the manifest and the container header alone, so a whole upload can be
inspected, screened and planned against before a single frame is decoded -- which
is what makes it possible to tell a user their manifest is wrong in seconds
rather than after an hour of decoding.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from gantry.contracts.connector import Connector, connector_descriptor
from gantry.errors import ConfigError
from gantry.spine import (
    ArraySource,
    ChannelSpec,
    Descriptor,
    EpisodeLabels,
    EpisodeMeta,
    EpisodeRecord,
)

VERSION = "0.1.0.dev0"

#: What the sidecar is called when the caller does not say.
MANIFEST = "clips.json"

#: Containers worth trying. Not a gate -- a path in the manifest is used as given,
#: and this only drives directory discovery.
SUFFIXES = (".mp4", ".mov", ".mkv", ".avi", ".MP4", ".MOV")

#: Fields a clip entry must carry, and why refusing is kinder than defaulting.
REQUIRED = {
    "path": "which file this is",
    "instruction": (
        "what the person was doing, in words, a language-conditioned policy "
        "trains on this sentence, so a clip without one cannot be used for what "
        "is being measured"
    ),
    "scene": (
        "where this was filmed, 'forty clips' and 'forty clips in one kitchen' "
        "are different datasets, and without a scene id nobody can tell them "
        "apart afterwards"
    ),
}

#: The name given to the video channel.
RGB = "ego_rgb"


@dataclass(frozen=True)
class Clip:
    """One entry from the manifest, after checking."""

    id: str
    path: Path
    instruction: str
    scene: str
    #: Optional and genuinely optional: absent means unknown, not false.
    hand: str | None = None
    outcome: bool | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def annotations(self) -> dict[str, Any]:
        return {
            "instruction": self.instruction,
            "scene": self.scene,
            **({"hand": self.hand} if self.hand else {}),
            **dict(self.metadata),
        }


def read_manifest(path: str | Path, *, root: Path | None = None) -> tuple[Clip, ...]:
    """Parse the sidecar, refusing every clip that is missing something load-bearing.

    All the problems at once, rather than the first one. Somebody uploading forty
    clips through a GUI should get one list of what to fix, not forty rounds of
    fix-one-reupload.
    """
    target = Path(path)
    base = root or target.parent
    try:
        data = json.loads(target.read_text())
    except FileNotFoundError as error:
        raise ConfigError(
            f"no clip manifest at {target}. An ego upload needs one: a JSON list "
            "with, per clip, its path, what the person was doing, and where"
        ) from error
    except json.JSONDecodeError as error:
        raise ConfigError(f"{target} is not valid JSON: {error}") from error

    entries = data.get("clips", data) if isinstance(data, Mapping) else data
    if not isinstance(entries, list):
        raise ConfigError(
            f"{target}: expected a list of clips (or an object with a 'clips' list), "
            f"got {type(entries).__name__}"
        )

    clips: list[Clip] = []
    problems: list[str] = []
    seen: set[str] = set()
    for index, raw in enumerate(entries):
        where = f"clip {index}"
        if not isinstance(raw, Mapping):
            problems.append(f"{where}: expected an object, got {type(raw).__name__}")
            continue
        missing = [key for key in REQUIRED if not str(raw.get(key, "")).strip()]
        if missing:
            problems.extend(f"{where}: no {key}, {REQUIRED[key]}" for key in missing)
            continue
        file = Path(str(raw["path"]))
        if not file.is_absolute():
            file = base / file
        if not file.exists():
            problems.append(f"{where}: {file} does not exist")
            continue
        identifier = str(raw.get("id") or file.stem)
        if identifier in seen:
            problems.append(f"{where}: duplicate id {identifier!r}")
            continue
        seen.add(identifier)
        clips.append(
            Clip(
                id=identifier,
                path=file,
                instruction=str(raw["instruction"]).strip(),
                scene=str(raw["scene"]).strip(),
                hand=str(raw["hand"]).strip() if raw.get("hand") else None,
                outcome=_outcome(raw.get("outcome")),
                metadata={
                    key: value
                    for key, value in raw.items()
                    if key not in {*REQUIRED, "id", "hand", "outcome"}
                },
            )
        )

    if problems:
        raise ConfigError(
            f"{target}: {len(problems)} problem(s) in the clip manifest\n  " + "\n  ".join(problems)
        )
    if not clips:
        raise ConfigError(f"{target}: the manifest lists no clips")
    return tuple(clips)


def _outcome(value: Any) -> bool | None:
    """Whether the attempt worked, if the uploader said.

    Tri-state and it matters. A dataset where every clip is an unlabelled failure
    trains nothing at all -- that happened here once, with a checkpoint fine-tuned
    on a split that was 0-for-52 -- and the only defence is that "unlabelled" and
    "failed" stay different values.
    """
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"true", "success", "yes", "1", "ok"}:
        return True
    if text in {"false", "failure", "fail", "no", "0"}:
        return False
    return None


def probe(path: Path) -> dict[str, Any]:
    """Frame count, size and rate from the container header, without decoding.

    Header-only on purpose: this is what lets a whole upload be validated in
    seconds. A container that cannot be probed is reported rather than guessed
    at, since a guessed frame count becomes a schema that disagrees with the data.
    """
    try:
        import av
    except ImportError as error:  # pragma: no cover - needs the extra
        raise ConfigError(
            "reading video needs a decoder: pip install 'gantry-connector-egovideo[video]'"
        ) from error
    try:
        with av.open(str(path)) as container:
            stream = next((s for s in container.streams if s.type == "video"), None)
            if stream is None:
                raise ConfigError(f"{path} has no video stream")
            rate = stream.average_rate
            frames = int(stream.frames or 0)
            duration = float(container.duration / 1_000_000) if container.duration else 0.0
            if not frames and rate and duration:
                frames = int(duration * float(rate))
            return {
                "frames": frames,
                "height": int(stream.codec_context.height),
                "width": int(stream.codec_context.width),
                "rate_hz": float(rate) if rate else None,
                "seconds": duration or (frames / float(rate) if frames and rate else 0.0),
            }
    except ConfigError:
        raise
    except Exception as error:  # noqa: BLE001 - a container that will not open
        raise ConfigError(f"{path} could not be read as video: {error}") from error


def decode(path: Path, *, stride: int = 1, limit: int | None = None) -> np.ndarray:
    """Frames as ``(t, h, w, 3)`` uint8.

    ``stride`` is here rather than in a later resampling step because decoding is
    the expensive part: an hour at 30 Hz is a hundred thousand frames and a policy
    that sees 5 Hz has no use for ninety thousand of them.
    """
    try:
        import av
    except ImportError as error:  # pragma: no cover - needs the extra
        raise ConfigError(
            "reading video needs a decoder: pip install 'gantry-connector-egovideo[video]'"
        ) from error
    frames: list[np.ndarray] = []
    with av.open(str(path)) as container:
        stream = next((s for s in container.streams if s.type == "video"), None)
        if stream is None:
            raise ConfigError(f"{path} has no video stream")
        for index, frame in enumerate(container.decode(stream)):
            if index % max(1, stride):
                continue
            frames.append(frame.to_ndarray(format="rgb24"))
            if limit is not None and len(frames) >= limit:
                break
    if not frames:
        raise ConfigError(f"{path} decoded to no frames")
    return np.stack(frames)


class EgoVideoConnector(Connector):
    """A folder of ego clips plus its manifest, as episodes."""

    def __init__(
        self,
        path: str | Path,
        *,
        manifest: str = MANIFEST,
        source: str = "ego",
        stride: int = 1,
        rate_hz: float | None = None,
        prober: Any = probe,
        decoder: Any = decode,
    ):
        self._root = Path(path)
        self._source = source
        self._stride = max(1, int(stride))
        self._rate_hz = rate_hz
        self._probe = prober
        self._decode = decoder
        manifest_path = self._root / manifest if self._root.is_dir() else self._root
        self._clips = read_manifest(manifest_path, root=self._root if self._root.is_dir() else None)
        self._by_id = {f"{source}/{clip.id}": clip for clip in self._clips}
        self._probed: dict[str, dict[str, Any]] = {}

    # -- contract ----------------------------------------------------------

    def descriptor(self) -> Descriptor:
        return connector_descriptor(
            name="egovideo",
            version=VERSION,
            lazy=True,
            stage_events=False,
            # Whether an attempt worked is optional in the manifest, so this is
            # true only when somebody actually labelled some. Claiming it
            # otherwise would let an analysis read an unlabelled upload as a run
            # of failures.
            outcomes=any(clip.outcome is not None for clip in self._clips),
            media=True,
            writes=False,
            clips=len(self._clips),
            scenes=len(self.scenes),
            stride=self._stride,
        )

    def episode_ids(self) -> tuple[str, ...]:
        return tuple(self._by_id)

    def schema(self, episode_id: str) -> tuple[ChannelSpec, ...]:
        """Answered from the container header, so a whole upload validates in seconds."""
        clip = self._clip(episode_id)
        header = self._header(episode_id, clip)
        from gantry_semantics_ego import ego_camera

        return (
            ego_camera(
                RGB,
                height=header["height"],
                width=header["width"],
                rate_hz=self._effective_rate(header),
            ),
        )

    def open(self, episode_id: str) -> EpisodeRecord:
        clip = self._clip(episode_id)
        header = self._header(episode_id, clip)
        frames = self._decode(clip.path, stride=self._stride)
        return EpisodeRecord(
            meta=EpisodeMeta(
                id=episode_id,
                source=self._source,
                task=clip.instruction,
                # Not a robot. Said explicitly because everything downstream
                # asks, and "human" is the answer that makes the retargeting
                # step visible rather than assumed.
                embodiment="human",
                extra={
                    "scene": clip.scene,
                    "file": str(clip.path),
                    "seconds": header.get("seconds"),
                    "source_rate_hz": header.get("rate_hz"),
                    "stride": self._stride,
                    **dict(clip.metadata),
                },
            ),
            schema=self.schema(episode_id),
            source=ArraySource({RGB: frames}),
            labels=EpisodeLabels(success=clip.outcome, annotations=clip.annotations),
        )

    # -- what a report wants -----------------------------------------------

    @property
    def clips(self) -> tuple[Clip, ...]:
        return self._clips

    @property
    def scenes(self) -> tuple[str, ...]:
        """Distinct filming locations, in first-seen order.

        Surfaced here because it is the number the ranking turns on, and because
        a user seeing "scenes: 1" beside "clips: 40" understands the problem
        without reading a word of prose.
        """
        return tuple(dict.fromkeys(clip.scene for clip in self._clips))

    def hours(self) -> float:
        """Total footage, from the headers. Zero if nothing could be probed."""
        total = 0.0
        for identifier, clip in self._by_id.items():
            try:
                total += float(self._header(identifier, clip).get("seconds") or 0.0)
            except ConfigError:
                continue
        return total / 3600.0

    def unlabelled(self) -> tuple[str, ...]:
        """Clips whose outcome nobody stated.

        Not a defect on its own -- most ego uploads are unlabelled and that is
        expected. It matters because a dataset that is secretly all failures is
        indistinguishable from one that is all successes until somebody looks,
        and this is the list to look at.
        """
        return tuple(identifier for identifier, clip in self._by_id.items() if clip.outcome is None)

    # -- internals ---------------------------------------------------------

    def _clip(self, episode_id: str) -> Clip:
        try:
            return self._by_id[episode_id]
        except KeyError:
            raise KeyError(
                f"no episode {episode_id!r}; this connector has {len(self._by_id)}"
            ) from None

    def _header(self, episode_id: str, clip: Clip) -> dict[str, Any]:
        if episode_id not in self._probed:
            self._probed[episode_id] = self._probe(clip.path)
        return self._probed[episode_id]

    def _effective_rate(self, header: Mapping[str, Any]) -> float | None:
        if self._rate_hz is not None:
            return self._rate_hz
        rate = header.get("rate_hz")
        return float(rate) / self._stride if rate else None
