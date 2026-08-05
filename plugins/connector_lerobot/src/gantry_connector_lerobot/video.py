"""Frames out of the mp4 side-cars, one window at a time.

A LeRobot dataset keeps its cameras out of the parquet, one mp4 per camera per
episode. That is a sensible way to store them and an awkward way to read them:
the parquet is columnar and seekable, the mp4 is a stream that has to be decoded
forward from a keyframe.

Two things follow, and both matter more than the decoding itself.

**Nothing is decoded until it is asked for.** A hundred episodes of two cameras
is a few gigabytes of pixels; opening them to look at a state vector would make
the lazy spine a lie. So a video channel is described from ``info.json`` -- which
already carries the shape, the codec and the frame rate -- and touched only when
somebody reads it.

**Frames are checked against the parquet.** A truncated mp4 is the failure this
format invites: the episode still loads, the state vector is still the right
length, and the frames are silently off by however many are missing. Every step
after that is measuring an image from the wrong moment. So the frame count is
compared against the episode length before any frame is handed out, and a
disagreement is a refusal rather than a shrug.

The decoder is optional
-----------------------
``av`` is not a dependency of the base install. Reading joint angles should not
require ffmpeg bindings, and the datasets people screen most often have no video
at all. Install ``gantry-connector-lerobot[video]`` to turn the cameras on; the
connector says clearly which of the two situations it is in.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np

from gantry.errors import ComponentError, ConfigError

#: Decode from the start rather than seeking when the window begins this early.
#: A seek lands on the preceding keyframe and decodes forward from there, so for
#: a window near the front it does the same work with more moving parts.
SEEK_THRESHOLD = 32

#: What a decoded frame is converted to. Three channels, eight bits, no scaling -- #: any normalisation is the consumer's decision and its business to declare.
PIXEL_FORMAT = "rgb24"


def decoder() -> Any:
    """The ``av`` module, or a refusal that says how to get it."""
    try:
        import av
    except ImportError as error:  # pragma: no cover - exercised by the extra-less env
        raise ConfigError(
            "reading LeRobot video needs a decoder: pip install "
            "'gantry-connector-lerobot[video]' (it pulls in av, the ffmpeg bindings). "
            "Pass video=False to work with the parquet channels alone"
        ) from error
    return av


def available() -> bool:
    """Whether frames could be read at all, without raising if they cannot."""
    try:
        import av  # noqa: F401
    except ImportError:
        return False
    return True


class VideoSource:
    """One camera, one episode. Decodes only the window it is asked for."""

    def __init__(self, path: Path, name: str, shape: tuple[int, ...], length: int, fps: float):
        self._path = path
        self._name = name
        self._shape = tuple(shape)
        self._length = length
        self._fps = fps or 1.0
        self._checked = False

    @property
    def num_steps(self) -> int:
        return self._length

    def read(self, start: int, stop: int) -> np.ndarray:
        """Frames ``[start, stop)`` as ``(steps, H, W, 3)`` uint8."""
        start, stop = max(0, start), min(stop, self._length)
        if stop <= start:
            return np.empty((0, *self._shape), dtype=np.uint8)

        av = decoder()
        frames: dict[int, np.ndarray] = {}
        try:
            with av.open(str(self._path)) as container:
                stream = container.streams.video[0]
                stream.thread_type = "AUTO"
                self._check_length(stream)
                if start > SEEK_THRESHOLD:
                    container.seek(self._pts(start, stream), stream=stream, backward=True)
                for position, frame in self._walk(container, stream):
                    if position >= stop:
                        break
                    if position >= start:
                        frames[position] = frame.to_ndarray(format=PIXEL_FORMAT)
        except (OSError, ValueError) as error:
            raise ComponentError(
                f"{self._name}: could not decode {self._path.name}: {type(error).__name__}: {error}"
            ) from error

        missing = [index for index in range(start, stop) if index not in frames]
        if missing:
            raise ComponentError(
                f"{self._name}: {self._path.name} has no frame at "
                f"{missing[0] if len(missing) == 1 else missing[:3]}"
                f"{'' if len(missing) <= 3 else f' and {len(missing) - 3} more'}; "
                f"the file is shorter than the episode or its timestamps are broken"
            )
        stacked = np.stack([frames[index] for index in range(start, stop)])
        self._check_shape(stacked)
        return stacked

    def _walk(self, container: Any, stream: Any):
        """Decoded frames paired with the step each one belongs to.

        The index comes from the frame's own timestamp rather than from a
        counter, because after a seek the decoder starts at whatever keyframe
        precedes the target and a counter would be off by that distance.
        """
        counter = 0
        for frame in container.decode(stream):
            if frame.pts is None:
                yield counter, frame
                counter += 1
                continue
            seconds = float(frame.pts * stream.time_base)
            yield int(round(seconds * self._fps)), frame

    def _pts(self, index: int, stream: Any) -> int:
        return int(index / self._fps / float(stream.time_base))

    def _check_length(self, stream: Any) -> None:
        """The mp4 and the parquet must agree about how long the episode is."""
        if self._checked:
            return
        self._checked = True
        declared = int(getattr(stream, "frames", 0) or 0)
        if declared and declared != self._length:
            raise ComponentError(
                f"{self._name}: {self._path.name} holds {declared} frame(s) and the "
                f"episode is {self._length} step(s) long; every frame after the "
                "difference belongs to a different moment than the state beside it"
            )

    def _check_shape(self, frames: np.ndarray) -> None:
        if frames.shape[1:] != self._shape:
            raise ComponentError(
                f"{self._name}: {self._path.name} decodes to {frames.shape[1:]} and "
                f"info.json declares {self._shape}"
            )


class MultiSource:
    """Parquet columns and camera streams behind one step source.

    An episode is one thing to a reader even though it is several files on disk,
    and which file a channel came from is not something a feedback module should
    have to know.
    """

    def __init__(self, table: Any, videos: dict[str, VideoSource]):
        self._table = table
        self._videos = videos

    @property
    def num_steps(self) -> int:
        return self._table.num_steps

    @property
    def channels(self) -> tuple[str, ...]:
        return tuple(self._table.channels) + tuple(self._videos)

    def read(
        self,
        channels: Sequence[str] | None = None,
        start: int = 0,
        stop: int | None = None,
    ) -> dict[str, np.ndarray]:
        wanted = tuple(channels) if channels is not None else self.channels
        unknown = [name for name in wanted if name not in self.channels]
        if unknown:
            raise KeyError(f"no such channel(s): {unknown}")

        columns = [name for name in wanted if name not in self._videos]
        out = self._table.read(columns, start, stop) if columns else {}
        if any(name in self._videos for name in wanted):
            first, last = (
                max(0, start),
                self.num_steps if stop is None else min(stop, self.num_steps),
            )
            for name in wanted:
                if name in self._videos:
                    out[name] = self._videos[name].read(first, last)
        return {name: out[name] for name in wanted}
