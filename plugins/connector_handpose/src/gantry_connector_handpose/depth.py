"""Metric position from a depth map, instead of from the hand's own size.

The perspective-n-point path in :mod:`.pnp` recovers distance by treating the
hand as a ruler of known length. It works — measured 0.36 to 0.70 m on real
kitchen footage — and it has one structural weakness: *every* coordinate scales
with the assumed focal length. Get the intrinsics 20% wrong and the whole
trajectory is 20% away from the truth, uniformly and invisibly.

Reading depth directly is better in a specific and limited way. Distance comes
from the depth model rather than from the focal length, so an intrinsics error
no longer scales it. The sideways coordinates still need intrinsics — unprojection
is ``x = (u - cx) * z / fx`` — so a focal length error skews the trajectory
rather than dilating it. Skew is the less damaging failure of the two, and more
importantly it is a *different* failure, so the two paths disagreeing is itself
a signal that something is wrong.

The distinction nobody documents clearly
----------------------------------------
Most monocular depth models output **relative** depth: a number that increases
with distance, in no unit, unique up to an unknown scale and shift. That is
useless here and looks identical to metric depth in every array. Only the
specifically fine-tuned metric checkpoints return metres.

So :class:`Depth` carries ``metric`` and ``licence``, and a relative model is
refused for lifting rather than quietly producing a trajectory in furlongs. This
is the same failure the ego vocabulary already exists to prevent, arriving from
a new direction.

Licences, since that is why this path exists at all
---------------------------------------------------
Depth Anything V2's *small* checkpoint is Apache-2.0; its larger ones are
CC-BY-NC. ZoeDepth is MIT. The wires below carry what they are, and it
propagates onto every derived episode, so a dataset built with a non-commercial
depth model says so rather than being discovered at legal review.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol

import numpy as np

from gantry.errors import ConfigError

from .pnp import Intrinsics

#: Half-width of the patch sampled around a keypoint, in pixels. A single pixel
#: of monocular depth is noisy and, on a hand edge, frequently belongs to the
#: background instead — which puts the wrist two metres away for one frame and
#: produces a trajectory that teleports.
PATCH = 5

#: A depth reading outside this range is not a hand in a kitchen.
NEAR, FAR = 0.05, 3.0


@dataclass(frozen=True)
class DepthMaps:
    """One clip's depth, and what it is worth.

    ``metric`` is the load-bearing field. Relative depth and metric depth are
    both ``(steps, h, w)`` float arrays and only one of them is in metres.
    """

    values: np.ndarray
    metric: bool
    licence: str = "unknown"
    model: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        array = np.asarray(self.values)
        if array.ndim != 3:
            raise ConfigError(f"expected (steps, height, width) depth, got {array.shape}")


class Depth(Protocol):
    """Whatever turns frames into depth."""

    def estimate(self, frames: np.ndarray) -> DepthMaps:
        """``(steps, h, w, 3)`` uint8 in, one :class:`DepthMaps` out."""


def sample(depth: np.ndarray, points: np.ndarray, *, patch: int = PATCH) -> np.ndarray:
    """Depth at each 2D point, as the median of a small patch around it.

    Median rather than the pixel value, because a keypoint on the edge of a hand
    lands on the background about as often as on the hand, and a background
    reading puts the wrist metres away for one frame. The median of a patch
    survives that; a mean does not, because the background value is an outlier
    rather than noise.
    """
    frame_depth = np.asarray(depth, dtype=float)
    pixels = np.asarray(points, dtype=float).reshape(-1, 2)
    height, width = frame_depth.shape[-2:]
    out = np.zeros(len(pixels), dtype=float)
    for index, (u, v) in enumerate(pixels):
        column, row = int(round(u)), int(round(v))
        if not (0 <= column < width and 0 <= row < height):
            continue
        window = frame_depth[
            max(0, row - patch) : row + patch + 1,
            max(0, column - patch) : column + patch + 1,
        ]
        finite = window[np.isfinite(window) & (window > 0)]
        out[index] = float(np.median(finite)) if finite.size else 0.0
    return out


def unproject(points: np.ndarray, depths: np.ndarray, intrinsics: Intrinsics) -> np.ndarray:
    """2D pixels plus depth to 3D in the camera frame.

    The asymmetry worth knowing: ``z`` comes from the depth model and ``x``/``y``
    come from the focal length. So an intrinsics error skews this trajectory,
    while it would have dilated a PnP one. Different failure, same cause, which
    is exactly why running both and comparing is worth the cost.
    """
    pixels = np.asarray(points, dtype=float).reshape(-1, 2)
    z = np.asarray(depths, dtype=float).reshape(-1)
    x = (pixels[:, 0] - intrinsics.cx) * z / intrinsics.fx
    y = (pixels[:, 1] - intrinsics.cy) * z / intrinsics.fy
    return np.stack([x, y, z], axis=1)


def lift(
    keypoints: np.ndarray,
    maps: DepthMaps,
    intrinsics: Intrinsics,
    *,
    patch: int = PATCH,
    near: float = NEAR,
    far: float = FAR,
) -> tuple[np.ndarray, np.ndarray]:
    """A sequence of 2D hands to metric 3D. Returns ``(positions, solved)``.

    ``keypoints`` is ``(steps, joints, 2)`` in pixels. Only the wrist is lifted —
    the finger joints come from the hand model, and lifting each one
    independently from a noisy depth map produces a hand that flexes at random.
    """
    if not maps.metric:
        raise ConfigError(
            f"{maps.model or 'this depth model'} returns relative depth, which is a "
            "number that increases with distance in no unit at all. Unprojecting it "
            "produces a trajectory in unknown units that looks exactly like one in "
            "metres. Use a metric checkpoint, or recover scale from the hand with "
            "the perspective-n-point path instead"
        )
    points = np.asarray(keypoints, dtype=float)
    if points.ndim != 3 or points.shape[-1] != 2:
        raise ConfigError(f"expected (steps, joints, 2) pixel keypoints, got {points.shape}")
    values = np.asarray(maps.values, dtype=float)
    if len(values) != len(points):
        raise ConfigError(f"{len(values)} depth frames for {len(points)} keypoint frames")

    positions = np.zeros((len(points), 3), dtype="float32")
    solved = np.zeros(len(points), dtype=bool)
    for index in range(len(points)):
        wrist = points[index, :1, :]
        if not np.any(wrist):
            continue
        z = sample(values[index], wrist, patch=patch)
        if not (near <= z[0] <= far):
            # Out of range is a reading about something else — the ceiling, a
            # hole in the depth map, the far wall seen through a gap between
            # fingers. Left unsolved rather than clamped, because a clamped
            # value is indistinguishable from a real one downstream.
            continue
        positions[index] = unproject(wrist, z, intrinsics)[0]
        solved[index] = True
    return positions, solved


# -- wires --------------------------------------------------------------------


def depth_anything(size: str = "small", *, device: str = "cuda", metric: bool = False) -> Depth:
    """Depth Anything V2, through transformers.

    ``size`` matters legally as well as numerically: the small checkpoint is
    Apache-2.0 and the larger ones are CC-BY-NC. The default is small for that
    reason, not for speed.

    ``metric`` says whether the checkpoint chosen is one of the metric-fine-tuned
    variants. It is False by default because the headline Depth Anything models
    are *relative*, and a caller who does not know that should be stopped by
    :func:`lift` rather than handed a trajectory in no units.
    """
    licences = {
        "small": "Apache-2.0",
        "base": "CC-BY-NC-4.0 (non-commercial)",
        "large": "CC-BY-NC-4.0 (non-commercial)",
    }
    if size not in licences:
        raise ConfigError(f"unknown size {size!r}; known: {sorted(licences)}")
    name = f"depth-anything/Depth-Anything-V2-{size.capitalize()}-hf"

    class DepthAnything:
        licence = licences[size]

        def estimate(self, frames: np.ndarray) -> DepthMaps:  # pragma: no cover - needs the model
            try:
                import torch
                from transformers import pipeline
            except ImportError as error:
                raise ConfigError(
                    "depth needs transformers and torch: pip install "
                    "'gantry-connector-handpose[depth]'"
                ) from error
            from PIL import Image

            pipe = pipeline(
                task="depth-estimation",
                model=name,
                device=0 if device == "cuda" and torch.cuda.is_available() else -1,
            )
            out = [
                np.asarray(pipe(Image.fromarray(np.asarray(frame)))["predicted_depth"])
                for frame in frames
            ]
            return DepthMaps(
                values=np.stack(out),
                metric=metric,
                licence=licences[size],
                model=name,
                metadata={"size": size},
            )

    return DepthAnything()


def zoedepth(variant: str = "ZoeD_N", *, device: str = "cuda") -> Depth:
    """ZoeDepth, which is metric by construction and MIT licensed.

    The commercially-clean metric option, and the reason this module has two
    wires rather than one: Depth Anything is better and its metric checkpoints
    are not all permissive, so which one to use is a licence decision as much as
    an accuracy one.
    """

    class ZoeDepth:
        licence = "MIT"

        def estimate(self, frames: np.ndarray) -> DepthMaps:  # pragma: no cover - needs the model
            try:
                import torch
            except ImportError as error:
                raise ConfigError(
                    "depth needs torch: pip install 'gantry-connector-handpose[depth]'"
                ) from error
            model = torch.hub.load("isl-org/ZoeDepth", variant, pretrained=True)
            model = model.to(device if torch.cuda.is_available() else "cpu").eval()
            out = []
            with torch.no_grad():
                for frame in frames:
                    tensor = (
                        torch.from_numpy(np.asarray(frame)).permute(2, 0, 1)[None].float() / 255.0
                    ).to(next(model.parameters()).device)
                    out.append(model.infer(tensor).squeeze().cpu().numpy())
            return DepthMaps(
                values=np.stack(out),
                metric=True,
                licence="MIT",
                model=f"ZoeDepth/{variant}",
            )

    return ZoeDepth()


def constant(metres: float = 0.5, *, shape: tuple[int, int] = (64, 64)) -> Depth:
    """A depth model that says everything is at one distance.

    For tests and for a deliberate baseline: a policy trained on data lifted with
    a constant depth has had all its depth information removed, and comparing
    against it says how much the depth model was actually contributing. Declares
    itself metric because it is — trivially, and wrongly, which is the point.
    """

    class Constant:
        licence = "n/a"

        def estimate(self, frames: np.ndarray) -> DepthMaps:
            steps = len(np.asarray(frames))
            return DepthMaps(
                values=np.full((steps, *shape), float(metres)),
                metric=True,
                licence="n/a",
                model=f"constant({metres} m)",
            )

    return Constant()


def agreement(
    pnp_positions: np.ndarray, depth_positions: np.ndarray, *, solved: np.ndarray | None = None
) -> dict[str, Any]:
    """How far the two independent paths disagree, in metres.

    Worth computing whenever both are available. They fail differently — a wrong
    focal length dilates the perspective-n-point answer and skews the depth one —
    so agreement is evidence that neither is badly wrong, and a large
    disagreement localises the problem without either being ground truth.
    """
    a = np.asarray(pnp_positions, dtype=float).reshape(-1, 3)
    b = np.asarray(depth_positions, dtype=float).reshape(-1, 3)
    mask = np.any(a != 0, axis=1) & np.any(b != 0, axis=1)
    if solved is not None:
        mask &= np.asarray(solved, dtype=bool)
    if not mask.any():
        return {"compared": 0, "agree": None}
    gap = np.linalg.norm(a[mask] - b[mask], axis=1)
    ratio = np.linalg.norm(a[mask], axis=1) / np.maximum(np.linalg.norm(b[mask], axis=1), 1e-6)
    return {
        "compared": int(mask.sum()),
        "median_gap_m": round(float(np.median(gap)), 4),
        "p90_gap_m": round(float(np.percentile(gap, 90)), 4),
        # A ratio consistently away from one is the signature of a wrong focal
        # length: PnP scales with it and depth does not.
        "median_scale_ratio": round(float(np.median(ratio)), 4),
    }
