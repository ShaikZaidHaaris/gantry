"""Accept raw egocentric video, not only a robot export.

Why this exists
---------------
Intake reads LeRobot v2 and nothing else, so it refuses the cheapest
robot-learning data in the world: a person wearing a camera in their own
kitchen. There is no `action` column in that footage because no robot was there,
and the gate that needs one therefore never sees it.

Gantry has answered this since before the product did. `manifests/ego/upload.json`
declares the chain, and every plugin in it ships in this repository:

    egovideo -> handpose -> egoactions

Frames, then hands in the frames, then an arm's command that would put a gripper
where the hand was. What was missing is the seam between that chain and the
upload box, which is all this module is.

Two ways to get the poses, and neither one invents anything
-----------------------------------------------------------
**Ours.** We run a hand tracker over the frames. Cheap for the contributor,
costs minutes of CPU here, and the estimate is only as good as monocular
tracking on their footage.

**Theirs.** A lab with its own tracker, or with a motion-capture rig, uploads
what it already has and we skip estimation entirely. `Estimator` is a Protocol
with one method, so "bring your own" is a reader rather than a special case.

The distinction is recorded on every episode, because a result from tracked
poses and a result from estimated ones are not the same claim.

What is refused rather than guessed
-----------------------------------
Every clip must declare what the person was doing and where it was filmed. The
connector already refuses a manifest without them, and this module does not
soften that. The reason is the whole point of the product: a language-conditioned
policy trains on that sentence, so a clip labelled with a sentence somebody made
up teaches a made-up thing, and `scene` is what stops forty clips of one kitchen
being counted as forty independent kitchens.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Mapping

Report = Callable[..., None]


def _quiet(*_args, **_kwargs) -> None:
    """Default report: say nothing. Callable without a worker."""


#: The sidecar that marks an archive as raw ego footage rather than an export.
MANIFEST = "clips.json"

#: Where a contributor's own poses live, when they have them. One `.npz` per
#: clip, named after the clip's file.
POSES = "poses"

#: What the conversion did, left in the dataset it produced. Read back by intake
#: so the answer reaches the report rather than stopping at the worker.
SIDECAR = "gantry_ego.json"

#: Frames per second the chain writes. The stride below decimates to it.
FPS = 10.0

#: Every Nth frame. Ego footage is 50-60fps and a policy consumes ~10, so
#: reading all of them costs minutes of tracking for frames nothing will use.
STRIDE = 6

#: What a policy consumes. Storing the source resolution instead is what filled
#: a disk and took the product down, and nothing downstream reads the extra
#: pixels.
SIZE = (224, 224)


#: Where the hand landmarker's weights are, when the operator has not said.
#: Overridden by ``BENCH_HAND_MODEL``.
MODEL_PATHS = (
    "/home/ubuntu/egorun/models/hand_landmarker.task",
    "/opt/gantry/models/hand_landmarker.task",
)


def model_file() -> Path:
    """The tracker's weights, or a refusal that names what to install.

    Checked once, before a frame is decoded. Without this the first clip fails
    deep inside MediaPipe with "ExternalFile must specify at least one of
    'file_content', 'file_name'...", every subsequent clip fails identically,
    and the gate reports that nothing in the footage had usable hands. Which
    blames a contributor for a file missing on our machine.
    """
    import os

    from gantry.errors import ConfigError

    override = os.environ.get("BENCH_HAND_MODEL", "").strip()
    for candidate in ([override] if override else []) + list(MODEL_PATHS):
        if candidate and Path(candidate).is_file():
            return Path(candidate)
    raise ConfigError(
        "the hand landmarker's weights are not on this worker. Download "
        "hand_landmarker.task and point BENCH_HAND_MODEL at it, or install it to "
        f"{MODEL_PATHS[1]}"
    )


def find(root: Path) -> Path | None:
    """The directory holding a clip manifest, or None if this is not ego footage."""
    hit = next(root.rglob(MANIFEST), None)
    return hit.parent if hit is not None else None


def describe(folder: Path) -> dict:
    """What the manifest says, before anything is decoded.

    Read separately from the conversion so intake can answer "what is in this"
    in the second it is meant to take, and refuse a broken manifest without
    having decoded a single frame.
    """
    entries = json.loads((folder / MANIFEST).read_text())
    clips = entries.get("clips", entries) if isinstance(entries, Mapping) else entries
    if not isinstance(clips, list):
        raise ValueError("the clip manifest is not a list of clips")

    missing = [
        f"clip {i}: no {field}"
        for i, c in enumerate(clips)
        for field in ("path", "instruction", "scene")
        if not (isinstance(c, Mapping) and str(c.get(field, "")).strip())
    ]
    present = [c for c in clips if isinstance(c, Mapping) and c.get("path")]
    absent = [str(c["path"]) for c in present if not (folder / str(c["path"])).exists()]
    supplied = folder / POSES

    return {
        "clips": len(clips),
        "scenes": sorted({str(c.get("scene", "")) for c in present if c.get("scene")}),
        "instructions": sorted({str(c.get("instruction", "")) for c in present if c.get("instruction")}),
        "missing_fields": missing,
        "missing_files": absent,
        # Their poses or ours. Detected rather than configured, because the
        # contributor already answered by what they did or did not upload.
        "poses": "yours" if supplied.is_dir() and any(supplied.iterdir()) else "ours",
    }


def check(found: dict) -> list[dict]:
    """Refusals a manifest can earn, in the shape intake already uses."""
    out: list[dict] = []
    if found["clips"] == 0:
        out.append(
            {
                "code": "ego.no_clips",
                "severity": "strong",
                "summary": "the clip manifest lists no clips",
                "prescription": None,
            }
        )
    if found["missing_fields"]:
        shown = "; ".join(found["missing_fields"][:5])
        out.append(
            {
                "code": "ego.manifest_incomplete",
                "severity": "strong",
                "summary": (
                    f"{len(found['missing_fields'])} clip entries are missing what they "
                    f"have to say: {shown}"
                ),
                "prescription": (
                    "Every clip needs the sentence describing what was done and an id for "
                    "where it was filmed. Neither is guessed for you: a policy trains on "
                    "the sentence, and without the place nobody can tell forty kitchens "
                    "from one kitchen forty times."
                ),
            }
        )
    if found["missing_files"]:
        shown = ", ".join(found["missing_files"][:4])
        out.append(
            {
                "code": "ego.file_missing",
                "severity": "strong",
                "summary": f"{len(found['missing_files'])} clips name a file that is not in the archive: {shown}",
                "prescription": "Check the paths in the manifest are relative to the folder it sits in.",
            }
        )
    if found["clips"] and len(found["scenes"]) == 1:
        out.append(
            {
                "code": "ego.one_scene",
                "severity": "moderate",
                "summary": (
                    f"all {found['clips']} clips were filmed in one place "
                    f"({found['scenes'][0]})"
                ),
                "prescription": (
                    "Clips from one room are not independent evidence. Held-out clips are "
                    "grouped by place, so a corpus with one place can be measured but "
                    "cannot be generalised from. Footage from more places is worth more "
                    "than more footage from this one."
                ),
            }
        )
    return out


def _per_resolution(build, rig: str):
    """An estimator that reads the frame size instead of being told it.

    Intrinsics were declared once for the whole upload, at one width and height.
    A corpus is not one resolution: in EPIC a few participants are 1280x720
    where the rest are 1920x1080, and a 720p frame solved with 1080p intrinsics
    does not fail. The focal length is wrong by the ratio of the widths, so
    perspective-n-point returns a hand at a plausible pose the wrong distance
    away, and every action retargeted from it is smoothly, confidently wrong.
    Nothing downstream can detect that, which is what makes it worth fixing
    here rather than warning about.

    Focal length in pixels is the lens's field of view times the width, so the
    same rig at another resolution is a different `Intrinsics` and `for_rig`
    already computes it. All that was missing was asking per clip.

    Built once per distinct resolution rather than per clip, because the
    detector and the hand model behind it are expensive to construct and do not
    depend on the size.

    One assumption remains and it is recorded rather than hidden: this treats a
    second resolution as the same lens scaled, not as a crop. A camera that
    crops to change resolution has a different field of view, and no arithmetic
    on the frame size can tell the two apart.
    """
    from gantry_connector_handpose import for_rig

    cache: dict[tuple[int, int], Any] = {}

    class PerResolution:
        #: Whatever the underlying estimator declares, resolved on first use.
        licence = "unknown"

        def estimate(self, frames):
            import numpy as np

            height, width = np.asarray(frames).shape[1:3]
            key = (int(width), int(height))
            if key not in cache:
                cache[key] = build(for_rig(rig, width=key[0], height=key[1]))
            estimator = cache[key]
            self.licence = getattr(estimator, "licence", "unknown")
            track = estimator.estimate(frames)
            # Said on the episode, because "solved at 1280x720" and "solved at
            # 1920x1080" are different measurements and the report should be
            # able to tell them apart afterwards.
            metadata = dict(getattr(track, "metadata", {}) or {})
            metadata["solved_at"] = f"{key[0]}x{key[1]}"
            metadata["rig"] = rig
            try:
                object.__setattr__(track, "metadata", metadata)
            except Exception:  # noqa: BLE001 - metadata is a nicety, the pose is not
                pass
            return track

        @property
        def resolutions(self) -> list[str]:
            return [f"{w}x{h}" for (w, h) in sorted(cache)]

    return PerResolution()


def _supplied(folder: Path, clips: list[dict]):
    """An estimator that reads the contributor's own poses instead of guessing.

    `Estimator` is one method, so their tracker plugs in where ours would. The
    per-clip file carries keypoints and confidence for each hand, and the
    confidence is not optional: without it every consumer has to treat a guess
    and a firm detection alike, which is exactly what monocular tracking gets
    wrong in the tail.
    """
    import numpy as np
    from gantry_connector_handpose import Track

    class Supplied:
        def __init__(self) -> None:
            self._by_clip: dict[str, Path] = {}
            for c in clips:
                stem = Path(str(c["path"])).stem
                for candidate in (folder / POSES / f"{stem}.npz",):
                    if candidate.exists():
                        self._by_clip[stem] = candidate

        def for_clip(self, stem: str) -> Path | None:
            return self._by_clip.get(stem)

        def estimate(self, frames):  # pragma: no cover - exercised with real uploads
            raise RuntimeError(
                "a supplied-pose upload is read per clip; this estimator is not called "
                "frame-wise"
            )

        def load(self, stem: str, steps: int) -> Track:
            path = self._by_clip[stem]
            data = np.load(path)
            hands = sorted({k.split(".")[0] for k in data.files if "." in k})
            keypoints = {h: data[f"{h}.keypoints"][:steps] for h in hands}
            confidence = {h: data[f"{h}.confidence"][:steps] for h in hands}
            world = {h: data[f"{h}.world"][:steps] for h in hands if f"{h}.world" in data.files}
            return Track(
                keypoints=keypoints,
                confidence=confidence,
                world=world,
                licence=str(data["licence"]) if "licence" in data.files else "unknown",
                metadata={"source": "supplied by the contributor"},
            )

    return Supplied()


def convert(
    folder: Path,
    out: Path,
    report: Report = _quiet,
    *,
    model_path: str | None = None,
    rig: str = "gopro_hero5_wide",
) -> dict:
    """Run the ego chain and write a LeRobot dataset at ``out``.

    Writing LeRobot rather than handing episodes straight to the next gate is
    deliberate: every gate after intake already reads that format, so an ego
    upload becomes an ordinary one the moment this returns, and nothing
    downstream learns a second shape.
    """
    from gantry_connector_egoactions import EgoActionConnector
    from gantry_connector_egovideo import EgoVideoConnector
    from gantry_connector_handpose import HandPoseConnector, mediapipe, rtm_with_mediapipe, rtmpose
    from gantry_connector_lerobot import LeRobotConnector
    from gantry_retargeter_hands import Hand, HandToArm, Mount, VIPERX_300

    found = describe(folder)
    report("reading the clip manifest", note=f"{found['clips']} clips")

    video = EgoVideoConnector(str(folder), stride=STRIDE)

    if found["poses"] == "yours":
        report("reading the poses you supplied")
        estimator: Any = _supplied(folder, json.loads((folder / MANIFEST).read_text()))
    else:
        # Resolved before anything is decoded, so a worker missing its weights
        # says so in a second rather than after tracking every clip and calling
        # the result a fault in the footage.
        weights = Path(model_path) if model_path else model_file()
        report("finding the hands", note="this is the slow part")
        # The detector and the hand model are built once and shared; only the
        # intrinsics vary, and they vary with the frames rather than with a
        # number written down here.
        detector = rtmpose(device="cpu", mode="lightweight")
        shape = mediapipe(model_path=str(weights))
        estimator = _per_resolution(
            lambda intrinsics: rtm_with_mediapipe(detector, shape, intrinsics=intrinsics),
            rig,
        )

    hands = HandPoseConnector(video, estimator=estimator, cache=False)
    retarget = {
        h: HandToArm(
            mount=Mount.aligned(),
            # A stand-in, and it says so. Hand span scales the whole trajectory,
            # so a wrong one is a smoothly wrong dataset rather than a broken
            # one, and the only honest fix is measuring the person who filmed.
            hand=Hand(closed=0.02, open=0.10, span=0.19, measured_by="adult average (stand-in)"),
            reach=VIPERX_300,
            name=f"hands.{h}",
        )
        for h in ("left", "right")
    }
    actions = EgoActionConnector(
        hands, retargeter=retarget, hold_missing=True, keep_video=True, size=SIZE, cache=False
    )

    from gantry.errors import ConfigError

    episodes, dropped, ours = [], [], []
    ids = list(actions.episode_ids())
    for index, eid in enumerate(ids, 1):
        report("turning hands into actions", current=index, total=len(ids))
        try:
            episodes.append(actions.open(eid))
        except ConfigError as error:
            # Not a judgement on the footage. A missing tracker, an unreadable
            # model file or a rig we cannot describe is this machine being set
            # up wrong, and every clip will fail identically. Kept apart from
            # `dropped` so the caller cannot report it as a fault in the data,
            # which is the one mistake this product must never make.
            ours.append({"clip": eid.split("/")[-1], "why": str(error)[:200]})
        except Exception as error:  # noqa: BLE001 - per clip, and why is worth keeping
            dropped.append({"clip": eid.split("/")[-1], "why": str(error)[:200]})

    if not episodes:
        return {"episodes": 0, "dropped": dropped, "ours": ours, "poses": found["poses"]}

    report("writing it out", note=f"{len(episodes)} clips")
    LeRobotConnector.write(episodes, str(out), fps=FPS, videos=True, accept_loss=True)

    # Written beside the dataset rather than returned, so it travels with the
    # data instead of living in a variable intake forgets. Where the poses came
    # from is not a detail of this run: a number from a lab's own tracker and a
    # number from our estimate are different claims, and the second one is only
    # as good as monocular tracking on that footage. Anything reading this
    # dataset later can see which it was.
    (out / "meta" / SIDECAR).write_text(
        json.dumps(
            {
                "poses": found["poses"],
                "resolutions": getattr(estimator, "resolutions", []),
                "clips_in": found["clips"],
                "episodes_out": len(episodes),
                "dropped": len(dropped),
                "stride": STRIDE,
                "fps": FPS,
                "rig": rig,
            },
            indent=1,
        )
        + "\n"
    )
    return {
        "episodes": len(episodes),
        "dropped": dropped,
        "ours": ours,
        "poses": found["poses"],
        # The resolutions actually solved, read off the frames. A corpus with
        # more than one is worth knowing about: this treats a second resolution
        # as the same lens scaled, and a camera that crops instead has a
        # different field of view that nothing here can detect.
        "resolutions": getattr(estimator, "resolutions", []),
        "scenes": len({e.meta.extra.get("scene", "?") for e in episodes}),
    }
