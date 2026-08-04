"""Gate 0 -- Intake. Can we read this at all?

Seconds, free, and it refuses rather than guesses. Everything it reports is
something it read; anything it could not read is reported as unread, never as
zero. A user whose dataset is fine should see their own numbers within a few
seconds of the upload finishing -- that first minute is where trust is won.

The refusals carry the pipeline's own dotted codes, because the same vocabulary
has to work in the UI, in the API, and in a support conversation.
"""

from __future__ import annotations

import json
import zipfile
from pathlib import Path
from typing import Any, Callable, Mapping

from .. import ego

#: What a gate is handed to say where it is: phase, and optionally how far
#: through and a note. Typed here so a gate can be run outside a worker -- in a
#: test, or by hand -- without one.
Report = Callable[..., None]


def _quiet(*_args, **_kwargs) -> None:
    """Default report: say nothing. A gate must run without a worker."""

#: What a channel has to be for the rest of the gauntlet to have any chance.
#: Not a whitelist of names -- a dataset may call things what it likes -- but
#: the shapes the benchmark can execute.
NEEDS_ACTION = "action"
NEEDS_STATE = "observation.state"


def _finding(code: str, severity: str, summary: str, fix: str | None = None) -> dict:
    return {"code": code, "severity": severity, "summary": summary, "prescription": fix}


#: Codes that mean *we* broke, not that the data was judged and found wanting.
#: The product's central distinction: `failed` can be retried because the fault
#: is ours to fix, `refused` cannot, because re-rolling until the answer is
#: liked is the one thing a benchmark must not offer. A missing tracker fails
#: every clip identically and says nothing whatever about the footage.
OURS = {"ego.tracker_missing", "ego.conversion_failed"}


def unpack(archive: Path, into: Path, report: Report = _quiet) -> tuple[Path | None, list[dict]]:
    """Unzip, and find the dataset inside whatever folder structure came.

    Two shapes are accepted. A LeRobot v2 export is used as it is. Raw
    egocentric video, which is a clip manifest beside the footage, is converted
    here into the first shape and then treated as one, so nothing downstream
    learns a second format. See :mod:`worker.ego` for why that path exists at
    all: there is no `action` column in kitchen video, and manufacturing one is
    the whole job.
    """
    into.mkdir(parents=True, exist_ok=True)
    try:
        with zipfile.ZipFile(archive) as zf:
            # A zip that unpacks outside its target is a security problem, not
            # a formatting one; refuse the whole archive rather than sanitise.
            for member in zf.namelist():
                target = (into / member).resolve()
                if not str(target).startswith(str(into.resolve())):
                    return None, [
                        _finding(
                            "intake.unsafe_archive",
                            "strong",
                            "the archive contains a path that would write outside its own folder",
                        )
                    ]
            zf.extractall(into)
    except zipfile.BadZipFile:
        return None, [_finding("intake.not_a_zip", "strong", "this file is not a zip archive")]

    info = next(into.rglob("meta/info.json"), None)
    if info is not None:
        return info.parents[1], []

    # No export. Before refusing, look for the other thing a contributor
    # plausibly has: their own footage plus a manifest saying what is in it.
    folder = ego.find(into)
    if folder is not None:
        return _from_ego(folder, into, report)

    return None, [
        _finding(
            "intake.no_lerobot_meta",
            "strong",
            "no meta/info.json anywhere in the archive, so this is not a LeRobot v2 dataset",
            "Export with LeRobot v2 (a meta/ folder beside data/), or upload raw egocentric "
            f"video with a {ego.MANIFEST} beside it saying what each clip shows and where it "
            "was filmed.",
        )
    ]


def _from_ego(folder: Path, into: Path, report: Report) -> tuple[Path | None, list[dict]]:
    """Turn raw ego footage into a LeRobot dataset, or say why it cannot.

    The manifest is checked before a single frame is decoded, because the
    refusals it can earn are ones the contributor fixes in a text file, and
    making them wait through hand tracking to hear about a missing sentence
    would be unkind and pointless.
    """
    try:
        found = ego.describe(folder)
    except (OSError, ValueError) as error:
        return None, [
            _finding(
                "ego.manifest_unreadable",
                "strong",
                f"the clip manifest could not be read: {error}",
                f"{ego.MANIFEST} should be a JSON list of clips, each with a path, an "
                "instruction and a scene.",
            )
        ]

    blocking = [f for f in ego.check(found) if f["severity"] == "strong"]
    if blocking:
        return None, blocking

    report(
        "turning your footage into episodes",
        note=(
            "using the poses you supplied"
            if found["poses"] == "yours"
            else "finding the hands, which takes a few minutes"
        ),
    )
    try:
        result = ego.convert(folder, into / "ego_lerobot", report)
    except Exception as error:  # noqa: BLE001 - ours, and it must not read as their fault
        return None, [
            _finding(
                "ego.conversion_failed",
                "strong",
                f"we could not turn this footage into episodes: {str(error)[:200]}",
                "This is our machinery rather than your data. The run can be tried again.",
            )
        ]

    # Ours before theirs. A worker without the tracker installed fails every
    # clip identically, and reporting that as "we found no hands in your
    # footage" blames a contributor for our own machine. It is the same
    # distinction the rest of the product turns on: `failed` is us, `refused`
    # is a judgement on the data, and only the first is worth retrying.
    if result.get("ours"):
        return None, [
            _finding(
                "ego.tracker_missing",
                "strong",
                "this worker cannot estimate hand poses, so your footage was never judged",
                f"Our machinery, not your data: {result['ours'][0]['why']}",
            )
        ]

    if not result["episodes"]:
        why = result["dropped"][0]["why"] if result["dropped"] else "no reason recorded"
        return None, [
            _finding(
                "ego.no_hands_found",
                "strong",
                f"no clip produced a usable hand trajectory ({len(result['dropped'])} tried)",
                f"The first one failed with: {why}. Hand tracking needs the hands to be "
                "visible and in focus for a reasonable share of the clip.",
            )
        ]
    return (into / "ego_lerobot"), []


#: Videos actually opened, spread evenly across the upload. A file that lists
#: in the archive and will not decode is invisible to every other check here,
#: and the next gate finds out by crashing -- which the product then reports as
#: our failure rather than as a file the contributor can go and fix.
#:
#: Sampled rather than exhaustive: opening ten thousand files would make the
#: free gate slow enough to stop being free. The count is reported so a report
#: never implies more was checked than was.
SAMPLE_VIDEOS = 8


def unreadable_videos(root: Path, limit: int = SAMPLE_VIDEOS) -> tuple[list[str], int]:
    """Videos that will not decode, and how many were opened to find out."""
    files = sorted(root.rglob("*.mp4"))
    if not files:
        return [], 0
    step = max(1, len(files) // limit)
    chosen = files[::step][:limit]

    try:
        import av
    except ImportError:  # pragma: no cover - decoding is optional at intake
        return [], 0

    broken = []
    for path in chosen:
        try:
            with av.open(str(path)) as container:
                stream = next((s for s in container.streams if s.type == "video"), None)
                if stream is None:
                    broken.append(path.name)
                    continue
                if not next(container.decode(stream), None):
                    broken.append(path.name)
        except Exception:  # noqa: BLE001 - any failure to open is a failure to read
            broken.append(path.name)
    return broken, len(chosen)


def describe(root: Path) -> dict:
    """What the dataset says about itself."""
    info = json.loads((root / "meta" / "info.json").read_text())
    features = info.get("features") or {}
    channels = [
        {
            "name": name,
            "dtype": spec.get("dtype"),
            "shape": spec.get("shape"),
            "width": (spec.get("shape") or [None])[0] if spec.get("dtype") != "video" else None,
        }
        for name, spec in features.items()
    ]
    return {
        "episodes": info.get("total_episodes"),
        "frames": info.get("total_frames"),
        "fps": info.get("fps"),
        "robot_type": info.get("robot_type"),
        "channels": channels,
        "videos": len(list(root.rglob("*.mp4"))),
        **dict(zip(("unreadable_videos", "videos_opened"), unreadable_videos(root))),
        "has_stats": (root / "meta" / "episodes_stats.jsonl").exists(),
        "has_sidecar": (root / "meta" / "gantry.jsonl").exists(),
        "tasks": [
            json.loads(line).get("task")
            for line in (root / "meta" / "tasks.jsonl").read_text().splitlines()
            if line.strip()
        ][:20]
        if (root / "meta" / "tasks.jsonl").exists()
        else [],
    }


def check(detected: dict) -> list[dict]:
    """The refusals. Each one blocks the gauntlet; each one says what to do."""
    out: list[dict] = []
    names = {c["name"] for c in detected["channels"]}

    if not detected.get("episodes"):
        out.append(
            _finding(
                "intake.no_episodes",
                "strong",
                "the dataset declares no episodes",
                "Check the export completed; meta/info.json says total_episodes is 0.",
            )
        )
    if NEEDS_ACTION not in names:
        out.append(
            _finding(
                "intake.no_action_channel",
                "strong",
                f"no {NEEDS_ACTION!r} channel, so there is nothing for a policy to imitate",
                "Every frame needs the command that was issued at it. If your actions are "
                "under a different name, rename them to 'action' on export.",
            )
        )
    if NEEDS_STATE not in names:
        out.append(
            _finding(
                "intake.no_state_channel",
                "strong",
                f"no {NEEDS_STATE!r} channel, so there is no record of where the arms were",
                "A policy is trained on what it saw and where it was. Export proprioception "
                "as 'observation.state'.",
            )
        )
    if not any(c["dtype"] == "video" or "image" in c["name"] for c in detected["channels"]):
        out.append(
            _finding(
                "intake.no_camera",
                "strong",
                "no camera channel, so a vision-language policy would be reading nothing",
                "Include at least one camera stream, exported as video.",
            )
        )

    action = next((c for c in detected["channels"] if c["name"] == NEEDS_ACTION), None)
    state = next((c for c in detected["channels"] if c["name"] == NEEDS_STATE), None)
    if action and state and action["width"] and state["width"] and action["width"] != state["width"]:
        out.append(
            _finding(
                "intake.width_mismatch",
                "moderate",
                f"action is {action['width']} wide and state is {state['width']}; they usually "
                "describe the same arms and normally match",
                "If that is deliberate say so at the next step, otherwise check the export.",
            )
        )
    if detected.get("videos") == 0 and any(c["dtype"] == "video" for c in detected["channels"]):
        out.append(
            _finding(
                "intake.videos_missing",
                "strong",
                "the schema declares video channels but the archive contains no .mp4 files",
                "Include the videos/ folder; the parquet files alone have no pixels in them.",
            )
        )
    broken = detected.get("unreadable_videos") or []
    if broken:
        opened = detected.get("videos_opened", 0)
        out.append(
            _finding(
                "intake.video_unreadable",
                "strong",
                f"{len(broken)} of the {opened} video(s) opened will not decode: "
                + ", ".join(broken[:3])
                + (f" and {len(broken) - 3} more" if len(broken) > 3 else ""),
                "Re-export those clips. A file that lists in the archive but has no "
                "decodable frames in it is worse than a missing one, because everything "
                "downstream counts it as present.",
            )
        )
    return out


def run(
    archive: Path,
    workdir: Path,
    report: Report = _quiet,
    params: Mapping[str, Any] | None = None,
) -> dict:
    """The gate contract: a verdict, findings, and what we detected.

    ``report`` says where we are. Called rather than returned because the whole
    point is to be heard while the gate is still running.
    """
    report("unpacking", note=archive.name)
    root, problems = unpack(archive, workdir / "unpacked", report)
    if root is None:
        ours = any(p["code"] in OURS for p in problems)
        return {
            "status": "failed" if ours else "refused",
            "detected": {},
            "findings": problems,
            "summary": problems[0]["summary"],
        }

    report("reading the manifest")
    report("opening a sample of the videos")
    detected = describe(root)
    report("checking", note=f"{detected.get('episodes', 0)} episodes")
    findings = check(detected)
    blocking = [f for f in findings if f["severity"] == "strong"]
    status = "refused" if blocking else "passed"
    if blocking:
        summary = blocking[0]["summary"]
    else:
        summary = (
            f"{detected['episodes']} episodes · {detected['frames']} frames · "
            f"{detected['fps']} fps · {len(detected['channels'])} channels"
        )
    return {
        "status": status,
        "detected": detected,
        "findings": findings,
        "summary": summary,
        "root": str(root),
    }
