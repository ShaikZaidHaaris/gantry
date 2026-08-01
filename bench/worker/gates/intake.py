"""Gate 0 — Intake. Can we read this at all?

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
from typing import Any

#: What a channel has to be for the rest of the gauntlet to have any chance.
#: Not a whitelist of names -- a dataset may call things what it likes -- but
#: the shapes the benchmark can execute.
NEEDS_ACTION = "action"
NEEDS_STATE = "observation.state"


def _finding(code: str, severity: str, summary: str, fix: str | None = None) -> dict:
    return {"code": code, "severity": severity, "summary": summary, "prescription": fix}


def unpack(archive: Path, into: Path) -> tuple[Path | None, list[dict]]:
    """Unzip, and find the LeRobot root inside whatever folder structure came."""
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
    if info is None:
        return None, [
            _finding(
                "intake.no_lerobot_meta",
                "strong",
                "no meta/info.json anywhere in the archive, so this is not a LeRobot v2 dataset",
                "Export with LeRobot v2 (a meta/ folder beside data/), or tell us the format "
                "you have and we will say whether a connector exists for it.",
            )
        ]
    return info.parents[1], []


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
    return out


def run(archive: Path, workdir: Path) -> dict:
    """The gate contract: a verdict, findings, and what we detected."""
    root, problems = unpack(archive, workdir / "unpacked")
    if root is None:
        return {"status": "refused", "detected": {}, "findings": problems, "summary": problems[0]["summary"]}

    detected = describe(root)
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
