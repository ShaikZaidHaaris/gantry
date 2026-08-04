"""Raw ego footage reaches intake, and the refusals it earns are the right ones.

These build archives on disk rather than mocking, because the bug this path is
most likely to have is about what a zip actually contains: a manifest one folder
deeper than expected, a clip naming a file nobody uploaded, a sentence somebody
left blank. None of those are visible to a test that hands the gate an object.

Nothing here decodes a frame or estimates a pose. The expensive half is covered
by running it against real footage; what is pinned here is the half a
contributor can fix in a text editor, which is the half that has to answer in
seconds.
"""

from __future__ import annotations

import json
import sys
import zipfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from worker import ego  # noqa: E402
from worker.gates import intake  # noqa: E402


def archive(tmp_path: Path, clips, *, files=("a.mp4", "b.mp4"), poses=None, folder="ego") -> Path:
    src = tmp_path / "src" / folder
    src.mkdir(parents=True)
    (src / ego.MANIFEST).write_text(json.dumps(clips))
    for name in files:
        (src / name).write_bytes(b"\x00\x00\x00\x18ftypmp42")
    if poses:
        (src / ego.POSES).mkdir()
        for name in poses:
            (src / ego.POSES / name).write_bytes(b"npz")
    out = tmp_path / "upload.zip"
    with zipfile.ZipFile(out, "w") as zf:
        for path in sorted(src.rglob("*")):
            zf.write(path, path.relative_to(tmp_path / "src").as_posix())
    return out


GOOD = [
    {"path": "a.mp4", "instruction": "wash the celery", "scene": "kitchen-P01"},
    {"path": "b.mp4", "instruction": "open the fridge", "scene": "kitchen-P02"},
]


def codes(problems) -> set[str]:
    return {p["code"] for p in problems}


def test_a_manifest_is_recognised_rather_than_refused(tmp_path):
    """The whole point: this archive used to be turned away at the door.

    It still cannot become episodes without decoding, which needs a tracker the
    test environment has no reason to carry, so what is asserted is that intake
    stopped saying "not a LeRobot dataset" and started trying.
    """
    found = intake.unpack(archive(tmp_path, GOOD), tmp_path / "work")
    assert "intake.no_lerobot_meta" not in codes(found[1])


def test_a_blank_instruction_is_refused_before_anything_is_decoded(tmp_path):
    """And refused for the reason it is refused, not a generic parse error.

    A clip labelled with a sentence somebody invented teaches an invented thing,
    so the manifest is the one place the product will not fill a gap for you.
    """
    clips = [dict(GOOD[0]), {"path": "b.mp4", "instruction": "  ", "scene": "kitchen-P02"}]
    root, problems = intake.unpack(archive(tmp_path, clips), tmp_path / "work")
    assert root is None
    assert "ego.manifest_incomplete" in codes(problems)
    assert "instruction" in problems[0]["summary"]


def test_a_missing_scene_is_refused(tmp_path):
    clips = [dict(GOOD[0]), {"path": "b.mp4", "instruction": "stir the pan"}]
    root, problems = intake.unpack(archive(tmp_path, clips), tmp_path / "work")
    assert root is None
    assert "ego.manifest_incomplete" in codes(problems)


def test_a_clip_naming_a_file_nobody_uploaded_is_named(tmp_path):
    clips = GOOD + [{"path": "missing.mp4", "instruction": "chop", "scene": "kitchen-P03"}]
    root, problems = intake.unpack(archive(tmp_path, clips), tmp_path / "work")
    assert root is None
    assert "ego.file_missing" in codes(problems)
    assert "missing.mp4" in problems[0]["summary"]


def test_one_kitchen_is_a_warning_and_not_a_refusal(tmp_path):
    """Measurable, but not generalisable, and the difference is the wording.

    Held-out clips are grouped by scene, so a corpus filmed in one room has one
    independent unit however many clips it has. That is worth saying and is not
    grounds for turning the upload away.
    """
    clips = [
        {"path": "a.mp4", "instruction": "wash up", "scene": "kitchen-P01"},
        {"path": "b.mp4", "instruction": "dry up", "scene": "kitchen-P01"},
    ]
    intake.unpack(archive(tmp_path, clips), tmp_path / "work")
    found = ego.describe(ego.find(tmp_path / "work"))
    assert found["scenes"] == ["kitchen-P01"]

    problems = ego.check(found)
    assert "ego.one_scene" in codes(problems)
    # Nothing here is strong, so `unpack` carries on to the conversion rather
    # than turning the upload away.
    assert all(p["severity"] != "strong" for p in problems)


def test_supplied_poses_are_detected_from_what_was_uploaded(tmp_path):
    """Ours or theirs is answered by the archive, not by a form field."""
    plain = archive(tmp_path / "one", GOOD)
    intake.unpack(plain, tmp_path / "one" / "work")
    folder = ego.find(tmp_path / "one" / "work")
    assert ego.describe(folder)["poses"] == "ours"

    withpose = archive(tmp_path / "two", GOOD, poses=("a.npz", "b.npz"))
    intake.unpack(withpose, tmp_path / "two" / "work")
    folder = ego.find(tmp_path / "two" / "work")
    assert ego.describe(folder)["poses"] == "yours"


def test_a_lerobot_export_still_takes_the_old_path(tmp_path):
    """The seam must not change what already worked."""
    src = tmp_path / "src" / "ds" / "meta"
    src.mkdir(parents=True)
    (src / "info.json").write_text(json.dumps({"total_episodes": 1}))
    out = tmp_path / "u.zip"
    with zipfile.ZipFile(out, "w") as zf:
        zf.write(src / "info.json", "ds/meta/info.json")
    root, problems = intake.unpack(out, tmp_path / "work")
    assert problems == []
    assert root is not None and (root / "meta" / "info.json").exists()


def test_neither_shape_still_refuses_and_now_says_both_ways_in(tmp_path):
    out = tmp_path / "u.zip"
    with zipfile.ZipFile(out, "w") as zf:
        zf.writestr("notes.txt", "hello")
    root, problems = intake.unpack(out, tmp_path / "work")
    assert root is None
    assert "intake.no_lerobot_meta" in codes(problems)
    assert ego.MANIFEST in problems[0]["prescription"]


@pytest.mark.parametrize("payload", ['{"clips": 3}', "not json at all", "[1, 2, 3]"])
def test_a_broken_manifest_says_so_rather_than_crashing(tmp_path, payload):
    src = tmp_path / "src" / "ego"
    src.mkdir(parents=True)
    (src / ego.MANIFEST).write_text(payload)
    out = tmp_path / "u.zip"
    with zipfile.ZipFile(out, "w") as zf:
        zf.write(src / ego.MANIFEST, f"ego/{ego.MANIFEST}")
    root, problems = intake.unpack(out, tmp_path / "work")
    assert root is None
    assert codes(problems) & {"ego.manifest_unreadable", "ego.no_clips", "ego.manifest_incomplete"}
