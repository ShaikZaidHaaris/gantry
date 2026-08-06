"""A LeRobot version the reader does not speak is a refusal, not our outage.

Found live: the first Hub fetch anyone ran was lerobot/pusht, which the Hub
had already converted to v3.0. Intake passed on the presence of meta/info.json
alone, the data report then died inside the connector, and the runner's
catch-all reported it as our machinery breaking -- "failed", retryable, with
nothing the visitor could act on. The dataset's only problem had a fix a
sentence long.
"""

from __future__ import annotations

import json
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gates import intake  # noqa: E402


def archive_with_version(tmp_path: Path, version) -> Path:
    zpath = tmp_path / "ds.zip"
    info = {"robot_type": "pusht", "fps": 10}
    if version is not None:
        info["codebase_version"] = version
    with zipfile.ZipFile(zpath, "w") as zf:
        zf.writestr("meta/info.json", json.dumps(info))
        zf.writestr("data/chunk-000/file.parquet", "x")
    return zpath


def test_a_v3_export_is_refused_with_the_fix_attached(tmp_path):
    root, findings = intake.unpack(archive_with_version(tmp_path, "v3.0"), tmp_path / "out")
    assert root is None
    assert findings and findings[0]["code"] == "intake.lerobot_version"
    assert "v3.0" in findings[0]["summary"]
    assert "v2" in (findings[0]["prescription"] or "")


def test_v2_exports_still_pass_through(tmp_path):
    for ok in ("v2.0", "v2.1"):
        root, findings = intake.unpack(archive_with_version(tmp_path, ok), tmp_path / f"out-{ok}")
        assert root is not None and findings == [], (ok, findings)


def test_an_undeclared_version_is_not_refused_on_suspicion(tmp_path):
    """Old exports may omit the field; absence is unknown, and unknown passes
    to the connector, which reads more than a version string."""
    root, findings = intake.unpack(archive_with_version(tmp_path, None), tmp_path / "out-none")
    assert root is not None and findings == []
