"""Files that sit beside the SPA, and the 200 that is worse than a 404.

Vite emits hashed bundles into ``dist/assets`` and copies everything in
``public/`` to the *root* of ``dist``. The API mounted ``/assets`` and nothing
else, so a request for ``/hero-rig.jpg`` matched no route, fell through to the
catch-all, and was answered with ``index.html`` under HTTP 200 and
``text/html``.

That is a bad failure to be handed. A 404 is visible in a network panel and in
a log; a 200 carrying the wrong content type is a browser quietly drawing
nothing where a photograph should be. It cannot be reproduced in development
either, because there Vite serves ``public/`` itself and every image loads.

These tests build a miniature ``dist`` and ask the same questions of it.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest

_TMP = Path(tempfile.mkdtemp(prefix="bench-static-test-"))
os.environ["BENCH_DATA"] = str(_TMP)

from app import main as mainmod  # noqa: E402
from fastapi.responses import FileResponse  # noqa: E402


@pytest.fixture()
def bundle(tmp_path, monkeypatch):
    """A dist directory shaped the way Vite leaves one."""
    dist = tmp_path / "dist"
    (dist / "assets").mkdir(parents=True)
    (dist / "index.html").write_text("<!doctype html><title>app</title>", encoding="utf-8")
    (dist / "assets" / "index-abc123.js").write_text("// bundle", encoding="utf-8")
    # public/ contents, which Vite copies to the root rather than into assets/.
    (dist / "hero-rig.jpg").write_bytes(b"\xff\xd8\xff\xe0JPEGDATA")
    (dist / "favicon.ico").write_bytes(b"icon")
    monkeypatch.setattr(mainmod, "WEB_DIST", dist)
    return dist


def served(path: str):
    """What the SPA handler would return for this path."""
    return mainmod.spa(path)


def test_a_public_file_is_served_as_itself(bundle):
    """The defect, stated as what the browser actually got.

    Before the fix this returned index.html with a 200, so an <img> pointed at
    it rendered nothing and reported no error.
    """
    response = served("hero-rig.jpg")
    assert isinstance(response, FileResponse)
    assert Path(response.path).name == "hero-rig.jpg", (
        "the image request was answered with something else, most likely the "
        "SPA shell, which a browser cannot draw"
    )
    assert Path(response.path).read_bytes().startswith(b"\xff\xd8\xff"), "not JPEG bytes"


def test_other_root_files_work_too(bundle):
    """Not a special case for one image: anything Vite copied from public/."""
    assert Path(served("favicon.ico").path).name == "favicon.ico"


def test_a_client_route_still_falls_back_to_the_app(bundle):
    """The fallback is the reason this handler exists, and must survive.

    A hard reload on /submissions/sub_123 names no file, and has to be answered
    with the app so the router can take it from there.
    """
    for route in ("submissions/sub_123", "compare", ""):
        assert Path(served(route).path).name == "index.html", route


def test_a_missing_file_is_the_app_rather_than_an_error(bundle):
    """An unknown path is a client route we have not seen, not a 404."""
    assert Path(served("nope.jpg").path).name == "index.html"


def test_a_path_cannot_climb_out_of_the_bundle(bundle, tmp_path):
    """The check that makes serving arbitrary paths safe.

    This handler turns a URL into a filesystem path, which is the shape of every
    traversal bug. The database and the env file live above the bundle, so a
    path that escapes it is the one thing that must not be possible.
    """
    secret = tmp_path / "env"
    secret.write_text("BENCH_WORKER_TOKEN=hunter2", encoding="utf-8")

    for attempt in ("../env", "../../env", "..%2f..%2fenv", "./../env"):
        response = served(attempt)
        assert Path(response.path).name == "index.html", (
            f"{attempt!r} escaped the bundle and served {response.path}"
        )
