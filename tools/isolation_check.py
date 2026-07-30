#!/usr/bin/env python3
"""Install one plugin, alone, and run its tests.

The modularity claim is that a plugin needs core plus whatever it declares, and
nothing else. That is easy to believe in a monorepo, where one shared virtual
environment quietly satisfies every import whether it was declared or not — a
plugin can depend on a sibling it never mentions and nobody finds out until
somebody installs it on its own.

So this builds a fresh interpreter per plugin and installs *only* core, that
plugin, and the siblings it actually declares. An undeclared import fails here,
loudly, which is the whole point.

Sibling dependencies resolve from the local tree because none of these are
published. That is a packaging detail rather than a loosening: an undeclared
sibling is still absent from the install set, so it still fails.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
import venv
from pathlib import Path

import tomllib

ROOT = Path(__file__).resolve().parents[1]
PLUGINS = ROOT / "plugins"


def distributions() -> dict[str, Path]:
    """Distribution name -> plugin directory, for everything in this tree."""
    found = {}
    for plugin in sorted(PLUGINS.iterdir()):
        pyproject = plugin / "pyproject.toml"
        if pyproject.exists():
            found[tomllib.loads(pyproject.read_text())["project"]["name"]] = plugin
    return found


def declared(plugin: Path, *, include_dev: bool) -> list[str]:
    meta = tomllib.loads((plugin / "pyproject.toml").read_text())["project"]
    names = list(meta.get("dependencies", []))
    if include_dev:
        names += meta.get("optional-dependencies", {}).get("dev", [])
    return [re.split(r"[><=\[;\s]", name)[0].strip() for name in names]


def closure(plugin: Path, siblings: dict[str, Path]) -> list[Path]:
    """Every local sibling this plugin needs, transitively.

    Direct dependencies are not enough. A plugin that declares one sibling which
    itself declares another gets the second one fetched from PyPI, where it does
    not exist, and the check reports an install failure that looks like an
    undeclared import. That is a false accusation — the declarations were
    correct and the tool was not walking them far enough.
    """
    seen: dict[str, Path] = {}
    frontier = [plugin]
    while frontier:
        current = frontier.pop()
        for name in declared(current, include_dev=current == plugin):
            path = siblings.get(name)
            if path is None or path == plugin or name in seen:
                continue
            seen[name] = path
            frontier.append(path)
    return list(seen.values())


def check(plugin: Path, siblings: dict[str, Path]) -> tuple[bool, str]:
    target = Path("/tmp") / f"gantry-iso-{plugin.name}"
    shutil.rmtree(target, ignore_errors=True)
    venv.create(target, with_pip=True)
    pip = target / "bin" / "pip"

    install = ["-e", str(ROOT)]
    for path in closure(plugin, siblings):
        install += ["-e", str(path)]
    install += ["-e", f"{plugin}[dev]"]

    result = subprocess.run(
        [str(pip), "install", "-q", *install], capture_output=True, text=True
    )
    if result.returncode:
        shutil.rmtree(target, ignore_errors=True)
        last = (result.stderr.strip().splitlines() or ["unknown"])[-1]
        return False, f"install failed: {last[:88]}"

    tests = subprocess.run(
        # No -q here: the project config already passes one, and a second
        # suppresses the very line that says how many tests ran.
        [str(target / "bin" / "python"), "-m", "pytest", str(plugin / "tests")],
        capture_output=True,
        text=True,
        cwd=ROOT,
    )
    shutil.rmtree(target, ignore_errors=True)
    return tests.returncode == 0, _count(tests.stdout)


def _count(output: str) -> str:
    """The line that actually says how many tests ran.

    Not simply the last line: for a suite of any size pytest's progress dots
    wrap, so the final line is a partial row and reporting it would show a
    number that is not the test count.
    """
    for line in reversed(output.strip().splitlines()):
        if re.search(r"\d+ (passed|failed|error)", line):
            return line.strip()
    return "no test summary found"


def main(argv: list[str]) -> int:
    siblings = distributions()
    wanted = argv or sorted(
        p.name for p in PLUGINS.iterdir() if (p / "pyproject.toml").exists()
    )
    failures = 0
    for name in wanted:
        ok, summary = check(PLUGINS / name, siblings)
        print(f"{name:26} {'ok  ' if ok else 'FAIL'} {summary}")
        failures += not ok
    print(f"\n{len(wanted) - failures}/{len(wanted)} plugins install and pass on their own")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
