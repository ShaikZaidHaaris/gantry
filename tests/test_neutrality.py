"""The invariants that keep core neutral, enforced by the build.

Two rules, both easy to break by accident and expensive to discover late:

1. Core imports nothing but numpy and the standard library.
2. Core mentions no specific dataset format, robot, simulator, policy, or
   vendor — not in code, not in a default value.

If a future change needs core to know about a particular anything, that is the
signal it belongs in a plugin instead.
"""

from __future__ import annotations

import ast
import pathlib
import re

CORE = pathlib.Path(__file__).resolve().parents[1] / "src" / "gantry"
ALLOWED_THIRD_PARTY = {"numpy"}

# Names that would mean core has taken a side. Checked case-insensitively
# against source text outside of this file.
FORBIDDEN_NAMES = [
    # dataset formats / hubs
    "lerobot",
    "robomimic",
    "rlds",
    "tfds",
    "rosbag",
    "huggingface",
    # simulators
    "robosuite",
    "mujoco",
    "libero",
    "isaac",
    "maniskill",
    "pybullet",
    "gymnasium",
    # policies / model families
    "gr00t",
    "groot",
    "octo",
    "openvla",
    "spatialvla",
    "diffusion_policy",
    "act_policy",
    # frameworks a plugin might need but core must not
    "torch",
    "tensorflow",
    "jax",
    "transformers",
    # embodiments
    "franka",
    "panda",
    "ur5",
    "widowx",
    "aloha",
    "yam",
    "unitree",
]


def _core_files() -> list[pathlib.Path]:
    return sorted(CORE.rglob("*.py"))


def test_core_files_exist() -> None:
    assert _core_files(), f"no core sources found under {CORE}"


def test_core_imports_only_numpy() -> None:
    offenders: dict[str, set[str]] = {}
    for path in _core_files():
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                roots = {alias.name.split(".")[0] for alias in node.names}
            elif isinstance(node, ast.ImportFrom):
                if node.level:  # relative import, always fine
                    continue
                roots = {(node.module or "").split(".")[0]}
            else:
                continue
            for root in roots:
                if not root or root in ALLOWED_THIRD_PARTY:
                    continue
                if _is_stdlib(root):
                    continue
                offenders.setdefault(str(path.relative_to(CORE)), set()).add(root)
    assert not offenders, (
        "core must stay numpy-only; move these behind a plugin: "
        f"{ {k: sorted(v) for k, v in offenders.items()} }"
    )


def _is_stdlib(module: str) -> bool:
    import sys

    return module in sys.stdlib_module_names


def test_core_names_no_specific_implementation() -> None:
    offenders: dict[str, list[str]] = {}
    # Word boundaries, not substrings: "yaml" contains a robot's name, and a
    # test that fires on innocent words is a test somebody eventually deletes.
    patterns = {
        name: re.compile(rf"(?<![a-z0-9]){re.escape(name)}(?![a-z0-9])") for name in FORBIDDEN_NAMES
    }
    for path in _core_files():
        text = path.read_text().lower()
        hits = [name for name, pattern in patterns.items() if pattern.search(text)]
        if hits:
            offenders[str(path.relative_to(CORE))] = hits
    assert not offenders, (
        f"core named a specific implementation, which breaks plane neutrality: {offenders}"
    )
