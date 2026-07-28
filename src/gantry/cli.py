"""The command line: see what is installed, check it, plan it, run it.

Four verbs, and the two in the middle are the ones that make the framework
usable. ``check`` runs a plugin's conformance kit, so an adapter author never
has to guess whether they got it right. ``plan`` resolves without executing, so
a mistake costs a second instead of a GPU hour — and prints the refusal, which
is where most of the value of resolution lives.

Exit codes are meaningful: 0 for a pass, 1 for a refusal or a failure, 2 for
misuse. A CI job can gate on this without parsing text.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

from . import __version__
from .conformance import KITS
from .errors import GantryError
from .manifest import Manifest
from .resolve import Registry
from .runner import plan_manifest, run_manifest
from .spine import PLANES

EXIT_OK, EXIT_REFUSED, EXIT_MISUSE = 0, 1, 2


def _registry() -> Registry:
    registry = Registry()
    registry.discover()
    return registry


# --------------------------------------------------------------------------
# verbs
# --------------------------------------------------------------------------


def cmd_list(args: argparse.Namespace) -> int:
    registry = _registry()
    installed = registry.describe_all()
    if args.json:
        print(json.dumps(installed, indent=2, sort_keys=True))
        return EXIT_OK
    if not installed:
        print("nothing installed. Plugins register through packaging entry points;")
        print("installing one is all it takes to make it appear here.")
        return EXIT_OK
    for plane in PLANES:
        names = installed.get(plane, [])
        print(f"{plane:12} {', '.join(names) if names else '-'}")
    return EXIT_OK


def cmd_checks(args: argparse.Namespace) -> int:
    """What every conformance kit verifies. The specification, enumerated."""
    kits = {name: enumerate_checks() for name, enumerate_checks in KITS.items()}
    if args.plane and args.plane not in kits:
        print(f"no kit for {args.plane!r}; have {', '.join(sorted(kits))}", file=sys.stderr)
        return EXIT_MISUSE
    for name, checks in kits.items():
        if args.plane and name != args.plane:
            continue
        print(f"{name} ({len(checks)} checks)")
        for code, description in checks:
            print(f"  {code:24} {description}")
    return EXIT_OK


def cmd_plan(args: argparse.Namespace) -> int:
    try:
        manifest = Manifest.load(args.manifest)
    except GantryError as error:
        print(str(error), file=sys.stderr)
        return EXIT_MISUSE

    verdict = plan_manifest(manifest, _registry())
    print(verdict.explain())
    if not verdict.ok:
        return EXIT_REFUSED
    kind = "evaluate then analyse" if manifest.evaluates else "analyse as recorded"
    print(f"  would {kind}: {len(manifest.cohorts)} cohort(s), "
          f"{len(manifest.feedback)} feedback module(s)")
    return EXIT_OK


def cmd_run(args: argparse.Namespace) -> int:
    try:
        manifest = Manifest.load(args.manifest)
    except GantryError as error:
        print(str(error), file=sys.stderr)
        return EXIT_MISUSE

    try:
        outcome = run_manifest(manifest, _registry(), save_to=args.save)
    except GantryError as error:
        print(f"run halted: {error}", file=sys.stderr)
        return EXIT_REFUSED

    if args.out:
        target = Path(args.out)
        target.write_text(json.dumps(outcome.as_dict(), indent=2, default=str) + "\n")
        print(f"wrote {target}")
    if args.json and not args.out:
        print(json.dumps(outcome.as_dict(), indent=2, default=str))
    else:
        print(outcome.explain())
    return EXIT_OK if outcome.ok else EXIT_REFUSED


# --------------------------------------------------------------------------
# entry point
# --------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="gantry",
        description="Plane-independent tooling for robot data, policies, evaluation and feedback.",
    )
    parser.add_argument("--version", action="version", version=f"gantry {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    listed = subparsers.add_parser("list", help="what is installed, by plane")
    listed.add_argument("--json", action="store_true", help="machine-readable output")
    listed.set_defaults(func=cmd_list)

    checks = subparsers.add_parser("checks", help="what the conformance kits verify")
    checks.add_argument("plane", nargs="?", help="connector, policy, evaluator or feedback")
    checks.set_defaults(func=cmd_checks)

    plan = subparsers.add_parser("plan", help="resolve a manifest without running it")
    plan.add_argument("manifest")
    plan.set_defaults(func=cmd_plan)

    run = subparsers.add_parser("run", help="execute a manifest")
    run.add_argument("manifest")
    run.add_argument("-o", "--out", help="write the full outcome summary as JSON")
    run.add_argument(
        "--save",
        metavar="DIR",
        help="persist each run as a versioned, re-readable record in DIR",
    )
    run.add_argument("--json", action="store_true", help="print JSON instead of a summary")
    run.set_defaults(func=cmd_run)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":  # pragma: no cover - entry point
    raise SystemExit(main())
