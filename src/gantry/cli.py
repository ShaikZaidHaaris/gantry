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
from typing import Any, Sequence

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


def cmd_curate(args: argparse.Namespace) -> int:
    """Propose a curation from installed curators, and say what it would cost.

    Reads the ledger for this signal's track record, so a proposal arrives
    alongside how often that signal has been right before rather than as a bare
    recommendation.
    """
    from .ledger import Ledger
    from .resolve import Registry

    registry = Registry.from_entry_points()
    names = registry.names("curation")
    if args.signal not in names:
        print(f"no curator named {args.signal!r}; installed: {list(names) or 'none'}")
        return 1
    print(f"curator {args.signal!r} is installed and ready")
    ledger = Ledger(args.ledger)
    prior = ledger.prior_for(args.signal)
    if prior is None:
        print(f"  no track record yet: {args.signal!r} has never been verified here")
    else:
        print(f"  track record: held {prior:.0%} of {ledger.tested(args.signal)} test(s)")
    print("  build the dataset and call .plan(episodes, runs) to propose")
    return 0


def cmd_ledger(args: argparse.Namespace) -> int:
    """What every curation signal has actually achieved here."""
    from .ledger import Ledger

    ledger = Ledger(args.dir)
    if args.json:
        print(
            json.dumps(
                {
                    "tested": len(ledger),
                    "by_signal": ledger.track_record(),
                    "by_task": ledger.by_task(),
                },
                indent=2,
            )
        )
    else:
        print(ledger.report())
    return 0


def _writer_for(registry: Any, name: str) -> Any:
    """The class that writes this format.

    Not an instance: a writer is asked to produce a dataset that does not exist
    yet, so there is nothing for it to have opened. The registry's factory for a
    connector is the class itself, and ``write`` is a classmethod on the
    contract for exactly this reason.
    """
    return registry.get("dataset", name).factory


def cmd_curate_apply(args: argparse.Namespace) -> int:
    """Apply a plan to a real dataset and write the curated one out.

    The plan comes from a file so that what was applied is the same artifact
    that was proposed and, later, the same one the ledger names. Re-deriving it
    here from a signal would mean the thing verified and the thing applied were
    two different objects that merely agreed at the time.
    """
    from .contracts.curation import CurationPlan
    from .curate import apply, mix_config
    from .errors import GantryError
    from .plan_io import read_plan
    from .resolve import Registry

    plan: CurationPlan = read_plan(args.plan)
    registry = Registry()
    registry.discover()
    try:
        connector = registry.get("dataset", args.reader).build({"path": args.dataset})
        episodes = list(connector)
    except Exception as error:  # noqa: BLE001 - reported, not raised
        print(f"cannot read {args.dataset!r} with {args.reader!r}: {error}")
        return 1
    survivors, applied = apply(plan, episodes, strict=not args.force)
    print(f"plan   : {plan.summary()}")
    print(f"applied: {applied.summary(source_size=len(episodes))}")
    for reason in applied.validate(source_size=len(episodes)).reasons:
        print(f"  [{reason.code}] {reason.message}")

    if applied.weights:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        mix = Path(str(args.out) + ".mix.json")
        mix.write_text(json.dumps(mix_config(plan), indent=2) + "\n")
        print(f"weights written to {mix} (configuration, not a change to the files)")

    if not args.write:
        print("nothing written: pass --write to produce the curated dataset")
        return 0

    # Reading and writing need not be the same format. The signal has to run
    # where the evidence is — success labels usually survive only in the
    # collection's native format — while the trainer reads whatever it reads.
    # Curating in one and emitting in the other is the ordinary case, not an
    # exotic one, so it is two arguments rather than an assumption.
    writer_name = args.writer or args.reader
    try:
        report = _writer_for(registry, writer_name).write(
            survivors, args.out, accept_loss=args.accept_loss
        )
    except GantryError as error:
        print(f"{writer_name!r}: {error}")
        return 1
    print(f"wrote {len(survivors)} episode(s) to {args.out}")
    # What the plan did travels with the data, so a checkpoint trained on this
    # is traceable to the curation that produced it rather than to a memory.
    Path(args.out).mkdir(parents=True, exist_ok=True)
    (Path(args.out) / "curation.json").write_text(
        json.dumps(
            {
                "signal": plan.signal,
                "rung": plan.rung,
                "predicted": str(plan.predicted),
                "summary": plan.summary(),
                "kept": len(applied.kept),
                "dropped": len(applied.dropped),
                "source": str(args.dataset),
                "evidence_seeds": list(plan.evidence_seeds),
            },
            indent=2,
        )
        + "\n"
    )
    print(f"provenance written to {Path(args.out) / 'curation.json'}")
    return 0 if getattr(report, "ok", True) else 1


def cmd_relink(args: argparse.Namespace) -> int:
    """Reconnect a converted dataset to the collection it came from, by content.

    For datasets written before converters recorded where episodes came from.
    Prints the mapping, and with --rewrite-plan translates a plan out of the
    source's vocabulary and into the target's, which is what makes a plan built
    from labels applicable where the labels no longer exist.
    """
    from .lineage import relink, rename
    from .plan_io import read_plan, write_plan
    from .resolve import Registry

    registry = Registry()
    registry.discover()
    try:
        src = list(registry.get("dataset", args.source_reader).build({"path": args.source}))
        dst = list(registry.get("dataset", args.target_reader).build({"path": args.target}))
    except Exception as error:  # noqa: BLE001 - reported, not raised
        print(f"cannot read: {error}")
        return 1

    found = relink(src, dst)
    print(f"source {len(src)} episode(s), target {len(dst)}")
    print(f"linked {len(found.links)}")
    for reason in found.validate(target_size=len(dst)).reasons:
        print(f"  [{reason.code}] {reason.message}")
    for converted, origin in list(found.links.items())[:3]:
        print(f"  {converted}  <-  {origin}")
    if len(found.links) > 3:
        print(f"  ... and {len(found.links) - 3} more")

    if args.rewrite_plan:
        plan = read_plan(args.rewrite_plan)
        translated = rename(plan.drops, found.links)
        from dataclasses import replace

        from .contracts.curation import CurationAction

        rewritten = replace(
            plan,
            actions=(CurationAction("drop", episodes=translated),),
            metadata={**dict(plan.metadata), "relinked_from": str(args.source)},
        )
        out = args.rewrite_plan_out or (str(args.rewrite_plan) + ".relinked.json")
        write_plan(rewritten, out)
        print(f"plan translated: {len(plan.drops)} name(s) -> {len(translated)} in the "
              f"target's vocabulary, written to {out}")
    return 0


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

    curate = subparsers.add_parser(
        "curate", help="propose what to do to the data, with the signal's track record"
    )
    curate.add_argument("signal", help="an installed curator, e.g. labels")
    curate.add_argument("--ledger", default="ledger", help="where outcomes are kept")
    curate.set_defaults(func=cmd_curate)

    applied = subparsers.add_parser(
        "curate-apply", help="apply a plan to a dataset and write the curated one"
    )
    applied.add_argument("plan", help="a plan written by a curator, as JSON")
    applied.add_argument("dataset", help="the source dataset")
    applied.add_argument("-o", "--out", required=True, help="where to write it")
    applied.add_argument("--reader", required=True, help="which connector reads the source")
    applied.add_argument(
        "--writer",
        help="which connector writes the result; defaults to the reader. Curating "
        "where the labels are and emitting where the trainer reads is ordinary.",
    )
    applied.add_argument("--write", action="store_true", help="actually write; otherwise dry run")
    applied.add_argument("--force", action="store_true", help="apply despite refusals")
    applied.add_argument("--accept-loss", action="store_true", help="write even if lossy")
    applied.set_defaults(func=cmd_curate_apply)

    linked = subparsers.add_parser(
        "relink",
        help="reconnect a converted dataset to its source by content, for datasets "
        "written before converters recorded it",
    )
    linked.add_argument("source", help="the collection the evidence lives in")
    linked.add_argument("target", help="the converted copy a trainer reads")
    linked.add_argument("--source-reader", required=True)
    linked.add_argument("--target-reader", required=True)
    linked.add_argument("--rewrite-plan", help="translate this plan into the target's names")
    linked.add_argument("--rewrite-plan-out")
    linked.set_defaults(func=cmd_relink)

    ledger = subparsers.add_parser(
        "ledger", help="which curation signals have actually worked, and where"
    )
    ledger.add_argument("dir", nargs="?", default="ledger")
    ledger.add_argument("--json", action="store_true")
    ledger.set_defaults(func=cmd_ledger)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":  # pragma: no cover - entry point
    raise SystemExit(main())
