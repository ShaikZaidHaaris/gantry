"""Does this connector actually honour the connector contract?

One check per invariant sentence in :mod:`gantry.contracts.connector`. Returns
a :class:`~gantry.spine.verdict.Verdict` rather than raising, and depends on no
test framework, so the same kit runs from pytest, from the CLI, or from a
plugin's own CI on a machine that has never heard of Gantry's test suite.

This is the enforcement mechanism the whole plugin ecosystem rests on. With
five implementations per plane there are thousands of combinations and no way
to test them; testing every plugin against its contract instead means any
combination the resolver accepts is trustworthy by construction.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np

from ..contracts.connector import (
    CAP_MEDIA,
    CAP_OUTCOMES,
    CAP_STAGE_EVENTS,
    CONNECTOR_CONTRACT,
    Connector,
)
from ..spine import ContractVersion, EpisodeRecord, Verdict, get_semantics

MEDIA_KINDS = {"image", "depth_image", "point_cloud", "audio", "blob"}
NUMERIC_KINDS = {"scalar", "vector", "matrix", "tensor"}


@dataclass(frozen=True)
class Context:
    """What the checks share, so the episodes are opened once."""

    connector: Connector
    ids: tuple[str, ...]
    sample: tuple[EpisodeRecord, ...]
    strict: bool


@dataclass(frozen=True)
class Check:
    code: str
    description: str
    run: Callable[[Context], Verdict]


# --------------------------------------------------------------------------
# checks
# --------------------------------------------------------------------------


def _descriptor_plane(context: Context) -> Verdict:
    descriptor = context.connector.descriptor()
    if descriptor.plane != "dataset":
        return Verdict.no(
            "conformance.plane",
            f"{descriptor.ref} declares plane {descriptor.plane!r}, expected 'dataset'",
        )
    return Verdict.yes()


def _descriptor_contract(context: Context) -> Verdict:
    descriptor = context.connector.descriptor()
    return descriptor.contract_version.satisfies(ContractVersion.parse(CONNECTOR_CONTRACT))


def _ids_deterministic(context: Context) -> Verdict:
    again = context.connector.episode_ids()
    if tuple(again) != context.ids:
        return Verdict.no(
            "conformance.ids_unstable",
            "episode_ids() returned a different order or set on a second call",
            hint="enumeration must be stable, or nothing downstream is reproducible",
        )
    return Verdict.yes()


def _ids_unique(context: Context) -> Verdict:
    duplicates = {i for i in context.ids if context.ids.count(i) > 1}
    if duplicates:
        return Verdict.no(
            "conformance.ids_duplicate",
            f"repeated episode id(s): {sorted(duplicates)[:5]}",
        )
    return Verdict.yes()


def _ids_nonempty(context: Context) -> Verdict:
    if not context.ids:
        return Verdict.note(
            "conformance.empty",
            "the connector serves no episodes, so most checks were skipped",
        )
    return Verdict.yes()


def _schema_matches_open(context: Context) -> Verdict:
    checks = []
    for episode in context.sample:
        declared = tuple(context.connector.schema(episode.meta.id))
        if declared != episode.schema:
            checks.append(
                Verdict.no(
                    "conformance.schema_drift",
                    f"{episode.meta.uid}: schema() disagrees with open().schema",
                    hint="the resolver plans from schema() and would plan the wrong run",
                )
            )
    return Verdict.all(checks)


def _records_valid(context: Context) -> Verdict:
    return Verdict.all(episode.validate(deep=True) for episode in context.sample)


def _reads_are_selective(context: Context) -> Verdict:
    checks = []
    for episode in context.sample:
        if len(episode) < 3 or not episode.schema:
            continue
        wanted = episode.schema[0].name
        start, stop = 1, min(3, len(episode))
        window = episode.read([wanted], start=start, stop=stop)
        if set(window) != {wanted}:
            checks.append(
                Verdict.no(
                    "conformance.read_extra_channels",
                    f"{episode.meta.uid}: asked for {[wanted]}, got {sorted(window)}",
                    hint="selective reads are what make a large dataset openable",
                )
            )
            continue
        if len(window[wanted]) != stop - start:
            checks.append(
                Verdict.no(
                    "conformance.read_window",
                    f"{episode.meta.uid}: asked for steps {start}:{stop}, "
                    f"got {len(window[wanted])} rows",
                )
            )
    return Verdict.all(checks)


def _reads_are_deterministic(context: Context) -> Verdict:
    checks = []
    for episode in context.sample:
        reopened = context.connector.open(episode.meta.id)
        for spec in episode.schema:
            if spec.name not in episode.source.channels:
                continue
            first = episode.array(spec.name)
            second = reopened.array(spec.name)
            if not np.array_equal(first, second):
                checks.append(
                    Verdict.no(
                        "conformance.read_unstable",
                        f"{episode.meta.uid}: channel {spec.name!r} differed between two opens",
                    )
                )
                break
    return Verdict.all(checks)


def _unknown_id_raises(context: Context) -> Verdict:
    bogus = "__gantry_conformance_no_such_episode__"
    try:
        context.connector.open(bogus)
    except KeyError:
        return Verdict.yes()
    except Exception as error:  # noqa: BLE001 - we are classifying the failure
        return Verdict.no(
            "conformance.unknown_id_wrong_error",
            f"opening an unknown id raised {type(error).__name__}, expected KeyError",
        )
    return Verdict.no(
        "conformance.unknown_id_silent",
        "opening an unknown id returned a record instead of raising KeyError",
        hint="a typo in a manifest must fail loudly, not silently read something else",
    )


def _ids_are_namespaced(context: Context) -> Verdict:
    checks = []
    for episode in context.sample:
        if not episode.meta.source:
            checks.append(
                Verdict.no(
                    "conformance.source_missing",
                    f"episode {episode.meta.id!r} has an empty source",
                    hint="merging two datasets that both number from zero silently "
                    "collapses them without a namespace",
                )
            )
    return Verdict.all(checks)


def _capability(cap: str, present: Callable[[EpisodeRecord], bool]) -> Callable[[Context], Verdict]:
    def check(context: Context) -> Verdict:
        if not context.sample:
            return Verdict.yes()
        declared = bool(context.connector.descriptor().provides.get(cap))
        observed = any(present(episode) for episode in context.sample)
        if declared and not observed:
            return Verdict.note(
                "conformance.capability_unproven",
                f"declares {cap}=True but none of the {len(context.sample)} sampled "
                "episodes show it",
                hint="not necessarily wrong on a large dataset, but worth confirming",
                capability=cap,
            )
        if observed and not declared:
            return Verdict.no(
                "conformance.capability_understated",
                f"declares {cap}=False but a sampled episode has it",
                hint="the resolver will refuse runs this connector could actually serve",
                capability=cap,
            )
        return Verdict.yes()

    return check


def _units_declared(context: Context) -> Verdict:
    """Advisory unless strict: physical channels should say what they are in."""
    checks = []
    for episode in context.sample:
        for spec in episode.schema:
            semantics = get_semantics(spec.semantics) if spec.semantics else None
            if semantics is None or semantics.dimension is None or spec.units is not None:
                continue
            message = (
                f"{episode.meta.uid}: channel {spec.name!r} is {spec.semantics!r} "
                "but declares no units"
            )
            hint = "undeclared units cannot be checked against a consumer"
            checks.append(
                Verdict.no("conformance.units_undeclared", message, hint)
                if context.strict
                else Verdict.note("conformance.units_undeclared", message, hint)
            )
    return Verdict.all(checks)


def _channels_described(context: Context) -> Verdict:
    """Advisory unless strict: a numeric channel should say what it is.

    A channel with neither units nor semantics can only be matched by name and
    width. That is matching by coincidence, and it is precisely how data ends
    up feeding a consumer that expected something else. A reader cannot invent
    the missing description — but it can be honest that it is missing, which is
    what this reports.
    """
    checks = []
    for episode in context.sample:
        for spec in episode.schema:
            if spec.kind not in NUMERIC_KINDS or spec.units or spec.semantics:
                continue
            message = (
                f"{episode.meta.uid}: channel {spec.name!r} declares neither units "
                "nor semantics"
            )
            hint = "supply a schema sidecar, or it can only be matched by name and width"
            checks.append(
                Verdict.no("conformance.channel_undescribed", message, hint)
                if context.strict
                else Verdict.note("conformance.channel_undescribed", message, hint)
            )
    return Verdict.all(checks)


CHECKS: tuple[Check, ...] = (
    Check("plane", "declares the dataset plane", _descriptor_plane),
    Check("contract", "implements a compatible connector contract", _descriptor_contract),
    Check("ids_nonempty", "serves at least one episode", _ids_nonempty),
    Check("ids_deterministic", "enumerates the same ids in the same order", _ids_deterministic),
    Check("ids_unique", "episode ids are unique", _ids_unique),
    Check("schema_matches_open", "schema() agrees with open().schema", _schema_matches_open),
    Check("records_valid", "sampled records validate against their own schema", _records_valid),
    Check("reads_selective", "reads honour the requested channels and window", _reads_are_selective),
    Check("reads_deterministic", "two opens return identical data", _reads_are_deterministic),
    Check("unknown_id", "an unknown id raises KeyError", _unknown_id_raises),
    Check("namespaced", "episode ids are namespaced by a source", _ids_are_namespaced),
    Check(
        "capability_stage_events",
        "the stage_events claim matches reality",
        _capability(CAP_STAGE_EVENTS, lambda e: bool(e.labels.stage_events)),
    ),
    Check(
        "capability_outcomes",
        "the outcomes claim matches reality",
        _capability(CAP_OUTCOMES, lambda e: e.labels.success is not None),
    ),
    Check(
        "capability_media",
        "the media claim matches reality",
        _capability(CAP_MEDIA, lambda e: any(s.kind in MEDIA_KINDS for s in e.schema)),
    ),
    Check("units_declared", "physical channels declare their units", _units_declared),
    Check("channels_described", "numeric channels declare units or meaning", _channels_described),
)


def connector_checks() -> tuple[tuple[str, str], ...]:
    """Every check this kit runs, as ``(code, description)``."""
    return tuple((check.code, check.description) for check in CHECKS)


def check_connector(
    connector: Connector,
    *,
    sample: int = 3,
    strict: bool = False,
    only: Sequence[str] | None = None,
) -> Verdict:
    """Run the kit. ``strict`` promotes advisory notes to failures."""
    try:
        ids = tuple(connector.episode_ids())
        opened = connector.sample(sample) if ids else ()
    except Exception as error:  # noqa: BLE001 - failing to enumerate is a failure
        return Verdict.no(
            "conformance.check_raised",
            f"enumerating the connector raised {type(error).__name__}: {error}",
            hint="episode_ids() and open() must work before any other check can run",
        )
    context = Context(connector=connector, ids=ids, sample=opened, strict=strict)
    selected = [c for c in CHECKS if only is None or c.code in set(only)]
    results = []
    for check in selected:
        try:
            results.append(check.run(context))
        except Exception as error:  # noqa: BLE001 - a raising check is itself a failure
            results.append(
                Verdict.no(
                    "conformance.check_raised",
                    f"check {check.code!r} raised {type(error).__name__}: {error}",
                    check=check.code,
                )
            )
    return Verdict.all(results)
