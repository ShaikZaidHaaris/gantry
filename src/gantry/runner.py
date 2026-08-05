"""Execute a manifest: resolve it, run what fits, refuse what does not.

Two shapes, decided by whether the manifest names a policy:

**Analyse.** Cohorts come straight from connectors, and the feedback modules
read the data as recorded. This is the "should I train on this" question, and it
needs no GPU.

**Evaluate.** Each cohort is handed to the evaluator with the policy, and the
feedback modules read the resulting runs instead. Same modules, because a
rollout is an episode.

Resolution is not advisory
--------------------------
Every module is resolved against every cohort before it is run: capabilities are
checked, channels are bound, and transforms are *applied* to what the module
reads. A module that does not fit is refused by name with the reason, and the
modules that do fit still run -- a plan that reported a conversion and then did
not perform it would be worse than having none, because the run would look
converted.

An embodiment, when named, is the reference the policy is checked against. It is
data rather than code, so it executes nothing; what it does is make "this policy
cannot drive this machine" a refusal before the first step instead of a
surprise during one.

Isolation is honoured. A component whose descriptor says it needs its own
environment gets one, or the run is refused -- never quietly imported into this
interpreter, which is the thing the declaration was asking to avoid.
"""

from __future__ import annotations

import platform
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Mapping, Sequence

from .contracts.feedback import Cohort, FeedbackModule, Report
from .errors import ConfigError, GantryError, must_halt
from .isolate import isolated_or_refuse, wants_isolation
from .manifest import ComponentSpec, Manifest
from .resolve import (
    AdapterRegistry,
    Registry,
    RetargeterRegistry,
    Wiring,
    adapt_all,
    bind,
    check_capabilities,
    requires_channels,
)
from .spine import ChannelSpec, EpisodeRecord, Provenance, Reason, RunRecord, Verdict
from .store import write_run


@dataclass(frozen=True)
class Outcome:
    """Everything one manifest produced, including what it declined to do."""

    manifest: Manifest
    reports: tuple[Report, ...] = ()
    runs: tuple[RunRecord, ...] = ()
    provenance: Provenance | None = None
    failures: tuple[str, ...] = ()
    refusals: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()
    saved: tuple[str, ...] = ()

    @property
    def ok(self) -> bool:
        return not self.failures and not self.refusals

    def as_dict(self) -> dict[str, Any]:
        return {
            "manifest": self.manifest.as_dict(),
            "provenance": self.provenance.as_dict() if self.provenance else None,
            "runs": [
                {
                    "digest": run.digest,
                    "episodes": len(run),
                    "metrics": {k: v.as_dict() for k, v in run.metrics.items()},
                }
                for run in self.runs
            ],
            "reports": [report.as_dict() for report in self.reports],
            "failures": list(self.failures),
            "refusals": list(self.refusals),
            "notes": list(self.notes),
            "saved": list(self.saved),
        }

    def explain(self) -> str:
        lines = [f"{self.manifest.name}: {len(self.reports)} report(s)"]
        for run in self.runs:
            metrics = ", ".join(f"{k}={v}" for k, v in run.metrics.items())
            lines.append(f"  run {run.digest}  {len(run)} episode(s)  {metrics}")
        for report in self.reports:
            lines.append("  " + report.explain().replace("\n", "\n  "))
        for refusal in self.refusals:
            lines.append(f"  REFUSED: {refusal}")
        for failure in self.failures:
            lines.append(f"  FAILED: {failure}")
        for saved in self.saved:
            lines.append(f"  saved {saved}")
        for note in self.notes:
            lines.append(f"  note: {note}")
        return "\n".join(lines)


# --------------------------------------------------------------------------
# building
# --------------------------------------------------------------------------


@dataclass
class Built:
    """Everything the manifest asked for, constructed."""

    cohorts: dict[str, Any] = field(default_factory=dict)
    feedback: list[FeedbackModule] = field(default_factory=list)
    policy: Any = None
    evaluator: Any = None
    embodiment: Any = None
    #: The single dataset, when the dataset plane is not the one varying.
    dataset: Any = None
    refs: list[Any] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    #: What building had to say -- chiefly which optional chain stages were
    #: skipped. A skip that never reaches the caller is a different pipeline
    #: running under the manifest's name without saying so.
    verdict: Verdict = field(default_factory=Verdict.yes)

    def close(self) -> None:
        for component in (*self.cohorts.values(), self.policy, self.evaluator, self.dataset):
            closer = getattr(component, "close", None)
            if callable(closer):
                try:
                    closer()
                except Exception:  # noqa: BLE001 - shutting down, never raise
                    pass


def build_component(
    registry: Registry, plane: str, spec: ComponentSpec
) -> tuple[Any, Any, Verdict]:
    """Construct one component, honouring the isolation it declared.

    A chained spec is built innermost first, each stage handed the one before
    it. Every stage's own refusals fire here, at plan time, which is the point:
    a chain that cannot work -- an estimator pointed at a dataset with no video,
    a retargeting step given image coordinates -- is refused before a single
    frame is decoded rather than an hour into a run.
    """
    if spec.chained:
        return _build_chain(registry, plane, spec)
    return _build_one(registry, plane, spec)


def _build_chain(registry: Registry, plane: str, spec: ComponentSpec) -> tuple[Any, Any, Verdict]:
    component: Any = None
    refs: list[str] = []
    notes: list[Reason] = []
    for link in spec.chain:
        try:
            component, ref, verdict = _build_one(registry, plane, link, source=component)
        except GantryError as error:
            if link.optional and component is not None:
                # A stage that may genuinely not apply. The run continues on the
                # stage before it and the record says which link was skipped and
                # why, because a silently shorter chain is a different pipeline
                # wearing the same name.
                notes.append(
                    Reason(
                        "chain.skipped",
                        f"{plane}: optional stage {link.name!r} was skipped ({error})",
                    )
                )
                refs.append(f"{link.name}(skipped)")
                continue
            raise
        refs.append(ref if isinstance(ref, str) else link.name)
        notes.extend(verdict.reasons)
    return component, " -> ".join(refs), Verdict(ok=True, reasons=tuple(notes))


def _build_one(
    registry: Registry, plane: str, spec: ComponentSpec, source: Any = None
) -> tuple[Any, Any, Verdict]:
    try:
        registration = registry.get(plane, spec.name)
    except KeyError as error:
        raise ConfigError(str(error)) from None

    try:
        descriptor = registration.descriptor(spec.config, source)
    except GantryError:
        raise
    except Exception as error:  # noqa: BLE001
        raise ConfigError(
            f"{plane}:{spec.name} could not describe itself: {type(error).__name__}: {error}"
        ) from error

    if wants_isolation(descriptor) and source is None:
        # Isolation and chaining do not compose: a stage in another process
        # cannot be handed a live object from this one. Declared isolation on a
        # chained stage is honoured by refusing rather than by quietly running
        # it in-process, which would be the isolation silently not happening.
        component, verdict = isolated_or_refuse(descriptor, plane, spec.name, spec.config)
        if component is None:
            raise ConfigError(verdict.explain())
        return component, descriptor.component_ref(config=spec.config), verdict

    if wants_isolation(descriptor) and source is not None:
        raise ConfigError(
            f"{plane}:{spec.name} declares isolation and is a stage in a chain. A "
            "component running in another process cannot be handed a live object "
            "from this one; put the isolated component first in the chain, or drop "
            "the isolation"
        )

    try:
        component = registration.build(spec.config, source)
    except GantryError:
        raise
    except Exception as error:  # noqa: BLE001
        raise ConfigError(
            f"{plane}:{spec.name} could not be built: {type(error).__name__}: {error}"
        ) from error
    return component, component.descriptor().component_ref(config=spec.config), Verdict.yes()


def build_all(manifest: Manifest, registry: Registry) -> Built:
    """Build every plane, with the cohorts on whichever one varies."""
    built = Built()
    # The cohorts are components of the varying plane -- datasets by default,
    # policies when comparing checkpoints, evaluators when comparing worlds.
    verdicts: list[Verdict] = []
    for name, spec in manifest.cohorts.items():
        component, ref, verdict = build_component(registry, manifest.varies, spec)
        built.cohorts[name] = component
        built.refs.append(ref)
        built.notes.extend(reason.message for reason in verdict.reasons)
        verdicts.append(verdict)
    for spec in manifest.feedback:
        component, ref, verdict = build_component(registry, "feedback", spec)
        built.feedback.append(component)
        built.refs.append(ref)
        verdicts.append(verdict)
    # Everything single-valued. A plane supplied by the cohorts has no single
    # component and is skipped here; the runner reads it per cohort instead.
    for plane, attribute in (
        ("embodiment", "embodiment"),
        ("policy", "policy"),
        ("evaluation", "evaluator"),
        ("dataset", "dataset"),
    ):
        spec = manifest.components.get(plane)
        if spec is None or manifest.varies == plane:
            continue
        if plane in ("policy", "evaluation") and not manifest.evaluates:
            continue
        component, ref, verdict = build_component(registry, plane, spec)
        setattr(built, attribute, component)
        built.refs.append(ref)
        verdicts.append(verdict)
    built.verdict = Verdict.all(verdicts)
    return built


# --------------------------------------------------------------------------
# fitting
# --------------------------------------------------------------------------


def _transform_registries() -> tuple[AdapterRegistry, RetargeterRegistry]:
    """Everything installed that can close a gap, discovered like any plugin."""
    from importlib.metadata import entry_points

    adapters = AdapterRegistry()
    for entry in entry_points(group="gantry.adapters"):
        try:
            adapters.add(entry.load())
        except Exception:  # noqa: BLE001 - a broken plugin must not break the run
            continue

    retargeters = RetargeterRegistry()
    for entry in entry_points(group="gantry.retargeters"):
        try:
            loaded = entry.load()
            retargeters.add(loaded() if isinstance(loaded, type) else loaded)
        except Exception:  # noqa: BLE001
            continue
    return adapters, retargeters


@dataclass(frozen=True)
class Fit:
    """How one consumer fits one provider."""

    wiring: Wiring | None
    verdict: Verdict


def fit_consumer(
    requirement: Any,
    schema: Sequence[ChannelSpec],
    provider: Any,
    adapters: AdapterRegistry,
    retargeters: RetargeterRegistry,
) -> Fit:
    """Check capabilities and bind channels, before anything is analysed.

    Takes the cohort's schema rather than its episodes because the schema is
    all this ever needed. Asking for the episodes made every caller materialise
    a whole cohort to answer a question about its channel names -- which for a
    chain that decodes video and estimates pose is minutes of work to read one
    tuple.
    """
    schema = tuple(schema)
    capability = check_capabilities(requirement, provider.descriptor())
    wiring, binding = bind(requirement, schema, adapters, retargeters)
    verdict = Verdict.all([capability, binding])
    return Fit(wiring if verdict.ok else None, verdict)


def check_policy_against_embodiment(
    policy: Any, embodiment: Any, adapters: AdapterRegistry, retargeters: RetargeterRegistry
) -> Verdict:
    """Can this policy actually drive this machine?

    The embodiment declares what its actuators accept; the policy declares what
    it emits. Checking the pair here turns "the arm moved somewhere unexpected"
    into a refusal before the first step.
    """
    descriptor = getattr(embodiment, "action", None)
    if not descriptor:
        return Verdict.note(
            "runner.embodiment_actionless",
            "the embodiment declares no action channels, so the policy was not checked against it",
        )
    requirement = requires_channels("policy", "policy", policy.action_spec())
    _, verdict = bind(requirement, tuple(descriptor), adapters, retargeters)
    if verdict.ok:
        return verdict
    return (
        Verdict.no(
            "runner.policy_embodiment",
            f"{policy.name} cannot command this embodiment",
            hint="the policy's action channel does not fit any the machine accepts",
        )
        & verdict
    )


# --------------------------------------------------------------------------
# running
# --------------------------------------------------------------------------


def run_manifest(
    manifest: Manifest,
    registry: Registry | None = None,
    *,
    stamp_time: bool = True,
    save_to: str | None = None,
) -> Outcome:
    """Build, resolve, run, report, and optionally persist."""
    manifest.validate().raise_if_refused(f"cannot run manifest {manifest.name!r}")
    registry = registry or discovered_registry()
    adapters, retargeters = _transform_registries()

    built = build_all(manifest, registry)
    try:
        return _execute(manifest, built, adapters, retargeters, stamp_time, save_to)
    finally:
        built.close()


def _execute(
    manifest: Manifest,
    built: Built,
    adapters: AdapterRegistry,
    retargeters: RetargeterRegistry,
    stamp_time: bool,
    save_to: str | None,
) -> Outcome:
    runs, sources, failures, notes = _gather(manifest, built)
    notes = list(built.notes) + notes
    refusals: list[str] = []

    if built.embodiment is not None and built.policy is not None:
        verdict = check_policy_against_embodiment(
            built.policy, built.embodiment, adapters, retargeters
        )
        if not verdict.ok:
            refusals.append(verdict.explain())
        notes.extend(r.message for r in verdict.reasons if verdict.ok)

    reports: list[Report] = []
    applied: list[Any] = []

    for module in built.feedback:
        cohorts: list[Cohort] = []
        blocked: Verdict | None = None
        for source in sources:
            fit = fit_consumer(
                module.requirement(),
                source.episodes[0].schema if source.episodes else (),
                source.provider,
                adapters,
                retargeters,
            )
            if fit.wiring is None:
                blocked = fit.verdict
                break
            # The run's provenance travels with the cohort. A module that needs
            # to know what the run was *set up* to do -- rather than only what
            # happened -- reads it from there.
            cohorts.append(
                Cohort(
                    source.name,
                    adapt_all(source.episodes, fit.wiring),
                    provenance=source.provenance,
                )
            )
            applied.extend(fit.wiring.adapters)
        if blocked is not None:
            refusals.append(f"{module.name}: {blocked.explain()}")
            continue
        try:
            reports.append(module.run(cohorts))
        except GantryError as error:
            if must_halt(error):
                raise
            failures.append(f"feedback:{module.name}: {error}")
        except Exception as error:  # noqa: BLE001 - one analysis, not the run
            failures.append(f"feedback:{module.name}: {type(error).__name__}: {error}")

    if refusals:
        runnable = [report.module for report in reports]
        notes.append("runnable as requested: " + (", ".join(runnable) if runnable else "nothing"))
    notes.extend(
        f"{step.name} is lossy: {'; '.join(step.losses)}" for step in applied if step.losses
    )

    provenance = Provenance(
        components=tuple(built.refs),
        protocol=dict(manifest.protocol),
        adapters=tuple({(s.name, s.version): s for s in applied}.values()),
        created_at=datetime.now(timezone.utc).isoformat() if stamp_time else None,
        host=platform.node() if stamp_time else None,
        notes=tuple(notes),
    )

    saved: list[str] = []
    if save_to:
        from pathlib import Path

        directory = Path(save_to)
        directory.mkdir(parents=True, exist_ok=True)
        for index, run in enumerate(runs):
            saved.append(str(write_run(run, directory / f"{manifest.name}-{index:02d}")))

    return Outcome(
        manifest,
        tuple(reports),
        tuple(runs),
        provenance,
        tuple(failures),
        tuple(refusals),
        tuple(notes),
        tuple(saved),
    )


@dataclass(frozen=True)
class Source:
    """One named set of episodes, and everything known about where it came from.

    ``provider`` is whatever produced the episodes, so a capability check asks
    the component that actually knows: reading a connector's episodes and then
    asking the evaluator about them, or the reverse, would check the wrong claim.

    ``provenance`` is present only for episodes a run produced, and is what lets
    a module see what the run was *set up* to do rather than only what happened.
    """

    name: str
    episodes: tuple[EpisodeRecord, ...]
    provider: Any
    provenance: Provenance | None = None


def _gather(
    manifest: Manifest, built: Built
) -> tuple[list[RunRecord], list[Source], list[str], list[str]]:
    """Episodes either as recorded, or as produced by running the policy."""
    runs: list[RunRecord] = []
    sources: list[Source] = []
    failures: list[str] = []
    notes: list[str] = []

    varies = manifest.varies
    for name, component in built.cohorts.items():
        # Whichever plane varies, this cohort supplies it; the rest are held.
        provider = component if varies == "dataset" else built.dataset
        policy = component if varies == "policy" else built.policy
        evaluator = component if varies == "evaluation" else built.evaluator
        embodiment = component if varies == "embodiment" else built.embodiment
        if embodiment is not None and evaluator is not None:
            # The world is rebuilt around the body whether the body is the axis
            # or is being held. Only rebuilding it when it varies was a real
            # hole: a manifest naming one embodiment recorded that name in
            # provenance and simulated whatever the evaluator already had, so
            # the run described a machine it never ran on.
            #
            # An evaluator with a body welded in refuses here rather than
            # running identical worlds under different labels.
            try:
                evaluator = evaluator.for_embodiment(embodiment)
            except GantryError as error:
                failures.append(f"embodiment:{name}: {error}")
                continue
        if provider is None:
            failures.append(f"{name}: nothing supplies the dataset plane; name one, or vary on it")
            continue
        episodes = tuple(provider)
        if not manifest.evaluates:
            sources.append(Source(name, episodes, provider))
            continue

        from .contracts.evaluator import Protocol

        protocol = Protocol(**_protocol_fields(manifest.protocol))
        # An evaluator whose world is the data under test gets this cohort's
        # episodes. One bound instance per cohort, never a rebound shared one:
        # the cohorts are compared against each other, so a single evaluator
        # that rebound itself would score the second against the first's
        # answer key.
        evaluator = (
            evaluator.bind(episodes) if getattr(evaluator, "needs_dataset", False) else evaluator
        )
        task = _task_for(evaluator, name, manifest)
        try:
            record = evaluator.evaluate(policy, task, protocol)
        except GantryError as error:
            if must_halt(error):
                raise
            failures.append(f"evaluation:{name}: {error}")
            continue
        runs.append(record)
        sources.append(Source(name, record.episodes, evaluator, record.provenance))
        notes.extend(record.provenance.notes)

    return runs, sources, failures, notes


def _protocol_fields(protocol: Mapping[str, Any]) -> dict[str, Any]:
    known = {"epochs", "seed_base", "execute", "max_trials"}
    fields = {key: value for key, value in protocol.items() if key in known}
    extra = {key: value for key, value in protocol.items() if key not in known}
    if extra:
        fields["extra"] = extra
    return fields


def _task_for(evaluator: Any, cohort: str, manifest: Manifest) -> Any:
    """Ask the evaluator for a task. Contractual since evaluator@1.1.

    This used to probe for the method, which meant an evaluator could pass every
    conformance check and still be undriveable. It is required now, so anything
    the resolver accepted can be started.
    """
    return evaluator.task_for(f"{manifest.name}:{cohort}")


def discovered_registry() -> Registry:
    registry = Registry()
    registry.discover()
    return registry


# --------------------------------------------------------------------------
# planning
# --------------------------------------------------------------------------


def _cohort_schema(connector: Any) -> tuple[ChannelSpec, ...]:
    """The cohort's channels, for the price of one episode.

    A connector that computes its episodes cannot declare a schema without
    producing one of them, so a plan pays for exactly one and no more. The
    alternative -- reading every episode to look at the first -- turns planning
    a chain that decodes video into a job rather than a check.
    """
    ids = connector.episode_ids()
    return tuple(connector.schema(ids[0])) if ids else ()


def plan_manifest(manifest: Manifest, registry: Registry | None = None) -> Verdict:
    """Resolve without running anything.

    Builds the cohorts, because a connector's schema and capabilities are only
    knowable once it has opened its source -- and refusing a run for a reason
    that could have been found in a second is the whole point of planning.

    "Without running anything" is meant literally, and a plan that reads a whole
    cohort has stopped being one. It costs one episode per cohort, not one per
    feedback module per cohort.
    """
    registry = registry or discovered_registry()
    checks = [check_manifest(manifest, registry)]
    if not checks[0].ok:
        return Verdict.all(checks)

    adapters, retargeters = _transform_registries()
    try:
        built = build_all(manifest, registry)
    except GantryError as error:
        return Verdict.all(checks + [Verdict.no("plan.build_failed", str(error))])
    checks.append(built.verdict)

    try:
        if built.embodiment is not None and built.policy is not None:
            checks.append(
                check_policy_against_embodiment(
                    built.policy, built.embodiment, adapters, retargeters
                )
            )
        schemas = {name: _cohort_schema(c) for name, c in built.cohorts.items()}
        for module in built.feedback:
            for name, connector in built.cohorts.items():
                fit = fit_consumer(
                    module.requirement(), schemas[name], connector, adapters, retargeters
                )
                if fit.wiring is None:
                    checks.append(
                        Verdict.no(
                            "plan.module_refused",
                            f"{module.name} cannot run on cohort {name!r}",
                            module=module.name,
                            cohort=name,
                        )
                        & fit.verdict
                    )
                elif fit.wiring.adapters:
                    checks.append(
                        Verdict.note(
                            "plan.adapted",
                            f"{module.name} on {name!r} needs "
                            f"{', '.join(step.name for step in fit.wiring.adapters)}",
                        )
                    )
    finally:
        built.close()
    return Verdict.all(checks)


def check_manifest(manifest: Manifest, registry: Registry | None = None) -> Verdict:
    """Is everything this manifest names actually installed?"""
    registry = registry or discovered_registry()
    checks = [manifest.validate()]
    wanted: list[tuple[str, ComponentSpec]] = [
        ("dataset", spec) for spec in manifest.cohorts.values()
    ]
    wanted += [("feedback", spec) for spec in manifest.feedback]
    wanted += list(manifest.components.items())

    for plane, spec in wanted:
        if not registry.has(plane, spec.name):
            available = registry.names(plane)
            checks.append(
                Verdict.no(
                    "manifest.not_installed",
                    f"{plane}:{spec.name} is not installed",
                    hint=(
                        f"installed on the {plane} plane: {', '.join(available)}"
                        if available
                        else f"nothing is installed on the {plane} plane"
                    ),
                    plane=plane,
                    name=spec.name,
                )
            )
    return Verdict.all(checks)
