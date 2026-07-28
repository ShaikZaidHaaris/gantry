"""Turning records into plain data and back, once.

Two places need this: writing a run to disk, and passing one across a process
boundary. They are the same problem, and having each solve it separately is how
a record written by one and read by the other quietly loses a field.

Everything here is symmetric by construction — every ``*_to_dict`` has a
``*_from_dict`` that inverts it — and the round-trip is tested rather than
assumed, because "we serialise it" and "we can read it back" are different
claims and only the second one is worth anything.

Step data is deliberately absent. Arrays are bulk, they belong in a format built
for them, and each caller decides where: a sidecar file on disk, a temporary
buffer over a pipe. This module handles everything describable, which is
everything that has to survive intact.
"""

from __future__ import annotations

from typing import Any, Mapping

from .spine import (
    AdapterStep,
    ChannelSpec,
    ComponentRef,
    Descriptor,
    EpisodeLabels,
    EpisodeMeta,
    Measurement,
    Provenance,
    StageEvent,
)


# -- channels ---------------------------------------------------------------


def spec_to_dict(spec: ChannelSpec) -> dict[str, Any]:
    return {
        "name": spec.name,
        "kind": spec.kind,
        "shape": list(spec.shape),
        "dtype": spec.dtype,
        "units": spec.units,
        "frame": spec.frame,
        "rate_hz": spec.rate_hz,
        "semantics": spec.semantics,
        "dim_labels": list(spec.dim_labels) if spec.dim_labels else None,
        "optional": spec.optional,
        # Load-bearing metadata keys travel with the spec. Dropping them here
        # would disarm every refusal they exist to cause, at exactly the two
        # boundaries that were added for portability: the store and the
        # isolation worker. A quaternion would cross either one and arrive
        # compatible with a channel it must not bind to.
        "discriminators": list(spec.discriminators),
        "metadata": dict(spec.metadata),
    }


def spec_from_dict(payload: Mapping[str, Any]) -> ChannelSpec:
    return ChannelSpec(
        name=payload["name"],
        kind=payload["kind"],
        shape=tuple(payload.get("shape") or ()),
        dtype=payload.get("dtype", "float32"),
        units=payload.get("units"),
        frame=payload.get("frame"),
        rate_hz=payload.get("rate_hz"),
        semantics=payload.get("semantics"),
        dim_labels=tuple(payload["dim_labels"]) if payload.get("dim_labels") else None,
        optional=payload.get("optional", False),
        discriminators=tuple(payload.get("discriminators") or ()),
        metadata=payload.get("metadata") or {},
    )


# -- episodes ---------------------------------------------------------------


def meta_to_dict(meta: EpisodeMeta) -> dict[str, Any]:
    return {
        "id": meta.id,
        "source": meta.source,
        "embodiment": meta.embodiment,
        "task": meta.task,
        "collected_by": meta.collected_by,
        "license": meta.license,
        "extra": dict(meta.extra),
    }


def meta_from_dict(payload: Mapping[str, Any]) -> EpisodeMeta:
    return EpisodeMeta(
        id=payload["id"],
        source=payload["source"],
        embodiment=payload.get("embodiment"),
        task=payload.get("task"),
        collected_by=payload.get("collected_by"),
        license=payload.get("license"),
        extra=payload.get("extra") or {},
    )


def labels_to_dict(labels: EpisodeLabels) -> dict[str, Any]:
    return {
        "success": labels.success,
        "stage_events": [
            {"name": event.name, "step": event.step, "detail": dict(event.detail)}
            for event in labels.stage_events
        ],
        "annotations": dict(labels.annotations),
    }


def labels_from_dict(payload: Mapping[str, Any]) -> EpisodeLabels:
    return EpisodeLabels(
        success=payload.get("success"),
        stage_events=tuple(
            StageEvent(event["name"], int(event["step"]), event.get("detail") or {})
            for event in payload.get("stage_events") or ()
        ),
        annotations=payload.get("annotations") or {},
    )


# -- provenance -------------------------------------------------------------


def provenance_from_dict(payload: Mapping[str, Any]) -> Provenance:
    """Inverts :meth:`Provenance.as_dict`, which is the forward direction."""
    return Provenance(
        components=tuple(
            ComponentRef(
                plane=component["plane"],
                name=component["name"],
                version=component["version"],
                config_digest=component.get("config_digest"),
                artifact_digest=component.get("artifact_digest"),
                detail=component.get("detail") or {},
            )
            for component in payload.get("components") or ()
        ),
        protocol=payload.get("protocol") or {},
        adapters=tuple(
            AdapterStep(step["name"], step["version"], tuple(step.get("losses") or ()))
            for step in payload.get("adapters") or ()
        ),
        created_at=payload.get("created_at"),
        host=payload.get("host"),
        notes=tuple(payload.get("notes") or ()),
    )


def measurement_from_dict(payload: Mapping[str, Any]) -> Measurement:
    ci = payload.get("ci")
    return Measurement(
        value=float(payload["value"]),
        n=payload.get("n"),
        ci=(float(ci[0]), float(ci[1])) if ci else None,
        units=payload.get("units"),
        method=payload.get("method"),
        detail=payload.get("detail") or {},
    )


# -- descriptors ------------------------------------------------------------


def descriptor_from_dict(payload: Mapping[str, Any]) -> Descriptor:
    """Inverts :meth:`Descriptor.as_dict`."""
    return Descriptor.from_dict(payload)
