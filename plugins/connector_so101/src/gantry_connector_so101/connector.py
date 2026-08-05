"""Read an SO-ARM101 corpus-factory recording as Gantry episodes.

The SO-101 rig is a **data source**, not an evaluator. A leader-follower pair
produces recorded episodes and a per-episode metadata sidecar; this connector
exposes both as :class:`~gantry.spine.episode.EpisodeRecord` values so the
feedback plane can grade the corpus. That is the corpus factory in
``CORPUS-PROTOCOL.md``, and the corpus is the product.

What this adds over the plain LeRobot connector
-----------------------------------------------
Nothing here re-reads parquet, re-resolves video, or re-implements selective
reads. :class:`~gantry_connector_lerobot.LeRobotConnector` already does all of
that and is composed, not copied. Three things are added, and each of them is a
refusal or a declaration that reader cannot make:

**1. Embodiment binding, and the width guard.** A recorded dataset is float
arrays. This connector binds one to a declared embodiment ref and refuses a
mismatch in either direction — see :mod:`.binding`. It then describes the joint
channels with the embodiment's own units, frame, meaning tags and
discriminators, so they stop being matchable by name and width alone.

**2. Calibration provenance.** Every episode carries the SHA-256 of both arms'
calibration JSONs and a digest over the pair, and a corpus containing more than
one digest is **refused**: it spans a recalibration, and pooling across one is
pooling two different mappings from these numbers to poses. LeRobot #3758 is an
offset that is invisible in the training loss, so the hash is the only thing
that makes it findable after the fact — which only works if a corpus that
straddles one cannot be read as if it did not.

**3. Corpus-factory metadata.** ``corpus_id``, ``condition_label``,
``operator_id``, ``session_id``, ``block_id``,
``episode_index_within_session``, the treatment assignment and the RNG seed that
produced it, object and container start pose, outcome tag, drift dose for a
derived corpus, dropped-frame counts per camera, and leader-follower tracking
error per joint. A corpus missing the block fields stays readable and
**declares** that it cannot support the block analysis, rather than quietly
producing one.

Capabilities, and why each is what it is
----------------------------------------
``lazy = True``
    Genuinely: :meth:`schema` reads no step data, and :meth:`open` hands back
    the LeRobot reader's own source, so a window read still projects columns and
    slices rows in the parquet.

``media`` — whatever the LeRobot reader could actually resolve
    True when the mp4 side-cars are present and decodable, false when the
    metadata declares cameras nobody downloaded. Delegated, because "this
    dataset has no images" and "this install cannot open them" are different
    problems and that reader already tells them apart.

``outcomes`` — true only when the sidecar carries an outcome tag
    Computed from the parsed rows, not assumed from the format.

``stage_events`` — false for a teleop recording, and normally false here
    The funnel needs object pose at every step. The physical rig has no
    perception, and the sidecar records one pose per episode, not per step. An
    overstated ``True`` yields an empty funnel that reads as a finished
    analysis. The contract's wording is "at least some episodes carry stage
    events", which is a fact about a *corpus*: a run recorded in sim does have
    poses and could carry them. So this is computed from what the episodes
    actually have, and the only way to put them there is to hand them in
    (``stage_events=``) together with a written ``stage_events_source``. There
    is no detector here, because a detector without a pose source would be a
    guess wearing a milestone's name.

What this corpus is not, and the description that says so
---------------------------------------------------------
Three things follow from describing the joint channels properly, and all three
are visible in ``tests/test_duration_confound.py``:

* **joint percent is not a position.** The channels declare
  ``so101.joint_position_normalized`` — dimensionless percent of each joint's
  calibrated travel — so nothing binds them to a consumer asking for a point in
  space. Path efficiency over percentages, where a jaw's travel and a shoulder's
  rotation are the same unit and the published URDF has no linear gripper joint
  to make a Cartesian path out of, would be a number with no referent. It comes
  back unavailable, which is the correct answer;
* **the gripper is a dimension, not a channel.** It is element 5 of the six
  (``gripper_index`` in the channel metadata), so a consumer wanting a scalar
  actuation signal is not silently handed the sixth column. Extracting it is a
  declared retargeter's job;
* **per-step-mean statistics over an absolute command inherit the episode's
  length.** One grasp and one release per episode is a fixed amount of gripper
  travel divided by a variable number of steps. Anything averaging
  ``|Δaction|`` over steps therefore falls as 1/T on a corpus with nothing wrong
  with it. That is a property of an absolute joint-command channel, it is real
  rather than an artefact of any fixture, and it is written down here because
  the alternative is that somebody reads it as a finding.
"""

from __future__ import annotations

import os
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from gantry_connector_lerobot import AUTO, LeRobotConnector

from gantry.contracts.connector import Connector, connector_descriptor
from gantry.errors import ConfigError
from gantry.spine import (
    ChannelSpec,
    Descriptor,
    EpisodeLabels,
    EpisodeMeta,
    EpisodeRecord,
    StageEvent,
)

from .binding import (
    ACTION,
    BIMANUAL_REF,
    CALIBRATION,
    CALIBRATION_DIGEST,
    CONTROL_HZ,
    FOLLOWER_REF,
    NORMALIZATION,
    STATE,
    Binding,
    MountingGeometry,
    get_binding,
)
from .sidecar import (
    EpisodeMetadata,
    Join,
    Sidecar,
    absent_sidecar,
    find_sidecar,
    join_rows,
    read_sidecar,
)

VERSION = "0.1.0.dev0"

#: How an outcome tag becomes the spine's boolean ``success``.
#:
#: ``recovered`` maps to True and keeps its tag. CORPUS-PROTOCOL.md calls it "a
#: distinct tag, not a flavour of success" because the E-RECOV condition is
#: *defined* by that count — but a recovered episode is one where the operator
#: missed the grasp, re-approached, and finished the task. Mapping it to False
#: would make the recovery corpus read as 40 % failures and the screen's
#: absolute mode would flag a corpus that is doing exactly what it was recorded
#: to do; mapping it to None would delete the outcomes of the one condition
#: whose outcomes are the treatment. The tag survives verbatim in
#: ``labels.annotations['outcome']``, so nothing that needs the three-way
#: distinction has to recover it from a boolean.
OUTCOME_SUCCESS: Mapping[str, bool | None] = {
    "success": True,
    "recovered": True,
    "fail": False,
}

#: Channels this connector describes from the binding. Everything else is left
#: exactly as the LeRobot reader described it — including cameras, which it can
#: locate but whose mounting nobody here can verify. A ``frame`` tag on a camera
#: is a claim about where it was bolted, and this connector has no evidence for
#: one.
DESCRIBED = (STATE, ACTION)

#: LeRobot writes ``timestamp`` as seconds since the start of the episode. This
#: is the one non-joint channel described here, and it is a declaration about
#: corpora produced by *our* recorder rather than about the format in general.
TIMESTAMP = "timestamp"


class SO101Connector(Connector):
    """One SO-ARM101 recording: a LeRobot dataset plus its design metadata."""

    def __init__(
        self,
        path: str | os.PathLike[str],
        *,
        embodiment: str,
        source: str | None = None,
        sidecar: str | os.PathLike[str] | bool = True,
        join: str = "auto",
        normalization: str = NORMALIZATION,
        calibration_variant: str = CALIBRATION,
        mounting: MountingGeometry | None = None,
        allow_mixed_calibration: bool = False,
        strict_metadata: bool = False,
        outcome_success: Mapping[str, bool | None] = OUTCOME_SUCCESS,
        stage_events: Mapping[str, Sequence[StageEvent]] | None = None,
        stage_events_source: str | None = None,
        video: bool | str = AUTO,
        include_bookkeeping: bool = False,
    ):
        self._root = Path(path)
        # Composed, not reimplemented. Parquet indexing, the chunked path
        # templates, video resolution and the selective-read source all live in
        # that reader, and a second opinion about any of them would be a second
        # place for them to be wrong.
        self._inner = LeRobotConnector(
            self._root,
            source=source,
            include_bookkeeping=include_bookkeeping,
            video=video,
        )
        self._binding = get_binding(embodiment)
        self._normalization = normalization
        self._calibration_variant = calibration_variant
        self._mounting = mounting

        self._sidecar = self._load_sidecar(sidecar)
        if strict_metadata:
            self._require_complete_metadata()
        self._calibration = self._resolve_calibration(allow_mixed_calibration)
        self._join = self._join_metadata(join)

        self._outcome_success = dict(outcome_success)
        self._stage_events = self._check_stage_events(stage_events, stage_events_source)
        self._stage_events_source = stage_events_source

        self._labels = {
            episode_id: self._labels_for(episode_id) for episode_id in self._inner.episode_ids()
        }
        self._schema = self._describe(self._inner.schema(self._inner.episode_ids()[0]))

    # -- metadata ----------------------------------------------------------

    def _load_sidecar(self, requested: str | os.PathLike[str] | bool) -> Sidecar:
        if requested is False:
            return absent_sidecar()
        if requested is True:
            found = find_sidecar(self._root)
            if found is None:
                return absent_sidecar()
        else:
            found = Path(requested)
            if not found.is_file():
                # An explicitly named sidecar that is not there is a
                # configuration error, unlike one that was merely looked for:
                # the caller said this corpus has metadata, and it does not.
                raise ConfigError(
                    f"{found}: no such metadata sidecar. Pass sidecar=False to read this "
                    "corpus without its design metadata, which is a different and "
                    "declared thing from reading it with metadata that is missing."
                )
        return read_sidecar(found, joint_names=self._binding.joint_names)

    def _require_complete_metadata(self) -> None:
        problems = []
        if not self._sidecar.present:
            problems.append("no sidecar was found")
        else:
            if not self._sidecar.supports_block_analysis:
                problems.append(self._sidecar.block_analysis["why_not"])
            if self._sidecar.missing_calibration_columns:
                problems.append(
                    f"the calibration columns {list(self._sidecar.missing_calibration_columns)} "
                    "are absent, so a recalibration inside this corpus would leave no trace"
                )
        if problems:
            raise ConfigError(
                f"{self._root}: strict_metadata was asked for and this corpus does not "
                "carry a complete design record. " + " ".join(problems)
            )

    def _resolve_calibration(self, allow_mixed: bool) -> dict[str, Any]:
        provenance = self._sidecar.calibration_provenance
        digests = provenance["digests"]
        if len(digests) > 1 and not allow_mixed:
            counts = {
                digest: sum(1 for row in self._sidecar.rows if row.calibration_digest == digest)
                for digest in digests
            }
            raise ConfigError(
                f"{self._root}: this corpus spans {len(digests)} calibrations "
                f"({counts}). Refusing to pool them. The two sets of episodes are the same "
                "dtype, the same width, the same units and the same joint order, and every "
                "number in one is offset from the other by whatever the recalibration "
                "changed — LeRobot #3758 is ~17 deg of that becoming ~6 cm of gripper "
                "error with a healthy training loss. A corpus spanning a recalibration is "
                "a confounded corpus; open the halves separately, or pass "
                "allow_mixed_calibration=True to state that you mean to pool them."
            )
        return {
            **provenance,
            "digest": digests[0] if len(digests) == 1 else None,
            "pooled_across_calibrations": len(digests) > 1,
        }

    def _join_metadata(self, method: str) -> Join | None:
        if not self._sidecar.present:
            return None
        ids = self._inner.episode_ids()
        lengths: dict[str, int] = {}
        if method != "index":
            # Lengths come from meta/episodes.jsonl through the LeRobot reader's
            # own source, so this reads no parquet — the join stays as cheap as
            # enumeration.
            lengths = {episode_id: len(self._inner.open(episode_id)) for episode_id in ids}
        return join_rows(
            self._sidecar, ids, lengths, index_of=_episode_index, method=method
        )

    def _metadata_for(self, episode_id: str) -> EpisodeMetadata | None:
        if self._join is None:
            return None
        return self._join.by_episode_id.get(episode_id)

    # -- labels ------------------------------------------------------------

    def _check_stage_events(
        self,
        stage_events: Mapping[str, Sequence[StageEvent]] | None,
        source: str | None,
    ) -> dict[str, tuple[StageEvent, ...]]:
        if not stage_events:
            return {}
        if not (source or "").strip():
            raise ConfigError(
                "stage_events were supplied without a stage_events_source. Milestones on a "
                "teleop corpus cannot come from this rig — it has no perception and the "
                "sidecar records one object pose per episode, not per step — so where they "
                "came from is the only thing that makes them checkable, and it is carried "
                "into every record they label."
            )
        known = set(self._inner.episode_ids())
        unknown = sorted(set(stage_events) - known)
        if unknown:
            raise ConfigError(
                f"stage_events name episode(s) {unknown[:5]} that this corpus does not "
                "contain. Milestones landing on the wrong episode would produce a funnel "
                "that is complete and describes another recording."
            )
        return {
            episode_id: tuple(events) for episode_id, events in stage_events.items() if events
        }

    def _labels_for(self, episode_id: str) -> EpisodeLabels:
        metadata = self._metadata_for(episode_id)
        annotations: dict[str, Any] = {}
        success: bool | None = None
        if metadata is not None:
            if metadata.outcome is not None:
                if metadata.outcome not in self._outcome_success:
                    raise ConfigError(
                        f"{episode_id}: outcome tag {metadata.outcome!r} is not one of "
                        f"{sorted(self._outcome_success)}. Refusing rather than treating an "
                        "unknown tag as a failure: 'we do not know what this tag means' and "
                        "'this demonstration did not work' are different corpora."
                    )
                success = self._outcome_success[metadata.outcome]
            annotations = {
                # The grouping keys a feedback module needs to form cohorts,
                # kept where a module that never reads meta.extra will still see
                # them. The full record lives on the episode's meta.
                "outcome": metadata.outcome,
                "corpus_id": metadata.corpus_id,
                "condition_label": metadata.condition_label,
                "operator_id": metadata.operator_id,
                "session_id": metadata.session_id,
                "block_id": metadata.block_id,
                "episode_index_within_session": metadata.episode_index_within_session,
            }
        return EpisodeLabels(
            success=success,
            stage_events=self._stage_events.get(episode_id, ()),
            annotations=annotations,
        )

    # -- schema ------------------------------------------------------------

    def _describe(self, inner_schema: Sequence[ChannelSpec]) -> tuple[ChannelSpec, ...]:
        described = []
        for spec in inner_schema:
            if spec.name in DESCRIBED:
                described.append(self._joint_channel(spec))
            elif spec.name == TIMESTAMP and spec.kind == "scalar":
                described.append(replace(spec, units="s", semantics="time"))
            else:
                # Left exactly as the LeRobot reader described it. That reader
                # declines to invent units and so does this one; a numeric
                # channel this connector has no declaration for stays
                # undescribed and strict conformance says so, which is the
                # honest outcome rather than a plausible guess.
                described.append(spec)
        return tuple(described)

    def _joint_channel(self, spec: ChannelSpec) -> ChannelSpec:
        binding = self._binding
        binding.check(spec.name, spec.width, spec.dim_labels)
        if np.dtype(spec.dtype).kind != "f":
            raise ConfigError(
                f"{spec.name!r} is stored as {spec.dtype!r}. {binding.ref} declares these "
                "channels as percent of calibrated travel in float32; an integer column is "
                "a different quantisation of the same range and reading it as the declared "
                "one would silently round every command."
            )
        extra: dict[str, Any] = {
            # The calibration *files*, alongside the embodiment's own
            # `calibration_variant` (the convention). Two corpora agreeing on
            # the convention and disagreeing here are the same rig either side
            # of a recalibration, which no structural check can see.
            CALIBRATION_DIGEST: self._calibration.get("digest"),
            "stored_dtype": spec.dtype,
            "source_dataset": str(self._root),
        }
        if spec.name == ACTION:
            # LeRobot records what the leader was at, i.e. the absolute target
            # the follower was commanded to. Carried through as the
            # embodiment's action semantics, never as a displacement.
            extra["recorded_by"] = "leader-follower teleoperation"
        channel = binding.build(
            spec.name,
            semantics=binding.semantics_for(spec.name),
            normalization=self._normalization,
            calibration_variant=self._calibration_variant,
            rate_hz=spec.rate_hz or CONTROL_HZ,
            extra=extra,
            mounting=self._mounting,
        )
        return replace(channel, discriminators=(*channel.discriminators, CALIBRATION_DIGEST))

    # -- contract ----------------------------------------------------------

    def descriptor(self) -> Descriptor:
        inner = self._inner.descriptor()
        return connector_descriptor(
            name="so101",
            version=VERSION,
            # schema() reads no steps and open() returns the LeRobot reader's
            # own source, so a channel-and-window read is still a projected,
            # sliced parquet read.
            lazy=True,
            # Computed, not asserted. Nothing here detects a milestone; the only
            # way an episode has one is that a caller handed it in with a
            # written source, which is a per-corpus fact and is treated as one.
            stage_events=any(bool(labels.stage_events) for labels in self._labels.values()),
            outcomes=any(labels.success is not None for labels in self._labels.values()),
            # Delegated: that reader knows which mp4s it can actually decode.
            media=bool(self._inner.cameras),
            path=str(self._root),
            embodiment=self._binding.ref,
            dof=self._binding.dof,
            joint_names=list(self._binding.joint_names),
            episodes=len(self._inner.episode_ids()),
            normalization=self._normalization,
            calibration_variant=self._calibration_variant,
            calibration=self._calibration,
            corpus=self._corpus_summary(),
            block_analysis=self._sidecar.block_analysis,
            metadata_join=self._join.as_dict() if self._join else None,
            stage_events_source=self._stage_events_source,
            mixed=self._mixed_fields(),
            lerobot=dict(inner.metadata),
        )

    def episode_ids(self) -> tuple[str, ...]:
        return self._inner.episode_ids()

    def schema(self, episode_id: str) -> tuple[ChannelSpec, ...]:
        self._require(episode_id)
        return self._schema

    def open(self, episode_id: str) -> EpisodeRecord:
        inner = self._inner.open(self._require(episode_id))
        metadata = self._metadata_for(episode_id)
        return EpisodeRecord(
            meta=EpisodeMeta(
                id=episode_id,
                source=inner.meta.source,
                # The declared ref, not LeRobot's free-string robot_type. The
                # width guard is what makes this a checked claim rather than a
                # label.
                embodiment=self._binding.ref,
                task=inner.meta.task,
                # Pseudonymous by protocol; it is the operator axis of the
                # design and belongs on the record that carries the numbers.
                collected_by=metadata.operator_id if metadata else None,
                extra={
                    **dict(inner.meta.extra),
                    "embodiment_ref": self._binding.ref,
                    "calibration": {
                        "sha256": dict(metadata.calibration) if metadata else {},
                        "digest": metadata.calibration_digest if metadata else None,
                        "variant": self._calibration_variant,
                        "normalization": self._normalization,
                    },
                    "corpus": metadata.as_dict() if metadata else None,
                    "metadata_join": self._join.as_dict() if self._join else None,
                    "block_analysis": self._sidecar.block_analysis,
                    "stage_events_source": self._stage_events_source,
                },
            ),
            schema=self._schema,
            # The LeRobot reader's source, untouched. This is what makes the
            # lazy claim true rather than aspirational.
            source=inner.source,
            labels=self._labels[episode_id],
        )

    def _require(self, episode_id: str) -> str:
        if episode_id not in self._labels:
            raise KeyError(f"{self._root}: no episode {episode_id!r}")
        return episode_id

    # -- what a reader wants to know ---------------------------------------

    def _corpus_summary(self) -> dict[str, Any]:
        rows = self._sidecar.rows
        if not rows:
            return {"sidecar": None, "producer": None}
        return {
            "sidecar": str(self._sidecar.path),
            "episodes": len(rows),
            "corpus_ids": sorted({row.corpus_id for row in rows if row.corpus_id}),
            "conditions": sorted({row.condition_label for row in rows if row.condition_label}),
            "operators": sorted({row.operator_id for row in rows if row.operator_id}),
            "sessions": sorted({row.session_id for row in rows if row.session_id}),
            "blocks": sorted({row.block_id for row in rows if row.block_id}),
            "outcomes": {
                tag: sum(1 for row in rows if row.outcome == tag)
                for tag in sorted({row.outcome for row in rows if row.outcome})
            },
            # Absent means "not declared", never "zero". A drift arm whose
            # manifest was lost must not read as a clean one.
            "drift": sorted(
                {
                    f"{row.drift.get('drift_joint')}@{row.drift.get('drift_deg')}deg"
                    for row in rows
                    if row.drift
                }
            )
            or None,
            "derived": any(row.drift for row in rows),
        }

    def _mixed_fields(self) -> dict[str, list[Any]]:
        """Fields that vary across the corpus where a campaign expects one value.

        Reported, not refused. A recording horizon that differs between
        conditions breaks frame-matching and is the defect that invalidated
        three published results — but this is a *reader*, and a corpus
        deliberately assembled from two horizons is a legitimate thing to look
        at. What is not legitimate is not knowing, so the disagreement is on the
        descriptor where a screen or a manifest can see it.
        """
        rows = self._sidecar.rows
        mixed: dict[str, list[Any]] = {}
        for field_name in ("episode_time_s", "control_hz", "sim_commit", "lerobot_commit"):
            values = sorted(
                {getattr(row, field_name) for row in rows if getattr(row, field_name) is not None},
                key=repr,
            )
            if len(values) > 1:
                mixed[field_name] = values
        return mixed

    # -- extras ------------------------------------------------------------

    @property
    def lerobot(self) -> LeRobotConnector:
        """The reader this one composes. Exposed so nothing has to wrap it twice."""
        return self._inner

    @property
    def binding(self) -> Binding:
        return self._binding

    @property
    def sidecar(self) -> Sidecar:
        return self._sidecar

    @property
    def supports_block_analysis(self) -> bool:
        """Whether CORPUS-PROTOCOL's confound analysis is possible on this corpus."""
        return self._sidecar.supports_block_analysis

    @property
    def calibration_digests(self) -> tuple[str, ...]:
        return tuple(self._calibration["digests"])

    def metadata(self, episode_id: str) -> EpisodeMetadata | None:
        """One episode's design record, or ``None`` when the corpus has none."""
        return self._metadata_for(self._require(episode_id))


def _episode_index(episode_id: str) -> int:
    """The LeRobot episode index behind an id.

    Parsed from the id the LeRobot reader publishes rather than reached into its
    private index: the id format is part of what that connector offers, and
    reaching past it would make this reader break on a refactor it cannot see.
    """
    _, _, digits = episode_id.rpartition("_")
    try:
        return int(digits)
    except ValueError:
        raise ConfigError(
            f"cannot read an episode index out of {episode_id!r}; expected the LeRobot "
            "reader's 'episode_NNNNNN' form. Without it a keyed join has no key."
        ) from None


__all__ = [
    "BIMANUAL_REF",
    "DESCRIBED",
    "FOLLOWER_REF",
    "OUTCOME_SUCCESS",
    "TIMESTAMP",
    "VERSION",
    "SO101Connector",
]
