"""The per-episode design metadata a corpus-factory recording writes beside it.

``CORPUS-PROTOCOL.md`` calls this list load-bearing and means it literally:
without ``session_id`` / ``block_id`` / ``episode_index_within_session`` the
randomised-block design cannot be *tested* afterwards — the time-trend covariate
has no x — and without the two calibration hashes a mid-campaign recalibration
is undetectable retrospectively. Both defects are only discoverable once the
teleop hours are already spent.

The producer is ``harness/so101/record_session.py``
(``metadata_columns()`` / ``write_episode_csv()``). Its column order is a
contract between two repositories, so this module states the columns it reads
and refuses on a malformed value rather than coercing one: a blank cell is
"unknown", a cell that will not parse is a defect, and the two must not become
the same thing.

What is *missing* and what is *wrong* are treated differently, on purpose:

* a **missing** sidecar, or a sidecar without the block columns, is readable.
  The corpus then declares that it cannot support the block analysis, so a
  caller is refused that analysis instead of being handed one computed from
  nothing (``supports_block_analysis`` and the descriptor's ``block_analysis``);
* a **conflicting** sidecar — a value that will not parse, a row count that does
  not match the dataset, two calibration digests in one corpus — is a refusal.

The join
--------
``metadata_columns()`` emits no LeRobot episode index. That is the one gap in
the contract between the recorder and this reader, and it matters more than it
looks: a join that is off by one attributes every episode to the wrong
condition, operator and session, which is the experiment's entire independent
variable, and nothing downstream can see it. So the join is either keyed
(an ``episode_index`` column) or ordinal-and-checked (row order against the
dataset's own frame counts), and an ordinal join that nothing can check is
refused rather than assumed. See :func:`join_rows`.
"""

from __future__ import annotations

import csv
import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from gantry.errors import ConfigError

#: Where the columns come from. Named so a reader of this file can find the
#: writer, and so a column-set drift has one place to be re-read from.
PRODUCER = "harness/so101/record_session.py: metadata_columns() / write_episode_csv()"

#: The file the recorder is expected to leave beside (or inside) the dataset.
#: Several spellings are accepted because the recorder takes the path as an
#: argument and campaigns name it differently; all of them are looked for, and
#: which one was found is recorded.
SIDECAR_NAMES = ("episodes.csv", "episodes_meta.csv", "episode_metadata.csv", "meta/episodes.csv")

#: Columns without which the confound analysis in CORPUS-PROTOCOL.md is not
#: possible. Their absence does not stop a corpus being read; it stops the
#: corpus claiming to support a block analysis.
BLOCK_ANALYSIS_COLUMNS = (
    "condition_label",
    "treatment",
    "operator_id",
    "session_id",
    "block_id",
    "episode_index_within_session",
    "order_seed",
)

#: Columns that carry the #3758 lesson. Without both, a recalibration inside a
#: campaign leaves no trace at all.
CALIBRATION_COLUMNS = ("calib_sha256_leader", "calib_sha256_follower")

#: Column names that may carry the LeRobot episode index, in preference order.
JOIN_COLUMNS = ("episode_index", "lerobot_episode_index")

#: Optional drift columns for a derived corpus (corpus axis 1). Absent on a
#: recorded corpus, and absence means "not declared", never "zero".
DRIFT_COLUMNS = ("drift_joint", "drift_deg", "drift_mode", "drift_ramp", "drift_mm")

_TRUE = {"true", "1", "yes", "t"}
_FALSE = {"false", "0", "no", "f", ""}


def _blank(value: str | None) -> bool:
    return value is None or value.strip() == ""


def _as_int(row: Mapping[str, str], column: str, where: str) -> int | None:
    raw = row.get(column)
    if _blank(raw):
        return None
    try:
        return int(str(raw).strip())
    except ValueError:
        raise ConfigError(
            f"{where}: column {column!r} holds {raw!r}, which is not an integer. "
            "Refusing rather than dropping it: this column is metadata the analysis "
            "keys on, and a silently missing key reads as a corpus that never had one."
        ) from None


def _as_float(row: Mapping[str, str], column: str, where: str) -> float | None:
    raw = row.get(column)
    if _blank(raw):
        return None
    try:
        return float(str(raw).strip())
    except ValueError:
        raise ConfigError(
            f"{where}: column {column!r} holds {raw!r}, which is not a number."
        ) from None


def _as_bool(row: Mapping[str, str], column: str, where: str) -> bool | None:
    raw = row.get(column)
    if raw is None:
        return None
    text = str(raw).strip().lower()
    if text in _TRUE:
        return True
    if text in _FALSE:
        return False if text else None
    raise ConfigError(f"{where}: column {column!r} holds {raw!r}, which is not a boolean")


def _as_str(row: Mapping[str, str], column: str) -> str | None:
    raw = row.get(column)
    return None if _blank(raw) else str(raw).strip()


def _pose(row: Mapping[str, str], columns: Sequence[str], where: str) -> tuple[float, ...] | None:
    """A pose, or ``None`` when no component was recorded.

    A partially recorded pose raises. Two of three coordinates is not a pose
    with a gap in it: ``score_tasks.zone_for_episode`` reads the components
    positionally, so a missing one shifts every later component into the wrong
    slot.
    """
    values = [_as_float(row, column, where) for column in columns]
    present = [value for value in values if value is not None]
    if not present:
        return None
    if len(present) != len(values):
        missing = [c for c, v in zip(columns, values) if v is None]
        raise ConfigError(
            f"{where}: pose {list(columns)} is missing component(s) {missing}. A pose is "
            "read positionally downstream, so a partial one silently shifts the remaining "
            "components into the wrong slots."
        )
    return tuple(present)


@dataclass(frozen=True)
class EpisodeMetadata:
    """One row of the recorder's per-episode CSV, typed.

    Every field is optional because a corpus may legitimately predate a column;
    what is *not* optional is that a present value parses. ``raw`` keeps the
    whole row so a column this reader has never heard of still reaches a record
    rather than being dropped on the floor.
    """

    corpus_id: str | None = None
    condition_label: str | None = None
    treatment: str | None = None
    operator_id: str | None = None
    session_id: str | None = None
    block_id: str | None = None
    episode_index_within_session: int | None = None
    started_at: str | None = None
    ended_at: str | None = None
    order_seed: int | None = None
    condition_order: tuple[str, ...] = ()
    pose_seed: int | None = None
    object_start_pose: tuple[float, ...] | None = None
    drawn_object_start_pose: tuple[float, ...] | None = None
    container_pose: tuple[float, ...] | None = None
    outcome: str | None = None
    episode_time_s: float | None = None
    control_hz: int | None = None
    frames_nominal: int | None = None
    num_frames: int | None = None
    early_exit: bool | None = None
    embodiment: str | None = None
    sim_commit: str | None = None
    lerobot_commit: str | None = None
    calibration: Mapping[str, str] = field(default_factory=dict)
    tracking_error: Mapping[str, Mapping[str, float]] = field(default_factory=dict)
    dropped_frames: Mapping[str, int] = field(default_factory=dict)
    fatigue: int | None = None
    notes: str | None = None
    drift: Mapping[str, Any] | None = None
    episode_index: int | None = None
    raw: Mapping[str, str] = field(default_factory=dict)

    @property
    def calibration_digest(self) -> str | None:
        """One id for both arms' calibration files, or ``None`` if either is absent.

        ``None`` is not a digest and never compares equal to one. A corpus that
        recorded no hashes must not pool silently with one that did — the whole
        point of the hash is that a recalibration is invisible in the numbers.
        """
        leader = self.calibration.get("leader")
        follower = self.calibration.get("follower")
        if not leader or not follower:
            return None
        payload = f"leader={leader}|follower={follower}"
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]

    def as_dict(self) -> dict[str, Any]:
        """The declaration as it rides in an episode record."""
        return {
            "corpus_id": self.corpus_id,
            "condition_label": self.condition_label,
            "treatment": self.treatment,
            "operator_id": self.operator_id,
            "session_id": self.session_id,
            "block_id": self.block_id,
            "episode_index_within_session": self.episode_index_within_session,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "order_seed": self.order_seed,
            "condition_order": list(self.condition_order),
            "pose_seed": self.pose_seed,
            "object_start_pose": list(self.object_start_pose or ()) or None,
            "drawn_object_start_pose": list(self.drawn_object_start_pose or ()) or None,
            "container_pose": list(self.container_pose or ()) or None,
            "outcome": self.outcome,
            "episode_time_s": self.episode_time_s,
            "control_hz": self.control_hz,
            "frames_nominal": self.frames_nominal,
            "num_frames": self.num_frames,
            "early_exit": self.early_exit,
            "embodiment": self.embodiment,
            "sim_commit": self.sim_commit,
            "lerobot_commit": self.lerobot_commit,
            "calibration_sha256": dict(self.calibration),
            "calibration_digest": self.calibration_digest,
            "tracking_error": {k: dict(v) for k, v in self.tracking_error.items()},
            "dropped_frames": dict(self.dropped_frames),
            "fatigue": self.fatigue,
            "notes": self.notes,
            "drift": dict(self.drift) if self.drift else None,
        }


@dataclass(frozen=True)
class Sidecar:
    """Every row of one recording's metadata, plus what it cannot support."""

    path: Path | None
    rows: tuple[EpisodeMetadata, ...]
    columns: tuple[str, ...]
    missing_block_columns: tuple[str, ...]
    missing_calibration_columns: tuple[str, ...]

    @property
    def present(self) -> bool:
        return self.path is not None

    @property
    def supports_block_analysis(self) -> bool:
        """Can the confound analysis in CORPUS-PROTOCOL.md be run on this corpus?

        A column being in the header is not enough: an empty ``session_id`` on
        every row is a header and no data. So this asks whether every row
        actually carries every block field.
        """
        if not self.present or not self.rows or self.missing_block_columns:
            return False
        return all(
            row.session_id
            and row.block_id
            and row.episode_index_within_session is not None
            and row.order_seed is not None
            and (row.condition_label or row.treatment)
            and row.operator_id
            for row in self.rows
        )

    @property
    def block_analysis(self) -> dict[str, Any]:
        """The declaration a caller reads instead of discovering this the hard way."""
        if not self.present:
            return {
                "supported": False,
                "why_not": (
                    "no per-episode metadata sidecar was found, so session, block and "
                    "within-session index are unknown. The randomised-block design cannot "
                    "be tested after the fact without them: the time-trend covariate has "
                    "no x. This corpus is readable and is not block-analysable."
                ),
                "missing_columns": list(BLOCK_ANALYSIS_COLUMNS),
                "producer": PRODUCER,
            }
        if self.supports_block_analysis:
            return {"supported": True, "missing_columns": [], "producer": PRODUCER}
        blank = sorted(
            {
                column
                for column in BLOCK_ANALYSIS_COLUMNS
                if column not in self.missing_block_columns
                and any(_blank(row.raw.get(column)) for row in self.rows)
            }
        )
        return {
            "supported": False,
            "why_not": (
                "the sidecar does not carry a complete block design for every episode "
                f"(absent columns: {list(self.missing_block_columns)}; present but blank "
                f"on some row: {blank}). Reported rather than filled in: a block analysis "
                "computed over episodes whose session is unknown is an answer to a "
                "different question."
            ),
            "missing_columns": list(self.missing_block_columns),
            "blank_columns": blank,
            "producer": PRODUCER,
        }

    @property
    def calibration_provenance(self) -> dict[str, Any]:
        digests = sorted({row.calibration_digest for row in self.rows if row.calibration_digest})
        unknown = sum(1 for row in self.rows if row.calibration_digest is None)
        return {
            "digests": digests,
            "episodes_without_a_digest": unknown,
            "columns_missing": list(self.missing_calibration_columns),
            "why_it_matters": (
                "LeRobot #3758: a ~17 deg calibration offset produced ~6 cm of gripper "
                "error, was invisible in the training loss because the policy learns the "
                "biased mapping, and left no trace in the numbers. The hash of both arms' "
                "calibration files is the only thing that makes a mid-campaign "
                "recalibration findable retrospectively."
            ),
        }


def find_sidecar(root: Path, names: Sequence[str] = SIDECAR_NAMES) -> Path | None:
    """The first metadata CSV present under a dataset root, or ``None``."""
    for name in names:
        candidate = root / name
        if candidate.is_file():
            return candidate
    return None


def read_sidecar(path: Path, *, joint_names: Sequence[str] = ()) -> Sidecar:
    """Parse one metadata CSV. ``joint_names`` says which tracking columns to expect.

    The tracking-error columns are named per joint (``trk_mean_shoulder_pan``),
    so which ones exist is a fact about the *embodiment* the corpus was recorded
    on. Taking them from the binding rather than from whatever the header
    happens to contain is what makes a six-joint sidecar beside a twelve-joint
    corpus visible instead of half-read.
    """
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        columns = tuple(reader.fieldnames or ())
        raw_rows = [dict(row) for row in reader]

    if not columns:
        raise ConfigError(
            f"{path}: the metadata sidecar has no header row. An empty sidecar and a "
            "sidecar whose columns were never written look identical to a reader that "
            "guesses; this one refuses."
        )

    rows = tuple(
        _parse_row(raw, f"{path}:{index + 2}", joint_names)
        for index, raw in enumerate(raw_rows)
    )
    return Sidecar(
        path=path,
        rows=rows,
        columns=columns,
        missing_block_columns=tuple(c for c in BLOCK_ANALYSIS_COLUMNS if c not in columns),
        missing_calibration_columns=tuple(c for c in CALIBRATION_COLUMNS if c not in columns),
    )


def absent_sidecar() -> Sidecar:
    """What a corpus with no metadata beside it looks like. Not an error."""
    return Sidecar(
        path=None,
        rows=(),
        columns=(),
        missing_block_columns=tuple(BLOCK_ANALYSIS_COLUMNS),
        missing_calibration_columns=tuple(CALIBRATION_COLUMNS),
    )


def _parse_row(raw: Mapping[str, str], where: str, joint_names: Sequence[str]) -> EpisodeMetadata:
    order = _as_str(raw, "condition_order")
    calibration = {
        arm: value
        for arm, column in (("leader", "calib_sha256_leader"), ("follower", "calib_sha256_follower"))
        for value in [_as_str(raw, column)]
        if value
    }
    for arm, digest in calibration.items():
        if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest.lower()):
            raise ConfigError(
                f"{where}: the {arm} calibration hash {digest!r} is not a 64-character "
                "SHA-256 digest. A truncated or renamed hash compares unequal to the same "
                "calibration recorded properly, which turns the recalibration check into "
                "a refusal on every corpus."
            )

    tracking: dict[str, dict[str, float]] = {}
    for joint in joint_names:
        base = joint[:-4] if joint.endswith(".pos") else joint
        mean = _as_float(raw, f"trk_mean_{base}", where)
        maximum = _as_float(raw, f"trk_max_{base}", where)
        if mean is None and maximum is None:
            continue
        if mean is None or maximum is None:
            raise ConfigError(
                f"{where}: joint {joint!r} has only one of trk_mean/trk_max. The pair is "
                "the leader-follower fidelity proxy and half of it is not a proxy: this "
                "rig has no force sensing anywhere, so tracking error is the only "
                "evidence of how well the demonstration was actually followed."
            )
        tracking[joint] = {"mean": mean, "max": maximum}

    dropped = {
        column[len("dropped_") :]: value
        for column in raw
        if column.startswith("dropped_")
        for value in [_as_int(raw, column, where)]
        if value is not None
    }

    drift = {
        column: raw[column].strip()
        for column in DRIFT_COLUMNS
        if column in raw and not _blank(raw.get(column))
    }

    episode_index = None
    for column in JOIN_COLUMNS:
        if column in raw:
            episode_index = _as_int(raw, column, where)
            if episode_index is not None:
                break

    return EpisodeMetadata(
        corpus_id=_as_str(raw, "corpus_id"),
        condition_label=_as_str(raw, "condition_label"),
        treatment=_as_str(raw, "treatment"),
        operator_id=_as_str(raw, "operator_id"),
        session_id=_as_str(raw, "session_id"),
        block_id=_as_str(raw, "block_id"),
        episode_index_within_session=_as_int(raw, "episode_index_within_session", where),
        started_at=_as_str(raw, "started_at"),
        ended_at=_as_str(raw, "ended_at"),
        order_seed=_as_int(raw, "order_seed", where),
        condition_order=tuple(part for part in (order or "").split("|") if part),
        pose_seed=_as_int(raw, "pose_seed", where),
        object_start_pose=_pose(raw, ("object_start_x", "object_start_y", "object_start_yaw"), where),
        drawn_object_start_pose=_pose(raw, ("drawn_start_x", "drawn_start_y", "drawn_start_yaw"), where),
        container_pose=_pose(raw, ("container_x", "container_y", "container_z"), where),
        outcome=_as_str(raw, "outcome"),
        episode_time_s=_as_float(raw, "episode_time_s", where),
        control_hz=_as_int(raw, "control_hz", where),
        frames_nominal=_as_int(raw, "frames_nominal", where),
        # `num_frames` is the recorder's own spelling for what the dataset
        # actually contains; `frames_recorded` is the same number under the
        # other module's name. Read whichever is present, prefer num_frames,
        # and refuse if the two disagree — a stale length mis-sizes every
        # frame-matched arm built from this corpus.
        num_frames=_frames(raw, where),
        early_exit=_as_bool(raw, "early_exit", where),
        embodiment=_as_str(raw, "embodiment"),
        sim_commit=_as_str(raw, "sim_commit"),
        lerobot_commit=_as_str(raw, "lerobot_commit"),
        calibration=calibration,
        tracking_error=tracking,
        dropped_frames=dropped,
        fatigue=_as_int(raw, "fatigue", where),
        notes=_as_str(raw, "notes"),
        drift=drift or None,
        episode_index=episode_index,
        raw=dict(raw),
    )


def _frames(raw: Mapping[str, str], where: str) -> int | None:
    declared = _as_int(raw, "num_frames", where)
    recorded = _as_int(raw, "frames_recorded", where)
    if declared is not None and recorded is not None and declared != recorded:
        raise ConfigError(
            f"{where}: num_frames={declared} and frames_recorded={recorded} disagree. "
            "They are the same quantity under two module's names, so a disagreement "
            "means one of them is stale — and a stale length silently mis-sizes every "
            "frame-matched training arm built from this corpus."
        )
    return declared if declared is not None else recorded


# --------------------------------------------------------------------------
# the join
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Join:
    """How sidecar rows were matched to episodes, and how well that was checked."""

    method: str
    by_episode_id: Mapping[str, EpisodeMetadata]
    verified_against: str
    note: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {"method": self.method, "verified_against": self.verified_against, "note": self.note}


def join_rows(
    sidecar: Sidecar,
    episode_ids: Sequence[str],
    lengths: Mapping[str, int],
    *,
    index_of: Any,
    method: str = "auto",
) -> Join:
    """Match rows to episodes, or refuse to guess.

    ``index_of`` maps an episode id to its LeRobot episode index.

    A misjoin is the worst failure this connector can have and the only one with
    no symptom: every episode keeps its numbers and acquires another episode's
    condition, operator and session. The experiment's independent variable is
    then wrong everywhere and every downstream check still passes. So there are
    exactly two accepted joins — a key, or file order cross-checked against the
    dataset's own frame counts — and when the cross-check cannot distinguish a
    correct order from a shuffled one, this raises instead.
    """
    if method not in ("auto", "index", "ordinal"):
        raise ConfigError(f"unknown join method {method!r}; expected auto, index or ordinal")
    rows = sidecar.rows
    keyed = all(row.episode_index is not None for row in rows) and bool(rows)

    if method == "index" and not keyed:
        raise ConfigError(
            f"{sidecar.path}: join='index' was asked for but no {JOIN_COLUMNS[0]!r} column "
            f"is present (columns: {list(sidecar.columns)}). {PRODUCER} does not currently "
            "emit one; add it, or pass join='ordinal' to accept file order."
        )
    if keyed and method != "ordinal":
        by_index = {row.episode_index: row for row in rows}
        if len(by_index) != len(rows):
            raise ConfigError(
                f"{sidecar.path}: the episode index column repeats a value, so it cannot "
                "be a join key."
            )
        mapping = {}
        for episode_id in episode_ids:
            row = by_index.pop(index_of(episode_id), None)
            if row is None:
                raise ConfigError(
                    f"{sidecar.path}: no metadata row for episode {episode_id!r}. A corpus "
                    "with metadata for some episodes and not others silently splits into a "
                    "labelled part and an unlabelled one that pools with it."
                )
            mapping[episode_id] = row
        if by_index:
            raise ConfigError(
                f"{sidecar.path}: {len(by_index)} metadata row(s) name episodes the dataset "
                f"does not contain ({sorted(by_index)[:5]}). Either the sidecar belongs to "
                "another block or episodes were deleted after recording."
            )
        return Join("index", mapping, verified_against=JOIN_COLUMNS[0])

    # -- ordinal ---------------------------------------------------------
    if len(rows) != len(episode_ids):
        raise ConfigError(
            f"{sidecar.path}: {len(rows)} metadata row(s) for {len(episode_ids)} episode(s). "
            "Without a join key the rows can only be matched by order, and an order join "
            "over unequal counts attributes every episode after the first gap to the wrong "
            "condition, operator and session."
        )
    ordered = _order_rows(rows)
    frame_counts = [row.num_frames for row in ordered]
    known = [count for count in frame_counts if count is not None]
    distinct = len(set(known))

    if len(known) == len(ordered) and distinct > 1:
        for episode_id, row in zip(episode_ids, ordered):
            if lengths.get(episode_id) != row.num_frames:
                raise ConfigError(
                    f"{sidecar.path}: ordinal join rejected. Row {row.session_id}/"
                    f"{row.episode_index_within_session} declares {row.num_frames} frames "
                    f"and episode {episode_id!r} contains {lengths.get(episode_id)}. The "
                    "rows are not in the dataset's order, so this join would have labelled "
                    "every episode with another episode's condition."
                )
        return Join(
            "ordinal",
            dict(zip(episode_ids, ordered)),
            verified_against=f"num_frames, {distinct} distinct value(s) across {len(ordered)} episodes",
        )

    if method != "ordinal":
        raise ConfigError(
            f"{sidecar.path}: refusing an unverifiable ordinal join. There is no "
            f"{JOIN_COLUMNS[0]!r} column, and the frame counts "
            + (
                f"are identical on all {len(ordered)} episodes"
                if distinct == 1
                else "are not recorded on every row"
            )
            + ", so nothing distinguishes the correct row order from a shuffled one. A "
            "misjoin has no symptom: every episode keeps its numbers and acquires another "
            f"episode's condition. Add an {JOIN_COLUMNS[0]!r} column to the sidecar "
            f"({PRODUCER}), or pass join='ordinal' to accept file order as the key."
        )
    return Join(
        "ordinal",
        dict(zip(episode_ids, ordered)),
        verified_against="nothing",
        note=(
            "file order was accepted as the join key on the caller's instruction; the "
            "frame counts could not distinguish it from a shuffled order, so the "
            "condition, operator and session on every episode rest on that instruction"
        ),
    )


def _order_rows(rows: Sequence[EpisodeMetadata]) -> tuple[EpisodeMetadata, ...]:
    """File order, unless the rows carry a within-session ordinal for every row.

    Sorting by ``(session_id, episode_index_within_session)`` is what the
    recorder's own ordering means; falling back to file order when any row lacks
    it keeps a partially indexed sidecar from being silently reordered.
    """
    if any(row.episode_index_within_session is None for row in rows):
        return tuple(rows)
    return tuple(sorted(rows, key=lambda row: (row.session_id or "", row.episode_index_within_session)))


__all__ = [
    "BLOCK_ANALYSIS_COLUMNS",
    "CALIBRATION_COLUMNS",
    "DRIFT_COLUMNS",
    "JOIN_COLUMNS",
    "PRODUCER",
    "SIDECAR_NAMES",
    "EpisodeMetadata",
    "Join",
    "Sidecar",
    "absent_sidecar",
    "find_sidecar",
    "join_rows",
    "read_sidecar",
]
