"""Every run that was ever done, and what it teaches about what to do next.

:mod:`gantry.store` can write a run to disk and read it back unchanged. That is
persistence. This is the question one layer up: given everything already run,
what should the next decision be?

The gap this closes is small to describe and expensive to leave open. Until now
every evaluation produced a record and then dropped it — results lived in
whatever scratch directory a sweep happened to write to, the baseline a
comparison was measured against was remembered by a person, and the number of
attempts a signal had already made was an argument somebody had to get right. So
the layer measured well and remembered nothing, which means it could not get
better at asking.

Three specific things it fixes
------------------------------
**Baselines stop being folklore.** "Is this checkpoint worse than the last one"
requires knowing which run *was* the last one. Pinning makes that a fact on disk
rather than a claim in a commit message — and a pin rather than "most recent",
because the most recent run is sometimes the broken one and a regression gate
that silently re-baselines on failure is not a gate.

**Sizing stops needing a guess.** :func:`~gantry.spine.inference.trials_needed`
wants a baseline rate, and until now the caller invented one. After a few runs
on a task the history knows what that rate is and how much it moves, so the
question is answered from evidence. It returns nothing rather than a default: an
invented rate produces an invented trial count, and an invented trial count is
how an underpowered run gets approved.

**Selection corrects itself.** A signal on its twentieth attempt is not making
the same claim as one on its first, and the correction needs a count. Nobody
should have to remember the count.

What it deliberately does not do
--------------------------------
It does not decide what is comparable. Whether two runs may be averaged depends
on whether their provenance agrees on the axis under comparison, the feedback
plane already knows how to check that, and a second implementation here would
give two places to disagree. This answers questions about what exists.
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

from .spine import Measurement, RunRecord
from .store import read_run, write_run

#: Subdirectories under the history root.
RUNS, PINS = "runs", "pins"


@dataclass(frozen=True)
class Row:
    """One run, summarised for deciding what to do next.

    A summary rather than the full record: history is read to make a decision,
    and rehydrating megabytes of step arrays to answer "how many trials was
    that" is the kind of cost that makes people stop asking. The full record is
    still on disk and :meth:`History.record_for` will load it.
    """

    key: str
    task: str | None = None
    embodiment: str | None = None
    policy: str | None = None
    evaluation: str | None = None
    #: Which judge decided these labels. ``None`` means nobody recorded one,
    #: which is not the same as "the simulator did" and is kept distinguishable.
    scorer: str | None = None
    rate: float | None = None
    n: int | None = None
    trials: int = 0
    #: ``{scene_id: success}``, ``None`` where a trial errored. Kept because a
    #: paired comparison needs them and they are small.
    outcomes: Mapping[str, bool | None] = field(default_factory=dict)
    created_at: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def scored(self) -> int:
        return sum(1 for value in self.outcomes.values() if value is not None)


def summarise(record: RunRecord) -> dict[str, Any]:
    """The parts of a run worth keeping for a later decision."""
    provenance = getattr(record, "provenance", None)
    components = {
        component.plane: component.name
        for component in (getattr(provenance, "components", ()) or ())
    }
    # The version travels too, in a separate field. Filtering on "the policy
    # called ph_official" is the common question; pinning the exact build is
    # the provenance question, and conflating them makes the first one fail
    # silently for anyone who did not memorise a version string.
    refs = {
        f"{component.plane}_ref": f"{component.name}@{component.version}"
        for component in (getattr(provenance, "components", ()) or ())
    }
    outcomes: dict[str, Any] = {}
    task: str | None = None
    for episode in getattr(record, "episodes", ()) or ():
        meta = getattr(episode, "meta", None)
        labels = getattr(episode, "labels", None)
        scene = str(getattr(meta, "id", "") or len(outcomes))
        success = getattr(labels, "success", None)
        outcomes[scene] = None if success is None else bool(success)
        task = task or getattr(meta, "task", None)

    rate = (getattr(record, "metrics", {}) or {}).get("success_rate")
    return {
        "task": task or components.get("task"),
        "embodiment": components.get("embodiment"),
        "policy": components.get("policy"),
        "evaluation": components.get("evaluation"),
        "scorer": components.get("scorer"),
        "rate": None if rate is None else float(getattr(rate, "value", rate)),
        "n": None if rate is None else getattr(rate, "n", None),
        "trials": len(outcomes),
        "outcomes": outcomes,
        "created_at": getattr(provenance, "created_at", None),
        "protocol": dict(getattr(provenance, "protocol", {}) or {}),
        **refs,
    }


_SUMMARY_FIELDS = {
    "task", "embodiment", "policy", "evaluation", "scorer",
    "rate", "n", "trials", "outcomes", "created_at",
}


class History:
    """What has been run, and the priors that fall out of it.

    Content-addressed: a run's key is a hash of its summary, so recording the
    same run twice is one row and a re-judged run does not silently duplicate.
    One file per run rather than one growing index, so two processes writing at
    once cannot corrupt it and a diff shows what a session added — the same
    choice the ledger makes, for the same reasons.
    """

    def __init__(self, root: str | Path = "history"):
        self._root = Path(root)

    @property
    def root(self) -> Path:
        return self._root

    # -- writing -----------------------------------------------------------

    def put(self, record: RunRecord, *, keep_record: bool = True, **extra: Any) -> str:
        """Index a run, and by default keep the full record beside the summary."""
        payload = summarise(record)
        payload.update(extra)
        key = _key_for(payload)
        directory = self._root / RUNS
        directory.mkdir(parents=True, exist_ok=True)
        (directory / f"{key}.json").write_text(json.dumps(payload, indent=2) + "\n")
        if keep_record:
            # Through the versioned writer, so the archived copy round-trips.
            write_run(record, directory / f"{key}.run.json")
        return key

    def record_for(self, key: str) -> RunRecord | None:
        """The full run behind a summary, if it was kept."""
        path = self._root / RUNS / f"{key}.run.json"
        return read_run(path) if path.exists() else None

    def pin(self, key: str, *, task: str, embodiment: str | None = None) -> Path:
        """Make this run the reference for its task and body."""
        directory = self._root / PINS
        directory.mkdir(parents=True, exist_ok=True)
        target = directory / f"{_pin_name(task, embodiment)}.json"
        target.write_text(
            json.dumps({"task": task, "embodiment": embodiment, "run": key}, indent=2)
            + "\n"
        )
        return target

    # -- reading -----------------------------------------------------------

    def __iter__(self) -> Iterator[Row]:
        directory = self._root / RUNS
        if not directory.exists():
            return iter(())
        rows = []
        for path in sorted(directory.glob("*.json")):
            if path.name.endswith(".run.json"):
                continue
            raw = json.loads(path.read_text())
            rows.append(
                Row(
                    key=path.stem,
                    task=raw.get("task"),
                    embodiment=raw.get("embodiment"),
                    policy=raw.get("policy"),
                    evaluation=raw.get("evaluation"),
                    scorer=raw.get("scorer"),
                    rate=raw.get("rate"),
                    n=raw.get("n"),
                    trials=int(raw.get("trials") or 0),
                    outcomes=raw.get("outcomes") or {},
                    created_at=raw.get("created_at"),
                    metadata={k: v for k, v in raw.items() if k not in _SUMMARY_FIELDS},
                )
            )
        return iter(rows)

    def __len__(self) -> int:
        return len(list(iter(self)))

    def get(self, key: str) -> Row | None:
        return next((row for row in self if row.key == key), None)

    def query(self, **filters: Any) -> tuple[Row, ...]:
        """Runs matching every filter given. ``None`` values do not filter."""
        wanted = {k: v for k, v in filters.items() if v is not None}
        unknown = set(wanted) - {f.name for f in Row.__dataclass_fields__.values()}
        if unknown:
            raise ValueError(f"history has no field(s) {sorted(unknown)}")
        return tuple(
            row
            for row in self
            if all(getattr(row, key) == value for key, value in wanted.items())
        )

    # -- what it is for ----------------------------------------------------

    def baseline_for(self, task: str, embodiment: str | None = None) -> Row | None:
        """The pinned reference for this task and body, falling back to any body."""
        for name in (_pin_name(task, embodiment), _pin_name(task, None)):
            path = self._root / PINS / f"{name}.json"
            if path.exists():
                return self.get(str(json.loads(path.read_text()).get("run", "")))
        return None

    def rate_for(
        self, task: str, embodiment: str | None = None
    ) -> Measurement | None:
        """What success rate this task has historically produced.

        The number sizing needs, from evidence rather than from whoever is
        typing. ``None`` until there is evidence.
        """
        rates = [
            row.rate
            for row in self.query(task=task, embodiment=embodiment)
            if row.rate is not None
        ]
        if not rates:
            return None
        mean = sum(rates) / len(rates)
        spread = (
            (sum((r - mean) ** 2 for r in rates) / (len(rates) - 1)) ** 0.5
            if len(rates) > 1
            else None
        )
        return Measurement(
            value=round(mean, 4),
            n=sum(
                row.trials
                for row in self.query(task=task, embodiment=embodiment)
                if row.rate is not None
            ),
            method=f"mean of {len(rates)} recorded run(s)",
            detail={
                "runs": len(rates),
                "spread": None if spread is None else round(spread, 4),
                "min": round(min(rates), 4),
                "max": round(max(rates), 4),
            },
        )

    def attempts(self, **filters: Any) -> int:
        """How many runs already match these filters.

        The count a selection correction needs. Asked of history rather than
        supplied by a caller, because the caller's memory is exactly what makes
        "best of twelve checkpoints" get reported as a finding.
        """
        return len(self.query(**filters))

    def paired(self, left: str, right: str) -> tuple[int, int, int]:
        """Wins for ``right``, wins for ``left``, and shared scenes.

        Keyed on scene identity rather than position: two runs that skipped
        different trials line up wrongly otherwise, and the pairing is the whole
        source of a paired test's power.
        """
        a, b = self.get(left), self.get(right)
        if a is None or b is None:
            return (0, 0, 0)
        shared = [
            scene
            for scene, value in a.outcomes.items()
            if value is not None and b.outcomes.get(scene) is not None
        ]
        right_wins = sum(1 for s in shared if b.outcomes[s] and not a.outcomes[s])
        left_wins = sum(1 for s in shared if a.outcomes[s] and not b.outcomes[s])
        return (right_wins, left_wins, len(shared))

    def matrix_for(self, policy: str) -> tuple[tuple[float, ...], ...]:
        """This policy's rates, grouped task by task.

        The shape the aggregates want. Grouping lives here rather than in
        :mod:`~gantry.spine.inference` because it is a question about
        provenance, and the statistics deliberately know nothing about that.
        """
        by_task: dict[str, list[float]] = defaultdict(list)
        for row in self.query(policy=policy):
            if row.rate is not None:
                by_task[row.task or "(untasked)"].append(row.rate)
        return tuple(tuple(values) for _, values in sorted(by_task.items()))

    # -- reporting ---------------------------------------------------------

    def report(self) -> str:
        rows = list(self)
        if not rows:
            return "history is empty: no run has been recorded yet"
        grouped: dict[str, list[Row]] = defaultdict(list)
        for row in rows:
            grouped[row.task or "(untasked)"].append(row)
        lines = [f"{len(rows)} run(s) recorded", ""]
        for task, group in sorted(grouped.items()):
            rate = self.rate_for(task)
            pinned = self.baseline_for(task)
            summary = (
                f"rate {rate.value:.0%} over {rate.detail['runs']} run(s)"
                if rate
                else "nothing scored"
            )
            lines.append(
                f"  {task:24} {len(group):3} run(s)   {summary}"
                + (f"   [baseline {pinned.key[:8]}]" if pinned else "")
            )
        unscored = [row for row in rows if row.scorer is None]
        if unscored:
            lines += [
                "",
                f"  {len(unscored)} run(s) name no scorer, so their labels record "
                "who decided nowhere",
            ]
        return "\n".join(lines)


def _key_for(payload: Mapping[str, Any]) -> str:
    """A content address for a run summary.

    blake2b at 8 bytes rather than :func:`~gantry.spine.seed_from`, which is
    deliberately 32-bit because its job is seeding a generator. Sixty-four bits
    is the right size for an address that has to stay unique across years of
    runs, and a collision here would silently merge two different experiments.
    """
    encoded = json.dumps(payload, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.blake2b(encoded.encode(), digest_size=8).hexdigest()


def _pin_name(task: str, embodiment: str | None) -> str:
    return f"{task}__{embodiment or 'any'}".replace("/", "_")


def matrix_of(rows: Sequence[Row]) -> tuple[tuple[float, ...], ...]:
    """Rows grouped into the task-by-runs matrix the aggregates take."""
    by_task: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        if row.rate is not None:
            by_task[row.task or "(untasked)"].append(row.rate)
    return tuple(tuple(values) for _, values in sorted(by_task.items()))
