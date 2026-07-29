"""Every curation that was tested, and whether it worked.

This is the part that compounds. A curation signal is a commodity — there are
several good ones published and more coming — but *which signal works on which
kind of task, at which data regime* is not published anywhere, because
answering it requires running the loop many times and nobody's evaluation is
cheap enough to do that.

So the outcomes are kept, all of them, and the refuted ones especially. A ledger
holding only successes is a brochure: it cannot tell you that a method failed
four times on insertion tasks before working on a pick-and-place, which is
exactly the sentence worth having.

Addressed by content, so the same outcome recorded twice is one row and a run
that is re-judged does not silently duplicate. Written as one file per outcome
rather than one growing file, so two processes appending at once cannot corrupt
it and a diff shows what a session added.
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from .contracts.curation import CurationOutcome
from .spine import seed_from


@dataclass(frozen=True)
class Entry:
    """One tested plan, as it comes back off disk."""

    key: str
    signal: str
    rung: str
    held: bool
    delta: float
    p: float | None
    predicted: str
    summary: str
    tasks: tuple[str, ...]
    cost: Mapping[str, Any]
    raw: Mapping[str, Any]


class Ledger:
    """The record of what data interventions actually did."""

    def __init__(self, root: str | Path = "ledger"):
        self._root = Path(root)

    @property
    def root(self) -> Path:
        return self._root

    # -- writing -----------------------------------------------------------

    def record(self, outcome: CurationOutcome, **extra: Any) -> str:
        """Append one outcome. Returns its key; recording it twice is one row."""
        payload = dict(outcome.as_dict())
        payload["tasks"] = list(outcome.plan.predicted.tasks)
        payload.update(extra)
        # Keyed on what was done and what it was measured against, so a rerun
        # of the same experiment lands on the same row rather than inflating
        # the count of evidence.
        key = seed_from(
            outcome.plan.signal,
            outcome.plan.summary(),
            outcome.baseline_run,
            outcome.curated_run,
        )
        name = f"{outcome.plan.signal}-{key:016x}.json"
        self._root.mkdir(parents=True, exist_ok=True)
        (self._root / name).write_text(json.dumps(payload, indent=2) + "\n")
        return name

    # -- reading -----------------------------------------------------------

    def __iter__(self) -> Iterable[Entry]:
        if not self._root.exists():
            return iter(())
        entries = []
        for path in sorted(self._root.glob("*.json")):
            raw = json.loads(path.read_text())
            entries.append(
                Entry(
                    key=path.name,
                    signal=str(raw.get("signal", "?")),
                    rung=str(raw.get("rung", "?")),
                    held=bool(raw.get("held", False)),
                    delta=float((raw.get("delta") or {}).get("value", 0.0)),
                    p=raw.get("p"),
                    predicted=str(raw.get("predicted", "")),
                    summary=str(raw.get("summary", "")),
                    tasks=tuple(raw.get("tasks") or ()),
                    cost=raw.get("cost") or {},
                    raw=raw,
                )
            )
        return iter(entries)

    def __len__(self) -> int:
        return len(list(iter(self)))

    # -- what it is for ----------------------------------------------------

    def tested(self, signal: str) -> int:
        """How many plans this signal has already had tested.

        The number verification needs in order to correct for selection. A
        signal on its twentieth attempt is not making the same claim as one on
        its first, and this is the only place that fact is kept.
        """
        return sum(1 for entry in self if entry.signal == signal)

    def track_record(self) -> Mapping[str, Mapping[str, Any]]:
        """Per signal: how often it held, and by how much when it did.

        The table nobody else can produce, because producing it means having
        run the loop rather than having published once.
        """
        grouped: dict[str, list[Entry]] = defaultdict(list)
        for entry in self:
            grouped[entry.signal].append(entry)
        out = {}
        for signal, entries in sorted(grouped.items()):
            held = [e for e in entries if e.held]
            out[signal] = {
                "tested": len(entries),
                "held": len(held),
                "rate": round(len(held) / len(entries), 3) if entries else 0.0,
                "median_delta_when_held": (
                    round(sorted(e.delta for e in held)[len(held) // 2], 4) if held else None
                ),
                "rungs": sorted({e.rung for e in entries}),
                "gpu_minutes": sum(
                    float(e.cost.get("gpu_minutes", 0) or 0) for e in entries
                ),
            }
        return out

    def by_task(self) -> Mapping[str, Mapping[str, Any]]:
        """Per task: which signals have been tried, and which held.

        A signal that works on lifting and not on insertion is the finding this
        exists to surface, and it only appears once the same signal has been
        tested on both.
        """
        grouped: dict[str, list[Entry]] = defaultdict(list)
        for entry in self:
            for task in entry.tasks or ("(unscoped)",):
                grouped[task].append(entry)
        return {
            task: {
                "tested": len(entries),
                "held": sorted({e.signal for e in entries if e.held}),
                "refuted": sorted({e.signal for e in entries if not e.held}),
            }
            for task, entries in sorted(grouped.items())
        }

    def prior_for(self, signal: str, task: str | None = None) -> float | None:
        """How often this signal has held, here. ``None`` until there is evidence.

        Deliberately returns nothing rather than a default. A prior invented
        for a signal nobody has tested is the framework asserting something it
        does not know, and the whole point of the ledger is to stop guessing.
        """
        relevant = [
            e for e in self
            if e.signal == signal and (task is None or task in e.tasks)
        ]
        if not relevant:
            return None
        return round(sum(1 for e in relevant if e.held) / len(relevant), 3)

    def report(self) -> str:
        """The ledger as a page of text, for a terminal or a commit message."""
        entries = list(self)
        if not entries:
            return "ledger is empty: no curation plan has been tested yet"
        lines = [f"{len(entries)} curation plan(s) tested", ""]
        for signal, stats in self.track_record().items():
            median = stats["median_delta_when_held"]
            lines.append(
                f"  {signal:14} {stats['held']}/{stats['tested']} held"
                + (f", median {median:+.3f} when it did" if median is not None else "")
                + (f"  [{stats['gpu_minutes']:.0f} gpu-min]" if stats["gpu_minutes"] else "")
            )
        lines.append("")
        for task, stats in self.by_task().items():
            lines.append(
                f"  {task:20} held: {', '.join(stats['held']) or '-'}"
                f"   refuted: {', '.join(stats['refuted']) or '-'}"
            )
        return "\n".join(lines)
