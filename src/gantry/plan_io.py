"""A curation plan as a file.

The plan has to survive leaving the process that proposed it, for a reason that
is the whole point of the design rather than a convenience: what gets applied,
what gets verified, and what the ledger names must be *the same object*. If the
plan were re-derived at each step from the signal that produced it, three
things would exist that merely agreed at the time -- and the one that eventually
disagreed would do so silently, months later, when the dataset had changed
underneath it.

So a plan is written once, and every later step reads that file.

Round-tripping is checked in the tests rather than assumed. A field silently
dropped here is a claim that stops travelling: the evidence seeds are the
obvious case, since losing them turns a leakage refusal into a clean pass.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from .contracts.curation import (
    CollectionOrder,
    CurationAction,
    CurationPlan,
    Prediction,
)


def order_to_dict(order: CollectionOrder) -> dict[str, Any]:
    return {
        "task": order.task,
        "n": order.n,
        "seeds": list(order.seeds),
        "stage": order.stage,
        "note": order.note,
    }


def order_from_dict(raw: Mapping[str, Any]) -> CollectionOrder:
    return CollectionOrder(
        task=str(raw["task"]),
        n=int(raw["n"]),
        seeds=tuple(int(s) for s in raw.get("seeds") or ()),
        stage=raw.get("stage"),
        note=str(raw.get("note", "")),
    )


def action_to_dict(action: CurationAction) -> dict[str, Any]:
    out: dict[str, Any] = {"kind": action.kind}
    if action.episodes:
        out["episodes"] = list(action.episodes)
    if action.cohort is not None:
        out["cohort"] = action.cohort
    if action.weight is not None:
        out["weight"] = action.weight
    if action.order is not None:
        out["order"] = order_to_dict(action.order)
    if action.detail:
        out["detail"] = dict(action.detail)
    return out


def action_from_dict(raw: Mapping[str, Any]) -> CurationAction:
    return CurationAction(
        kind=str(raw["kind"]),
        episodes=tuple(str(e) for e in raw.get("episodes") or ()),
        cohort=raw.get("cohort"),
        weight=raw.get("weight"),
        order=order_from_dict(raw["order"]) if raw.get("order") else None,
        detail=dict(raw.get("detail") or {}),
    )


def plan_to_dict(plan: CurationPlan) -> dict[str, Any]:
    return {
        "signal": plan.signal,
        "rung": plan.rung,
        "predicted": {
            "metric": plan.predicted.metric,
            "direction": plan.predicted.direction,
            "magnitude": plan.predicted.magnitude,
            "tasks": list(plan.predicted.tasks),
        },
        "actions": [action_to_dict(action) for action in plan.actions],
        # Written even when empty, so a reader can tell "this signal read no
        # rollouts" from "somebody's serialiser dropped the field".
        "evidence_seeds": list(plan.evidence_seeds),
        "metadata": dict(plan.metadata),
    }


def plan_from_dict(raw: Mapping[str, Any]) -> CurationPlan:
    predicted = raw.get("predicted") or {}
    return CurationPlan(
        actions=tuple(action_from_dict(a) for a in raw.get("actions") or ()),
        signal=str(raw.get("signal", "")),
        rung=str(raw.get("rung", "screening")),
        predicted=Prediction(
            metric=str(predicted.get("metric", "success_rate")),
            direction=str(predicted.get("direction", "+")),
            magnitude=float(predicted.get("magnitude", 0.0)),
            tasks=tuple(predicted.get("tasks") or ()),
        ),
        evidence_seeds=tuple(int(s) for s in raw.get("evidence_seeds") or ()),
        metadata=dict(raw.get("metadata") or {}),
    )


def write_plan(plan: CurationPlan, path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(plan_to_dict(plan), indent=2) + "\n")
    return target


def read_plan(path: str | Path) -> CurationPlan:
    return plan_from_dict(json.loads(Path(path).read_text()))
