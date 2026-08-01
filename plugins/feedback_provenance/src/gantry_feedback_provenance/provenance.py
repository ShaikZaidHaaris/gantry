"""What this dataset may legally be used for, traced through everything that made it.

Every other module in the feedback plane asks whether a number is true. This one
asks whether the dataset is *yours*, and it is here because that question has a
specific and expensive way of going unanswered.

The best monocular hand reconstructions available — HaWoR, WiLoR, anything built
on MANO — are CC-BY-NC-ND and registration-gated. The best monocular depth models
are split: one checkpoint Apache, the next CC-BY-NC, same repository, same
filename pattern. Nothing about the arrays they return records which one produced
them. So a training set built through a research-only model is indistinguishable
from a clean one in every file it contains, and the discovery happens at legal
review, after the model is trained and the customer is waiting.

Licences do not merge, they dominate
------------------------------------
A dataset derived through five components carries the *most restrictive* licence
in the chain, not the average and not the one that appears most often. One
CC-BY-NC step makes the whole thing non-commercial no matter how permissive
everything else was. That is why this walks the lineage rather than reading a
field: the encumbrance is usually four steps upstream of the file somebody is
looking at.

Unknown is worse than restricted
--------------------------------
A component that declares CC-BY-NC is a known cost — it can be swapped, or the
dataset can be marked research-only and used honestly. A component that declares
nothing cannot be planned around at all: it might be fine, and nobody can act on
"might". So ``unknown`` is reported at a *higher* severity than
``non_commercial``, which surprises people and is the right way round.

What this is not
----------------
Legal advice, and it says so. It reports what the components declared and what
that implies under the ordinary reading of those licences. A lawyer decides;
this makes sure they are shown the whole chain rather than the last file in it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

from gantry.contracts.feedback import (
    Cohort,
    FeedbackModule,
    Finding,
    Report,
    feedback_descriptor,
)
from gantry.resolve import Requirement, requires_channels
from gantry.spine import Descriptor, Measurement

VERSION = "0.1.0.dev0"

#: How restrictive a licence is. Higher dominates; the chain takes the maximum.
#:
#: ``unknown`` sits above ``non_commercial`` on purpose. A declared restriction
#: is a cost you can plan around — swap the component, or ship the dataset as
#: research-only and say so. A missing declaration cannot be planned around at
#: all, and "probably fine" is not a position anybody can defend later.
PERMISSIVE, ATTRIBUTION, NON_COMMERCIAL, UNKNOWN = 0, 1, 2, 3

RANK = {
    PERMISSIVE: "permissive",
    ATTRIBUTION: "attribution required",
    NON_COMMERCIAL: "non-commercial",
    UNKNOWN: "undeclared",
}

#: Patterns matched against a declared licence string, most specific first.
#: Deliberately a small, readable table rather than a licence-parsing library:
#: what matters here is being *wrong loudly* rather than subtly right, and a
#: string this code does not recognise becomes ``unknown``, which is the
#: conservative answer.
PATTERNS: tuple[tuple[str, int], ...] = (
    (r"\bcc[- ]?by[- ]?nc", NON_COMMERCIAL),
    (r"\bnon[- ]?commercial\b", NON_COMMERCIAL),
    (r"\bresearch[- ]only\b", NON_COMMERCIAL),
    (r"\bmano\b", NON_COMMERCIAL),
    (r"\bcc[- ]?by[- ]?sa", ATTRIBUTION),
    (r"\bcc[- ]?by\b", ATTRIBUTION),
    (r"\bgpl\b|\bagpl\b", ATTRIBUTION),
    (r"\bapache\b", PERMISSIVE),
    (r"\bmit\b", PERMISSIVE),
    (r"\bbsd\b", PERMISSIVE),
    (r"\bcc0\b|\bpublic domain\b", PERMISSIVE),
)

#: Keys anywhere in an episode's metadata that name something with a licence.
#: Open-ended by design — a new component that records its licence under a new
#: key should be picked up without editing this module.
LICENCE_KEYS = ("licence", "license", "_licence", "_license")


@dataclass(frozen=True)
class Claim:
    """One component's declared licence, and where it was declared."""

    where: str
    text: str
    rank: int

    @property
    def restriction(self) -> str:
        return RANK[self.rank]


def classify(text: str | None) -> int:
    """How restrictive a declared licence string is.

    Anything unrecognised is :data:`UNKNOWN` rather than assumed permissive. The
    asymmetry matters: reading an unfamiliar licence as permissive produces a
    confident wrong answer, and reading it as unknown produces a question.
    """
    if not text or not str(text).strip():
        return UNKNOWN
    lowered = str(text).lower()
    for pattern, rank in PATTERNS:
        if re.search(pattern, lowered):
            return rank
    return UNKNOWN


def claims_in(episode: Any) -> tuple[Claim, ...]:
    """Every licence this episode or its metadata declares.

    Reads the episode's own ``license`` field and any licence-shaped key in its
    metadata or annotations, because the components that produced it record
    theirs in different places and no single field is authoritative.
    """
    out: list[Claim] = []
    meta = getattr(episode, "meta", None)
    declared = getattr(meta, "license", None)
    if declared:
        out.append(Claim("episode", str(declared), classify(declared)))
    sources: list[tuple[str, Mapping[str, Any]]] = []
    if meta is not None:
        sources.append(("meta", dict(getattr(meta, "extra", {}) or {})))
    labels = getattr(episode, "labels", None)
    if labels is not None:
        sources.append(("annotations", dict(getattr(labels, "annotations", {}) or {})))
    for where, mapping in sources:
        for key, value in mapping.items():
            if any(name in str(key).lower() for name in LICENCE_KEYS) and value:
                out.append(Claim(f"{where}.{key}", str(value), classify(value)))
    return tuple(out)


def worst(claims: Iterable[Claim]) -> int:
    """The governing restriction. Licences dominate rather than average."""
    ranks = [claim.rank for claim in claims]
    return max(ranks) if ranks else UNKNOWN


class Provenance(FeedbackModule):
    """What a dataset may be used for, from what everything upstream declared."""

    def __init__(self, *, intent: str = "commercial", require_declared: bool = True):
        """``intent`` is what the dataset is *for*, and it changes the verdict.

        A research benchmark is unbothered by a CC-BY-NC component; a product is
        stopped by it. Stating the intent is what lets one module serve both
        without either getting an answer shaped for the other.
        """
        if intent not in ("commercial", "research"):
            raise ValueError(f"intent must be 'commercial' or 'research', got {intent!r}")
        self._intent = intent
        self._require = bool(require_declared)

    def descriptor(self) -> Descriptor:
        return feedback_descriptor(
            name="provenance",
            version=VERSION,
            min_cohorts=1,
            prescribes=True,
            holds=(),
            intent=self._intent,
            # Said in the descriptor because it will end up quoted in a report
            # and somebody will otherwise take it for a legal opinion.
            disclaimer="reports what components declared; not legal advice",
        )

    def requirement(self) -> Requirement:
        return requires_channels(
            "provenance",
            "feedback",
            description="licences declared by the components that produced these episodes",
        )

    def analyse(self, cohorts: Sequence[Cohort]) -> Report:
        findings: list[Finding] = []
        measurements: dict[str, Measurement] = {}
        notes: list[str] = []

        for cohort in cohorts:
            claims = [claim for episode in cohort.episodes for claim in claims_in(episode)]
            if not claims:
                findings.append(
                    Finding(
                        code="provenance.nothing_declared",
                        summary=(
                            f"{cohort.name} carries no licence information at all, so "
                            "what it may be used for is unknown rather than unrestricted"
                        ),
                        severity="strong",
                        prescription=(
                            "Every component that touches the data should record its "
                            "licence, and the connectors here already do. An episode "
                            "with none came from somewhere that does not, and that "
                            "component is the one to find."
                        ),
                        cohorts=(cohort.name,),
                    )
                )
                continue

            governing = worst(claims)
            by_rank: dict[int, list[Claim]] = {}
            for claim in claims:
                by_rank.setdefault(claim.rank, []).append(claim)
            distinct = {claim.text: claim for claim in claims}  # one row per distinct declaration
            measurements[f"{cohort.name}.restriction"] = Measurement(
                value=float(governing),
                n=len(distinct),
                method="most restrictive licence in the lineage",
                detail={
                    "governing": RANK[governing],
                    "components": len(distinct),
                    "declared": sorted(distinct),
                },
            )
            notes.append(
                f"{cohort.name}: {len(distinct)} distinct declaration(s), governing "
                f"restriction {RANK[governing]!r}"
            )
            findings.extend(self._findings_for(cohort, governing, by_rank, sorted(distinct)))

        return Report(
            module="provenance",
            findings=tuple(findings),
            measurements=measurements,
            notes=tuple(notes),
            cohorts=tuple(cohort.name for cohort in cohorts),
        )

    # -- one cohort's verdict ---------------------------------------------

    def _findings_for(
        self,
        cohort: Cohort,
        governing: int,
        by_rank: Mapping[int, list[Claim]],
        declared: Sequence[str],
    ) -> list[Finding]:
        findings: list[Finding] = []
        commercial = self._intent == "commercial"

        if UNKNOWN in by_rank:
            offenders = sorted({claim.where for claim in by_rank[UNKNOWN]})
            unrecognised = sorted({claim.text for claim in by_rank[UNKNOWN]})
            findings.append(
                Finding(
                    code="provenance.undeclared",
                    # Above non-commercial on purpose. A stated restriction can
                    # be planned around; an unstated one cannot.
                    severity="strong",
                    summary=(
                        f"{cohort.name}: {len(offenders)} component(s) declare a licence "
                        "this module does not recognise, or none at all"
                    ),
                    evidence={"where": offenders[:8], "text": unrecognised[:8]},
                    prescription=(
                        "Find out what these permit before anything is built on them. "
                        "An unrecognised licence is not a small risk. It is an "
                        "unbounded one, because it cannot be swapped, priced, or "
                        "disclosed until somebody reads it."
                    ),
                    cohorts=(cohort.name,),
                )
            )

        if NON_COMMERCIAL in by_rank:
            offenders = sorted({claim.text for claim in by_rank[NON_COMMERCIAL]})
            findings.append(
                Finding(
                    code="provenance.non_commercial",
                    severity="strong" if commercial else "info",
                    summary=(
                        f"{cohort.name} was produced through non-commercial "
                        f"component(s) ({', '.join(offenders[:3])}"
                        + (f" and {len(offenders) - 3} more" if len(offenders) > 3 else "")
                        + ")"
                        + (
                            ", so the dataset itself cannot be used commercially"
                            if commercial
                            else "; fine for research use"
                        )
                    ),
                    evidence={
                        "non_commercial": offenders,
                        "where": sorted({c.where for c in by_rank[NON_COMMERCIAL]})[:8],
                    },
                    prescription=(
                        "Swap the offending component for a permissive one and rebuild. "
                        "The pipeline is designed for exactly this: the estimator is an "
                        "argument, and what changes with it is recorded. Rebuilding is "
                        "cheaper than the alternative, which is discovering this after "
                        "a model has been trained and shipped."
                        if commercial
                        else "Nothing to do for research use. Record it, because a "
                        "dataset that later becomes a product inherits this."
                    ),
                    cohorts=(cohort.name,),
                )
            )

        if ATTRIBUTION in by_rank and governing == ATTRIBUTION:
            findings.append(
                Finding(
                    code="provenance.attribution",
                    severity="weak",
                    summary=(
                        f"{cohort.name} may be used commercially with attribution to "
                        f"{len(by_rank[ATTRIBUTION])} component(s)"
                    ),
                    evidence={"attribution": sorted({c.text for c in by_rank[ATTRIBUTION]})},
                    prescription=(
                        "Carry the attributions into whatever ships. They are a "
                        "condition of use rather than a courtesy."
                    ),
                    cohorts=(cohort.name,),
                )
            )

        if governing == PERMISSIVE:
            findings.append(
                Finding(
                    code="provenance.clean",
                    severity="info",
                    summary=(
                        f"{cohort.name} is permissively licensed throughout "
                        f"({len(declared)} declaration(s)), so it may be used commercially"
                    ),
                    evidence={"declared": list(declared)},
                    cohorts=(cohort.name,),
                )
            )
        return findings


def chain(episode: Any) -> list[dict[str, Any]]:
    """One episode's declarations as plain rows, for a report table."""
    return [
        {"where": claim.where, "licence": claim.text, "restriction": claim.restriction}
        for claim in claims_in(episode)
    ]


def usable_for(cohort: Cohort) -> str:
    """One word for what this cohort may be used for. For a GUI badge."""
    claims = [claim for episode in cohort.episodes for claim in claims_in(episode)]
    return RANK[worst(claims)] if claims else RANK[UNKNOWN]
