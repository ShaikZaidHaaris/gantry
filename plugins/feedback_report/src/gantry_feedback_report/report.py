"""The document a contributor reads, assembled from what the modules concluded.

Every module in the feedback plane produces ``Finding`` objects with codes,
severities and prescriptions. Nothing turned them into something a person could
read, so this does — and the ordering it imposes is the whole design, because a
report that puts the numbers first is a report that gets misread.

The order is a claim about what invalidates what
------------------------------------------------
1. **Can this be used at all** — licence. A dataset built through non-commercial
   weights is not a data-quality question, it is a stop. Putting it after the
   results means somebody reads the results first and remembers those.

2. **Is the question well posed** — coverage. Kitchen footage measured on
   tabletop tasks produces a real number about nothing. The delta below is
   meaningless until this passes, so it comes before the delta.

3. **Did the data carry information** — the control. Not "did the number move",
   which fine-tuning does regardless.

4. **How much, and is it real** — the delta, with its interval and its power.

5. **What to change** — filming advice for the contributor, extraction findings
   for us, kept apart because they have different owners.

6. **What we could not tell you** — every abstention, named. This section is not
   an appendix. A report whose refusals are buried reads as more confident than
   it is, and the refusals are the reason to trust the rest of it.

Two rules the assembler enforces
--------------------------------
A blocking finding suppresses the sections it invalidates. If the licence is
encumbered or the coverage is a mismatch, the delta is still *computed* and is
still in the record — it is not presented as an answer, because it is not one.

And nothing is upgraded in transit. The assembler cannot turn a ``weak`` finding
into a headline; it can only order, group and quote. Every sentence in the output
is traceable to a module that was willing to sign it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

VERSION = "0.1.0.dev0"

#: The sections, in the order a reader should meet them, with the codes that
#: belong to each. A code from a module nobody has written yet lands in
#: ``other`` rather than being dropped.
SECTIONS: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    (
        "usable",
        "Can this be used",
        ("provenance.",),
    ),
    (
        "scope",
        "Is this the right question",
        ("coverage.",),
    ),
    (
        "signal",
        "Did the data carry information",
        ("control.",),
    ),
    (
        "effect",
        "How much it moved, and whether that is real",
        ("compare.", "power.", "rank."),
    ),
    (
        "filming",
        "What to change about the filming",
        ("capture.",),
    ),
    (
        "pipeline",
        "What we should change on our side",
        ("extraction.",),
    ),
)

#: Findings that stop the sections after them from being read as answers.
#: Deliberately short: a blocker is a claim that everything downstream is
#: uninterpretable, and that is a strong thing to say.
#:
#: A code appearing here is necessary and not sufficient — the module must also
#: have marked the finding ``strong``. Severity is the module's own judgement of
#: how much its finding matters, and the assembler is not entitled to overrule
#: it. The case that forced this: provenance emits ``non_commercial`` for a
#: research-intent run at ``info``, because for research it is a note rather
#: than a stop, and blocking on the code alone made every research report lead
#: with a blocker that its own module had said was fine.
BLOCKING = {
    "provenance.non_commercial": "the dataset cannot be used commercially",
    "provenance.undeclared": "what the dataset may be used for is unknown",
    "provenance.nothing_declared": "no component declared a licence",
    "coverage.mismatch": "the evaluated tasks are not what this data is about",
    "control.control_wins": "the control beat the real data, which means something is wrong",
    "control.no_control": "there is no control, so nothing here is attributable to the data",
}

#: Sections a blocker makes uninterpretable. Everything from here on is recorded
#: and not presented as an answer.
BLOCKS = {
    "usable": ("signal", "effect"),
    "scope": ("signal", "effect"),
    "signal": ("effect",),
}


@dataclass(frozen=True)
class Section:
    title: str
    key: str
    findings: tuple[Any, ...] = ()
    suppressed_by: str | None = None

    @property
    def blocked(self) -> bool:
        return self.suppressed_by is not None


@dataclass
class Assembled:
    """The whole report, as data. Rendering is a separate and dumber problem."""

    dataset: str
    sections: tuple[Section, ...]
    blockers: tuple[tuple[str, str], ...] = ()
    abstentions: tuple[str, ...] = ()
    measurements: Mapping[str, Any] = field(default_factory=dict)
    notes: tuple[str, ...] = ()

    @property
    def verdict(self) -> str:
        """One line. The thing that goes at the top and in an email subject."""
        if self.blockers:
            return self.blockers[0][1]
        signal = self._first("control.data_carried_information")
        if signal:
            return "the data carried information beyond what fine-tuning alone provides"
        if self._first("control.not_separated"):
            return "not separated from its control at this number of trials"
        return "no conclusion was reached"

    def _first(self, code: str) -> Any | None:
        for section in self.sections:
            for finding in section.findings:
                if getattr(finding, "code", "") == code:
                    return finding
        return None

    def as_dict(self) -> dict[str, Any]:
        return {
            "dataset": self.dataset,
            "verdict": self.verdict,
            "blockers": [{"code": c, "why": w} for c, w in self.blockers],
            "sections": [
                {
                    "key": s.key,
                    "title": s.title,
                    "blocked_by": s.suppressed_by,
                    "findings": [
                        {
                            "code": getattr(f, "code", ""),
                            "severity": getattr(f, "severity", "info"),
                            "summary": getattr(f, "summary", ""),
                            "prescription": getattr(f, "prescription", None),
                        }
                        for f in s.findings
                    ],
                }
                for s in self.sections
            ],
            "could_not_tell_you": list(self.abstentions),
            "measurements": {
                k: (v.as_dict() if hasattr(v, "as_dict") else v)
                for k, v in self.measurements.items()
            },
            "notes": list(self.notes),
        }


def assemble(reports: Iterable[Any], *, dataset: str = "this upload") -> Assembled:
    """Every module's report, in the order a person should meet them."""
    findings: list[Any] = []
    measurements: dict[str, Any] = {}
    notes: list[str] = []
    for report in reports:
        findings.extend(getattr(report, "findings", ()) or ())
        measurements.update(dict(getattr(report, "measurements", {}) or {}))
        notes.extend(getattr(report, "notes", ()) or ())

    blockers = tuple(
        (f.code, BLOCKING[f.code])
        for f in findings
        if getattr(f, "code", "") in BLOCKING and getattr(f, "severity", "info") == "strong"
    )
    blocked_sections: dict[str, str] = {}
    for code, _why in blockers:
        for key, _title, prefixes in SECTIONS:
            if any(code.startswith(p) for p in prefixes):
                for downstream in BLOCKS.get(key, ()):
                    blocked_sections.setdefault(downstream, code)

    claimed: set[int] = set()
    sections: list[Section] = []
    for key, title, prefixes in SECTIONS:
        mine = [f for f in findings if any(getattr(f, "code", "").startswith(p) for p in prefixes)]
        claimed.update(id(f) for f in mine)
        sections.append(
            Section(
                title=title,
                key=key,
                # Worst first inside a section; the reader should not have to
                # hunt for the one that matters.
                findings=tuple(sorted(mine, key=_severity)),
                suppressed_by=blocked_sections.get(key),
            )
        )
    other = [f for f in findings if id(f) not in claimed]
    if other:
        sections.append(
            Section(title="Other", key="other", findings=tuple(sorted(other, key=_severity)))
        )

    return Assembled(
        dataset=dataset,
        sections=tuple(sections),
        blockers=blockers,
        # The refusals, gathered from wherever they were raised. Not an
        # appendix: a report whose abstentions are buried reads as more
        # confident than it is.
        abstentions=tuple(_abstentions(findings, blocked_sections, notes)),
        measurements=measurements,
        notes=tuple(notes),
    )


def _severity(finding: Any) -> int:
    return {"strong": 0, "weak": 1, "info": 2}.get(getattr(finding, "severity", "info"), 3)


def _abstentions(
    findings: Sequence[Any], blocked: Mapping[str, str], notes: Sequence[str]
) -> list[str]:
    out: list[str] = []
    for section, code in blocked.items():
        title = next((t for k, t, _ in SECTIONS if k == section), section)
        out.append(f"{title.lower()} — not answerable because {BLOCKING.get(code, code)}")
    for finding in findings:
        code = getattr(finding, "code", "")
        if code.endswith(("not_separated", "nothing_measured", "no_task_list")):
            out.append(getattr(finding, "summary", code))
    for note in notes:
        if "not measured is not the same as fine" in note or "nothing to read" in note:
            out.append(note)
    return out


# -- rendering, which is deliberately the dumb part --------------------------


def as_markdown(assembled: Assembled) -> str:
    """The report as text. No judgement here — ordering and wording came from
    the modules, and this only lays them out."""
    lines = [f"# {assembled.dataset}", "", f"**{assembled.verdict}**", ""]
    if assembled.blockers:
        lines += ["## Read this first", ""]
        for code, why in assembled.blockers:
            lines.append(f"- **{why}**  `{code}`")
        lines.append("")
    for section in assembled.sections:
        if not section.findings and not section.blocked:
            continue
        lines.append(f"## {section.title}")
        if section.blocked:
            lines += [
                "",
                f"_Not presented as an answer: {BLOCKING.get(section.suppressed_by, '')}. "
                "The numbers were still computed and are in the record._",
            ]
        lines.append("")
        for finding in section.findings:
            mark = {"strong": "**", "weak": "", "info": ""}.get(
                getattr(finding, "severity", "info"), ""
            )
            lines.append(f"- {mark}{getattr(finding, 'summary', '')}{mark}")
            prescription = getattr(finding, "prescription", None)
            if prescription:
                lines.append(f"  - {prescription}")
        lines.append("")
    if assembled.abstentions:
        lines += ["## What we could not tell you", ""]
        lines += [f"- {line}" for line in assembled.abstentions]
        lines.append("")
    return "\n".join(lines)


def write(assembled: Assembled, path: str | Path) -> Path:
    """Both forms side by side: the markdown a person reads and the JSON a
    system consumes, so a claim in one can always be traced to the other."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(as_markdown(assembled))
    target.with_suffix(".json").write_text(json.dumps(assembled.as_dict(), indent=2))
    return target
