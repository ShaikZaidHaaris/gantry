"""A person watches the video, reads the rubric, and says yes or no.

This is the scorer that makes the other two mean anything. A simulator's
predicate is free and a model's judgement is cheap, but neither can be checked
against anything until somebody has looked — and the whole argument for writing
every criterion twice is that the sentence a person reads on a bench is the same
sentence the predicate was standing in for.

So the deliverable is small on purpose: one static HTML page, one video per
trial, the rubric printed verbatim beside it, and three buttons. No server, no
framework, no account. The page writes a JSONL you keep, and this module reads it
back as judgements.

Why the rubric appears verbatim and unabridged
----------------------------------------------
Because the thing being measured is whether *that sentence* produces agreement.
Summarising it, or showing the criterion name instead, would measure whether
people agree about something else — and the agreement number would then be about
a rubric nobody will ever use.

Why "can't tell" is a button and not an omission
------------------------------------------------
A person forced to choose will choose, and their guess enters the corpus looking
exactly like a judgement. An explicit abstention is cheaper than a wrong label
and far cheaper than a wrong agreement score computed from wrong labels. The
abstention rate is itself a measurement: a rubric that produces many of them is
a rubric that needs rewriting, and that is worth knowing before it goes to a
bench.

Why the rater is named
----------------------
Two people disagreeing is information; two people disagreeing where you cannot
tell which was which is noise. Inter-rater agreement needs the labels grouped by
who produced them, so the session records a rater id and refuses to merge
sessions that do not have one.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from gantry.contracts.scorer import (
    VIDEO,
    Evidence,
    Judgement,
    Scorer,
    scorer_descriptor,
)
from gantry.spine import Descriptor, Verdict

VERSION = "0.1.0.dev0"

#: What the page writes and this module reads. One JSON object per line, so a
#: session can be appended to, interrupted, and resumed without a database.
LINES = "judgements.jsonl"


@dataclass(frozen=True)
class Session:
    """One person's scoring of one set of trials."""

    rater: str
    #: ``{trial: {criterion: passed}}``, where ``None`` is an abstention.
    judgements: Mapping[str, Mapping[str, bool | None]] = field(default_factory=dict)
    #: Free text the rater left, per trial. Kept because "the video cut off
    #: before it settled" is the note that fixes a rubric.
    notes: Mapping[str, str] = field(default_factory=dict)
    task: str | None = None

    def validate(self) -> Verdict:
        checks = []
        if not self.rater.strip():
            checks.append(
                Verdict.no(
                    "session.anonymous",
                    "this session records no rater",
                    hint="agreement between raters needs the labels grouped by who "
                    "produced them; without a name they cannot be",
                )
            )
        if not self.judgements:
            checks.append(Verdict.no("session.empty", f"{self.rater!r} judged nothing"))
        abstained = sum(
            1
            for verdicts in self.judgements.values()
            for value in verdicts.values()
            if value is None
        )
        total = sum(len(v) for v in self.judgements.values())
        if total and abstained / total > 0.3:
            checks.append(
                Verdict.note(
                    "session.abstains_often",
                    f"{self.rater!r} could not tell on {abstained} of {total} judgement(s)",
                    hint="a rubric that produces this many abstentions needs "
                    "rewriting before it reaches a bench",
                )
            )
        return Verdict.all(checks)

    @property
    def abstention_rate(self) -> float:
        total = sum(len(v) for v in self.judgements.values())
        if not total:
            return 0.0
        return sum(1 for v in self.judgements.values() for x in v.values() if x is None) / total


def read_session(path: str | Path) -> Session:
    """Read a session back from the JSONL the page wrote."""
    lines = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    if not lines:
        return Session(rater="")
    rater = str(lines[0].get("rater", ""))
    judgements: dict[str, dict[str, bool | None]] = {}
    notes: dict[str, str] = {}
    task = None
    for row in lines:
        trial = str(row.get("trial", ""))
        criterion = str(row.get("criterion", ""))
        if not trial or not criterion:
            continue
        verdict = row.get("passed")
        judgements.setdefault(trial, {})[criterion] = None if verdict is None else bool(verdict)
        if row.get("note"):
            notes[trial] = str(row["note"])
        task = task or row.get("task")
    return Session(rater=rater, judgements=judgements, notes=notes, task=task)


def write_page(
    trials: Sequence[Mapping[str, Any]],
    task: Any,
    path: str | Path,
    *,
    rater: str = "",
) -> Path:
    """Write the scoring page: video, rubric, three buttons, per trial.

    ``trials`` is ``[{"trial": id, "video": path}, ...]``. Videos are referenced
    relative to the page so the whole thing is a directory you can move, zip, or
    hand to somebody who has never installed any of this.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    criteria = [
        {"check": c.check, "rubric": c.rubric} for c in (getattr(task, "success", ()) or ())
    ]
    payload = {
        "task": getattr(task, "name", "unknown"),
        "instruction": getattr(task, "instruction", ""),
        "criteria": criteria,
        "trials": [dict(trial) for trial in trials],
        "rater": rater,
        "lines": LINES,
    }
    target.write_text(_PAGE.replace("__PAYLOAD__", json.dumps(payload)))
    return target


#: The page. Deliberately one file with no dependencies: a scoring tool that
#: needs a build step is a scoring tool nobody uses on a Friday afternoon.
_PAGE = """<!doctype html>
<meta charset="utf-8">
<title>gantry — score trials</title>
<style>
  :root { color-scheme: light dark; }
  body { font: 16px/1.55 system-ui, sans-serif; max-width: 46rem; margin: 2rem auto;
         padding: 0 1rem; }
  video { width: 100%; background: #000; border-radius: 6px; }
  .rubric { background: color-mix(in srgb, canvas 92%, canvasText);
            border-left: 3px solid currentColor; padding: .75rem 1rem;
            margin: 1rem 0; white-space: pre-wrap; }
  .row { display: flex; gap: .5rem; margin: .75rem 0; }
  button { flex: 1; padding: .7rem; font: inherit; cursor: pointer;
           border: 1px solid currentColor; border-radius: 6px; background: canvas;
           color: canvasText; }
  button:hover { background: color-mix(in srgb, canvas 88%, canvasText); }
  button.on { background: canvasText; color: canvas; }
  input[type=text] { width: 100%; padding: .5rem; font: inherit; box-sizing: border-box; }
  .meta { opacity: .7; font-size: .85em; }
  #done { padding: 1rem; border: 1px dashed currentColor; border-radius: 6px; }
  h2 { margin-bottom: .25rem; }
</style>
<body>
<h1>Score trials</h1>
<p class="meta" id="head"></p>
<label class="meta">Your name or initials
  <input type="text" id="rater" placeholder="required — agreement needs to know who judged">
</label>
<hr>
<div id="trial"></div>
<div id="done" hidden></div>
<script>
const D = __PAYLOAD__;
const answers = [];
let at = 0;
document.getElementById("rater").value = D.rater || "";
document.getElementById("head").textContent =
  D.task + " — " + D.instruction + " — " + D.trials.length + " trial(s)";

function render() {
  const host = document.getElementById("trial");
  if (at >= D.trials.length) { finish(); return; }
  const t = D.trials[at];
  host.innerHTML = "";
  const h = document.createElement("h2");
  h.textContent = "Trial " + (at + 1) + " of " + D.trials.length;
  host.append(h);
  const id = document.createElement("p");
  id.className = "meta";
  id.textContent = t.trial;
  host.append(id);
  if (t.video) {
    const v = document.createElement("video");
    v.src = t.video; v.controls = true; v.autoplay = true; v.loop = true; v.muted = true;
    host.append(v);
  }
  // Every criterion, with its rubric printed in full. Not summarised: the
  // sentence is the thing being measured.
  const picks = {};
  for (const c of D.criteria) {
    const box = document.createElement("div");
    const q = document.createElement("div");
    q.className = "rubric";
    q.textContent = c.rubric;
    box.append(q);
    const row = document.createElement("div");
    row.className = "row";
    for (const [label, value] of [["Yes", true], ["No", false], ["Can't tell", null]]) {
      const b = document.createElement("button");
      b.textContent = label;
      b.onclick = () => {
        picks[c.check] = value;
        [...row.children].forEach(x => x.classList.remove("on"));
        b.classList.add("on");
      };
      row.append(b);
    }
    box.append(row);
    host.append(box);
  }
  const note = document.createElement("input");
  note.type = "text";
  note.placeholder = "anything worth noting (optional)";
  host.append(note);
  const next = document.createElement("div");
  next.className = "row";
  const go = document.createElement("button");
  go.textContent = "Next trial";
  go.onclick = () => {
    const rater = document.getElementById("rater").value.trim();
    if (!rater) { alert("Please put your name or initials at the top."); return; }
    if (Object.keys(picks).length < D.criteria.length) {
      alert("Please answer every question — 'Can't tell' counts as an answer.");
      return;
    }
    for (const c of D.criteria) {
      answers.push({rater, task: D.task, trial: t.trial, criterion: c.check,
                    passed: picks[c.check], note: note.value || ""});
    }
    at += 1;
    render();
  };
  next.append(go);
  host.append(next);
}

function finish() {
  document.getElementById("trial").innerHTML = "";
  const box = document.getElementById("done");
  box.hidden = false;
  const lines = answers.map(a => JSON.stringify(a)).join("\\n") + "\\n";
  box.innerHTML = "<p><strong>Done — " + D.trials.length +
    " trial(s) scored.</strong></p><p>Save this file next to the page as <code>" +
    D.lines + "</code>.</p>";
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([lines], {type: "application/x-ndjson"}));
  a.download = D.lines;
  a.textContent = "Download " + D.lines;
  box.append(a);
}
render();
</script>
"""


class HumanScorer(Scorer):
    """Reads back what a person decided, watching video against the rubric.

    Deliberately not interactive. The scorer does not block waiting for
    somebody — a person scores when they have twenty minutes, and this reads the
    file afterwards. That separation is what lets the same judgements be
    re-analysed later without asking anyone to watch anything twice.
    """

    def __init__(self, session: Session | str | Path, *, name: str = "human"):
        self._session = session if isinstance(session, Session) else read_session(session)
        self._name = name

    @property
    def session(self) -> Session:
        return self._session

    def descriptor(self) -> Descriptor:
        return scorer_descriptor(
            name=self._name,
            version=VERSION,
            evidence=(VIDEO,),
            # A person is not deterministic and this does not claim otherwise;
            # the recorded file is, which is a different property and is what
            # makes re-analysis possible.
            deterministic=False,
            cost="human",
            abstains=True,
            rater=self._session.rater,
            trials_scored=len(self._session.judgements),
            abstention_rate=round(self._session.abstention_rate, 4),
        )

    @classmethod
    def annotate(cls, trials, task, path, **options):
        """The page a person scores on. See :func:`write_page`."""
        return write_page(trials, task, path, rater=str(options.get("rater", "")))

    @classmethod
    def recorded_labels(cls, path):
        """A rater's name and their verdicts, in the shape agreement needs."""
        session = read_session(path)
        session.validate().raise_if_refused(f"cannot read {path}")
        return session.rater, session.judgements

    def judge(self, evidence: Evidence, task: Any) -> tuple[Judgement, ...]:
        recorded = self._session.judgements.get(evidence.trial, {})
        note = self._session.notes.get(evidence.trial, "")
        out = []
        for criterion in getattr(task, "success", ()) or ():
            if criterion.check in recorded:
                verdict = recorded[criterion.check]
                why = f"{self._session.rater or 'a rater'} watched the video" + (
                    f": {note}" if note else ""
                )
            else:
                # Nobody scored this trial. An abstention rather than a failure:
                # an unscored trial is missing data, and counting it as a
                # failure would make an incomplete session look like a bad
                # policy.
                verdict = None
                why = (
                    f"{evidence.trial!r} was not scored in {self._session.rater or 'this'} session"
                )
            out.append(
                Judgement(
                    criterion=criterion.check,
                    passed=verdict,
                    rationale=why,
                    used=(VIDEO,),
                )
            )
        return tuple(out)


def merge(sessions: Iterable[Session]) -> dict[str, Session]:
    """Group sessions by rater, refusing anonymous ones.

    What agreement needs: labels attributable to a person. Two sessions from the
    same rater are combined; a session with no rater is dropped and the caller
    finds out, because silently folding it into somebody else's labels would
    corrupt the one number this is all for.
    """
    grouped: dict[str, Session] = {}
    for session in sessions:
        if not session.rater.strip():
            continue
        existing = grouped.get(session.rater)
        if existing is None:
            grouped[session.rater] = session
            continue
        judgements = {**existing.judgements, **session.judgements}
        notes = {**existing.notes, **session.notes}
        grouped[session.rater] = Session(
            rater=session.rater,
            judgements=judgements,
            notes=notes,
            task=existing.task or session.task,
        )
    return grouped
