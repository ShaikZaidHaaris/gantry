"""A model reads the same rubric a person would, and its answer is checked.

This is the scorer that makes rubric-based evaluation affordable. A person can
score twenty videos in an afternoon; a model can score twenty thousand overnight
for the price of lunch. That is the difference between rubrics as documentation
and rubrics as measurement.

It is also the scorer that is easiest to be wrong with. A model that has not been
compared against a person produces labels shaped exactly like correct ones, and
every aggregate computed from them inherits the error in silence. So this plugin
is deliberately built so that using it *without* calibration is awkward: the
descriptor records the model, the prompt hash and the temperature, the labels are
attributable, and :mod:`gantry_feedback_calibrate` will refuse the findings of a
judge nobody has checked.

The wire is a callable
----------------------
No SDK is a hard dependency. ``ask`` is any function taking ``(prompt, frames)``
and returning text — which means a lab with its own inference stack substitutes
that, an air-gapped setup substitutes a local model, and the tests substitute a
dictionary. Baking in one vendor's client would make the plugin a bet on that
vendor rather than on the method.

Every ask is transcribed
------------------------
The prompt, frames offered, raw reply and parse result are recorded. Three
reasons, all learned from the LLM-judging literature: a disagreement with a
person is only diagnosable if you can read what the model actually said; a
re-analysis six months later must not require paying for inference again; and a
prompt that drifted is otherwise indistinguishable from a model that changed.

What it is made to do badly
---------------------------
Guess. The prompt instructs abstention when the frames do not settle the
criterion, an unparseable reply abstains rather than defaulting, and a reply
naming a criterion the task did not ask about is dropped. A judge that answers
everything looks accurate on the subset it should have declined, and the
abstention rate is the only thing that catches it.
"""

from __future__ import annotations

import hashlib
import io
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from gantry.contracts.scorer import (
    VIDEO,
    Evidence,
    Judgement,
    Scorer,
    scorer_descriptor,
)
from gantry.errors import ConfigError
from gantry.spine import Descriptor

VERSION = "0.1.0.dev0"

#: How many frames are offered per trial. Enough to see a motion, few enough
#: that the model is not being asked to summarise a film. Fixed rather than
#: tuned per task, because a judge whose evidence budget varies by task is a
#: judge whose agreement cannot be compared across tasks.
FRAMES = 8

#: What a reply must contain. Strict on purpose: a parser that accepts prose
#: will happily read "it seems like it might have worked" as a yes.
_ANSWER = re.compile(
    r"^\s*(?P<criterion>[A-Za-z0-9_.\-]+)\s*:\s*(?P<verdict>yes|no|unclear)\b",
    re.IGNORECASE | re.MULTILINE,
)

PROMPT = """You are grading one attempt by a robot arm, from frames sampled in \
order across the attempt.

Task given to the robot: {instruction}

Answer each criterion below using ONLY its written standard. Do not apply your \
own idea of what success should mean, and do not give partial credit.

{criteria}

Rules:
- Answer exactly one line per criterion, formatted `<name>: yes` or `<name>: no` \
or `<name>: unclear`.
- Answer `unclear` whenever the frames do not settle the question — for example \
if the object leaves view, if the final state is not visible, or if the standard \
requires seeing something the frames do not show. `unclear` is a correct and \
useful answer; guessing is not.
- After the lines, add one short sentence per criterion beginning `why <name>:` \
describing what you actually saw.
"""


@dataclass(frozen=True)
class Answer:
    """One reply, parsed."""

    verdicts: Mapping[str, bool | None] = field(default_factory=dict)
    reasons: Mapping[str, str] = field(default_factory=dict)
    #: Whatever came back, kept verbatim. A disagreement is only diagnosable if
    #: you can read what was said rather than what was extracted.
    raw: str = ""
    parsed: bool = True


@dataclass(frozen=True)
class Transcript:
    """Everything one asking consisted of, so it never has to be paid for twice."""

    trial: str
    prompt: str
    prompt_hash: str
    frames: int
    raw: str
    verdicts: Mapping[str, bool | None] = field(default_factory=dict)
    model: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "trial": self.trial,
            "prompt_hash": self.prompt_hash,
            "frames": self.frames,
            "raw": self.raw,
            "verdicts": dict(self.verdicts),
            "model": self.model,
            "prompt": self.prompt,
        }


def build_prompt(task: Any) -> str:
    """The prompt, with every rubric quoted verbatim.

    Verbatim and unabridged, exactly as the human page shows it. The whole
    argument for measuring agreement is that both judges were given the same
    sentence; paraphrasing for the model would make the agreement number about
    two different standards and it would still look like a number.
    """
    criteria = getattr(task, "success", ()) or ()
    if not criteria:
        raise ConfigError(
            f"{getattr(task, 'name', 'this task')!r} declares no success criterion, "
            "so there is nothing to grade against"
        )
    block = "\n\n".join(
        f"Criterion `{criterion.check}`:\n{criterion.rubric.strip()}" for criterion in criteria
    )
    return PROMPT.format(instruction=getattr(task, "instruction", "(not stated)"), criteria=block)


def prompt_hash(prompt: str) -> str:
    """A short digest of the prompt, recorded with every judgement.

    So that a prompt which drifted is distinguishable from a model that changed.
    Without it the two are the same observation and neither can be fixed.
    """
    return hashlib.blake2b(prompt.encode(), digest_size=6).hexdigest()


def parse_answer(text: str, criteria: Sequence[str]) -> Answer:
    """Read verdicts out of a reply, abstaining rather than guessing.

    Unmatched criteria abstain: a model that skipped a question has not answered
    it, and filling in a default would invent a label. Criteria the task never
    asked about are dropped — a judge that volunteers extra verdicts is not
    grading the task in front of it.
    """
    wanted = list(criteria)
    verdicts: dict[str, bool | None] = {name: None for name in wanted}
    # Matched case-insensitively. A model that replies "Lifted: yes" has
    # answered the question, and dropping it for capitalisation would abstain
    # on a settled criterion — the one direction of error that looks like
    # caution while actually discarding data.
    by_lower = {name.lower(): name for name in wanted}
    found = False
    for match in _ANSWER.finditer(text or ""):
        name = by_lower.get(match.group("criterion").lower())
        if name is None:
            continue
        raw = match.group("verdict").lower()
        verdicts[name] = {"yes": True, "no": False, "unclear": None}[raw]
        found = True

    reasons: dict[str, str] = {}
    for name in wanted:
        why = re.search(
            rf"^\s*why\s+{re.escape(name)}\s*:\s*(?P<text>.+)$",
            text or "",
            re.IGNORECASE | re.MULTILINE,
        )
        if why:
            reasons[name] = why.group("text").strip()
    return Answer(verdicts=verdicts, reasons=reasons, raw=text or "", parsed=found)


def frames_from_video(path: str | Path, count: int = FRAMES) -> list[bytes]:
    """``count`` frames spread evenly across a video, as PNG bytes.

    Evenly spread rather than the first N, because the end of an attempt is
    where most criteria are settled and a judge shown only the opening has been
    set up to abstain.
    """
    try:
        import av
    except ImportError as error:  # pragma: no cover - needs the decoder
        raise ConfigError(
            "sampling frames needs a decoder: pip install 'gantry-scorer-vlm[video]'"
        ) from error
    try:
        from PIL import Image  # noqa: F401 - av.to_image() needs it
    except ImportError as error:  # pragma: no cover - needs the encoder
        raise ConfigError(
            "encoding frames needs an image library: pip install 'gantry-scorer-vlm[video]'"
        ) from error

    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        total = stream.frames or 0
        wanted = (
            {round(i * (total - 1) / max(1, count - 1)) for i in range(count)}
            if total
            else set(range(count))
        )
        out: list[bytes] = []
        for index, frame in enumerate(container.decode(video=0)):
            if index in wanted or not total:
                # Encoded explicitly rather than through ``_repr_png_``, which
                # is an IPython display hook and returns something else — or
                # nothing — outside a notebook. Frames that are not really PNG
                # reach a model as garbage and it dutifully abstains, which
                # looks like an honest judge on unclear footage.
                buffer = io.BytesIO()
                frame.to_image().save(buffer, format="PNG")
                out.append(buffer.getvalue())
                if len(out) >= count:
                    break
    return out


def replay(transcripts: Mapping[str, str]) -> Callable[[str, Sequence[bytes]], str]:
    """An ``ask`` that returns recorded replies instead of calling a model.

    What makes this plugin testable and re-analysable. A stored transcript is a
    complete record of what the judge said, so agreement can be recomputed, a
    parser can be fixed, and a rubric can be re-scored — none of which should
    require paying for inference twice or having network access at all.
    """
    remaining = dict(transcripts)

    def ask(prompt: str, frames: Sequence[bytes]) -> str:
        # Keyed on nothing but call order when the caller did not key by trial:
        # a replay is a tape, and a tape plays in order.
        if not remaining:
            raise ConfigError("the transcript ran out of recorded replies")
        key = next(iter(remaining))
        return remaining.pop(key)

    return ask


def replay_by_trial(
    transcripts: Mapping[str, str],
) -> Callable[[str, Sequence[bytes], str], str]:
    """A trial-keyed replay, for when order is not guaranteed."""

    def ask(prompt: str, frames: Sequence[bytes], trial: str = "") -> str:
        if trial not in transcripts:
            raise ConfigError(f"no recorded reply for trial {trial!r}")
        return transcripts[trial]

    return ask


class VlmScorer(Scorer):
    """Asks a vision-language model to grade a trial against its written rubric.

    ``ask`` is any callable taking ``(prompt, frames)`` and returning text. That
    is the whole integration surface: no SDK is imported here, so pointing this
    at a different model, a local one, or a recorded transcript is a change of
    argument rather than a change of plugin.
    """

    def __init__(
        self,
        ask: Callable[..., str],
        *,
        model: str = "unnamed",
        temperature: float = 0.0,
        frames: int = FRAMES,
        name: str = "vlm",
    ):
        self._ask = ask
        self._model = model
        self._temperature = temperature
        self._frames = frames
        self._name = name
        self._transcripts: list[Transcript] = []

    @property
    def transcripts(self) -> tuple[Transcript, ...]:
        """Every asking, in order. Write these out; they are the audit trail."""
        return tuple(self._transcripts)

    def descriptor(self) -> Descriptor:
        return scorer_descriptor(
            name=self._name,
            version=VERSION,
            evidence=(VIDEO,),
            # Even at temperature zero this is not promised. A served model can
            # change under you without changing its name, and claiming
            # determinism would make every agreement number computed against it
            # quietly unfalsifiable.
            deterministic=False,
            cost="cheap",
            abstains=True,
            model=self._model,
            temperature=self._temperature,
            frames=self._frames,
        )

    def judge(self, evidence: Evidence, task: Any) -> tuple[Judgement, ...]:
        criteria = [c.check for c in (getattr(task, "success", ()) or ())]
        if not criteria:
            return ()

        prompt = build_prompt(task)
        digest = prompt_hash(prompt)
        try:
            frames = frames_from_video(evidence.video, self._frames) if evidence.video else []
        except Exception:  # noqa: BLE001
            # A video that will not decode is missing evidence, not a crash. The
            # model is then asked with no frames, sees nothing, and abstains —
            # which is the honest outcome. Raising here would make an unreadable
            # file indistinguishable from a broken judge.
            frames = []

        try:
            reply = _call(self._ask, prompt, frames, evidence.trial)
        except Exception as error:  # noqa: BLE001 - a failed ask abstains
            self._transcripts.append(
                Transcript(
                    trial=evidence.trial,
                    prompt=prompt,
                    prompt_hash=digest,
                    frames=len(frames),
                    raw=f"<{type(error).__name__}: {error}>",
                    model=self._model,
                )
            )
            return tuple(
                Judgement(
                    criterion=name,
                    passed=None,
                    rationale=(
                        f"{self._name}: asking failed ({type(error).__name__}), so "
                        "nothing was judged"
                    ),
                    used=(VIDEO,),
                )
                for name in criteria
            )

        answer = parse_answer(reply, criteria)
        self._transcripts.append(
            Transcript(
                trial=evidence.trial,
                prompt=prompt,
                prompt_hash=digest,
                frames=len(frames),
                raw=answer.raw,
                verdicts=answer.verdicts,
                model=self._model,
            )
        )

        out = []
        for name in criteria:
            verdict = answer.verdicts.get(name)
            if not answer.parsed:
                why = (
                    f"{self._name}: the reply did not contain a verdict in the "
                    f"required form, so nothing was read from it"
                )
            elif verdict is None:
                why = answer.reasons.get(name) or (
                    f"{self._name}: answered unclear — the frames did not settle it"
                )
            else:
                why = answer.reasons.get(name) or (
                    f"{self._name} ({self._model}): answered {'yes' if verdict else 'no'}"
                )
            out.append(Judgement(criterion=name, passed=verdict, rationale=why, used=(VIDEO,)))
        return tuple(out)

    # -- the contract's optional halves ------------------------------------

    @classmethod
    def annotate(cls, trials, task, path, **options):
        """Write the prompt this judge would be given, for a person to read.

        Not a page to click through — the surface a model needs is its prompt,
        and the reason to write it out is that a prompt nobody has read is a
        prompt nobody can criticise. Reviewing it beside the rubric is the
        cheapest way to catch a judge being asked the wrong question.
        """
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        prompt = build_prompt(task)
        target.write_text(
            json.dumps(
                {
                    "task": getattr(task, "name", "unknown"),
                    "prompt": prompt,
                    "prompt_hash": prompt_hash(prompt),
                    "trials": [dict(trial) for trial in trials],
                },
                indent=2,
            )
            + "\n"
        )
        return target

    @classmethod
    def recorded_labels(cls, path):
        """A model's verdicts, read back from written transcripts."""
        rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
        if not rows:
            raise ConfigError(f"{path} holds no transcript")
        judge = str(rows[0].get("model") or "vlm")
        labels: dict[str, dict[str, bool | None]] = {}
        for row in rows:
            trial = str(row.get("trial", ""))
            if not trial:
                continue
            labels[trial] = {
                str(k): (None if v is None else bool(v))
                for k, v in (row.get("verdicts") or {}).items()
            }
        return judge, labels

    def write_transcripts(self, path: str | Path) -> Path:
        """Persist the audit trail as JSONL, one asking per line."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            "".join(json.dumps(transcript.as_dict()) + "\n" for transcript in self._transcripts)
        )
        return target


def _call(ask: Callable[..., str], prompt: str, frames: Sequence[bytes], trial: str) -> str:
    """Call ``ask``, passing the trial id only if it will accept one.

    A two-argument callable is the documented contract; a three-argument one is
    convenient for a trial-keyed replay. Supporting both here keeps the contract
    small without making every caller thread an argument they do not use.
    """
    try:
        return ask(prompt, frames, trial)  # type: ignore[call-arg]
    except TypeError:
        return ask(prompt, frames)
