"""Turn a submission's results into at most four ways to improve the dataset.

Every number on the verdict page says what happened; none of it says what to do
next. The gap between "0% of scenes got past 'moved'" and "collect thirty more
demonstrations of the lift, filmed slower" is judgement, and this module buys
that judgement from a language model, grounded in the measurements the gates
already made.

Boundaries, stated up front:

* **The model never sees the dataset.** It reads the gates' outputs: findings,
  verdict sentences, ladder rates, what intake detected. It cannot invent a
  measurement, only interpret the ones on the page, and the prompt forbids
  advice that does not cite one.
* **Advice is not a verdict.** It is stored separately, rendered separately,
  labelled as generated, and nothing downstream depends on it. A submission
  with no coaching is merely a submission with no coaching.
* **Failure is silent by design.** No key configured, the API down, the model
  returning soup: every path logs one line and stores nothing. A gate result
  must never be held hostage by an advice call.

The model is OpenAI's cheapest current one, ``gpt-5-nano``, overridable with
``BENCH_FEEDBACK_MODEL``. A digest is a few hundred tokens and the reply a few
hundred more, so a full run's coaching costs a fraction of a cent against a
robot test that costs hours of GPU.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request

OPENAI_URL = "https://api.openai.com/v1/chat/completions"

#: Cheapest current OpenAI model. Named in one place, overridable without a
#: deploy, because "latest cheapest" is a fact about their price list, not ours.
MODEL = os.environ.get("BENCH_FEEDBACK_MODEL", "gpt-5-nano")

#: Four, exactly as specified. The truncation is enforced here rather than
#: trusted to the prompt, because a prompt is a request and a slice is a rule.
MAX_POINTS = 4

SYSTEM = (
    "You turn robot-dataset evaluation measurements into advice for the uploader. "
    "Input: intake facts, findings (each with a code), check verdicts, and a "
    "robot-test ladder comparing their data against a shuffled control and a "
    "baseline.\n"
    "Hard rules, all of them:\n"
    "- points: at most 4. Each is ONE imperative sentence, at most 18 words.\n"
    "- Every sentence must reuse a number or a named stage from the input. Never "
    "invent a number, a cause, or a fact that is not in the input.\n"
    "- Say what to DO: add footage, refilm, fix the export. Name the failing "
    "stage when the ladder shows one. If hand tracking was estimated or dropped, "
    "say how to film so it tracks.\n"
    "- No hedging, no praise, no preamble, no 'consider', no 'ensure'.\n"
    "- fixes: one entry per finding code. ONE imperative sentence, at most 14 "
    "words, grounded only in that finding. Use \"\" when the finding is "
    "informational and needs no action.\n"
    'Output JSON only: {"points": ["..."], "fixes": {"<code>": "<sentence>"}}'
)

#: The longest sentence the UI will show, enforced here and again at the API.
#: A cap is a rule; the prompt's word limit is a request.
MAX_SAY = 160


def digest(record: dict) -> dict:
    """The facts worth sending, and nothing else.

    Built by allow-list, not by dumping the record: the submission carries the
    uploader's email and the full event log, and neither belongs in a request to
    a third party. What goes is what a careful human coach would read off the
    screen.
    """
    gates = {g.get("key"): g for g in record.get("gates", [])}
    out: dict = {"benchmark": (record.get("benchmark") or {}).get("key")}

    detected = (record.get("dataset") or {}).get("detected") or {}
    out["intake"] = {
        k: detected.get(k)
        for k in ("episodes", "frames", "fps", "channels", "source", "poses")
        if detected.get(k) is not None
    }

    for key, keep in (("g1", "data report"), ("g2", "signal check"), ("g3", "robot test")):
        gate = gates.get(key)
        if not gate or gate.get("status") in (None, "queued", "running"):
            continue
        entry: dict = {
            "status": gate.get("status"),
            "verdict": (gate.get("verdict") or {}).get("summary", ""),
            # Code and summary both: the summary is what the advice is grounded
            # in, the code is the handle the reply keys its per-finding line to,
            # and the allow-list below refuses any code we did not send.
            "findings": [
                {"code": f.get("code", ""), "summary": f.get("summary", "")}
                for f in (gate.get("findings") or [])[:8]
                if f.get("summary")
            ],
        }
        if key == "g3":
            detail = gate.get("detail") or {}
            entry["ladder"] = [
                {
                    "rung": row.get("rung") or row.get("name"),
                    "rates": {
                        arm: (cell or {}).get("rate")
                        for arm, cell in (row.get("cells") or {}).items()
                        if (cell or {}).get("measured")
                    },
                }
                for row in (detail.get("ladder") or [])[:8]
            ]
        out[keep] = entry
    return out


def known_codes(facts: dict) -> set[str]:
    """The finding codes that actually travelled, and therefore the only keys a
    reply may use. Anything else in the reply is the model free-associating."""
    codes: set[str] = set()
    for section in facts.values():
        if isinstance(section, dict):
            for finding in section.get("findings", []):
                if isinstance(finding, dict) and finding.get("code"):
                    codes.add(finding["code"])
    return codes


def ask(facts: dict, *, key: str, model: str = "") -> dict:
    """One call, one JSON reply: at most four points, one short line per finding.

    Two gpt-5-family behaviours shaped this, both learned from the first live
    call rather than the docs. Reasoning tokens are spent from the same
    ``max_completion_tokens`` budget as the answer, and the model reasons
    freely by default: a trivial probe spent 704 tokens thinking before its
    one-line reply, and the real digest blew the whole budget on thought and
    returned empty content, which parsed as "Expecting value: line 1 column
    0". So the effort is pinned to minimal -- this is a summarisation of
    measurements, not a puzzle -- with a retry without the field for any model
    that refuses it, and an empty reply names the budget as the cause instead
    of failing as a JSON error three layers up.
    """
    body = {
        "model": model or MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": json.dumps(facts, ensure_ascii=False)},
        ],
        "response_format": {"type": "json_object"},
        "reasoning_effort": "minimal",
        # `max_tokens` is the old name and these models refuse it.
        "max_completion_tokens": 4000,
    }

    def post(payload: dict) -> dict:
        request = urllib.request.Request(
            OPENAI_URL,
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"},
        )
        with urllib.request.urlopen(request, timeout=90) as response:
            return json.loads(response.read())

    try:
        reply = post(body)
    except urllib.error.HTTPError as error:
        detail = error.read()[:400].decode(errors="replace")
        if error.code == 400 and "reasoning_effort" in detail:
            reply = post({k: v for k, v in body.items() if k != "reasoning_effort"})
        else:
            raise urllib.error.HTTPError(error.url, error.code, detail, error.hdrs, None)

    choice = reply["choices"][0]
    text = choice["message"].get("content") or ""
    if not text.strip():
        reason = choice.get("finish_reason", "?")
        raise RuntimeError(
            f"the model returned no content (finish_reason={reason}); with a "
            "gpt-5-family model that usually means reasoning consumed the whole "
            "completion budget"
        )
    reply_body = json.loads(text)
    points = [
        str(p).strip()[:MAX_SAY]
        for p in reply_body.get("points", [])
        if str(p).strip()
    ][:MAX_POINTS]

    # The guardrails on the per-finding lines are structural, not stylistic:
    # only codes we sent survive, blanks mean "nothing to say" and are dropped,
    # and length is capped because the prompt's word limit is a request while a
    # slice is a rule.
    allowed = known_codes(facts)
    raw = reply_body.get("fixes", {})
    fixes = {}
    if isinstance(raw, dict):
        for code, say in raw.items():
            text_say = str(say).strip()
            if code in allowed and text_say:
                fixes[code] = text_say[:MAX_SAY]
    return {"points": points, "fixes": fixes}


def maybe_coach(api: str, sub_id: str, call) -> str:
    """Generate and store advice for one submission. Returns a one-line outcome.

    ``call`` is the worker's own API helper, passed in rather than imported, so
    tests hand a fake and the transport stays in one place.
    """
    key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not key:
        return "coach: skipped, no OPENAI_API_KEY"

    try:
        record = call(api, f"/api/submissions/{sub_id}/for-worker", {})
    except Exception as error:  # noqa: BLE001 - advice must never break a gate
        return f"coach: could not read the record ({type(error).__name__})"

    facts = digest(record)
    if len(facts) <= 2:  # benchmark + intake alone: nothing worth interpreting
        return "coach: skipped, no finished checks to read"

    try:
        advice = ask(facts, key=key)
    except urllib.error.HTTPError as error:
        detail = ""
        try:
            detail = error.read()[:200].decode(errors="replace")
        except Exception:  # noqa: BLE001
            pass
        return f"coach: openai refused ({error.code}) {detail}"
    except Exception as error:  # noqa: BLE001
        return f"coach: openai call failed ({type(error).__name__}: {error})"

    if not advice["points"] and not advice["fixes"]:
        return "coach: model returned nothing usable, stored nothing"

    try:
        call(
            api,
            f"/api/submissions/{sub_id}/coach",
            {"points": advice["points"], "fixes": advice["fixes"], "model": MODEL},
        )
    except Exception as error:  # noqa: BLE001
        return f"coach: could not store the advice ({type(error).__name__})"
    return f"coach: stored {len(advice['points'])} point(s), {len(advice['fixes'])} finding note(s)"
