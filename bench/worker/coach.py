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
    "You review robot-learning dataset evaluations and tell the uploader how to "
    "improve their dataset. You are given measurements: intake facts, data-report "
    "findings, a signal-check verdict, and a robot-test verdict with a per-stage "
    "ladder comparing their data against a shuffled control and a baseline.\n"
    "Rules:\n"
    "- At most 4 points. Fewer is better if the data supports fewer.\n"
    "- Every point must cite a number or finding from the input. No generic advice.\n"
    "- Each point says what to DO: what footage to add, what to refilm, what to "
    "fix in the export. Name the failing stage when the ladder shows one.\n"
    "- If hand tracking was estimated or dropped, say how to film so it tracks.\n"
    "- Plain language, one sentence per point, no hedging, no preamble.\n"
    'Reply as JSON: {"points": ["...", "..."]}'
)


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
            "findings": [
                f.get("summary", "") for f in (gate.get("findings") or [])[:8] if f.get("summary")
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


def ask(facts: dict, *, key: str, model: str = "") -> list[str]:
    """One call, one JSON reply, at most four points back.

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
    points = json.loads(text).get("points", [])
    return [str(p).strip() for p in points if str(p).strip()][:MAX_POINTS]


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
        points = ask(facts, key=key)
    except urllib.error.HTTPError as error:
        detail = ""
        try:
            detail = error.read()[:200].decode(errors="replace")
        except Exception:  # noqa: BLE001
            pass
        return f"coach: openai refused ({error.code}) {detail}"
    except Exception as error:  # noqa: BLE001
        return f"coach: openai call failed ({type(error).__name__}: {error})"

    if not points:
        return "coach: model returned no usable points, stored nothing"

    try:
        call(api, f"/api/submissions/{sub_id}/coach", {"points": points, "model": MODEL})
    except Exception as error:  # noqa: BLE001
        return f"coach: could not store the advice ({type(error).__name__})"
    return f"coach: stored {len(points)} point(s)"
