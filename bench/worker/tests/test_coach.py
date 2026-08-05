"""The advice layer, tested at its boundaries rather than its taste.

What a language model says cannot be asserted on; what can be is everything
around it: what facts leave this machine, that four is a ceiling and not a
request, and that every failure path returns a sentence instead of raising.
The last one is the contract that matters, because this runs inside the
worker's job loop directly after a verdict is stored, and an advice call that
can take down a gate result would cost more than the advice is worth.
"""

from __future__ import annotations

import io
import json
import sys
import urllib.error
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import coach  # noqa: E402


def record(with_g3: bool = True) -> dict:
    gates = [
        {
            "key": "g1",
            "status": "passed",
            "verdict": {"summary": "we produced a report"},
            "findings": [{"code": "language.one_instruction", "summary": "1 distinct instruction across 58 clips"}],
        },
        {
            "key": "g2",
            "status": "passed",
            "verdict": {"summary": "your footage predicts the robot's movements"},
            "findings": [],
        },
    ]
    if with_g3:
        gates.append(
            {
                "key": "g3",
                "status": "passed",
                "verdict": {"summary": "solved 2/50 against the baseline's 12/100"},
                "findings": [{"code": "ladder.bottleneck", "summary": "100% of scenes are lost between 'moved' and 'lifted'"}],
                "detail": {
                    "ladder": [
                        {
                            "rung": "moved",
                            "cells": {
                                "your data": {"measured": True, "rate": 0.4},
                                "baseline": {"measured": True, "rate": 0.6},
                            },
                        },
                        {
                            "rung": "lifted all 3",
                            "cells": {
                                "your data": {"measured": True, "rate": 0.0},
                                "baseline": {"measured": False},
                            },
                        },
                    ]
                },
            }
        )
    return {
        "id": "sub_x",
        "email": "secret@lab.edu",
        "benchmark": {"key": "pick_dual_bottles"},
        "gates": gates,
        "dataset": {"detected": {"episodes": 58, "fps": 30, "poses": "estimated"}},
        "events": [{"kind": "should.never.travel"}],
    }


# -- the digest: what leaves this machine ------------------------------------


def test_the_digest_is_an_allow_list_not_a_dump():
    """The record holds the uploader's email and the event log; a request to a
    third party must hold neither. Asserted on the serialised form, since that
    is the thing that would travel."""
    wire = json.dumps(coach.digest(record()))
    assert "secret@lab.edu" not in wire
    assert "should.never.travel" not in wire
    assert "sub_x" not in wire, "the submission id is not a fact about the dataset"


def test_the_digest_carries_the_facts_advice_needs():
    facts = coach.digest(record())
    assert facts["intake"]["episodes"] == 58
    assert facts["intake"]["poses"] == "estimated", "hand-tracking provenance is the point"
    assert "lost between" in facts["robot test"]["findings"][0]["summary"]
    assert facts["robot test"]["findings"][0]["code"] == "ladder.bottleneck"
    ladder = facts["robot test"]["ladder"]
    assert ladder[0] == {"rung": "moved", "rates": {"your data": 0.4, "baseline": 0.6}}
    assert ladder[1]["rates"] == {"your data": 0.0}, "an unmeasured cell must not travel as 0"


# -- the ceiling -------------------------------------------------------------


def fake_openai(points, fixes=None):
    payload = {"points": points, "fixes": fixes or {}}
    reply = {"choices": [{"message": {"content": json.dumps(payload)}}]}

    class Response(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    return lambda request, timeout: Response(json.dumps(reply).encode())


def test_four_is_a_slice_not_a_request(monkeypatch):
    """The prompt asks for four; the code enforces it. A prompt is a request."""
    monkeypatch.setattr(coach.urllib.request, "urlopen", fake_openai(["a", "b", "c", "d", "e", "f"]))
    got = coach.ask({"x": 1}, key="k")
    assert len(got["points"]) == coach.MAX_POINTS == 4


def test_blank_points_are_dropped_before_the_ceiling(monkeypatch):
    monkeypatch.setattr(coach.urllib.request, "urlopen", fake_openai(["  ", "add wrist camera", ""]))
    assert coach.ask({"x": 1}, key="k")["points"] == ["add wrist camera"]


# -- failure is a sentence, never an exception -------------------------------


def test_no_key_means_skipped_and_no_network(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    def explode(*a, **k):
        raise AssertionError("no key must mean no call of any kind")

    got = coach.maybe_coach("http://api", "sub_x", explode)
    assert got.startswith("coach: skipped")


def test_an_openai_refusal_is_reported_not_raised(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "k")
    calls = []

    def transport(api, path, payload=None):
        calls.append(path)
        return record()

    def refuse(request, timeout):
        raise urllib.error.HTTPError(
            coach.OPENAI_URL, 429, "rate limited", {}, io.BytesIO(b"slow down")
        )

    monkeypatch.setattr(coach.urllib.request, "urlopen", refuse)
    got = coach.maybe_coach("http://api", "sub_x", transport)
    assert "429" in got
    assert calls == ["/api/submissions/sub_x/for-worker"], "nothing may be stored after a refusal"


def test_nothing_finished_means_nothing_asked(monkeypatch):
    """Intake alone is not worth a model call: there is no measurement to
    interpret yet, and advice from an empty page would be invented."""
    monkeypatch.setenv("OPENAI_API_KEY", "k")
    bare = {"benchmark": {"key": "b"}, "gates": [], "dataset": {"detected": {"episodes": 3}}}

    def explode(request, timeout):
        raise AssertionError("no finished gates must mean no OpenAI call")

    monkeypatch.setattr(coach.urllib.request, "urlopen", explode)
    got = coach.maybe_coach("http://api", "sub_x", lambda api, path, payload=None: bare)
    assert "no finished checks" in got


def test_the_happy_path_stores_through_the_transport(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "k")
    stored = {}

    def transport(api, path, payload=None):
        if path.endswith("/for-worker"):
            return record()
        stored[path] = payload
        return {"stored": len(payload["points"])}

    monkeypatch.setattr(
        coach.urllib.request, "urlopen",
        fake_openai(
            ["film the lift slower", "add 30 demos of the handoff"],
            {"language.one_instruction": "Write one instruction per clip."},
        ),
    )
    got = coach.maybe_coach("http://api", "sub_x", transport)
    assert got == "coach: stored 2 point(s), 1 finding note(s)"
    payload = stored["/api/submissions/sub_x/coach"]
    assert payload["points"] == ["film the lift slower", "add 30 demos of the handoff"]
    assert payload["fixes"] == {"language.one_instruction": "Write one instruction per clip."}


# -- the guardrails on the per-finding lines ---------------------------------


def test_a_code_we_never_sent_is_refused(monkeypatch):
    """The allow-list is the whole defence against free association.

    A reply keyed to an invented code would put a generated sentence on the
    page beside a finding it was never grounded in.
    """
    facts = coach.digest(record())
    monkeypatch.setattr(
        coach.urllib.request, "urlopen",
        fake_openai(["p"], {
            "language.one_instruction": "Write one instruction per clip.",
            "invented.by_the_model": "Buy a better robot.",
        }),
    )
    got = coach.ask(facts, key="k")
    assert "language.one_instruction" in got["fixes"]
    assert "invented.by_the_model" not in got["fixes"], (
        "a code the digest never sent survived; the model can now write beside "
        "any finding it invents"
    )


def test_a_blank_line_means_nothing_to_say_and_is_dropped(monkeypatch):
    facts = coach.digest(record())
    monkeypatch.setattr(
        coach.urllib.request, "urlopen",
        fake_openai(["p"], {"language.one_instruction": "   "}),
    )
    assert coach.ask(facts, key="k")["fixes"] == {}


def test_length_is_a_slice_here_too(monkeypatch):
    facts = coach.digest(record())
    monkeypatch.setattr(
        coach.urllib.request, "urlopen",
        fake_openai(["p"], {"language.one_instruction": "x" * 500}),
    )
    got = coach.ask(facts, key="k")
    assert len(got["fixes"]["language.one_instruction"]) == coach.MAX_SAY
