"""A model grading against a written rubric, and everything that keeps it honest.

The wire is replaced by a recorded tape throughout. That is not a convenience for
testing — it is the same mechanism a re-analysis six months from now uses, so
exercising it here is exercising the real path rather than a stub of it.
"""

from __future__ import annotations

import json

import pytest
from gantry_scorer_vlm import (
    VlmScorer,
    build_prompt,
    parse_answer,
    prompt_hash,
    replay,
    replay_by_trial,
)

from gantry.conformance import check_scorer
from gantry.contracts.scorer import VIDEO, Evidence
from gantry.contracts.task import Criterion, TaskDefinition
from gantry.errors import ConfigError

RUBRIC = (
    "The cube is clear of the table by at least 4 cm and held in the gripper. "
    "A cube nudged off the edge or dropped immediately does not count."
)
HELD = "The cube stays in the gripper for at least one full second."


def task(*criteria):
    return TaskDefinition(
        name="lift_cube",
        instruction="lift the cube",
        success=criteria or (Criterion("lifted", {"height": 0.04}, RUBRIC),),
    )


def saying(text: str) -> VlmScorer:
    return VlmScorer(lambda prompt, frames, trial="": text, model="test")


# -- the prompt --------------------------------------------------------------


def test_the_prompt_quotes_every_rubric_verbatim():
    """Paraphrasing for the model would make the agreement number about two
    different standards, and it would still look like a number."""
    prompt = build_prompt(task(Criterion("lifted", {}, RUBRIC), Criterion("held", {}, HELD)))
    assert RUBRIC in prompt
    assert HELD in prompt


def test_the_prompt_carries_the_instruction_the_robot_was_given():
    assert "lift the cube" in build_prompt(task())


def test_the_prompt_instructs_abstention_rather_than_guessing():
    prompt = build_prompt(task())
    assert "unclear" in prompt
    assert "guessing is not" in prompt


def test_the_prompt_forbids_partial_credit():
    """Graded answers cost roughly twenty points of agreement."""
    assert "partial credit" in build_prompt(task())


def test_a_task_with_no_criteria_is_refused():
    with pytest.raises(ConfigError, match="no success criterion"):
        build_prompt(TaskDefinition(name="empty", instruction="do nothing"))


def test_the_prompt_hash_separates_a_drifted_prompt_from_a_changed_model():
    one = prompt_hash(build_prompt(task()))
    other = prompt_hash(build_prompt(task(Criterion("lifted", {}, "Something else."))))
    assert one != other
    assert len(one) == 12


# -- parsing -----------------------------------------------------------------


def test_a_clean_verdict_is_read():
    answer = parse_answer("lifted: yes\nwhy lifted: held clear of the table", ["lifted"])
    assert answer.verdicts == {"lifted": True}
    assert answer.reasons["lifted"] == "held clear of the table"
    assert answer.parsed


def test_unclear_is_an_abstention_not_a_failure():
    answer = parse_answer("lifted: unclear", ["lifted"])
    assert answer.verdicts["lifted"] is None
    assert answer.parsed


def test_prose_abstains_rather_than_being_interpreted():
    """A parser that accepts prose reads 'it seems like it might have worked'
    as a yes."""
    answer = parse_answer("I think it probably worked, hard to say", ["lifted"])
    assert answer.verdicts["lifted"] is None
    assert not answer.parsed


def test_a_criterion_the_task_never_asked_about_is_dropped():
    answer = parse_answer("lifted: no\ngrasped: yes", ["lifted"])
    assert answer.verdicts == {"lifted": False}


def test_a_skipped_criterion_abstains():
    answer = parse_answer("lifted: yes", ["lifted", "held"])
    assert answer.verdicts["lifted"] is True
    assert answer.verdicts["held"] is None


def test_an_empty_reply_abstains():
    assert parse_answer("", ["lifted"]).verdicts["lifted"] is None


def test_case_and_surrounding_text_do_not_break_it():
    answer = parse_answer("Here is my grading.\nLIFTED: Yes\nThat's all.", ["lifted"])
    assert answer.verdicts["lifted"] is True


# -- judging -----------------------------------------------------------------


def test_it_judges_against_the_rubric_and_records_why():
    scorer = saying("lifted: yes\nwhy lifted: the cube is clear and held")
    judged = scorer.score(Evidence("seed_0", video="seed_0.mp4"), task())
    assert judged[0].passed is True
    assert "clear and held" in judged[0].rationale


def test_it_returns_one_judgement_per_criterion_in_order():
    scorer = saying("held: no\nlifted: yes")
    judged = scorer.score(
        Evidence("s", video="s.mp4"),
        task(Criterion("lifted", {}, RUBRIC), Criterion("held", {}, HELD)),
    )
    assert [j.criterion for j in judged] == ["lifted", "held"]
    assert [j.passed for j in judged] == [True, False]


def test_one_abstention_leaves_the_whole_trial_unknown():
    scorer = saying("lifted: yes\nheld: unclear")
    judged = scorer.score(
        Evidence("s", video="s.mp4"),
        task(Criterion("lifted", {}, RUBRIC), Criterion("held", {}, HELD)),
    )
    assert scorer.overall(judged) is None


def test_a_failed_ask_abstains_rather_than_crashing_the_run():
    def explodes(prompt, frames, trial=""):
        raise RuntimeError("the endpoint is down")

    scorer = VlmScorer(explodes, model="test")
    judged = scorer.judge(Evidence("s", video=None), task())
    assert judged[0].passed is None
    assert "asking failed" in judged[0].rationale


def test_an_undecodable_video_abstains():
    """Missing evidence, not a broken judge."""
    scorer = saying("lifted: unclear\nwhy lifted: no frames were provided")
    assert scorer.judge(Evidence("s", video="/dev/null"), task())[0].passed is None


# -- what it declares --------------------------------------------------------


def test_it_does_not_claim_determinism_even_at_temperature_zero():
    """A served model can change under you without changing its name, and
    claiming determinism would make agreement numbers unfalsifiable."""
    scorer = VlmScorer(lambda p, f, t="": "lifted: yes", model="x", temperature=0.0)
    assert scorer.deterministic is False


def test_it_declares_itself_cheap_and_video_reading():
    scorer = saying("lifted: yes")
    assert scorer.cost == "cheap"
    assert scorer.needs == (VIDEO,)


def test_the_descriptor_records_the_model_and_temperature():
    provides = (
        VlmScorer(lambda p, f, t="": "lifted: yes", model="claude-x", temperature=0.3)
        .descriptor()
        .metadata
    )
    assert provides["model"] == "claude-x"
    assert provides["temperature"] == 0.3


def test_it_passes_the_conformance_kit():
    scorer = saying("lifted: unclear\nwhy lifted: nothing was visible")
    verdict = check_scorer(scorer, Evidence("s", video=None), task())
    assert verdict.ok, verdict.explain()


# -- the audit trail ---------------------------------------------------------


def test_every_asking_is_transcribed():
    scorer = saying("lifted: yes\nwhy lifted: seen")
    scorer.judge(Evidence("seed_0", video=None), task())
    scorer.judge(Evidence("seed_1", video=None), task())
    assert len(scorer.transcripts) == 2
    first = scorer.transcripts[0]
    assert first.trial == "seed_0"
    assert first.raw == "lifted: yes\nwhy lifted: seen"
    assert first.model == "test"
    assert first.prompt_hash


def test_a_failed_ask_is_transcribed_too():
    def explodes(prompt, frames, trial=""):
        raise RuntimeError("down")

    scorer = VlmScorer(explodes, model="test")
    scorer.judge(Evidence("s", video=None), task())
    assert "RuntimeError" in scorer.transcripts[0].raw


def test_transcripts_round_trip_so_reanalysis_needs_no_inference(tmp_path):
    """The property that makes a stored judgement re-checkable: fixing a parser
    or recomputing agreement must not require paying for the model again."""
    scorer = saying("lifted: yes\nwhy lifted: seen")
    for i in range(3):
        scorer.judge(Evidence(f"seed_{i}", video=None), task())
    path = scorer.write_transcripts(tmp_path / "t.jsonl")
    judge, labels = VlmScorer.recorded_labels(path)
    assert judge == "test"
    assert set(labels) == {"seed_0", "seed_1", "seed_2"}
    assert labels["seed_0"]["lifted"] is True


def test_reading_an_empty_transcript_is_refused(tmp_path):
    path = tmp_path / "empty.jsonl"
    path.write_text("")
    with pytest.raises(ConfigError, match="no transcript"):
        VlmScorer.recorded_labels(path)


def test_a_tape_replays_what_was_recorded():
    scorer = VlmScorer(replay_by_trial({"seed_0": "lifted: no"}), model="taped")
    assert scorer.judge(Evidence("seed_0", video=None), task())[0].passed is False


def test_a_tape_that_runs_out_abstains_rather_than_inventing():
    scorer = VlmScorer(replay({}), model="taped")
    assert scorer.judge(Evidence("seed_0", video=None), task())[0].passed is None


# -- the surface it produces -------------------------------------------------


def test_annotating_writes_the_prompt_out_for_a_person_to_read(tmp_path):
    """A prompt nobody has read is a prompt nobody can criticise."""
    path = VlmScorer.annotate(
        [{"trial": "seed_0", "video": "seed_0.mp4"}], task(), tmp_path / "prompt.json"
    )
    payload = json.loads(path.read_text())
    assert RUBRIC in payload["prompt"]
    assert payload["prompt_hash"]
    assert payload["trials"][0]["trial"] == "seed_0"


# -- frame extraction --------------------------------------------------------


def test_frames_are_real_png_spread_across_the_attempt(tmp_path):
    """Pinned because it degraded silently once.

    Frames that are not really PNG reach a model as garbage, it dutifully
    abstains, and the result is indistinguishable from an honest judge looking
    at unclear footage — a failure with no symptom.
    """
    av = pytest.importorskip("av")
    pytest.importorskip("PIL")
    import numpy as np
    from gantry_scorer_vlm import frames_from_video

    path = tmp_path / "trial.mp4"
    with av.open(str(path), "w") as container:
        stream = container.add_stream("libx264", rate=20)
        stream.width, stream.height = 64, 64
        stream.pix_fmt = "yuv420p"
        for i in range(30):
            image = np.full((64, 64, 3), i * 8, dtype=np.uint8)
            for packet in stream.encode(av.VideoFrame.from_ndarray(image, format="rgb24")):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)

    frames = frames_from_video(path, count=8)
    assert len(frames) == 8
    assert all(frame[:8] == b"\x89PNG\r\n\x1a\n" for frame in frames)
    # Spread rather than the opening eight: most criteria are settled at the end,
    # and a judge shown only the start has been set up to abstain.
    assert len({bytes(frame) for frame in frames}) == 8


# -- the wires ---------------------------------------------------------------


def test_a_wire_sends_frames_in_order_with_the_prompt_last():
    """Order matters: the frames are a sequence in time, and a model told to
    read them out of order is being asked a different question."""
    from gantry_scorer_vlm import anthropic_wire

    seen = {}

    class FakeMessages:
        def create(self, **kwargs):
            seen.update(kwargs)

            class Block:
                type = "text"
                text = "lifted: yes"

            class Reply:
                content = [Block()]

            return Reply()

    class FakeClient:
        messages = FakeMessages()

    ask = anthropic_wire("claude-test", client=FakeClient())
    assert ask("the prompt", [b"one", b"two"]) == "lifted: yes"
    content = seen["messages"][0]["content"]
    assert [block["type"] for block in content] == ["image", "image", "text"]
    assert content[-1]["text"] == "the prompt"
    assert seen["temperature"] == 0.0


def test_caching_means_a_trial_is_paid_for_once():
    """An interrupted scoring pass, or one rubric changing, must not re-bill
    every trial."""
    from gantry_scorer_vlm import cached

    calls = []

    def ask(prompt, frames, trial=""):
        calls.append(trial)
        return "lifted: yes"

    store = {}
    wrapped = cached(ask, store)
    wrapped("p", [], "seed_0")
    wrapped("p", [], "seed_0")
    wrapped("p", [], "seed_1")
    assert calls == ["seed_0", "seed_1"]
    assert set(store) == {"seed_0", "seed_1"}
