"""The ego pipeline end to end, from an upload to a report.

Six plugins, none of which imports another except through a declared dependency,
wired the way the product wires them. This lives in ``tests/integration`` because
a plugin's own suite may not import a sibling — and the thing worth testing here
is precisely that they compose, which no single plugin's tests can show.

The path:

    a folder of clips + a manifest      connector_egovideo
      -> the same clips with hands      connector_handpose
      -> one arm's pose commands        retargeter_hands
      -> is this data even in scope     feedback_coverage
      -> what to change about filming   feedback_capture

What this test is really checking is that the refusals survive composition. Each
plugin refuses things on its own and passes its own tests; the question is
whether an unscaled estimate stays unscaled all the way to the retargeter, and
whether a count written at ingest is still readable by a module four steps later.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
from gantry_connector_egovideo import RGB, EgoVideoConnector
from gantry_connector_handpose import HandPoseConnector, Track
from gantry_feedback_capture import Capture, top_fixes
from gantry_feedback_coverage import Coverage
from gantry_retargeter_hands import (
    VIPERX_300,
    Hand,
    HandToArm,
    Mount,
    arm_command,
    assemble,
    hand_command,
)

from gantry.contracts.feedback import Cohort
from gantry.errors import ConfigError
from gantry.spine import ChannelSpec

STEPS = 24
SIZE = 8

#: What the benchmark evaluates. Kitchen work, so the matching upload is in scope
#: and the tabletop one is not.
EVALUATES = [
    "pick up the mug and put it in the sink",
    "open the fridge door",
    "wipe the counter",
    "put the plate in the cupboard",
]


def hand_at(x: float) -> np.ndarray:
    """Twenty-one MediaPipe joints with a well-defined palm frame."""
    points = np.zeros((21, 3), dtype="float32")
    points[0] = (x, 0.0, 0.0)
    points[5] = (x + 0.05, 0.0, 0.0)
    points[17] = (x, 0.05, 0.0)
    points[4] = (x + 0.05, -0.05, 0.0)
    points[8] = (x + 0.10, 0.0, 0.0)
    return points


class Tracker:
    """A hand estimator, scripted so the test controls what the report will say."""

    def __init__(self, confidence: float = 0.95, drift: float = 0.004):
        self.confidence = confidence
        self.drift = drift

    def estimate(self, frames):
        steps = len(frames)
        points = np.stack([hand_at(index * self.drift) for index in range(steps)])
        return Track(
            keypoints={"left": points, "right": points},
            confidence={
                hand: np.full(steps, self.confidence, dtype="float32") for hand in ("left", "right")
            },
            convention="mediapipe",
            metadata={"estimator": "scripted"},
        )


def upload(tmp_path, clips, name="upload"):
    """Write a folder of clips and a manifest, the way a GUI would."""
    root = tmp_path / name
    root.mkdir()
    entries = []
    for index, (instruction, scene) in enumerate(clips):
        (root / f"{index}.mp4").write_bytes(b"stand-in for video")
        entries.append({"path": f"{index}.mp4", "instruction": instruction, "scene": scene})
    (root / "clips.json").write_text(json.dumps(entries))
    return root


def header(path, **over):
    return {
        "frames": STEPS,
        "height": SIZE,
        "width": SIZE,
        "rate_hz": 30.0,
        "seconds": STEPS / 30.0,
        **over,
    }


def frames(path, stride=1, limit=None):
    return np.zeros((STEPS // max(1, stride), SIZE, SIZE, 3), dtype="uint8")


def ingest(root, *, tracker=None):
    """The two connector steps, as the pipeline runs them."""
    video = EgoVideoConnector(root, prober=header, decoder=frames)
    return video, HandPoseConnector(video, estimator=tracker or Tracker())


def cohort_of(connector, name="theirs"):
    return Cohort(
        name=name,
        episodes=tuple(connector.open(i) for i in connector.episode_ids()),
    )


# -- the whole path ----------------------------------------------------------


def test_an_upload_becomes_arm_commands_with_its_lineage_intact(tmp_path):
    """The end-to-end claim: a folder of video comes out the other side as
    something a robot could execute, and every step of how it got there is
    recoverable from the record."""
    root = upload(
        tmp_path,
        [
            ("pick up the mug and put it in the sink", "kitchen-1"),
            ("open the fridge door", "kitchen-2"),
        ],
    )
    video, hands = ingest(root)

    episode = hands.open(hands.episode_ids()[0])
    assert episode.meta.derived_from == ("ego/0",)
    assert episode.meta.embodiment == "human"
    assert episode.meta.extra["scene"] == "kitchen-1"
    assert RGB in episode.channel_names

    # The estimate is labelled as an estimate, four steps from where it was made.
    assert episode.channel("right_wrist").metadata["scale"] == "unscaled"

    retargeter = HandToArm(
        mount=Mount.aligned(),
        hand=Hand(closed=0.02, open=0.10, span=0.19, measured_by="tape"),
        reach=VIPERX_300,
    )
    source = hand_command("right_wrist", hand="right", scale="unscaled", rotation_repr="quat_wxyz")
    target = arm_command("right_arm", arm="right", rotation_repr="euler_xyz")

    values = np.concatenate(
        [episode.array("right_wrist"), episode.array("right_aperture")[:, None]], axis=1
    )
    commands = retargeter.apply(values, source, target)
    assert commands.shape == (STEPS, 7)
    assert np.isfinite(commands).all()
    # The gripper came out as a fraction of that person's own measured travel.
    assert 0.0 <= commands[:, -1].min() <= commands[:, -1].max() <= 1.0


def test_the_unscaled_estimate_is_still_refused_at_the_far_end(tmp_path):
    """The composition test that matters most. The connector marked the hands
    unscaled; four steps later a retargeter with no measured hand span still
    refuses them, rather than the marking having quietly worn off in transit."""
    root = upload(tmp_path, [("pick up the mug", "kitchen-1")])
    _video, hands = ingest(root)
    episode = hands.open(hands.episode_ids()[0])

    unmeasured = HandToArm(
        mount=Mount.aligned(),
        hand=Hand(closed=0.02, open=0.10),  # no span
    )
    source = hand_command("right_wrist", scale=episode.channel("right_wrist").metadata["scale"])
    verdict = unmeasured.accepts(source, arm_command("right_arm"))

    assert not verdict.ok
    assert "hands.unscaled_without_reference" in verdict.explain()


def test_both_hands_assemble_into_a_bimanual_command_in_the_declared_order(tmp_path):
    """The other end of the discriminator policy_pi0 declares: a fourteen-wide
    vector is the same object whichever arm comes first."""
    root = upload(tmp_path, [("pick up the mug", "kitchen-1")])
    _video, hands = ingest(root)
    episode = hands.open(hands.episode_ids()[0])

    person = Hand(closed=0.02, open=0.10, span=0.19)
    retargeter = HandToArm(mount=Mount.aligned(), hand=person)
    target = arm_command("arm", rotation_repr="euler_xyz")

    per_arm = {}
    for hand in ("left", "right"):
        values = np.concatenate(
            [episode.array(f"{hand}_wrist"), episode.array(f"{hand}_aperture")[:, None]],
            axis=1,
        )
        per_arm[hand] = retargeter.apply(
            values, hand_command(f"{hand}_wrist", hand=hand, scale="unscaled"), target
        )

    bimanual = ChannelSpec(
        "action",
        "vector",
        (14,),
        "float32",
        semantics="actuation",
        dim_labels=tuple(
            [f"left_{index}" for index in range(7)] + [f"right_{index}" for index in range(7)]
        ),
    )
    merged = assemble(per_arm, bimanual)
    assert merged.shape == (STEPS, 14)
    assert np.allclose(merged[:, :7], per_arm["left"])
    assert np.allclose(merged[:, 7:], per_arm["right"])


# -- the report --------------------------------------------------------------


def test_kitchen_footage_on_a_kitchen_benchmark_reads_as_in_scope(tmp_path):
    root = upload(
        tmp_path,
        [
            ("pick up the mug and put it in the sink", "kitchen-1"),
            ("open the fridge door", "kitchen-2"),
            ("wipe the counter", "kitchen-3"),
            ("put the plate in the cupboard", "kitchen-1"),
        ],
    )
    _video, hands = ingest(root)
    report = Coverage(evaluates=EVALUATES).analyse([cohort_of(hands)])

    assert [finding.code for finding in report.findings] == ["coverage.ample"]
    assert report.measurements["theirs.coverage"].value == 1.0


def test_tabletop_footage_on_a_kitchen_benchmark_reads_as_a_task_mismatch(tmp_path):
    """The unfairness the whole coverage module exists to prevent, checked
    through the real ingest path rather than on hand-built episodes."""
    root = upload(
        tmp_path,
        [
            ("stack the red block on the green block", "lab-1"),
            ("insert the peg into the hole", "lab-1"),
            ("push the block to the target", "lab-1"),
        ],
    )
    _video, hands = ingest(root)
    report = Coverage(evaluates=EVALUATES).analyse([cohort_of(hands)])

    codes = [finding.code for finding in report.findings]
    assert "coverage.mismatch" in codes
    finding = next(f for f in report.findings if f.code == "coverage.mismatch")
    assert "task match rather than about this data" in finding.summary


def test_the_counts_written_at_ingest_are_readable_by_the_report(tmp_path):
    """Four steps and two planes apart. The estimator wrote hands_visible; the
    feedback module reads it off the record without either knowing about the
    other."""
    root = upload(tmp_path, [("pick up the mug", "kitchen-1")] * 6)
    _video, hands = ingest(root, tracker=Tracker(confidence=0.2))

    report = Capture().analyse([cohort_of(hands)])
    codes = [finding.code for finding in report.findings]

    assert "capture.hands_offscreen" in codes
    assert report.measurements["theirs.hands_visible"].value == 0.0
    # Six clips, one location, one instruction: both variety checks fire too.
    assert "capture.single_scene" in codes
    assert "capture.one_instruction" in codes


def test_the_reach_report_feeds_the_filming_advice(tmp_path):
    """The retargeter measured what fraction of the reaching the arm could
    follow; the capture module turns it into a sentence about standing closer."""
    root = upload(tmp_path, [("pick up the mug", "kitchen-1")])
    _video, hands = ingest(root)
    episode = hands.open(hands.episode_ids()[0])

    retargeter = HandToArm(
        mount=Mount.aligned(),
        hand=Hand(closed=0.02, open=0.10, span=0.19),
        reach=VIPERX_300,
    )
    values = np.concatenate(
        [episode.array("right_wrist"), episode.array("right_aperture")[:, None]], axis=1
    )
    reach = retargeter.reach_report(values, hand_command("right_wrist", scale="unscaled"))
    assert reach["measured"] is True
    assert 0.0 <= reach["in_reach"] <= 1.0

    # That number is exactly the key feedback_capture reads.
    with_reach = Cohort(
        name="theirs",
        episodes=tuple(
            record.with_labels(
                type(record.labels)(
                    success=record.labels.success,
                    stage_events=record.labels.stage_events,
                    annotations={**dict(record.labels.annotations), "in_reach": 0.3},
                )
            )
            for record in cohort_of(hands).episodes
        ),
    )
    report = Capture().analyse([with_reach])
    finding = next(f for f in report.findings if f.code == "capture.out_of_workspace")
    assert "Stand closer" in finding.prescription


def test_a_good_upload_and_a_poor_one_get_visibly_different_reports(tmp_path):
    """What the product actually delivers: two uploads through one pipeline, and
    a difference somebody can act on."""
    careful = upload(
        tmp_path,
        [
            ("pick up the mug and put it in the sink", "kitchen-1"),
            ("open the fridge door", "kitchen-2"),
            ("wipe the counter", "kitchen-3"),
            ("put the plate in the cupboard", "kitchen-4"),
        ],
        name="careful",
    )
    careless = upload(
        tmp_path,
        [("pick up the mug", "kitchen-1")] * 4,
        name="careless",
    )

    _v1, good = ingest(careful)
    _v2, bad = ingest(careless, tracker=Tracker(confidence=0.3, drift=0.9))

    good_report = Capture().analyse([cohort_of(good, "careful")])
    bad_report = Capture().analyse([cohort_of(bad, "careless")])

    # Nothing fixable. Not "capture.clean" — three checks had nothing to read on
    # this path (no workspace measured, no stabilisation flag, no outcomes), and
    # a clean bill of health is only issued when every check actually ran.
    assert [f.code for f in good_report.findings] == []
    assert any("not measured is not the same as fine" in note for note in good_report.notes)

    fixes = top_fixes(bad_report)
    assert len(fixes) == 3
    assert any("out of frame" in fix for fix in fixes)
    assert any("faster than the arm can follow" in fix for fix in fixes)
    # And nothing in the advice promises a score.
    assert not any("you would score" in fix for fix in fixes)


# -- the refusals at the front door survive too ------------------------------


def test_a_clip_with_no_instruction_never_reaches_the_pipeline(tmp_path):
    root = tmp_path / "bad"
    root.mkdir()
    (root / "0.mp4").write_bytes(b"x")
    (root / "clips.json").write_text(json.dumps([{"path": "0.mp4", "scene": "kitchen-1"}]))

    with pytest.raises(ConfigError, match="no instruction"):
        EgoVideoConnector(root, prober=header, decoder=frames)
