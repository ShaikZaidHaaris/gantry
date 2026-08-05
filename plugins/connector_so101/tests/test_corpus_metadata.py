"""Calibration provenance, the design metadata, and the join that carries it.

Every refusal here protects a failure with no symptom: a corpus that straddles a
recalibration trains and scores exactly like one that does not, and a misjoined
sidecar gives every episode another episode's condition while every structural
check stays green.
"""

from __future__ import annotations

import csv

import pytest
from gantry_connector_so101 import FOLLOWER_REF, SO101Connector
from gantry_connector_so101.fixtures import calibration_hashes, write_corpus

from gantry.errors import ConfigError
from gantry.fixtures import make_clean, make_duration_confound


def corpus(tmp_path, name="c", suite=None, **kwargs):
    suite = suite or make_clean(n=6)
    return write_corpus(suite.episodes, tmp_path / name, **kwargs)


# -- calibration provenance ------------------------------------------------


def test_every_episode_carries_both_arms_calibration_hashes(tmp_path):
    connector = SO101Connector(corpus(tmp_path), embodiment=FOLLOWER_REF)
    calibration = connector.open("episode_000000").meta.extra["calibration"]
    assert set(calibration["sha256"]) == {"leader", "follower"}
    assert calibration["sha256"] == calibration_hashes("cal-a")
    assert calibration["digest"] == connector.calibration_digests[0]


def test_the_digest_is_a_channel_discriminator_so_two_corpora_cannot_pool_silently(tmp_path):
    """The convention matching is not enough: these are the calibration *files*."""
    from gantry.spine import compatible

    a = SO101Connector(corpus(tmp_path, "a"), embodiment=FOLLOWER_REF)
    b = SO101Connector(
        corpus(tmp_path, "b", calibration="cal-b"), embodiment=FOLLOWER_REF
    )
    action_a = a.open("episode_000000").channel("action")
    action_b = b.open("episode_000000").channel("action")
    assert action_a.metadata["calibration_variant"] == action_b.metadata["calibration_variant"]
    verdict = compatible(action_a, action_b)
    assert not verdict.ok
    assert "metadata.mismatch" in verdict.codes()


def test_a_corpus_spanning_a_recalibration_is_refused(tmp_path):
    root = corpus(tmp_path, "mixed", calibration=lambda i: "cal-a" if i < 3 else "cal-b")
    with pytest.raises(ConfigError, match="spans 2 calibrations"):
        SO101Connector(root, embodiment=FOLLOWER_REF)


def test_pooling_across_a_recalibration_is_possible_only_by_saying_so(tmp_path):
    root = corpus(tmp_path, "mixed", calibration=lambda i: "cal-a" if i < 3 else "cal-b")
    connector = SO101Connector(
        root, embodiment=FOLLOWER_REF, allow_mixed_calibration=True
    )
    assert len(connector.calibration_digests) == 2
    assert connector.descriptor().metadata["calibration"]["pooled_across_calibrations"] is True
    # And the channel then declares no single digest, so it cannot claim to
    # match another corpus's calibration either.
    assert connector.open("episode_000000").channel("action").metadata["calibration_digest"] is None


def test_a_truncated_calibration_hash_is_refused(tmp_path):
    root = corpus(tmp_path, "short")
    path = root / "episodes.csv"
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    for row in rows:
        row["calib_sha256_leader"] = row["calib_sha256_leader"][:16]
    _rewrite(path, rows)
    with pytest.raises(ConfigError, match="not a 64-character SHA-256"):
        SO101Connector(root, embodiment=FOLLOWER_REF)


# -- the design metadata ---------------------------------------------------


def test_the_load_bearing_fields_reach_the_record(tmp_path):
    connector = SO101Connector(corpus(tmp_path), embodiment=FOLLOWER_REF)
    episode = connector.open("episode_000002")
    design = episode.meta.extra["corpus"]
    for field in (
        "corpus_id",
        "condition_label",
        "operator_id",
        "session_id",
        "block_id",
        "episode_index_within_session",
        "order_seed",
        "object_start_pose",
        "container_pose",
        "outcome",
        "calibration_sha256",
        "tracking_error",
        "dropped_frames",
    ):
        assert design[field] is not None, field
    assert episode.meta.collected_by == "O1"
    assert design["treatment"] == design["condition_label"]
    assert set(design["tracking_error"]) == set(connector.binding.joint_names)


def test_outcome_tags_become_success_and_keep_their_own_tag(tmp_path):
    tags = ("success", "fail", "recovered", "success", "fail", "recovered")
    connector = SO101Connector(
        corpus(tmp_path, outcome=lambda i: tags[i]), embodiment=FOLLOWER_REF
    )
    got = [
        (episode.labels.success, episode.labels.annotations["outcome"])
        for episode in connector.episodes()
    ]
    assert got[0] == (True, "success")
    assert got[1] == (False, "fail")
    # 'recovered' is a distinct tag and still a demonstration that finished the
    # task. Mapping it to False would make the recovery condition read as a
    # 40 %-failure corpus, which is the one thing it is not.
    assert got[2] == (True, "recovered")
    assert connector.descriptor().provides["outcomes"] is True


def test_an_unknown_outcome_tag_is_refused_rather_than_read_as_failure(tmp_path):
    root = corpus(tmp_path, outcome="aborted")
    with pytest.raises(ConfigError, match="is not one of"):
        SO101Connector(root, embodiment=FOLLOWER_REF)


def test_a_drift_dose_is_carried_and_absence_means_undeclared(tmp_path):
    clean = SO101Connector(corpus(tmp_path, "clean"), embodiment=FOLLOWER_REF)
    assert clean.descriptor().metadata["corpus"]["drift"] is None
    assert clean.descriptor().metadata["corpus"]["derived"] is False

    drifted = SO101Connector(
        corpus(
            tmp_path,
            "drifted",
            drift={"drift_joint": "shoulder_pan.pos", "drift_deg": "17", "drift_mode": "state"},
        ),
        embodiment=FOLLOWER_REF,
    )
    assert drifted.descriptor().metadata["corpus"]["derived"] is True
    assert drifted.open("episode_000000").meta.extra["corpus"]["drift"]["drift_deg"] == "17"


def test_the_condition_and_session_keys_are_on_the_labels_too(tmp_path):
    """A feedback module forms cohorts from labels; it should not have to read meta."""
    connector = SO101Connector(corpus(tmp_path), embodiment=FOLLOWER_REF)
    annotations = connector.open("episode_000000").labels.annotations
    assert annotations["condition_label"] == "E-PRAC"
    assert annotations["session_id"] == "S001"
    assert annotations["episode_index_within_session"] == 0


# -- the block-analysis declaration ----------------------------------------


def test_a_complete_corpus_declares_that_it_supports_the_block_analysis(tmp_path):
    connector = SO101Connector(corpus(tmp_path), embodiment=FOLLOWER_REF)
    assert connector.supports_block_analysis
    assert connector.descriptor().metadata["block_analysis"]["supported"] is True


def test_a_corpus_without_the_block_columns_is_readable_and_says_it_cannot(tmp_path):
    """Readable, and refusing to imply an analysis it cannot support."""
    connector = SO101Connector(
        corpus(tmp_path, "noblocks", block_columns=False), embodiment=FOLLOWER_REF
    )
    assert len(connector.episode_ids()) == 6
    assert connector.supports_block_analysis is False
    declaration = connector.descriptor().metadata["block_analysis"]
    assert declaration["supported"] is False
    assert "session_id" in declaration["missing_columns"]
    assert "time-trend" in declaration["why_not"] or "different question" in declaration["why_not"]
    # And it is on every record, not only on the descriptor.
    assert connector.open("episode_000000").meta.extra["block_analysis"]["supported"] is False


def test_a_corpus_with_no_sidecar_at_all_is_readable_and_says_why(tmp_path):
    connector = SO101Connector(
        corpus(tmp_path, "bare", sidecar=False), embodiment=FOLLOWER_REF
    )
    assert len(connector.episode_ids()) == 6
    assert connector.supports_block_analysis is False
    assert "no per-episode metadata sidecar" in connector.descriptor().metadata["block_analysis"]["why_not"]
    assert connector.open("episode_000000").meta.extra["corpus"] is None
    assert connector.descriptor().provides["outcomes"] is False


def test_strict_metadata_turns_the_declaration_into_a_refusal(tmp_path):
    root = corpus(tmp_path, "noblocks", block_columns=False)
    with pytest.raises(ConfigError, match="complete design record"):
        SO101Connector(root, embodiment=FOLLOWER_REF, strict_metadata=True)


def test_a_sidecar_named_and_absent_is_an_error_not_a_shrug(tmp_path):
    root = corpus(tmp_path, "bare", sidecar=False)
    with pytest.raises(ConfigError, match="no such metadata sidecar"):
        SO101Connector(root, embodiment=FOLLOWER_REF, sidecar=root / "episodes.csv")


# -- the join --------------------------------------------------------------


def test_a_keyed_join_is_used_when_the_sidecar_has_an_episode_index(tmp_path):
    connector = SO101Connector(corpus(tmp_path), embodiment=FOLLOWER_REF)
    assert connector.descriptor().metadata["metadata_join"]["method"] == "index"


def test_an_ordinal_join_is_accepted_when_the_frame_counts_can_check_it(tmp_path):
    root = corpus(
        tmp_path, "ordinal", suite=make_duration_confound(n=6), episode_index_column=False
    )
    connector = SO101Connector(root, embodiment=FOLLOWER_REF)
    join = connector.descriptor().metadata["metadata_join"]
    assert join["method"] == "ordinal"
    assert "num_frames" in join["verified_against"]


def test_an_ordinal_join_nothing_can_check_is_refused(tmp_path):
    """Every episode the same length, so a shuffled sidecar looks identical."""
    root = corpus(tmp_path, "vacuous", episode_index_column=False)
    with pytest.raises(ConfigError, match="unverifiable ordinal join"):
        SO101Connector(root, embodiment=FOLLOWER_REF)


def test_that_join_can_be_forced_and_then_says_it_rests_on_the_caller(tmp_path):
    root = corpus(tmp_path, "vacuous", episode_index_column=False)
    connector = SO101Connector(root, embodiment=FOLLOWER_REF, join="ordinal")
    join = connector.descriptor().metadata["metadata_join"]
    assert join["verified_against"] == "nothing"
    assert "caller's instruction" in join["note"]


def test_a_shuffled_sidecar_is_caught_by_the_frame_counts(tmp_path):
    root = corpus(
        tmp_path,
        "shuffled",
        suite=make_duration_confound(n=6),
        episode_index_column=False,
        block_columns=False,
    )
    path = root / "episodes.csv"
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    _rewrite(path, list(reversed(rows)))
    with pytest.raises(ConfigError, match="not in the dataset's order"):
        SO101Connector(root, embodiment=FOLLOWER_REF)


def test_a_sidecar_with_the_wrong_number_of_rows_is_refused(tmp_path):
    root = corpus(tmp_path, "short", episode_index_column=False)
    path = root / "episodes.csv"
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    _rewrite(path, rows[:-1])
    with pytest.raises(ConfigError, match="metadata row"):
        SO101Connector(root, embodiment=FOLLOWER_REF)


def test_a_malformed_number_is_refused_rather_than_dropped(tmp_path):
    root = corpus(tmp_path)
    path = root / "episodes.csv"
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    rows[2]["order_seed"] = "twenty"
    _rewrite(path, rows)
    with pytest.raises(ConfigError, match="not an integer"):
        SO101Connector(root, embodiment=FOLLOWER_REF)


def _rewrite(path, rows):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
