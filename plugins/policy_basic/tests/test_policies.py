"""Reference policies, and the conformance kit that judges them."""

from __future__ import annotations

import numpy as np
import pytest
from gantry_policy_basic import ConstantPolicy, NoisyReplayPolicy, ReplayPolicy

from gantry.conformance import check_policy, policy_checks
from gantry.contracts.policy import EpisodeContext, Observation
from gantry.fixtures import make_clean

SUITE = make_clean(n=4, seed=1)
ACTION = SUITE.episodes[0].channel("action")


def observations(n: int = 5):
    data = SUITE.episodes[0].read()
    return [Observation(i, {k: v[i] for k, v in data.items()}) for i in range(n)]


@pytest.mark.parametrize(
    "policy",
    [
        ReplayPolicy(ACTION),
        ReplayPolicy(ACTION, chunk=8),
        ConstantPolicy(ACTION),
        ConstantPolicy(ACTION, value=0.5),
        NoisyReplayPolicy(ACTION, sigma=0.1),
    ],
    ids=["replay", "replay-chunked", "constant", "constant-half", "noisy"],
)
def test_conforms(policy):
    verdict = check_policy(policy, observations(), strict=True)
    assert verdict.ok, verdict.explain()


def test_the_kit_is_discoverable():
    assert "determinism" in dict(policy_checks())


def test_replay_returns_exactly_what_was_recorded():
    policy = ReplayPolicy(ACTION)
    for observation in observations():
        assert np.allclose(policy.act(observation)[0], observation["action"])


def test_a_chunked_policy_repeats_across_its_horizon():
    assert ReplayPolicy(ACTION, chunk=8).act(observations()[0]).shape[0] == 8


def test_constant_ignores_the_observation():
    policy = ConstantPolicy(ACTION, value=0.25)
    first, second = (policy.act(o)[0] for o in observations()[:2])
    assert np.allclose(first, second) and np.allclose(first, 0.25)


def test_noise_is_reproducible_within_an_episode():
    policy = NoisyReplayPolicy(ACTION, sigma=0.1, seed=3)
    policy.reset(EpisodeContext("ep"))
    first = policy.act(observations()[0])
    policy.reset(EpisodeContext("ep"))
    assert np.array_equal(first, policy.act(observations()[0]))


def test_different_episodes_get_different_noise():
    policy = NoisyReplayPolicy(ACTION, sigma=0.1, seed=3)
    policy.reset(EpisodeContext("a"))
    a = policy.act(observations()[0])
    policy.reset(EpisodeContext("b"))
    assert not np.array_equal(a, policy.act(observations()[0]))


def test_a_negative_sigma_is_refused():
    with pytest.raises(ValueError, match="must not be negative"):
        NoisyReplayPolicy(ACTION, sigma=-1.0)


def test_a_chunk_below_one_is_refused():
    with pytest.raises(ValueError, match="at least 1"):
        ReplayPolicy(ACTION, chunk=0)


# -- the kit must catch a policy that misbehaves ---------------------------


class _Liar(ReplayPolicy):
    """Claims determinism and does not have it."""

    def act(self, observation):
        return super().act(observation) + np.random.default_rng().normal(0, 1, (1, 4))


def test_a_false_determinism_claim_is_caught():
    verdict = check_policy(_Liar(ACTION), observations())
    assert "conformance.not_deterministic" in verdict.codes()


class _Overrun(ReplayPolicy):
    def act(self, observation):
        return np.repeat(super().act(observation), 4, axis=0)


def test_a_chunk_larger_than_declared_is_caught():
    verdict = check_policy(_Overrun(ACTION), observations())
    assert "conformance.chunk_overrun" in verdict.codes()


class _Single(ReplayPolicy):
    def act(self, observation):
        return np.asarray(observation["action"])  # a bare action, not a chunk


def test_returning_a_bare_action_instead_of_a_chunk_is_caught():
    verdict = check_policy(_Single(ACTION), observations())
    assert "conformance.chunk_rank" in verdict.codes()


def test_the_kit_needs_something_to_run_on():
    assert "conformance.no_observations" in check_policy(ReplayPolicy(ACTION), []).codes()
