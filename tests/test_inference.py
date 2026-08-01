"""The statistics, checked against the properties that make them worth having.

Not checked against hand-computed constants, mostly. A test asserting that some
function returns 0.0296 tells you the number did not change; it does not tell
you the number is right. What these check instead is the behaviour that makes
each function the correct choice — that a time-uniform interval really does hold
at every sample size, that an aggregate really is robust to one wild task, that
a chance-corrected agreement really does punish a judge who is guessing.

The numeric cross-checks against scipy, rliable, ppi_py and krippendorff live in
``plugins/feedback_core/tests`` where those libraries are declared. Core stays
numpy-only, and the agreement between our arithmetic and theirs is checked
mechanically rather than asserted here.
"""

from __future__ import annotations

import numpy as np
import pytest

from gantry.spine.inference import (
    AGREEMENT_TENTATIVE,
    AGREEMENT_TRUSTED,
    agreement_verdict_code,
    alpha_spent,
    barnard,
    cohen_kappa,
    confidence_sequence,
    holm,
    iqm,
    krippendorff_alpha,
    mmrv,
    performance_profile,
    ppi_mean,
    prob_improvement,
    stratified_bootstrap,
    trials_needed,
)

# -- sizing ------------------------------------------------------------------


def test_smaller_effects_cost_more_trials():
    assert (
        trials_needed(0.35, 0.30)
        < trials_needed(0.35, 0.10)
        < trials_needed(0.35, 0.05)
        < trials_needed(0.35, 0.03)
    )


def test_a_predicted_effect_of_nothing_is_uncountable():
    """No number of trials distinguishes a policy from itself."""
    assert trials_needed(0.35, 0.0, cap=500) == 500


def test_the_sizes_it_reports_are_the_ones_this_project_actually_needed():
    # 20 trials was the habit; these are the numbers that showed it was wrong.
    # Even a thirty-point difference -- enormous, the kind nobody would need
    # statistics to notice -- needs more than twenty paired trials to be sure
    # of catching. A ten-point one needs an order of magnitude more.
    assert trials_needed(0.35, 0.30) > 20
    assert trials_needed(0.35, 0.10) > 5 * trials_needed(0.35, 0.30) / 2


def test_the_power_argument_is_honoured_rather_than_decorative():
    """It used to be accepted and ignored, so every number returned was the
    count at which the *expected* result was just barely significant -- a coin
    flip on catching a real effect, sold as a plan."""
    assert (
        trials_needed(0.12, 0.08, power=0.5)
        < trials_needed(0.12, 0.08, power=0.8)
        < trials_needed(0.12, 0.08, power=0.9)
    )


def test_a_budget_meets_the_power_it_was_sized_for():
    """The contract, checked directly rather than through the search."""
    from gantry.spine.inference import _detection_rate

    needed = trials_needed(0.12, 0.08, power=0.8)
    assert _detection_rate(needed, 0.16, 0.75, 0.05) >= 0.8
    assert _detection_rate(needed - 1, 0.16, 0.75, 0.05) < 0.8


# -- stopping ----------------------------------------------------------------


def test_a_confidence_sequence_narrows_and_never_widens_below_its_bound():
    rng = np.random.default_rng(0)
    xs = (rng.random(200) < 0.6).astype(float)
    cs = confidence_sequence(xs)
    widths = [hi - lo for lo, hi in zip(cs.lower, cs.upper)]
    assert widths[-1] < widths[19] < widths[4]
    assert all(0.0 <= lo <= hi <= 1.0 for lo, hi in zip(cs.lower, cs.upper))


def test_every_interval_in_the_sequence_covers_the_truth():
    """The property that makes peeking legal: all of them hold, not just the last."""
    truth = 0.6
    misses = 0
    for seed in range(40):
        rng = np.random.default_rng(seed)
        cs = confidence_sequence((rng.random(120) < truth).astype(float))
        if any(not (lo <= truth <= hi) for lo, hi in zip(cs.lower, cs.upper)):
            misses += 1
    # At alpha=0.05 the guarantee is over the whole path, so a handful of
    # sequences may miss; a majority missing would mean the bound is broken.
    assert misses <= 4, f"{misses}/40 sequences excluded the truth at some point"


def test_it_reports_when_a_comparison_first_separated():
    rng = np.random.default_rng(1)
    cs = confidence_sequence((rng.random(400) < 0.9).astype(float))
    first = cs.separated_from(0.3)
    assert first is not None and first > 1


def test_it_reports_nothing_when_the_comparison_never_separated():
    rng = np.random.default_rng(2)
    cs = confidence_sequence((rng.random(30) < 0.5).astype(float))
    assert cs.separated_from(0.5) is None


def test_it_refuses_observations_outside_the_unit_interval():
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        confidence_sequence([0.0, 1.0, 2.0])


def test_correcting_for_a_fixed_number_of_looks_is_expressible():
    assert alpha_spent(1) == pytest.approx(0.05)
    assert alpha_spent(10) == pytest.approx(0.005)


# -- exact tests -------------------------------------------------------------


def test_barnard_separates_this_projects_real_comparisons():
    """The measured numbers, and what they actually support.

    mh vs ph on the training region is 33/50 against 28/50 — a ten point gap
    that does not survive a test. On the wider region it is 29/50 against 9/50,
    which does. Reporting the first as a finding would have been the error.
    """
    narrow = barnard(33, 17, 28, 22)
    wide = barnard(29, 21, 9, 41)
    assert narrow > 0.05, "a 10pp gap at n=50 is not separable"
    assert wide < 0.001, "a 40pp gap at n=50 is decisively separable"


def test_identical_arms_are_never_significant():
    assert barnard(10, 10, 10, 10) == pytest.approx(1.0)


def test_an_empty_arm_is_not_a_finding():
    assert barnard(0, 0, 5, 5) == 1.0


def test_holm_is_stricter_than_the_bare_threshold():
    """A p of 0.031 is a finding once and not the best of four tries."""
    alone = holm([0.031])
    among_four = holm([0.031, 0.04, 0.2, 0.5])
    assert alone == (True,)
    assert among_four[0] is False


def test_holm_keeps_what_survives_correction():
    assert holm([0.001, 0.5, 0.6])[0] is True


def test_holm_steps_down_rather_than_testing_each_alone():
    # Once the smallest p fails its threshold, nothing larger can pass.
    assert holm([0.03, 0.031]) == (False, False)


# -- aggregating -------------------------------------------------------------


def test_the_interquartile_mean_ignores_one_wild_task():
    ordinary = [0.5] * 10
    with_outlier = ordinary + [100.0]
    assert iqm(with_outlier) == pytest.approx(0.5)
    assert np.mean(with_outlier) > 9


def test_the_interquartile_mean_falls_back_on_tiny_samples():
    assert iqm([0.4, 0.6]) == pytest.approx(0.5)
    assert np.isnan(iqm([]))


def test_the_bootstrap_resamples_within_tasks_not_across_them():
    """Pooling runs across tasks gives an interval that is too narrow.

    Two tasks, each internally consistent but far apart. Resampling within
    tasks preserves the spread; a naive pooled resample would sometimes draw
    only from one task and report a spuriously tight interval.
    """
    matrix = [[0.9, 0.9, 0.9, 0.9], [0.1, 0.1, 0.1, 0.1]]
    point, lo, hi = stratified_bootstrap(matrix, iqm, resamples=400, seed=0)
    assert lo <= point <= hi
    # Every within-task resample reproduces the same two values, so the
    # aggregate cannot move: that is the correct answer here, and a pooled
    # resample would instead wander.
    assert hi - lo == pytest.approx(0.0, abs=1e-9)


def test_the_bootstrap_is_deterministic_for_a_given_seed():
    matrix = [[0.6, 0.4, 0.7], [0.1, 0.2, 0.0]]
    first = stratified_bootstrap(matrix, iqm, resamples=200, seed=7)
    again = stratified_bootstrap(matrix, iqm, resamples=200, seed=7)
    assert first == again


def test_a_profile_distinguishes_two_methods_with_the_same_mean():
    """The reason to publish a curve instead of a number."""
    everywhere_mediocre = [[0.5] * 4, [0.5] * 4]
    half_brilliant = [[1.0] * 4, [0.0] * 4]
    assert np.mean(np.concatenate(everywhere_mediocre)) == pytest.approx(
        np.mean(np.concatenate(half_brilliant))
    )
    a = dict(performance_profile(everywhere_mediocre, [0.25, 0.75]))
    b = dict(performance_profile(half_brilliant, [0.25, 0.75]))
    assert a[0.75] == 0.0 and b[0.75] == 0.5


def test_probability_of_improvement_is_scale_free():
    assert prob_improvement([1, 2, 3], [0, 0, 0]) == pytest.approx(1.0)
    assert prob_improvement([0, 0, 0], [1, 2, 3]) == pytest.approx(0.0)
    assert prob_improvement([1, 1], [1, 1]) == pytest.approx(0.5)


# -- comparing one measurement system to another -----------------------------


def test_a_faithful_proxy_has_no_rank_violations():
    real = [0.60, 0.45, 0.30, 0.10]
    proxy = [0.55, 0.42, 0.28, 0.12]
    assert mmrv(real, proxy) == pytest.approx(0.0)


def test_swapping_a_wide_pair_costs_more_than_swapping_a_close_one():
    real = [0.60, 0.45, 0.30, 0.29]
    swaps_far = [0.45, 0.60, 0.30, 0.29]  # inverts a 15pp gap
    swaps_near = [0.60, 0.45, 0.29, 0.30]  # inverts a 1pp gap
    assert mmrv(real, swaps_far) > mmrv(real, swaps_near) > 0


def test_a_fully_inverted_proxy_is_the_worst_case():
    """The mean of each policy's worst violation, not the single worst overall.

    For [0.6, 0.4, 0.2, 0.0] fully reversed, the two extreme policies each have
    a 0.6 violation and the two middle ones 0.4, so the mean is 0.5. Worth
    pinning: it is the difference between reporting a summary and reporting a
    maximum, and the summary is what makes MMRV comparable across suites.
    """
    real = [0.6, 0.4, 0.2, 0.0]
    inverted = mmrv(real, list(reversed(real)))
    assert inverted == pytest.approx(0.5)
    assert inverted > mmrv(real, [0.55, 0.42, 0.21, 0.05])


def test_mmrv_needs_matched_policy_sets():
    with pytest.raises(ValueError, match="matched lengths"):
        mmrv([0.5, 0.4], [0.5])


def test_a_biased_but_correlated_proxy_tightens_the_interval():
    """The whole argument for owning a validated simulator.

    Twenty expensive trials, three hundred cheap ones from a world that is
    optimistic by fifteen points. The rectified estimate should land nearer the
    truth than either source alone, with an interval narrower than the
    expensive trials could give by themselves.
    """
    rng = np.random.default_rng(3)

    def scenes(n):
        difficulty = rng.random(n)
        return (difficulty < 0.45).astype(float), (difficulty < 0.60).astype(float)

    paired_real, paired_proxy = scenes(20)
    _, proxy_only = scenes(300)
    out = ppi_mean(paired_real, paired_proxy, proxy_only)

    naive_width = 2 * 1.96 * paired_real.std(ddof=1) / np.sqrt(20)
    assert out.high - out.low < naive_width
    assert out.leverage > 0.1
    assert abs(out.value - 0.45) < abs(proxy_only.mean() - 0.45)


def test_a_useless_proxy_degrades_to_the_paired_trials_alone():
    """It must not be possible for a bad proxy to make the answer worse."""
    rng = np.random.default_rng(4)
    paired_real = (rng.random(20) < 0.45).astype(float)
    junk_paired = (rng.random(20) < 0.5).astype(float)
    junk_only = (rng.random(300) < 0.5).astype(float)
    out = ppi_mean(paired_real, junk_paired, junk_only)
    assert out.leverage == pytest.approx(0.0, abs=0.05)
    assert out.value == pytest.approx(paired_real.mean(), abs=0.05)


def test_prediction_powered_inference_needs_paired_trials():
    with pytest.raises(ValueError, match="at least 2 paired"):
        ppi_mean([1.0], [1.0], [1.0] * 50)


def test_it_refuses_ragged_paired_input():
    with pytest.raises(ValueError, match="equal length"):
        ppi_mean([1.0, 0.0], [1.0], [1.0] * 10)


# -- do two judges agree -----------------------------------------------------


def test_agreement_punishes_a_judge_who_is_guessing():
    truth = [1, 1, 0, 0, 1, 0, 1, 0, 0, 1]
    almost = [1, 1, 0, 0, 1, 0, 1, 0, 1, 1]
    coin = [1, 0, 1, 0, 0, 1, 0, 1, 1, 0]
    assert cohen_kappa(truth, almost) >= AGREEMENT_TRUSTED
    assert cohen_kappa(truth, coin) < 0


def test_raw_agreement_flatters_a_judge_on_skewed_data():
    """Why chance correction is not optional.

    A task solved five percent of the time: two judges who both say "failed"
    every time agree ninety-five percent of the time and have established
    nothing. Kappa says so.
    """
    mostly_failure = [0] * 19 + [1]
    lazy = [0] * 20
    raw = sum(a == b for a, b in zip(mostly_failure, lazy)) / 20
    assert raw == 0.95
    assert cohen_kappa(mostly_failure, lazy) <= 0.0


def test_an_abstention_is_dropped_rather_than_scored_as_a_verdict():
    """'I cannot tell from this video' is honesty, not a wrong answer."""
    a = [1, 1, 0, 0]
    with_abstain = [1, None, 0, 0]
    assert cohen_kappa(a, with_abstain) == pytest.approx(1.0)


def test_agreement_among_three_judges_tolerates_missing_labels():
    a = [1, 1, 0, 0, 1, 0]
    b = [1, 1, 0, 0, 1, 0]
    c = [1, 1, 0, None, 1, 0]
    assert krippendorff_alpha([a, b, c]) == pytest.approx(1.0)


def test_three_judges_who_disagree_score_near_zero():
    rng = np.random.default_rng(5)
    judges = [list((rng.random(60) < 0.5).astype(int)) for _ in range(3)]
    assert krippendorff_alpha(judges) < 0.2


def test_one_judge_is_not_an_agreement():
    assert np.isnan(krippendorff_alpha([[1, 0, 1]]))


def test_the_verdict_boundaries_are_the_published_ones():
    assert agreement_verdict_code(0.85) == "judge.calibrated"
    assert agreement_verdict_code(0.70) == "judge.tentative"
    assert agreement_verdict_code(0.40) == "judge.uncalibrated"
    assert agreement_verdict_code(float("nan")) == "judge.unmeasured"
    assert AGREEMENT_TENTATIVE < AGREEMENT_TRUSTED
