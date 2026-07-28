"""The maths, checked against values that can be worked out by hand."""

from __future__ import annotations

import numpy as np
import pytest

from gantry_feedback_core import statistics as st


# -- proportions -----------------------------------------------------------


def test_wilson_stays_inside_the_unit_interval_at_the_extremes():
    """Where Wald fails: 0/20 and 20/20 are exactly the cases that matter."""
    low, high = st.wilson(0, 20)
    assert low == 0.0 and 0.0 < high < 0.25
    low, high = st.wilson(20, 20)
    assert high == 1.0 and 0.75 < low < 1.0


def test_wilson_brackets_the_point_estimate():
    low, high = st.wilson(15, 100)
    assert low < 0.15 < high
    assert (low, high) == pytest.approx((0.0931, 0.2328), abs=1e-3)


def test_wilson_narrows_as_n_grows():
    narrow = st.wilson(500, 1000)
    wide = st.wilson(5, 10)
    assert (narrow[1] - narrow[0]) < (wide[1] - wide[0])


def test_no_data_means_no_information_not_certainty():
    assert st.wilson(0, 0) == (0.0, 1.0)


def test_newcombe_difference_spans_zero_when_arms_agree():
    low, high = st.newcombe_difference(10, 20, 10, 20)
    assert low < 0 < high


def test_newcombe_difference_excludes_zero_when_arms_clearly_differ():
    low, high = st.newcombe_difference(90, 100, 10, 100)
    assert low > 0


def test_mcnemar_is_one_when_nothing_is_discordant():
    assert st.mcnemar(both=10, only_a=0, only_b=0, neither=10) == 1.0


def test_mcnemar_needs_a_lopsided_split_to_be_significant():
    assert st.mcnemar(0, 5, 5, 0) == 1.0
    assert st.mcnemar(0, 10, 0, 0) < 0.01


# -- rank statistics -------------------------------------------------------


def test_cliffs_delta_is_plus_one_when_every_pair_favours_a():
    assert st.cliffs_delta([4, 5, 6], [1, 2, 3]) == 1.0
    assert st.cliffs_delta([1, 2, 3], [4, 5, 6]) == -1.0


def test_cliffs_delta_is_zero_for_identical_samples():
    assert st.cliffs_delta([1, 2, 3], [1, 2, 3]) == 0.0


def test_mann_whitney_separates_clearly_different_samples():
    assert st.mann_whitney(list(range(20, 40)), list(range(20))) < 0.01


def test_mann_whitney_finds_nothing_in_identical_samples():
    assert st.mann_whitney([1, 2, 3, 4, 5], [1, 2, 3, 4, 5]) == pytest.approx(1.0, abs=0.05)


def test_mann_whitney_handles_ties_without_dividing_by_zero():
    assert 0.0 <= st.mann_whitney([1] * 10, [1] * 10) <= 1.0


def test_spearman_catches_a_monotone_but_curved_relationship():
    x = list(range(1, 20))
    assert st.spearman(x, [v**3 for v in x]) == pytest.approx(1.0, abs=1e-9)


# -- multiplicity ----------------------------------------------------------


def test_bh_rejects_nothing_when_every_p_is_large():
    corrected = st.benjamini_hochberg({f"s{i}": 0.6 for i in range(10)})
    assert not any(c.significant for c in corrected)


def test_bh_is_stricter_than_a_bare_threshold():
    """One statistic at p=0.04 among eleven nulls clears 0.05 and means nothing.

    Screening a dozen statistics and reporting whatever crosses 0.05
    manufactures about one finding from noise every time. Correction is what
    stops that becoming a prescription.
    """
    pvalues = {"lucky": 0.04, **{f"null{i}": 0.9 for i in range(11)}}
    assert pvalues["lucky"] < 0.05
    corrected = {c.name: c for c in st.benjamini_hochberg(pvalues)}
    assert not corrected["lucky"].significant
    assert corrected["lucky"].q > 0.4


def test_bh_still_rejects_when_the_whole_family_is_real():
    """Correction is not blanket suppression: twelve genuine effects survive."""
    corrected = st.benjamini_hochberg({f"s{i}": 0.04 for i in range(12)})
    assert all(c.significant for c in corrected)


def test_bh_keeps_a_genuinely_strong_result_among_noise():
    pvalues = {"real": 0.0001, **{f"noise{i}": 0.5 for i in range(10)}}
    corrected = {c.name: c for c in st.benjamini_hochberg(pvalues)}
    assert corrected["real"].significant
    assert not corrected["noise0"].significant


def test_q_values_are_monotone_in_p():
    corrected = st.benjamini_hochberg({"a": 0.001, "b": 0.01, "c": 0.2, "d": 0.9})
    qs = [c.q for c in corrected]
    assert qs == sorted(qs)


def test_empty_family_is_handled():
    assert st.benjamini_hochberg({}) == ()


# -- resampling ------------------------------------------------------------


def test_bootstrap_interval_brackets_the_sample_mean():
    values = list(np.random.default_rng(0).normal(5.0, 1.0, 200))
    low, high = st.bootstrap_ci(values, resamples=2000, seed=1)
    assert low < float(np.mean(values)) < high


def test_bootstrap_is_reproducible_from_its_seed():
    values = list(np.random.default_rng(3).normal(0.0, 1.0, 60))
    assert st.bootstrap_ci(values, seed=7) == st.bootstrap_ci(values, seed=7)
    assert st.bootstrap_ci(values, seed=7) != st.bootstrap_ci(values, seed=8)


def test_paired_bootstrap_uses_the_pairing():
    """Correlated arms give a tighter interval than treating them as independent."""
    rng = np.random.default_rng(0)
    base = rng.normal(0, 5, 40)
    a, b = base + 1.0, base
    paired = st.paired_bootstrap_ci(a, b, seed=0)
    assert paired[0] > 0  # the constant +1 shift is visible despite the noise
    assert (paired[1] - paired[0]) < 1.0


def test_paired_bootstrap_refuses_mismatched_lengths():
    with pytest.raises(ValueError, match="must match in length"):
        st.paired_bootstrap_ci([1, 2, 3], [1, 2])


# -- robust summaries ------------------------------------------------------


def test_robust_bounds_ignore_a_single_outlier():
    """A stray episode must not widen the band to admit anything."""
    clean = st.robust_bounds([10.0] * 20)
    with_outlier = st.robust_bounds([10.0] * 20 + [1000.0])
    assert with_outlier == pytest.approx(clean, abs=1e-6)


def test_robust_bounds_survive_a_realistic_contaminated_sample():
    rng = np.random.default_rng(0)
    clean = list(rng.normal(0.5, 0.05, 50))
    low, high = st.robust_bounds(clean)
    contaminated = st.robust_bounds(clean + [99.0, -99.0])
    assert contaminated[0] == pytest.approx(low, abs=0.05)
    assert contaminated[1] == pytest.approx(high, abs=0.05)


def test_robust_bounds_widen_with_real_spread():
    tight = st.robust_bounds(list(np.random.default_rng(0).normal(0, 1, 100)))
    loose = st.robust_bounds(list(np.random.default_rng(0).normal(0, 10, 100)))
    assert (loose[1] - loose[0]) > (tight[1] - tight[0])
