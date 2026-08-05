"""The statistics the feedback modules run on, implemented from scratch.

numpy only, no scipy, so this installs anywhere core installs.

Small-sample honesty is the theme. Robot evaluation runs at n=20 or n=100, not
n=10000, and at those sizes the textbook shortcuts mislead: a normal-approximation
interval on a proportion of 0/20 spans negative numbers, and a t-test on
bounded, skewed progress scores assumes a shape the data does not have. So:
Wilson rather than Wald, rank statistics rather than means, an effect size
beside every p-value, and Benjamini-Hochberg whenever more than one question is
asked at once.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np

Z95 = 1.959963984540054


def normal_cdf(z: float) -> float:
    return 0.5 * math.erfc(-z / math.sqrt(2.0))


# --------------------------------------------------------------------------
# proportions
# --------------------------------------------------------------------------


def wilson(successes: int, n: int, z: float = Z95) -> tuple[float, float]:
    """Score interval for a proportion.

    Wald (``p ± z·sqrt(p(1-p)/n)``) collapses to zero width at p=0 or p=1 and
    happily returns negative bounds, which is precisely the regime robot
    evaluation lives in. Wilson stays inside [0, 1] and stays sane at the ends.
    """
    if n <= 0:
        return (0.0, 1.0)
    p = successes / n
    denominator = 1.0 + z * z / n
    centre = (p + z * z / (2.0 * n)) / denominator
    half = z * math.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n)) / denominator
    return (max(0.0, centre - half), min(1.0, centre + half))


def newcombe_difference(
    successes_a: int, n_a: int, successes_b: int, n_b: int, z: float = Z95
) -> tuple[float, float]:
    """Interval for the difference of two independent proportions.

    Built from each side's Wilson interval, so it inherits the same
    well-behaved-at-the-boundary property.
    """
    if n_a <= 0 or n_b <= 0:
        return (-1.0, 1.0)
    p_a, p_b = successes_a / n_a, successes_b / n_b
    low_a, high_a = wilson(successes_a, n_a, z)
    low_b, high_b = wilson(successes_b, n_b, z)
    delta = p_a - p_b
    lower = delta - math.sqrt((p_a - low_a) ** 2 + (high_b - p_b) ** 2)
    upper = delta + math.sqrt((high_a - p_a) ** 2 + (p_b - low_b) ** 2)
    return (max(-1.0, lower), min(1.0, upper))


def mcnemar(both: int, only_a: int, only_b: int, neither: int) -> float:
    """Exact two-sided p-value for a paired comparison.

    Paired because the same scenes are run under both arms: comparing them as
    independent samples throws away the pairing and needs far more scenes to
    see the same effect.
    """
    discordant = only_a + only_b
    if discordant == 0:
        return 1.0
    smaller = min(only_a, only_b)
    tail = sum(math.comb(discordant, k) for k in range(smaller + 1)) / (2.0**discordant)
    return min(1.0, 2.0 * tail)


# --------------------------------------------------------------------------
# rank statistics
# --------------------------------------------------------------------------


def cliffs_delta(a: Sequence[float], b: Sequence[float]) -> float:
    """Effect size in [-1, 1]: how often a exceeds b, minus how often it trails.

    Reported next to every p-value. A significant difference that is tiny is a
    statement about sample size, not about the data, and only an effect size
    tells the two apart.
    """
    left, right = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if left.size == 0 or right.size == 0:
        return 0.0
    comparison = np.sign(left[:, None] - right[None, :])
    return float(comparison.sum() / (left.size * right.size))


def _ranks(values: np.ndarray) -> tuple[np.ndarray, float]:
    """Average ranks plus the tie correction term."""
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    correction = 0.0
    position = 0
    while position < len(values):
        stop = position
        while stop + 1 < len(values) and values[order[stop + 1]] == values[order[position]]:
            stop += 1
        average = (position + stop) / 2.0 + 1.0
        ranks[order[position : stop + 1]] = average
        tied = stop - position + 1
        if tied > 1:
            correction += tied**3 - tied
        position = stop + 1
    return ranks, correction


def mann_whitney(a: Sequence[float], b: Sequence[float]) -> float:
    """Two-sided p-value by normal approximation, with tie correction.

    Rank-based on purpose: progress scores are bounded and skewed, and a test
    assuming normality on them reports confidence it has not earned.
    """
    left, right = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    n_a, n_b = left.size, right.size
    if n_a == 0 or n_b == 0:
        return 1.0
    combined = np.concatenate([left, right])
    ranks, correction = _ranks(combined)
    total = n_a + n_b
    u = ranks[:n_a].sum() - n_a * (n_a + 1) / 2.0
    mean = n_a * n_b / 2.0
    variance = (n_a * n_b / 12.0) * ((total + 1) - correction / (total * (total - 1)))
    if variance <= 0:
        return 1.0
    z = (abs(u - mean) - 0.5) / math.sqrt(variance)
    return min(1.0, 2.0 * (1.0 - normal_cdf(max(z, 0.0))))


def spearman(a: Sequence[float], b: Sequence[float]) -> float:
    """Rank correlation. Used to expose a statistic that merely tracks duration."""
    left, right = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if left.size < 3:
        return 0.0
    rank_left, _ = _ranks(left)
    rank_right, _ = _ranks(right)
    if rank_left.std() < 1e-12 or rank_right.std() < 1e-12:
        return 0.0
    return float(np.corrcoef(rank_left, rank_right)[0, 1])


# --------------------------------------------------------------------------
# multiplicity and resampling
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Corrected:
    """One hypothesis after multiplicity correction."""

    name: str
    p: float
    q: float
    significant: bool


def benjamini_hochberg(pvalues: dict[str, float], alpha: float = 0.05) -> tuple[Corrected, ...]:
    """Control the false discovery rate across a family of tests.

    Screening a dozen statistics at once and reporting whatever clears 0.05
    manufactures roughly one finding from noise every time. This is the
    correction for asking many questions, and it is not optional when the
    output is a prescription someone will spend collection budget on.
    """
    if not pvalues:
        return ()
    items = sorted(pvalues.items(), key=lambda kv: kv[1])
    m = len(items)
    qs: list[float] = []
    running = 1.0
    for rank in range(m, 0, -1):
        _, p = items[rank - 1]
        running = min(running, p * m / rank)
        qs.append(running)
    qs.reverse()
    return tuple(Corrected(name, p, q, q <= alpha) for (name, p), q in zip(items, qs))


def bootstrap_ci(
    values: Sequence[float],
    statistic=np.mean,
    *,
    resamples: int = 10_000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float]:
    """Percentile interval. Seeded, so a reported interval is reproducible."""
    data = np.asarray(values, dtype=float)
    if data.size == 0:
        return (float("nan"), float("nan"))
    if data.size == 1:
        return (float(data[0]), float(data[0]))
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, data.size, size=(resamples, data.size))
    estimates = np.apply_along_axis(statistic, 1, data[draws])
    return (
        float(np.percentile(estimates, 100 * alpha / 2)),
        float(np.percentile(estimates, 100 * (1 - alpha / 2))),
    )


def paired_bootstrap_ci(
    a: Sequence[float],
    b: Sequence[float],
    *,
    resamples: int = 10_000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float]:
    """Interval on a paired difference, resampling pairs rather than arms."""
    left, right = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if left.size != right.size:
        raise ValueError(f"paired inputs must match in length: {left.size} vs {right.size}")
    if left.size == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, left.size, size=(resamples, left.size))
    differences = (left[draws] - right[draws]).mean(axis=1)
    return (
        float(np.percentile(differences, 100 * alpha / 2)),
        float(np.percentile(differences, 100 * (1 - alpha / 2))),
    )


# --------------------------------------------------------------------------
# robust summaries
# --------------------------------------------------------------------------


def median_abs_deviation(values: Sequence[float]) -> float:
    data = np.asarray(values, dtype=float)
    if data.size == 0:
        return 0.0
    return float(np.median(np.abs(data - np.median(data))))


def robust_bounds(values: Sequence[float], k: float = 3.0) -> tuple[float, float]:
    """Median ± k·MAD, scaled to be comparable with a standard deviation.

    Used to fit a threshold from a reference cohort. Robust because the
    reference set is small and one unusual episode should not move the bar for
    everything screened against it afterwards.
    """
    data = np.asarray(values, dtype=float)
    if data.size == 0:
        return (float("nan"), float("nan"))
    centre = float(np.median(data))
    spread = 1.4826 * median_abs_deviation(data)
    if spread == 0.0:
        # Fall back to the IQR, never to the standard deviation. A tight
        # reference with one stray episode has MAD zero and a huge standard
        # deviation, so that fallback would widen the band to admit anything
        # in precisely the case where the reference is most informative.
        quartiles = np.percentile(data, [25, 75])
        spread = float(quartiles[1] - quartiles[0]) / 1.349
    return (centre - k * spread, centre + k * spread)
