"""The statistics a robot evaluation actually needs, implemented rather than imported.

Everything here is numpy. That is deliberate and it is not asceticism: the
alternative is that ``pip install gantry-core`` pulls scipy, statsmodels,
rliable and a bootstrap library, and the one thing this framework promises — that
describing and judging an experiment costs nothing to install — stops being true.
So the algorithms are written out, and each one is checked against its reference
implementation inside a plugin's dev extras, where those libraries are allowed.
Vendoring the algorithm is cheap; vendoring the dependency is not.

Why these particular functions
------------------------------
Robot evaluation lives at a sample size where the usual shortcuts are wrong in
the unsafe direction. Twenty trials is normal. At twenty trials the normal
approximation says a difference is significant when it is not, a mean over
thirteen tasks is dominated by whichever task happened to be easy, and peeking
after every rollout inflates the false-positive rate to somewhere near a
coin flip. Each function below exists because its absence produced a wrong
answer in this project at least once.

The four families
-----------------
**Sizing.** :func:`trials_needed` answers "can this budget see that effect"
before the budget is spent, which is the only time the answer is useful.

**Stopping.** :func:`confidence_sequence` is valid at every sample size
simultaneously, so watching a comparison and stopping when it separates is
legitimate rather than p-hacking. This is the difference between a rule and a
temptation.

**Aggregating.** :func:`iqm`, :func:`stratified_bootstrap`,
:func:`performance_profile` and :func:`prob_improvement` summarise a matrix of
tasks by runs without letting one outlier task speak for the rest.

**Comparing measurements to each other.** :func:`mmrv` asks whether a cheap
evaluator ranks policies the way an expensive one does. :func:`ppi_mean` turns
many cheap trials plus a few expensive ones into a valid interval about the
expensive world. :func:`cohen_kappa` and :func:`krippendorff_alpha` ask whether
two judges of the same rubric agree. These are what make an evaluation
auditable rather than merely repeatable.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import comb, isfinite, log, sqrt
from typing import Any, Sequence

import numpy as np

# --------------------------------------------------------------------------
# sizing
# --------------------------------------------------------------------------


def trials_needed(
    baseline: float,
    magnitude: float,
    *,
    alpha: float = 0.05,
    power: float = 0.8,
    cap: int = 5000,
) -> int:
    """Paired trials needed to see a change of ``magnitude``, by exact search.

    Exact rather than the normal approximation, because robot evaluations run at
    a sample size where the approximation errs in the unsafe direction: it says
    twenty trials suffice when they do not, which is precisely the belief this
    function exists to prevent.

    Modelled on the paired test that will do the judging, so what matters is the
    rate at which the two arms *disagree* — the trials where both succeed or
    both fail carry no information about which is better and inflating them does
    not help.
    """
    target = min(max(baseline + magnitude, 0.0), 1.0)
    gain = abs(target - baseline)
    if gain <= 0:
        return cap
    # Conservative: assume improvement is the only source of disagreement, then
    # add an equal amount of noise-driven disagreement in the other direction.
    discordant_rate = min(1.0, gain * 2)
    for n in range(4, cap + 1):
        expected_discordant = n * discordant_rate
        if expected_discordant < 4:
            continue
        k = int(round(expected_discordant))
        wins = int(round(k * (gain / discordant_rate + 0.5)))
        wins = min(max(wins, 0), k)
        tail = sum(comb(k, i) for i in range(wins, k + 1)) / (2**k)
        if 2 * tail <= alpha:
            return n
    return cap


# --------------------------------------------------------------------------
# stopping
# --------------------------------------------------------------------------


def _psi_e(lam: float) -> float:
    """The empirical-Bernstein exponential term, ``(-ln(1-λ) - λ)/4``."""
    lam = min(max(lam, 0.0), 0.99)
    return (-log(1.0 - lam) - lam) / 4.0


@dataclass(frozen=True)
class Sequence_:
    """A confidence sequence: an interval per sample size, all valid at once."""

    lower: tuple[float, ...]
    upper: tuple[float, ...]

    @property
    def final(self) -> tuple[float, float]:
        return (self.lower[-1], self.upper[-1]) if self.lower else (0.0, 1.0)

    def separated_from(self, value: float) -> int | None:
        """First sample size at which the interval excludes ``value``, if any.

        This is the whole point of a confidence sequence: because every interval
        is valid simultaneously, stopping the moment one of them excludes the
        null is a legitimate rule rather than the thing that inflates a
        false-positive rate to near a coin flip.
        """
        for n, (lo, hi) in enumerate(zip(self.lower, self.upper), start=1):
            if value < lo or value > hi:
                return n
        return None


def confidence_sequence(
    outcomes: Sequence[float], *, alpha: float = 0.05, cap: float = 0.5
) -> Sequence_:
    """Time-uniform interval for the mean of bounded observations.

    Predictably-mixed empirical Bernstein, after Waudby-Smith and Ramdas: a
    betting-style construction whose intervals hold at *every* sample size at
    once. That property is what a caller actually wants and almost never has —
    an ordinary interval is valid only at the sample size it was computed for,
    so looking at it repeatedly and stopping when it looks good is exactly the
    error that makes small-n robotics results irreproducible.

    ``outcomes`` must lie in [0, 1]; success indicators do. ``cap`` bounds the
    betting fraction away from 1 for numerical safety.
    """
    values = np.asarray(list(outcomes), dtype=np.float64)
    if values.size == 0:
        return Sequence_((), ())
    if values.min() < -1e-9 or values.max() > 1 + 1e-9:
        raise ValueError("confidence_sequence expects observations in [0, 1]")
    values = np.clip(values, 0.0, 1.0)

    log_term = log(2.0 / alpha)
    lowers, uppers = [], []
    # Running estimates use a 1/2 and 1/4 prior so the first step is defined.
    running_sum = 0.0
    running_var_sum = 0.0
    weighted_x = 0.0
    weight = 0.0
    penalty = 0.0
    mu_prev = 0.5
    var_prev = 0.25

    for i, x in enumerate(values, start=1):
        # λ is chosen from data strictly before this observation, which is what
        # makes the mixture predictable and the bound valid.
        denom = var_prev * i * log(1.0 + i)
        lam = min(cap, sqrt(2.0 * log_term / denom)) if denom > 0 else cap
        lam = min(max(lam, 1e-12), cap)

        weighted_x += lam * x
        weight += lam
        penalty += 4.0 * (x - mu_prev) ** 2 * _psi_e(lam)

        half = (log_term + penalty) / weight if weight > 0 else float("inf")
        centre = weighted_x / weight if weight > 0 else 0.5
        lowers.append(float(max(0.0, centre - half)))
        uppers.append(float(min(1.0, centre + half)))

        running_sum += x
        mu_prev = (0.5 + running_sum) / (i + 1)
        running_var_sum += (x - mu_prev) ** 2
        var_prev = max((0.25 + running_var_sum) / (i + 1), 1e-12)

    return Sequence_(tuple(lowers), tuple(uppers))


def alpha_spent(peeks: int, alpha: float = 0.05) -> float:
    """Per-look threshold for a fixed number of looks, by Bonferroni.

    The honest fallback when a confidence sequence is not wanted: valid, and
    conservative enough that anybody peeking a hundred times will notice the
    cost. Provided so that "we peeked but corrected" is expressible, and so the
    difference between correcting and not is a line of code rather than an
    argument.
    """
    return alpha / max(1, peeks)


# --------------------------------------------------------------------------
# exact tests
# --------------------------------------------------------------------------


def barnard(a: int, b: int, c: int, d: int, *, grid: int = 200) -> float:
    """Barnard's exact unconditional test for a 2x2 table.

    ``[[a, b], [c, d]]`` with rows the two arms. Strictly more powerful than
    Fisher's exact test at the sample sizes robot evaluation runs at, because it
    does not condition on margins that were not fixed by the design — nobody
    decided in advance how many successes there would be.

    Maximised over the nuisance success probability on a grid, which is the
    standard construction. The grid is the only approximation; 200 points is
    ample at n below a few hundred.
    """
    n1, n2 = a + b, c + d
    if n1 == 0 or n2 == 0:
        return 1.0

    def statistic(x: int, y: int) -> float:
        p1, p2 = x / n1, y / n2
        pooled = (x + y) / (n1 + n2)
        var = pooled * (1 - pooled) * (1 / n1 + 1 / n2)
        return 0.0 if var <= 0 else (p1 - p2) / sqrt(var)

    observed = abs(statistic(a, c))
    if observed == 0.0:
        return 1.0

    # Pre-compute binomial weights per candidate π once per row total.
    xs = np.arange(n1 + 1)
    ys = np.arange(n2 + 1)
    log_c1 = np.array([log(comb(n1, int(k))) for k in xs])
    log_c2 = np.array([log(comb(n2, int(k))) for k in ys])
    extreme = np.array(
        [[abs(statistic(int(x), int(y))) >= observed - 1e-12 for y in ys] for x in xs]
    )

    worst = 0.0
    for pi in np.linspace(1e-6, 1 - 1e-6, grid):
        lp, lq = log(pi), log(1 - pi)
        p1 = np.exp(log_c1 + xs * lp + (n1 - xs) * lq)
        p2 = np.exp(log_c2 + ys * lp + (n2 - ys) * lq)
        total = float((np.outer(p1, p2) * extreme).sum())
        worst = max(worst, total)
    return float(min(1.0, worst))


def holm(pvalues: Sequence[float], *, alpha: float = 0.05) -> tuple[bool, ...]:
    """Which hypotheses survive Holm's step-down correction.

    Controls the family-wise error rate, which is the right guarantee when the
    question is "did *any* of these comparisons find something real" — the
    situation you are in the moment you evaluate more than one checkpoint and
    report the best. Benjamini-Hochberg controls a different thing and is the
    wrong tool for a selection claim.
    """
    values = list(pvalues)
    if not values:
        return ()
    order = sorted(range(len(values)), key=lambda i: values[i])
    survives = [False] * len(values)
    m = len(values)
    for rank, index in enumerate(order):
        threshold = alpha / (m - rank)
        if values[index] <= threshold:
            survives[index] = True
        else:
            break  # step-down: once one fails, the rest fail too
    return tuple(survives)


# --------------------------------------------------------------------------
# aggregating a matrix of tasks by runs
# --------------------------------------------------------------------------


def iqm(scores: Sequence[float]) -> float:
    """Interquartile mean — the mean of the middle half.

    Preferred to both the mean and the median when aggregating across tasks.
    The mean is dominated by whichever task happened to be easy; the median
    throws away three quarters of the information and has a wide interval at
    the sample sizes anybody actually runs. This sits between them and is what
    the deep-RL reliability literature settled on for exactly this regime.
    """
    values = np.sort(np.asarray(list(scores), dtype=np.float64))
    if values.size == 0:
        return float("nan")
    if values.size < 4:
        return float(values.mean())
    lo = int(np.floor(values.size * 0.25))
    hi = int(np.ceil(values.size * 0.75))
    middle = values[lo:hi]
    return float(middle.mean() if middle.size else values.mean())


def stratified_bootstrap(
    matrix: Sequence[Sequence[float]],
    statistic: Any = iqm,
    *,
    alpha: float = 0.05,
    resamples: int = 2000,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Point estimate and interval for a task-by-runs matrix.

    ``matrix[t][r]`` is run ``r``'s score on task ``t``. Runs are resampled
    *within* each task rather than across the flattened pool, because the runs
    on one task are exchangeable with each other and not with runs on a
    different task. Pooling them produces an interval that is too narrow and
    looks authoritative.
    """
    rows = [np.asarray(list(row), dtype=np.float64) for row in matrix]
    rows = [row for row in rows if row.size]
    if not rows:
        return (float("nan"), float("nan"), float("nan"))
    point = float(statistic(np.concatenate(rows)))

    rng = np.random.default_rng(seed)
    draws = np.empty(resamples, dtype=np.float64)
    for i in range(resamples):
        sampled = [row[rng.integers(0, row.size, row.size)] for row in rows]
        draws[i] = statistic(np.concatenate(sampled))
    lo = float(np.quantile(draws, alpha / 2))
    hi = float(np.quantile(draws, 1 - alpha / 2))
    return (point, lo, hi)


def performance_profile(
    matrix: Sequence[Sequence[float]], thresholds: Sequence[float] | None = None
) -> tuple[tuple[float, float], ...]:
    """Fraction of runs scoring above each threshold, as ``(tau, fraction)``.

    A curve rather than a number, and the reason to prefer one: two methods
    with the same aggregate can have completely different profiles — one
    mediocre everywhere, one excellent on half the tasks and hopeless on the
    rest. A table of means cannot tell those apart and a profile cannot hide it.
    """
    flat = (
        np.concatenate([np.asarray(list(row), dtype=np.float64) for row in matrix if len(row)])
        if any(len(r) for r in matrix)
        else np.array([])
    )
    if flat.size == 0:
        return ()
    taus = (
        np.asarray(list(thresholds), dtype=np.float64)
        if thresholds is not None
        else np.linspace(0.0, 1.0, 21)
    )
    return tuple((float(t), float((flat > t).mean())) for t in taus)


def prob_improvement(left: Sequence[float], right: Sequence[float]) -> float:
    """Probability a run of ``left`` beats a run of ``right``, ties at half.

    Reported instead of a difference of means when the question is "which
    should I use" rather than "how much better is it". Scale-free, robust, and
    it answers the question somebody actually asked.
    """
    a = np.asarray(list(left), dtype=np.float64)
    b = np.asarray(list(right), dtype=np.float64)
    if a.size == 0 or b.size == 0:
        return float("nan")
    comparisons = a[:, None] - b[None, :]
    wins = float((comparisons > 0).sum())
    ties = float((comparisons == 0).sum())
    return (wins + 0.5 * ties) / (a.size * b.size)


# --------------------------------------------------------------------------
# comparing one measurement system to another
# --------------------------------------------------------------------------


def mmrv(real: Sequence[float], proxy: Sequence[float]) -> float:
    """Mean maximum rank violation between an expensive and a cheap evaluator.

    Zero when the cheap evaluator orders policies exactly as the expensive one
    does; larger when it swaps a pair, and weighted by how far apart that pair
    really was — swapping two policies that genuinely differ by forty points is
    a serious failure, swapping two that differ by one is noise.

    This is the number that says whether a simulator may stand in for a bench,
    and it is a property of the *evaluator*, not of any policy. Correlation
    alone is not enough: a proxy can correlate beautifully and still invert the
    one comparison a decision rests on.
    """
    r = np.asarray(list(real), dtype=np.float64)
    p = np.asarray(list(proxy), dtype=np.float64)
    if r.size != p.size:
        raise ValueError(f"mmrv needs matched lengths, got {r.size} and {p.size}")
    if r.size < 2:
        return 0.0
    worst = np.zeros(r.size)
    for i in range(r.size):
        for j in range(r.size):
            if i == j:
                continue
            if np.sign(r[i] - r[j]) != np.sign(p[i] - p[j]):
                worst[i] = max(worst[i], abs(r[i] - r[j]))
    return float(worst.mean())


@dataclass(frozen=True)
class Rectified:
    """A mean about the expensive world, inferred mostly from the cheap one."""

    value: float
    low: float
    high: float
    #: How much of the interval's width the cheap trials bought. 1.0 means the
    #: proxy carried everything; 0.0 means it contributed nothing and the answer
    #: rests entirely on the paired trials.
    leverage: float
    n_paired: int
    n_proxy: int


def ppi_mean(
    paired_real: Sequence[float],
    paired_proxy: Sequence[float],
    proxy_only: Sequence[float],
    *,
    alpha: float = 0.05,
    tune: bool = True,
) -> Rectified:
    """Prediction-powered inference: many cheap trials, a few expensive ones.

    The cheap evaluator is biased and nobody knows by how much. A small paired
    set measures that bias; the large cheap-only set then estimates the mean
    with the bias removed. What comes back is a valid interval about the
    *expensive* world, usually much tighter than the paired trials alone could
    give — which is the entire argument for owning a simulator you have
    validated rather than one you hope is right.

    With ``tune``, the correction is scaled by the variance-minimising factor
    (PPI++), so a proxy that turns out to be useless degrades to "ignore the
    proxy" instead of actively widening the interval.
    """
    yr = np.asarray(list(paired_real), dtype=np.float64)
    yp = np.asarray(list(paired_proxy), dtype=np.float64)
    up = np.asarray(list(proxy_only), dtype=np.float64)
    if yr.size != yp.size:
        raise ValueError("paired_real and paired_proxy must have equal length")
    if yr.size < 2:
        raise ValueError("prediction-powered inference needs at least 2 paired trials")

    n, big = yr.size, up.size
    lam = 1.0
    if tune and big > 0:
        var_proxy = float(up.var(ddof=1)) if big > 1 else 0.0
        cov = float(np.cov(yr, yp, ddof=1)[0, 1]) if n > 1 else 0.0
        denom = var_proxy * (1 + n / big) if big else 0.0
        lam = float(np.clip(cov / denom, 0.0, 1.0)) if denom > 0 else 0.0

    if big == 0 or lam == 0.0:
        centre = float(yr.mean())
        half = 1.959963985 * float(yr.std(ddof=1)) / sqrt(n) if n > 1 else float("inf")
        return Rectified(centre, centre - half, centre + half, 0.0, n, big)

    rectifier = float((yr - lam * yp).mean())
    centre = lam * float(up.mean()) + rectifier
    var = (
        float(np.var(yr - lam * yp, ddof=1)) / n + (lam**2) * float(up.var(ddof=1)) / big
        if big > 1
        else float(np.var(yr - lam * yp, ddof=1)) / n
    )
    half = 1.959963985 * sqrt(max(var, 0.0))
    naive_half = 1.959963985 * float(yr.std(ddof=1)) / sqrt(n) if n > 1 else float("inf")
    leverage = (
        float(np.clip(1.0 - half / naive_half, 0.0, 1.0))
        if isfinite(naive_half) and naive_half > 0
        else 0.0
    )
    return Rectified(
        float(np.clip(centre, 0.0, 1.0)),
        float(np.clip(centre - half, 0.0, 1.0)),
        float(np.clip(centre + half, 0.0, 1.0)),
        leverage,
        n,
        big,
    )


# --------------------------------------------------------------------------
# do two judges of the same rubric agree
# --------------------------------------------------------------------------


def cohen_kappa(left: Sequence[Any], right: Sequence[Any]) -> float:
    """Chance-corrected agreement between two judges.

    Raw agreement is useless on skewed data: two judges who both call
    everything a failure agree ninety-five percent of the time on a task with a
    five percent success rate, and have demonstrated nothing. Kappa subtracts
    the agreement they would have reached by guessing.

    ``None`` on either side is an abstention and that pair is dropped, because
    "I cannot tell from this video" is a different act from a judgement and
    scoring it as one would punish exactly the honesty worth encouraging.
    """
    pairs = [(a, b) for a, b in zip(left, right) if a is not None and b is not None]
    if not pairs:
        return float("nan")
    n = len(pairs)
    observed = sum(1 for a, b in pairs if a == b) / n
    labels = {a for a, _ in pairs} | {b for _, b in pairs}
    expected = sum(
        (sum(1 for a, _ in pairs if a == label) / n) * (sum(1 for _, b in pairs if b == label) / n)
        for label in labels
    )
    if expected >= 1.0:
        return 1.0 if observed >= 1.0 else 0.0
    return float((observed - expected) / (1 - expected))


def krippendorff_alpha(table: Sequence[Sequence[Any]]) -> float:
    """Chance-corrected agreement among any number of judges, nominal labels.

    ``table[j][i]`` is judge ``j``'s label for item ``i``; ``None`` is an
    abstention. Kappa handles two judges; this handles three or more and
    tolerates the missing entries that arise whenever a person got bored or a
    model declined.

    The conventional reading, which this project adopts as verdict boundaries:
    at or above 0.80 the labels can be relied on, 0.67 to 0.80 supports a
    tentative claim, and below 0.67 the judges are not measuring the same thing
    and nothing computed from their labels should be reported as a finding.
    """
    rows = [list(row) for row in table]
    if len(rows) < 2:
        return float("nan")
    items = max((len(row) for row in rows), default=0)
    if items == 0:
        return float("nan")

    units: list[list[Any]] = []
    for i in range(items):
        seen = [row[i] for row in rows if i < len(row) and row[i] is not None]
        if len(seen) >= 2:
            units.append(seen)
    if not units:
        return float("nan")

    # Observed disagreement: mean over units of the pairwise mismatch rate,
    # weighted by how many judgements that unit received.
    num = den = 0.0
    for seen in units:
        m = len(seen)
        mismatches = sum(1 for a in seen for b in seen if a != b)
        num += mismatches / (m - 1)
        den += m
    observed = num / den if den else 0.0

    pool = [label for seen in units for label in seen]
    total = len(pool)
    if total < 2:
        return float("nan")
    counts: dict[Any, int] = {}
    for label in pool:
        counts[label] = counts.get(label, 0) + 1
    same = sum(c * (c - 1) for c in counts.values())
    expected = 1.0 - same / (total * (total - 1))
    if expected <= 0:
        return 1.0 if observed == 0 else 0.0
    return float(1.0 - observed / expected)


#: What a chance-corrected agreement score is worth, by convention. Used as
#: verdict boundaries rather than advice: below the floor, a judge's
#: conclusions are refused rather than reported with a caveat nobody reads.
AGREEMENT_TRUSTED = 0.80
AGREEMENT_TENTATIVE = 0.67


def agreement_verdict_code(score: float) -> str:
    """Which verdict a measured agreement earns."""
    if not isfinite(score):
        return "judge.unmeasured"
    if score >= AGREEMENT_TRUSTED:
        return "judge.calibrated"
    if score >= AGREEMENT_TENTATIVE:
        return "judge.tentative"
    return "judge.uncalibrated"
