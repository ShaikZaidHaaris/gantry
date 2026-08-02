"""Gate 2's arithmetic, and the outcomes it is allowed to reach.

The gate's own end-to-end behaviour is exercised against real fixtures; what is
pinned here is the part that decides what a run is *allowed to conclude*, since
that is where being wrong is silent and expensive.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gates.signal import ALPHA, _sign_test, smallest_conclusive  # noqa: E402


def test_a_clean_sweep_below_the_floor_cannot_reach_significance():
    """The reason the floor exists. With five paired clips a clean sweep is
    p=0.0625, so a gate holding out five would refuse everything while looking
    like it was measuring something."""
    floor = smallest_conclusive()
    assert _sign_test(floor - 1, 0) > ALPHA
    assert _sign_test(floor, 0) <= ALPHA


def test_the_floor_is_derived_rather_than_chosen():
    """A constant would be right today and silently wrong the first time
    anybody changed alpha."""
    assert smallest_conclusive(alpha=0.05) == 6
    assert smallest_conclusive(alpha=0.10) < smallest_conclusive(alpha=0.05)
    assert smallest_conclusive(alpha=0.01) > smallest_conclusive(alpha=0.05)


def test_the_test_is_two_sided():
    """A control that beats the data is a real and different finding -- labels
    misaligned, or a split that leaked -- and a one-sided test would report it
    as "no signal" and let it through."""
    assert _sign_test(8, 0) == _sign_test(0, 8)


def test_ties_carry_no_information():
    """Clips where both arms score identically say nothing about which is
    better, and counting them would dilute the result toward "no difference"."""
    assert _sign_test(6, 0) == pytest.approx(2 * (1 / 2**6))


def test_an_even_split_is_as_unremarkable_as_it_looks():
    assert _sign_test(4, 4) == 1.0


def test_no_disagreements_at_all_concludes_nothing():
    """Two arms that never differ have not been separated, and reporting p=0
    for a comparison with no evidence in it is the opposite of the truth."""
    assert _sign_test(0, 0) == 1.0
