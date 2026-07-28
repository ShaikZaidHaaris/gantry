"""The three gaps worth closing automatically, and nothing else.

Each of these closes one refusal code the spine emits, and each is a transform
nobody could disagree about: a unit conversion is a multiplication, a
permutation is a reordering, a resample is an interpolation. Anything requiring
a judgement — a seven-jointed arm onto a six-jointed one, a pose onto joint
angles — is a *retargeter*, belongs to whoever owns those two machines, and is
deliberately not here.

Two of the three are exactly reversible and say so. The third is not, and says
that instead: a resampler discards frames, that sentence rides in the run's
provenance, and it travels with every number the run produces. An adapter that
quietly dropped a third of the data is how a benchmark becomes a rumour.
"""

from .adapters import (
    PERMUTE,
    RESAMPLE,
    UNIT_CONVERT,
    default_registry,
    permutation_between,
    resample_to,
)

__all__ = [
    "PERMUTE",
    "RESAMPLE",
    "UNIT_CONVERT",
    "default_registry",
    "permutation_between",
    "resample_to",
]
