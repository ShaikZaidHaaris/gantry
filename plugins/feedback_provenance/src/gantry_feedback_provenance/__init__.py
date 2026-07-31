"""What a dataset may legally be used for, traced through everything that made it."""

from .provenance import (
    ATTRIBUTION,
    LICENCE_KEYS,
    NON_COMMERCIAL,
    PATTERNS,
    PERMISSIVE,
    RANK,
    UNKNOWN,
    VERSION,
    Claim,
    Provenance,
    chain,
    claims_in,
    classify,
    usable_for,
    worst,
)

__all__ = [
    "ATTRIBUTION",
    "LICENCE_KEYS",
    "NON_COMMERCIAL",
    "PATTERNS",
    "PERMISSIVE",
    "RANK",
    "UNKNOWN",
    "VERSION",
    "Claim",
    "Provenance",
    "chain",
    "claims_in",
    "classify",
    "usable_for",
    "worst",
]
