"""Describe a machine in a file, or derive a first draft from its data."""

from .declared import (
    KNOWN_CAPABILITIES,
    VERSION,
    from_file,
    from_mapping,
    from_schema,
    to_file,
)

__all__ = [
    "KNOWN_CAPABILITIES",
    "VERSION",
    "from_file",
    "from_mapping",
    "from_schema",
    "to_file",
]
