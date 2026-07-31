"""Assembles the modules' findings into the document a contributor reads."""

from .report import (
    BLOCKING,
    BLOCKS,
    SECTIONS,
    VERSION,
    Assembled,
    Section,
    as_markdown,
    assemble,
    write,
)

__all__ = [
    "BLOCKING",
    "BLOCKS",
    "SECTIONS",
    "VERSION",
    "Assembled",
    "Section",
    "as_markdown",
    "assemble",
    "write",
]
