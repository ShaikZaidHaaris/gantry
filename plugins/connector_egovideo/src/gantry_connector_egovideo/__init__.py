"""Read a folder of egocentric clips plus its manifest as episodes."""

from .egovideo import (
    MANIFEST,
    REQUIRED,
    RGB,
    SUFFIXES,
    VERSION,
    Clip,
    EgoVideoConnector,
    decode,
    probe,
    read_manifest,
)

__all__ = [
    "MANIFEST",
    "REQUIRED",
    "RGB",
    "SUFFIXES",
    "VERSION",
    "Clip",
    "EgoVideoConnector",
    "decode",
    "probe",
    "read_manifest",
]
