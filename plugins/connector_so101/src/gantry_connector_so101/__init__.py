"""Read an SO-ARM101 corpus-factory recording as Gantry episodes.

The rig is a data source. This connector binds a recorded LeRobot dataset to a
declared SO-101 embodiment, refuses a width or joint-order that disagrees with
it, carries both arms' calibration digests into every record and refuses to pool
a corpus that spans a recalibration, and attaches the per-episode design
metadata ``CORPUS-PROTOCOL.md`` needs — declaring, when that metadata is
missing, that the block analysis cannot be run rather than running one anyway.
"""

from .binding import BIMANUAL_REF, BINDINGS, CALIBRATION_DIGEST, FOLLOWER_REF, Binding, get_binding
from .connector import DESCRIBED, OUTCOME_SUCCESS, VERSION, SO101Connector
from .sidecar import (
    BLOCK_ANALYSIS_COLUMNS,
    CALIBRATION_COLUMNS,
    JOIN_COLUMNS,
    PRODUCER,
    SIDECAR_NAMES,
    EpisodeMetadata,
    Join,
    Sidecar,
    find_sidecar,
    join_rows,
    read_sidecar,
)

__all__ = [
    "BIMANUAL_REF",
    "BINDINGS",
    "BLOCK_ANALYSIS_COLUMNS",
    "CALIBRATION_COLUMNS",
    "CALIBRATION_DIGEST",
    "DESCRIBED",
    "FOLLOWER_REF",
    "JOIN_COLUMNS",
    "OUTCOME_SUCCESS",
    "PRODUCER",
    "SIDECAR_NAMES",
    "VERSION",
    "Binding",
    "EpisodeMetadata",
    "Join",
    "SO101Connector",
    "Sidecar",
    "find_sidecar",
    "get_binding",
    "join_rows",
    "read_sidecar",
]
