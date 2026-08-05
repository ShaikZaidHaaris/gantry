# gantry-connector-so101

Reads an **SO-ARM101 corpus-factory recording** — a LeRobot v2.x dataset plus the
per-episode metadata sidecar `harness/so101/record_session.py` writes — as Gantry
`EpisodeRecord`s.

The rig is a data source, not an evaluator. The arms produce recorded episodes; this
connector exposes them so the feedback plane can grade the corpus. The corpus is the
product.

```python
from gantry_connector_so101 import SO101Connector, FOLLOWER_REF

corpus = SO101Connector("datasets/bench02-e-prac-s001-b1", embodiment=FOLLOWER_REF)
corpus.supports_block_analysis          # can CORPUS-PROTOCOL's confound analysis run?
corpus.calibration_digests              # one digest, or the read was refused
corpus.metadata("episode_000000")       # the design record for one episode
```

## Why this is not `connector_lerobot`

It **composes** that reader — parquet indexing, chunked path templates, video resolution
and the selective-read source are all inherited, none of them reimplemented. It adds
three things, and each one is a refusal or a declaration that reader is right not to
make:

| | what it adds | the failure it prevents |
|---|---|---|
| 1 | **Embodiment binding + width guard.** The corpus is bound to a declared ref and refused if the width or joint order disagrees, in either direction. | A 12-wide bimanual corpus read as one arm, or the reverse. Downstream, nothing distinguishes them but width — and width does not distinguish "left arm only" from "right arm only", which is why the ref is declared and recorded. |
| 2 | **Calibration provenance.** Both arms' calibration SHA-256 on every episode, a digest over the pair as a channel *discriminator*, and a **refusal** to pool a corpus spanning two of them. | LeRobot #3758: an offset invisible in the training loss. The hash is the only thing that makes a mid-campaign recalibration findable retrospectively, which only works if a corpus that straddles one cannot be read as if it did not. |
| 3 | **Corpus-factory metadata.** corpus/condition/operator/session/block/index, treatment + the RNG seed that produced it, start poses, outcome tag, drift dose, dropped frames, per-joint tracking error. Missing block fields → readable, and **declares** it cannot support the block analysis. | A block analysis computed over episodes whose session is unknown, which is a confident answer to a different question. |

## Capabilities, honestly

| capability | value | why |
|---|---|---|
| `lazy` | `True` | `schema()` reads no steps; `open()` returns the LeRobot reader's own source, so a window read still projects columns and slices rows. |
| `media` | delegated | True only when the mp4s are present *and* decodable. That reader distinguishes "no images" from "no decoder"; this one repeats its answer. |
| `outcomes` | computed | True only when the sidecar carries an outcome tag. |
| `stage_events` | computed, normally `False` | The rig has no perception and the sidecar records one object pose per *episode*, not per step. A `True` here yields an empty funnel that reads as a finished analysis. A sim corpus that does have poses is a per-corpus fact: pass `stage_events=` with a written `stage_events_source`. There is no detector, because a detector with no pose source is a guess wearing a milestone's name. |

## The join

`record_session.metadata_columns()` emits no LeRobot episode index, and a join that is
off by one gives every episode another episode's condition with no symptom at all. So
the join is either **keyed** (an `episode_index` column) or **ordinal and checked**
(file order cross-checked against the dataset's own frame counts). An ordinal join that
nothing can check — every episode the same length — is refused; `join="ordinal"` forces
it and records that it rests on the caller's instruction.

**Recommended fix upstream:** add `episode_index` to `metadata_columns()`.

## What this corpus is *not*

Written down because the alternative is somebody reading it as a finding — see
`tests/test_duration_confound.py`:

- **Joint percent is not a position.** The channels declare
  `so101.joint_position_normalized`, so nothing binds them to a consumer wanting a point
  in space. Path efficiency over percent-of-travel — where a jaw and a shoulder share a
  unit, and the published URDF has no linear gripper joint — comes back *unavailable*.
- **The gripper is a dimension, not a channel** (element 5). A consumer wanting a scalar
  actuation signal is not handed the sixth column.
- **Per-step means over an absolute command inherit episode length.** One grasp and one
  release per episode is a fixed amount of gripper travel over a variable number of
  steps, so anything averaging `|Δaction|` falls as 1/T on a corpus with nothing wrong
  with it. Real, not a fixture artefact.

## No hardware, no dataset

`gantry_connector_so101.fixtures.write_corpus()` renders any `gantry.fixtures` suite as
a real on-disk LeRobot dataset plus the recorder's sidecar. The whole test suite runs on
a laptop with no arm, no serial port and no recorded data.
