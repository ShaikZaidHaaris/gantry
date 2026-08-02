# gantry-evaluator-so101

**A leader-follower pair of SO-ARM101 arms, as a Gantry embodiment and a Gantry evaluator.**

This is ARCHITECTURE.md plane 4, backend family 2 — *Real*. It is also the first implementation of
plane 2, which until now shipped as a contract with nothing behind it.

One rig, two jobs, one loop:

- **Evaluate.** A policy drives the physical follower and the run comes back as `RunRecord`s
  indistinguishable in type from a simulator's, so every feedback module already works on them.
- **Record.** `LeaderTeleop` wears the *policy* contract: the leader's measured pose is the action
  the follower should take, a chunk of one, at the control rate. So the corpus factory of
  `C:\robotics\CORPUS-PROTOCOL.md` is not a second loop with its own clamp and its own bugs — it is
  this evaluator's ordinary run loop with a different action source. `SO101Evaluator.record(...)`
  is the entry point; the corpus is `.episodes`.

```bash
pip install -e plugins/evaluator_so101            # no serial stack, no hardware
pip install -e "plugins/evaluator_so101[serial]"  # add pyserial when a rig is attached
pytest plugins/evaluator_so101                    # the whole suite, on a laptop, with no rig
```

**Status, 2026-07-28.** The four modules were written in parallel and are not yet integrated:
`mock_pair()` in `evaluator.py` constructs `MockTransport(joints=...)` and `SO101Pair(leader=<bus>)`,
neither of which is the signature those objects have, so every test that needs a rig fails at
construction. The suite passes end to end against the real evaluator, transport and embodiment once
`mock_pair()` returns an object carrying the eight attributes `_checked_pair` demands
(`joints`, `control_hz`, `calibration`, `read_leader`, `read_follower`, `write_follower`, `connect`,
`disconnect`). `arm.SO101Pair` does not expose those today, and `SO101Arm` expects
`transport.set_torque(ids, on)` where `Transport` offers `torque_enable()` / `torque_disable()`.

---

## SAFETY — read this before the arm is powered

**The clamp is mandatory.** LeRobot's `max_relative_target` defaults to `None`, which means *no
limit on how far one command may move a joint*. One bad action chunk — an untrained policy, a
wrong-width vector, a NaN — then slams the follower into the table at whatever speed the STS3215 can
manage. `SO101Evaluator(..., max_relative_target=None)` is refused by name, and so is a
non-positive one. Every clamped tick is counted, the counts travel into provenance as a declared
loss, and `clamped_fraction` is reported with the run: a run where the clamp bound on most steps
measured the clamp, not the policy.

Everything else, in roughly the order it will bite you:

| | |
|---|---|
| **No force sensing anywhere.** | Position control only. Nothing on this rig can tell you it is pushing. `gripper_stall()` infers contact from a servo whose command ran past its measured position *and* whose measured position stopped moving — both halves, because tracking error alone fires on every fast close. It is a proxy, not a load reading, and it says nothing about how hard the jaws are pushing. |
| **Keep a hand on the power.** | The follower's supply is the emergency stop. No software interlock is fast enough to matter once six servos are already moving. |
| **Clear the workspace before every session.** | The follower reproduces a policy's mistakes at full travel. Nothing valuable, nothing fragile, nothing attached to a person inside the swept volume. |
| **Never energise the follower with a hand in the gripper.** | The Dex1 two-finger gripper closes on the sixth DoF with no torque feedback and no back-off. |
| **Assign motor IDs on an otherwise empty bus.** | Every STS3215 ships as `id=1`. Two servos with the same ID answer at once and the bus becomes unreadable. See bring-up step 2. |
| **Power the arm before opening the port; close the port before cutting power.** | A half-written packet on a dying bus leaves a servo in an undefined state. |
| **Leader torque off, follower torque on.** | `connect()` does this. A leader with torque on cannot be backdriven, and an operator pulling against it will strip a horn. |

---

## The wiring

One leader-follower pair is **one operator station**, not two rigs. Two arms is the minimum useful
configuration, not a doubling of throughput: teleoperation is strictly serial, one human at a time,
which is why teleop hours — not GPU hours — are the binding constraint in `CORPUS-PROTOCOL.md`.

```
   LEADER (the operator holds this)          FOLLOWER (this one moves the world)
   6x STS3215, daisy-chained                 6x STS3215, daisy-chained
   torque OFF, backdriven                    torque ON, tracking
   ┌───────────────────────────┐             ┌───────────────────────────┐
   │ id 1  shoulder_pan        │             │ id 1  shoulder_pan        │
   │ id 2  shoulder_lift       │             │ id 2  shoulder_lift       │
   │ id 3  elbow_flex          │             │ id 3  elbow_flex          │
   │ id 4  wrist_flex          │             │ id 4  wrist_flex          │
   │ id 5  wrist_roll          │             │ id 5  wrist_roll          │
   │ id 6  gripper             │             │ id 6  gripper             │
   └────────────┬──────────────┘             └────────────┬──────────────┘
                │ 3-pin servo bus                         │ 3-pin servo bus
       ┌────────┴─────────┐                      ┌────────┴─────────┐
       │ Waveshare Serial │                      │ Waveshare Serial │
       │ Bus Servo driver │                      │ Bus Servo driver │
       └───┬─────────┬────┘                      └───┬─────────┬────┘
       USB │     12V │                           USB │     12V │
           │         └── supply, on a switch ────────┘         │
           └──────────────┬────────────────────────────────────┘
                          │
                   ┌──────┴───────┐
                   │     host     │   one SO101Pair, two ports, two calibration files
                   └──────┬───────┘
                          │
     ┌────────────────────┴────────────────────┐
     │                                         │
  evaluate: policy -> follower           record: LeaderTeleop -> follower
  observations + operator label          leader pose as the action
        -> RunRecord                           -> RunRecord (.episodes = the corpus)
```

The arms are electrically independent: separate driver boards, separate USB ports, separate
supplies, **separate calibration files**. The pair refuses to start if both arms point at the same
calibration file, because that means one of them is running the other's homing offsets — the #3758
failure, installed on purpose.

---

## Bring-up, in order

Steps 2 and 3 are per-arm and per-servo. There is no batch version of either, and inventing one is
how two servos end up sharing an ID.

**1. Find the port.** With the arm's USB unplugged, list ports; plug it in, list again; the new
entry is that arm. `list_serial_ports()` and `port_removed(before, after)` in `transport.py` do
exactly this. Do it one arm at a time and write down which is which — Windows reassigns `COM*` after
a reboot and Linux `/dev/ttyACM*` after a replug.

```bash
python -m lerobot.find_port
```

**2. Assign motor IDs, one servo at a time, on an otherwise empty bus.** Every STS3215 ships with
`id=1`. Connect **exactly one** servo, set its ID, unplug it, connect the next. Only when all six
have distinct IDs do you daisy-chain them. The order is the LeRobot order — pan, lift, elbow, wrist
flex, wrist roll, gripper — because that order *is* the action vector, and nothing downstream can
tell you it is wrong.

```bash
python -m lerobot.setup_motors --robot.type=so101_follower --robot.port=<PORT>
```

**3. Calibrate each arm separately.** Move every joint through its full travel so the range is
measured. The variant is `so101_new_calib`, and `load_calibration()` verifies rather than merely
parses: a missing joint, a duplicated motor id or an inverted range each produce an arm that moves
and is wrong. An *unmeasured* calibration is refused outright, because it makes every normalised
value a fiction.

```bash
python -m lerobot.calibrate --robot.type=so101_follower --robot.port=<FOLLOWER_PORT> --robot.id=<name>
python -m lerobot.calibrate --teleop.type=so101_leader   --teleop.port=<LEADER_PORT>   --teleop.id=<name>
```

**4. Verify teleoperation before recording anything.** Drive the follower from the leader over an
empty workspace and watch all six joints. A joint that lags, jitters or moves the wrong way is a
calibration or an ID problem, and it will be invisible in the data afterwards. `SO101Pair` reports
per-joint tracking error and gripper stalls for exactly this.

**5. Only then record.** A corpus recorded on an unverified rig is a corpus you find out about after
training.

---

## The calibration trap, and why it is a discriminator

LeRobot issue **#3758** (open): a roughly 17-degree joint calibration offset produced about 6 cm of
gripper error, and it was **invisible during training**. The policy learns the biased mapping, the
loss curve looks healthy, and nothing in the data is structurally wrong — the channel has the right
name, the right width, the right units and the right meaning tag.

Nothing generic catches that, so it is made structural in two places:

- The **calibration variant is a discriminator** on every joint channel, alongside the normalisation
  convention and the gripper kind. Two corpora recorded under different conventions refuse to bind,
  with the reason named, instead of averaging into one silently wrong mapping.
- The **calibration hash of the arms that actually ran** rides on the embodiment component's
  `artifact_digest`, so it is part of the component `ref`. Two runs across a recalibration are
  therefore not comparable while holding the embodiment fixed, and `RunSet.comparable` refuses on
  its own — nobody has to remember.

This is also why axis 1 of `CORPUS-PROTOCOL.md` — injected calibration drift, reported in Cartesian
millimetres rather than degrees — is the flagship experiment: the failure is real, free to inject
post-hoc, and nobody has measured what it costs.

---

## What this evaluator declares, and what follows

| Capability | Value | Why |
|---|---|---|
| `closed_loop` | **True** | The world responds to what the policy did. |
| `outcomes` | **True** | A person watching can always say what happened. A trial the policy crashed out of is not asked about: it stays unknown, with the error beside it, and drops out of the success-rate denominator rather than out of the record. |
| `seedable` | **False** | See below. This one is load-bearing. |
| `stage_events` | **False** unless a `PoseSource` is configured | See below. |

### `seedable = False`, and what it costs you

The contract says a seedable evaluator *"returns identical records for identical seeds, which is
what a paired comparison rests on."* A physical rig cannot do that. Object placement is a human act
with its own error; so are table position, lighting, servo temperature and gripper wear. There is no
number that reproduces a scene. Scenes here *do* carry seeds — the seed indexes a start pose in the
pre-registered list — and every episode says in words that the placement was set by hand, not
verified and not reproducible, so the presence of a seed cannot be read as reproducibility.

Declaring `True` would not produce an error. It would produce a resolver that cheerfully plans a
**paired** analysis over unpaired data, and a result that reads exactly like a valid one. So:

> **The paired-seed power calculation in `CORPUS-PROTOCOL.md` applies to the SIM tier only.**
> Tier A (sim teleop) gets identical object start poses across arms and the n=127 → 76 power gain
> that comes with them. Tier B — this hardware — does not, and its comparisons are unpaired, with
> the sample sizes that implies. A real-rig number is a transfer check on a rank already established
> in sim, never a paired contrast in its own right.

The sentence to quote is exported as `SEEDABLE_IS_FALSE`.

### `stage_events = False` unless a pose source exists

A funnel needs to know where the *object* is. On hardware that means a perception rig — cameras,
extrinsics, a pose estimator with a stated error budget — and it **does not exist yet**. Nothing
implements `PoseSource`, and that is the honest state of the hardware rather than an omission.

The only contact signal available is `gripper_stall()`, which yields a GRASP *candidate* and nothing
behind it: no lift, no place. A funnel with one populated rung and several empty ones reads as a
finished analysis of a policy that never lifts anything. So the stall proxy is recorded as an
**episode annotation**, never as a stage event, and no stage events are emitted at all until a pose
source is configured. The record's provenance says so in words, and the resolver refuses funnel
diagnosis on it rather than handing back an empty table. Configure a pose source and the declaration
flips, the events appear, and the conformance kit checks the claim in both directions.

The sentence to quote is exported as `STAGE_EVENTS_NEED_A_POSE_SOURCE`.

---

## Isolation

`isolation` stays `in-process`. The only non-core dependency is `pyserial`, which is pure Python and
conflicts with nothing — this is not robosuite. Nor would isolation help: the `evaluation` plane is
not proxyable in core, and putting a physical arm's safety clamp on the far side of an IPC boundary
would be a downgrade rather than a hardening. The serial library is an **optional extra**, so the
package installs and the whole test suite runs on a machine that will never see a robot; the refusal
when it is missing happens where the port would be opened, naming what to install, rather than as an
ImportError at package import that would take the mock path down with it.

---

## Testing with no hardware

`MockTransport` is not a convenience. An evaluator whose correctness can only be checked while
standing next to the rig is an evaluator whose correctness is checked once, by whoever was standing
there — and this one is the instrument the entire SO-101 campaign is measured with. Every test in
`tests/test_conformance.py` runs with no driver board, no USB device and no `pyserial` installed:
both conformance kits, the mandatory clamp, the wrong-width refusal, the stall-proxy-is-not-a-stage-
event rule, and the duration-confound silence. `VirtualClock` jumps to each deadline instead of
sleeping to it, so a 30 Hz loop is testable in milliseconds — and a record produced under it carries
`paced=False`, because nothing in a virtual-clock run is evidence about timing.

The one test worth naming: `test_duration_alone_moves_none_of_the_numbers` runs the same rig and the
same scripted operator twice, at the short and long episode lengths of
`gantry.fixtures.make_duration_confound()`, and requires every metric to agree. Trial length on
hardware varies by an order of magnitude for reasons that have nothing to do with the policy — a
slow approach, a long reset, an operator watching one more second before calling it. Any quality
computed as a raw count over a trial would score a patient policy as a worse one. Metrics honestly
*about* time are exempt by their declared units; nothing else is. Getting that suite to stay silent
is the point.

---

## Layout

| File | What it owns |
|---|---|
| `transport.py` | the servo bus: the Feetech STS/SMS packet codec, `Transport`, `SerialTransport` (needs the `serial` extra), `MockTransport`, tick↔normalised conversion, bus-level stall detection |
| `arm.py` | one six-servo arm and the pair: `SO101Arm`, `SO101Pair`, verified `load_calibration`, `SafetyLimits`, rate keeping, drift analysis, `GripperStallDetector` |
| `embodiment.py` | plane-2 data: `so101_embodiment`, `so101_leader`, the joint and camera channels, and the retargeters between normalised range and joint angles |
| `evaluator.py` | the operator station: `SO101Evaluator`, `Console` / `TerminalConsole` / `ScriptedConsole`, `PoseSource`, `gripper_stall`, `LeaderTeleop`, `mock_pair` |
| `__init__.py` | the public surface, and the one place the four modules' overlapping vocabularies are reconciled |

`__init__.py` refuses to import if the modules disagree about joint order, normalisation or
calibration variant. That is not defensive decoration: order is the entire content of a six-float
action vector, and a package whose modules disagree about it drives the wrong joint while every
structural check passes.

Where two modules define the same word, the package exports one of them: joint order is `JOINTS`
(`arm.JOINT_KEYS` and `embodiment.JOINT_NAMES` are the same tuple under other names), and
`so101_embodiment` is `embodiment`'s. `Calibration`, `JointCalibration` and `SafetyLimits` exist in
both `transport` and `arm` as genuinely different types — a bus-level register map versus an
arm-level verified file — so neither is re-exported; import them from the module you mean.
