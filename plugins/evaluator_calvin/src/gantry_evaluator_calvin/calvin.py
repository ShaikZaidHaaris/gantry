"""CALVIN, for the axis every other backend here is blind to: length.

Every task this project has measured is one goal deep. Lift the cube. Put the can
in the bin. Open the drawer. A policy that can do any of those has demonstrated
something real, and it has demonstrated nothing whatsoever about the failure that
dominates useful robot work: doing the second thing after the first, from
whatever state the first left behind.

That failure is invisible to a single-goal benchmark by construction. Success
rates are measured from a *reset* scene every time, so a policy that only works
from a canonical starting state scores identically to one that recovers. On the
fifth instruction of a chain those two policies are nowhere near each other.

CALVIN measures exactly that. A row is thirty-four possible instructions, an
evaluation is a chain of five drawn in sequence, and the chain stops at the first
one the policy fails. The headline number is not a success rate -- it is *average
chain length*, between zero and five, and the distinguishing property is that it
falls off a cliff for policies that need a clean start.

Chain length is the outcome, so success needs defining
------------------------------------------------------
This is the awkward part, and pretending it is not would be worse. The rollout
loop's ``success`` field is one boolean per trial, and a chain has five results.
Collapsing them loses information no matter how it is done, so the collapse is an
explicit argument rather than a convention:

``all_five`` -- success means the whole chain. Harsh, standard in the CALVIN
literature as "5/5", and the right default because it is the one that cannot be
gamed by a policy that reliably does the first subtask and nothing else.

``at_least(k)`` -- success means k or more. Useful while a policy is bad enough
that 5/5 is always zero and the benchmark has stopped being informative.

The chain length itself is *always* recorded, per episode, alongside which
subtask broke it. That is the number to report; the boolean exists because the
rest of the framework counts things.

Every subtask is a milestone
----------------------------
So a funnel over a CALVIN run is genuinely meaningful -- ``stage_events=True``
here, unlike almost every other backend, and the stage names are the instructions
themselves. "Where does the chain break" is then a table rather than an
investigation, and it is the single most useful diagnostic this suite produces.
"""

from __future__ import annotations

from typing import Any, Callable, Mapping, Sequence

import numpy as np

from gantry.contracts.evaluator import Scene, TaskSpec, evaluator_descriptor
from gantry.errors import ConfigError
from gantry.rollout import ClosedLoop, Step, Trial
from gantry.spine import ChannelSpec, Descriptor, Measurement

VERSION = "0.1.0.dev0"

#: CALVIN's environment splits. D→D is same-environment; the rest hold out a
#: whole scene, which is the harder and more interesting claim.
SPLITS = ("D_D", "ABC_D", "ABCD_D", "ABC_D_zero_shot")

#: How many instructions a standard chain is.
CHAIN = 5

#: The length of one subtask's attempt, in CALVIN's own convention.
SUBTASK_STEPS = 360

CONTROL_HZ = 30.0

#: Relative Cartesian: pose deltas, then a binary gripper.
ACTION = ChannelSpec(
    "action",
    "vector",
    (7,),
    "float32",
    semantics="actuation",
    rate_hz=CONTROL_HZ,
    dim_labels=("dx", "dy", "dz", "droll", "dpitch", "dyaw", "gripper"),
)


def all_five(reached: int) -> bool:
    """Success is the whole chain.

    The default because it is the collapse a policy cannot game. Anything looser
    rewards reliably completing the first instruction, which is the behaviour a
    long-horizon benchmark exists to stop rewarding.
    """
    return reached >= CHAIN


def at_least(count: int) -> Callable[[int], bool]:
    """Success is ``count`` subtasks or more.

    For when 5/5 is zero for every policy under test and the benchmark has
    stopped separating them. Honest to use and dishonest to leave unstated, which
    is why it lands in the descriptor and in the record.
    """
    if not 1 <= int(count) <= CHAIN:
        raise ConfigError(f"a chain is {CHAIN} long, so k must be 1..{CHAIN}; got {count}")

    def enough(reached: int) -> bool:
        return reached >= int(count)

    enough.__name__ = f"at_least_{int(count)}"
    return enough


def make_env(split: str = "D_D", **kwargs: Any) -> Any:
    """Construct CALVIN's environment. Imported here, not at module load."""
    try:
        from calvin_env.envs.play_table_env import get_env
    except ImportError as error:  # pragma: no cover - needs the simulator
        raise ConfigError(
            "running CALVIN needs the simulator: pip install "
            "'gantry-evaluator-calvin[sim]' (it wants pybullet and the CALVIN "
            "dataset for its task definitions and starting states)"
        ) from error
    return get_env(split, **kwargs)


class Chain:
    """One chain of instructions, as a world.

    The subtlety worth reading: this world does *not* reset between subtasks. The
    second instruction starts from wherever the first ended, which is the entire
    reason CALVIN measures something the single-goal suites cannot. A version of
    this adapter that reset per subtask would produce five independent single-goal
    trials wearing a chain's clothing, and its numbers would look plausible.
    """

    def __init__(
        self,
        env: Any,
        *,
        oracle: Any,
        subtask_steps: int = SUBTASK_STEPS,
        succeeded: Callable[[int], bool] = all_five,
    ):
        self.env = env
        self.oracle = oracle
        self.subtask_steps = int(subtask_steps)
        self.succeeded = succeeded
        self.instructions: tuple[str, ...] = ()
        self.reached = 0
        self.broke_on: str | None = None
        self._index = 0
        self._steps_in_subtask = 0
        self._start_info: Any = None

    # -- the chain ---------------------------------------------------------

    @property
    def current(self) -> str | None:
        return self.instructions[self._index] if self._index < len(self.instructions) else None

    def begin(self, scene: Scene) -> Mapping[str, Any]:
        metadata = scene.metadata or {}
        self.instructions = tuple(metadata.get("instructions", ()))
        if not self.instructions:
            raise ConfigError(
                f"CALVIN scene {scene.id!r} carries no instruction chain; a chain "
                "evaluation with nothing to chain is a single-goal run mislabelled"
            )
        self.reached = 0
        self.broke_on = None
        self._index = 0
        self._steps_in_subtask = 0
        observation = self.env.reset(
            **{k: v for k, v in metadata.items() if k in ("robot_obs", "scene_obs")}
        )
        self._start_info = self.env.get_info()
        return _flatten(observation)

    def advance(self, action: np.ndarray) -> Step:
        observation, _reward, _done, _info = self.env.step(action)
        self._steps_in_subtask += 1
        info = self.env.get_info()
        instruction = self.current
        reached: tuple[str, ...] = ()

        solved = instruction is not None and bool(
            self.oracle.get_task_info_for_set(self._start_info, info, {instruction})
        )
        if solved:
            reached = (str(instruction),)
            self.reached += 1
            self._advance_subtask(info)
        elif self._steps_in_subtask >= self.subtask_steps:
            # Out of time on this subtask. The chain stops here -- that is the
            # rule that makes chain length mean anything.
            self.broke_on = instruction
            return Step(
                observation=_flatten(observation), done=True, reached=reached, info=dict(info)
            )

        done = self._index >= len(self.instructions)
        return Step(
            observation=_flatten(observation),
            reward=1.0 if solved else 0.0,
            done=done,
            reached=reached,
            info=dict(info),
        )

    def _advance_subtask(self, info: Any) -> None:
        self._index += 1
        self._steps_in_subtask = 0
        # The next subtask is judged relative to where this one finished, not to
        # the episode's opening state.
        self._start_info = info

    def verdict(self, trial: Trial) -> bool | None:
        """Collapse the chain to one boolean, having recorded the length itself."""
        return self.succeeded(self.reached)

    def close(self) -> None:
        if callable(getattr(self.env, "close", None)):
            self.env.close()


def _flatten(observation: Any, prefix: str = "") -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    if not isinstance(observation, Mapping):
        return {"observation": np.asarray(observation)}
    for key, value in observation.items():
        name = f"{prefix}{key}"
        if isinstance(value, Mapping):
            out.update(_flatten(value, prefix=f"{name}."))
            continue
        array = np.asarray(value)
        if array.dtype.kind in "OUSV":
            continue
        out[name] = array
    return out


class CalvinEvaluator(ClosedLoop):
    """Runs a policy over CALVIN instruction chains, scored by chain length."""

    rate_hz = CONTROL_HZ
    embodiment = "franka"

    def __init__(
        self,
        split: str = "D_D",
        *,
        name: str = "calvin",
        chains: Sequence[Sequence[str]] = (),
        succeeded: Callable[[int], bool] = all_five,
        subtask_steps: int = SUBTASK_STEPS,
        observes: Sequence[str] | None = None,
        factory: Callable[..., Any] = make_env,
        oracle: Any = None,
        env_kwargs: Mapping[str, Any] | None = None,
    ):
        """``chains`` is the list of instruction sequences to attempt.

        Required, and required to be sequences rather than single instructions: a
        CALVIN evaluator handed one-instruction chains is a single-goal benchmark
        wearing this plugin's name, which would be the most misleading thing here.
        """
        if split not in SPLITS:
            raise ConfigError(f"unknown CALVIN split {split!r}; it ships {list(SPLITS)}")
        chains = tuple(tuple(str(step) for step in chain) for chain in chains)
        if not chains:
            raise ConfigError(
                f"{name}: needs its instruction chains, this suite measures what "
                "happens on the second instruction after the first, and there is "
                "nothing to measure without them"
            )
        singles = [chain for chain in chains if len(chain) < 2]
        if singles:
            raise ConfigError(
                f"{name}: {len(singles)} chain(s) have fewer than two instructions. "
                "A one-deep 'chain' makes this a single-goal benchmark reporting "
                "itself as a long-horizon one, which is worse than not running it"
            )
        self._split = split
        self._name = name
        self._chains = chains
        self._succeeded = succeeded
        self._subtask_steps = int(subtask_steps)
        self.observes = tuple(observes) if observes is not None else None
        self._factory = factory
        self._oracle = oracle
        self._env_kwargs = dict(env_kwargs or {})
        self._world: Chain | None = None

    # -- contract ----------------------------------------------------------

    def descriptor(self) -> Descriptor:
        return evaluator_descriptor(
            name=self._name,
            version=VERSION,
            # Every subtask is a milestone, so a funnel here is genuinely
            # meaningful -- which is not true of most backends, where it would
            # have one rung. "Where does the chain break" becomes a table.
            stage_events=True,
            outcomes=True,
            seedable=True,
            closed_loop=True,
            hosts_embodiment=False,
            split=self._split,
            chains=len(self._chains),
            # In the descriptor because a run scored at_least(2) and one scored
            # 5/5 are not comparable, and the difference must not live only in
            # somebody's memory of how they invoked it.
            success_means=getattr(self._succeeded, "__name__", "custom"),
            subtask_steps=self._subtask_steps,
            control_hz=CONTROL_HZ,
        )

    def action(self) -> ChannelSpec:
        return ACTION

    def declares(self) -> Mapping[str, ChannelSpec]:
        return {ACTION.name: ACTION}

    def task_for(
        self, name: str = "", chains: int | None = None, horizon: int | None = None
    ) -> TaskSpec:
        """One scene per chain, with the whole chain in its metadata.

        The horizon defaults to the full chain's worth of steps, because a
        horizon that cuts a chain short would report a policy as having broken on
        subtask three when in fact the harness ran out.
        """
        chosen = self._chains if chains is None else self._chains[: max(1, int(chains))]
        longest = max(len(chain) for chain in chosen)
        return TaskSpec(
            name=name or f"calvin/{self._split}",
            scenes=tuple(
                Scene(
                    id=f"chain-{index:03d}",
                    instruction=chain[0],
                    seed=index,
                    metadata={"instructions": list(chain)},
                )
                for index, chain in enumerate(chosen)
            ),
            horizon=(longest * self._subtask_steps) if horizon is None else int(horizon),
            stages=tuple(dict.fromkeys(step for chain in chosen for step in chain)),
        )

    @property
    def oracle(self) -> Any:
        """CALVIN's own task checker. Injected in tests, loaded here otherwise."""
        if self._oracle is None:  # pragma: no cover - needs the simulator
            try:
                import hydra
                from calvin_agent.evaluation.multistep_sequences import get_sequences  # noqa: F401
            except ImportError as error:
                raise ConfigError(
                    "checking CALVIN subtasks needs its task oracle: pip install "
                    "'gantry-evaluator-calvin[sim]'"
                ) from error
            self._oracle = hydra.utils.instantiate({"_target_": "calvin_env.envs.tasks.Tasks"})
        return self._oracle

    def world_for(self, scene: Scene) -> Chain:
        if self._world is None:
            self._world = Chain(
                self._factory(self._split, **self._env_kwargs),
                oracle=self.oracle,
                subtask_steps=self._subtask_steps,
                succeeded=self._succeeded,
            )
        return self._world

    def close(self) -> None:
        if self._world is not None:
            self._world.close()
            self._world = None

    # -- the number that matters ------------------------------------------

    def assemble(self, trial, scene, epoch, task, protocol):
        """Record the chain length, and where it broke, on the episode itself.

        The boolean exists because the rest of the framework counts things. This
        is the number to report, and burying it in a metric average would lose
        the per-chain detail that makes the diagnosis possible.
        """
        episode = super().assemble(trial, scene, epoch, task, protocol)
        chain = list((scene.metadata or {}).get("instructions", ()))
        reached = len(trial.events)
        labels = episode.labels
        return episode.with_labels(
            type(labels)(
                success=labels.success,
                stage_events=labels.stage_events,
                annotations={
                    **dict(labels.annotations),
                    "chain": chain,
                    "chain_length": reached,
                    "chain_of": len(chain),
                    "broke_on": chain[reached] if reached < len(chain) else None,
                    "success_means": getattr(self._succeeded, "__name__", "custom"),
                },
            )
        )

    def metrics(self, episodes: Sequence[Any]) -> dict[str, Measurement]:
        """Average chain length beside the success rate.

        The headline CALVIN number, and the one whose behaviour is interesting: it
        falls off a cliff for a policy that needs a clean starting state, while a
        5/5 rate is zero for such a policy and zero for a policy that does
        nothing at all.
        """
        out = dict(super().metrics(episodes))
        lengths = [
            int(episode.labels.annotations.get("chain_length", 0))
            for episode in episodes
            if episode.labels.annotations.get("chain_length") is not None
            and not episode.labels.annotations.get("error")
        ]
        if lengths:
            mean = sum(lengths) / len(lengths)
            spread = (
                (sum((value - mean) ** 2 for value in lengths) / (len(lengths) - 1)) ** 0.5
                if len(lengths) > 1
                else 0.0
            )
            half = 1.96 * spread / (len(lengths) ** 0.5) if lengths else 0.0
            out["chain_length"] = Measurement(
                value=round(mean, 4),
                n=len(lengths),
                ci=(round(max(0.0, mean - half), 4), round(min(float(CHAIN), mean + half), 4)),
                method="mean subtasks completed before the chain broke, normal interval",
                detail={
                    "of": CHAIN,
                    "distribution": {
                        str(k): lengths.count(k) for k in range(0, CHAIN + 1) if lengths.count(k)
                    },
                },
            )
        return out
