"""SimplerEnv, and the question no other backend here can answer.

Every other evaluator in this project answers "how well does this policy do in
simulation". None of them answers the question anybody actually cares about,
which is whether that number predicts anything on a real robot. A benchmark can
be internally immaculate — seeded, powered, paired, letter-displayed — and still
rank policies in an order that reverses the moment they touch hardware, and
nothing inside the benchmark would show it.

SimplerEnv exists to make that checkable. Its scenes are built to approximate
specific real setups whose real success rates were measured and published, for
policies that are publicly available. So there are two numbers per (policy,
task): what the simulator said, and what the real robot did. The relationship
between those two columns is the measurement, and this project already has the
statistic for it — ``gantry.spine.inference.mmrv`` — sitting in core with nothing
to feed it. This plugin is the feed.

Real ranking violation, not correlation
---------------------------------------
The tempting summary is a correlation coefficient, and it is the wrong one.
Correlation asks whether the numbers move together; a benchmark's job is to get
the *order* right. A proxy that reports every real rate at half its true value
correlates perfectly and misleads nobody, while one that is well correlated but
swaps the top two policies is useless for exactly the decision benchmarks get
used for. MMRV asks how badly the ordering is violated, which is the question,
and it is why the statistic already in core is that one.

What this plugin is careful about
---------------------------------
**Two variants, not one.** SimplerEnv ships "visual matching" (scene tuned to
look like the real photographs) and "variant aggregation" (lighting, texture and
layout deliberately perturbed). They measure different things and they do not
mix: a policy's visual-matching rate compared against another's variant rate is
not a comparison. The variant is part of the scene id and part of the record.

**Real rates are provenance, not results.** A published real success rate is
somebody else's measurement, on their robot, on their day. It is recorded as an
annotation with its source named, never merged into this run's success rate. The
pairing is stated so the transfer analysis can find it; the two numbers stay
distinguishable forever.

**A real rate quoted without a source is refused.** The whole value of the paired
column is that it can be checked. A number typed in from memory is worse than no
number, because it looks identical to one that can be.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from gantry.contracts.evaluator import Scene, TaskSpec, evaluator_descriptor
from gantry.errors import ConfigError
from gantry.rollout import ClosedLoop, Step
from gantry.spine import ChannelSpec, Descriptor

VERSION = "0.1.0.dev0"

#: The two conditions SimplerEnv ships. They are not interchangeable and a
#: comparison across them is not a comparison.
VISUAL_MATCHING = "visual_matching"
VARIANT_AGGREGATION = "variant_aggregation"
VARIANTS = (VISUAL_MATCHING, VARIANT_AGGREGATION)

#: Task ids, by the names SimplerEnv's own registry uses. The Bridge and RT-1
#: setups are the ones with published real numbers.
TASKS = (
    "google_robot_pick_coke_can",
    "google_robot_move_near",
    "google_robot_open_drawer",
    "google_robot_close_drawer",
    "widowx_spoon_on_towel",
    "widowx_carrot_on_plate",
    "widowx_stack_cube",
    "widowx_put_eggplant_in_basket",
)

#: The two real platforms these scenes approximate.
PLATFORMS = {
    "google_robot": "Google Robot (RT-1 / RT-2 evaluation setup)",
    "widowx": "WidowX 250 (BridgeData V2 evaluation setup)",
}

CONTROL_HZ = 5.0


@dataclass(frozen=True)
class RealResult:
    """A published real-robot rate, and where it came from.

    ``source`` is required and it is the entire point of the dataclass. A real
    rate is being used to judge whether a simulator is trustworthy; a rate that
    cannot itself be traced turns that judgement into an assertion, and it looks
    exactly like a traceable one in every table it appears in.
    """

    task: str
    policy: str
    rate: float
    trials: int
    source: str
    platform: str = ""

    def __post_init__(self) -> None:
        if not str(self.source).strip():
            raise ConfigError(
                f"a real rate for {self.policy!r} on {self.task!r} needs its source "
                "named — a paper, a table, a run id. The paired column exists so a "
                "sim number can be checked against something checkable, and an "
                "unsourced rate is indistinguishable in every table from a sourced one"
            )
        if not 0.0 <= float(self.rate) <= 1.0:
            raise ConfigError(
                f"a real rate of {self.rate} for {self.policy!r} is not a proportion; "
                "SimplerEnv's published numbers are rates in [0, 1]"
            )
        if int(self.trials) < 1:
            raise ConfigError(
                f"a real rate for {self.policy!r} over {self.trials} trials carries no "
                "information; the trial count is what makes its interval computable"
            )

    def as_dict(self) -> dict[str, Any]:
        return {
            "task": self.task,
            "policy": self.policy,
            "real_rate": round(float(self.rate), 4),
            "real_trials": int(self.trials),
            "real_source": self.source,
            "real_platform": self.platform,
        }


def make_env(task: str, *, variant: str = VISUAL_MATCHING, **kwargs: Any) -> Any:
    """Construct a SimplerEnv environment. Imported here, not at module load."""
    try:
        import simpler_env
    except ImportError as error:  # pragma: no cover - needs the simulator
        raise ConfigError(
            "running SimplerEnv needs the simulator: pip install "
            "'gantry-evaluator-simpler[sim]' (it wants ManiSkill2-real2sim and "
            "a GL context)"
        ) from error
    return simpler_env.make(task, **kwargs)


class Approximation:
    """One SimplerEnv scene, as a world.

    SimplerEnv follows the gymnasium five-value step and puts success in
    ``info["success"]``, evaluated per step.
    """

    def __init__(self, env: Any, *, success_key: str = "success"):
        self.env = env
        self.success_key = success_key
        self._instruction: str | None = None

    @property
    def instruction(self) -> str | None:
        """The language goal the environment itself chose for this episode.

        Asked rather than assumed, because SimplerEnv resamples the instruction
        for some tasks per reset — and a policy given the task's *nominal*
        instruction while the environment scores a different one is being tested
        on a mismatch that shows up only as poor performance.
        """
        return self._instruction

    def begin(self, scene: Scene) -> Mapping[str, Any]:
        observation, _info = self.env.reset(seed=scene.seed)
        getter = getattr(self.env, "get_language_instruction", None)
        self._instruction = str(getter()) if callable(getter) else scene.instruction
        return _flatten(observation)

    def advance(self, action: np.ndarray) -> Step:
        observation, reward, terminated, truncated, info = self.env.step(action)
        info = info or {}
        return Step(
            observation=_flatten(observation),
            reward=float(np.asarray(reward).reshape(-1)[0]),
            done=bool(np.asarray(terminated).reshape(-1)[0]),
            success=(
                bool(np.asarray(info[self.success_key]).reshape(-1)[0])
                if self.success_key in info
                else None
            ),
        )

    def close(self) -> None:
        if callable(getattr(self.env, "close", None)):
            self.env.close()


def _flatten(mapping: Any, prefix: str = "") -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    if not isinstance(mapping, Mapping):
        return {"observation": np.asarray(mapping)}
    for key, value in mapping.items():
        name = f"{prefix}{key}"
        if isinstance(value, Mapping):
            out.update(_flatten(value, prefix=f"{name}."))
            continue
        array = np.asarray(value)
        if array.dtype.kind in "OUSV":
            continue
        out[name] = array
    return out


class SimplerEvaluator(ClosedLoop):
    """Runs a policy over a SimplerEnv task, carrying the paired real rate along."""

    rate_hz = CONTROL_HZ

    def __init__(
        self,
        task: str = "google_robot_pick_coke_can",
        *,
        variant: str = VISUAL_MATCHING,
        name: str = "simpler",
        real: RealResult | Mapping[str, Any] | None = None,
        observes: Sequence[str] | None = None,
        factory: Callable[..., Any] = make_env,
        success_key: str = "success",
        horizon: int = 120,
        env_kwargs: Mapping[str, Any] | None = None,
    ):
        if variant not in VARIANTS:
            raise ConfigError(
                f"unknown SimplerEnv condition {variant!r}; it ships {list(VARIANTS)}. "
                "These measure different things — one matches the real photographs, "
                "the other perturbs lighting and texture on purpose — so a rate from "
                "each is not two measurements of one quantity"
            )
        self._task = task
        self._variant = variant
        self._name = name
        self._real = _real(real)
        self.observes = tuple(observes) if observes is not None else None
        self._factory = factory
        self._success_key = success_key
        self._horizon = horizon
        self._env_kwargs = dict(env_kwargs or {})
        self._world: Approximation | None = None
        self._action: ChannelSpec | None = None

    # -- contract ----------------------------------------------------------

    def descriptor(self) -> Descriptor:
        return evaluator_descriptor(
            name=self._name,
            version=VERSION,
            stage_events=False,
            outcomes=True,
            seedable=True,
            closed_loop=True,
            # The scene approximates one specific real platform. It cannot be
            # rebuilt around a different arm — that is the whole design.
            hosts_embodiment=False,
            task=self._task,
            variant=self._variant,
            platform=self.platform,
            control_hz=CONTROL_HZ,
            **({"real": self._real.as_dict()} if self._real else {}),
        )

    @property
    def platform(self) -> str:
        for prefix, description in PLATFORMS.items():
            if self._task.startswith(prefix):
                return description
        return "unknown"

    @property
    def real(self) -> RealResult | None:
        return self._real

    @property
    def world(self) -> Approximation:
        if self._world is None:
            self._world = Approximation(
                self._factory(self._task, variant=self._variant, **self._env_kwargs),
                success_key=self._success_key,
            )
        return self._world

    def action(self) -> ChannelSpec:
        space = getattr(self.world.env, "action_space", None)
        shape = getattr(space, "shape", None)
        if not shape:
            raise ConfigError(
                f"{self._name}: the environment for {self._task!r} exposes no action "
                "space shape, so what a policy must emit cannot be stated"
            )
        if self._action is None:
            self._action = ChannelSpec(
                "action",
                "vector",
                (int(shape[-1]),),
                "float32",
                semantics="actuation",
                rate_hz=CONTROL_HZ,
            )
        return self._action

    def embodiment_for(self, scene: Scene) -> str:
        return self._task.split("_")[0] + ("_robot" if self._task.startswith("google") else "")

    def task_for(self, name: str = "", scenes: int = 24, horizon: int | None = None) -> TaskSpec:
        """One scene per seed, with the condition in every id.

        The variant lands in the id rather than only in the descriptor so that a
        record holding both conditions can still be split apart. A table that
        averaged visual-matching and variant-aggregation rates together would be
        reporting a quantity that does not exist.
        """
        return TaskSpec(
            name=name or f"{self._task}/{self._variant}",
            scenes=tuple(
                Scene(
                    id=f"{self._task}/{self._variant}#{index:03d}",
                    seed=index,
                    metadata={"variant": self._variant},
                )
                for index in range(scenes)
            ),
            horizon=self._horizon if horizon is None else int(horizon),
        )

    def world_for(self, scene: Scene) -> Approximation:
        return self.world

    def close(self) -> None:
        if self._world is not None:
            self._world.close()
            self._world = None

    def assemble(self, trial, scene, epoch, task, protocol):
        """The paired real rate goes on every episode, as an annotation.

        On the episode rather than only in the descriptor because that is where
        the transfer analysis reads from, and as an annotation rather than a
        metric because it is *not a result of this run*. Keeping the two columns
        distinguishable forever is what stops somebody's published number
        becoming, three tables later, a thing this benchmark claims to have
        measured.
        """
        episode = super().assemble(trial, scene, epoch, task, protocol)
        extra: dict[str, Any] = {"variant": self._variant, "platform": self.platform}
        if self._real:
            extra.update(self._real.as_dict())
        if self._world is not None and self._world.instruction:
            extra["instruction_given"] = self._world.instruction
        labels = episode.labels
        return episode.with_labels(
            type(labels)(
                success=labels.success,
                stage_events=labels.stage_events,
                annotations={**dict(labels.annotations), **extra},
            )
        )


def _real(raw: RealResult | Mapping[str, Any] | None) -> RealResult | None:
    if raw is None or isinstance(raw, RealResult):
        return raw
    return RealResult(**dict(raw))
