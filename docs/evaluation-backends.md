# Evaluation backends

Which worlds a policy can be run against, what each one is *for*, and what it costs
to add. Written because the obvious way to grow a benchmark — add more tasks — is
mostly the wrong way. Thirteen tasks on one table measure one thing thirteen times.

Every entry below is judged by a single question: **what can this measure that
nothing already here can?** A backend that answers "more of the same" is listed as
low priority no matter how well-cited it is.

## The axes, and what covers each

| axis | what it answers | covered by |
| --- | --- | --- |
| skill | can it do the thing at all | robosuite, LIBERO, Meta-World |
| breadth | how many different things | Meta-World (50), RLBench (~100), ManiSkill (~40) |
| scene variety | did it learn the skill or the table | **RoboCasa** (120 kitchens) |
| controlled perturbation | *which* factor breaks it | **RLBench + COLOSSEUM** (14 factors) |
| embodiment transfer | does it survive a different arm | **ManiSkill** (20+ bodies), RoboCasa (mobile, humanoid) |
| long horizon | can it do the second thing after the first | **CALVIN** (5-instruction chains) |
| sim-to-real fidelity | does any of this predict hardware | **SimplerEnv** (published real rates) |
| hardware | the actual question | **bench** (a real machine, a person) |
| offline | prediction error with no simulator at all | offline replay |

The bolded five did not exist here before this round. The gaps they closed are the
reason to carry their dependencies; nothing on this list was added for coverage's
sake.

## Shipped

Twelve evaluators. Each is a separately installable plugin whose simulator is an
optional extra, so all of them install and plan on a laptop and none of them can
force a dependency on the others.

| plugin | world | bodies | seedable | funnel | notes |
| --- | --- | --- | --- | --- | --- |
| `evaluator_robosuite` | robosuite tasks | 7 arms | yes | no | region→sampler staging |
| `evaluator_libero` | LIBERO ×4 suites + 90 | Panda | by init state | no | scenes are a stored list |
| `evaluator_gym` | anything with `reset`/`step` | whatever it was | declared | optional | duck-typed; readers supplied |
| `evaluator_offline` | recorded episodes | from the data | n/a | no | no world; prediction error |
| `evaluator_waypoint` | synthetic | declared | yes | yes | for testing the harness itself |
| `evaluator_maniskill` | ~40 ManiSkill tasks | **20+** | yes | no | the only real transfer axis |
| `evaluator_robocasa` | ~100 kitchen tasks | 4 incl. mobile + GR1 | yes | no | 10 layouts × 12 styles |
| `evaluator_simpler` | 8 real-to-sim scenes | fixed per scene | yes | no | carries published real rates |
| `evaluator_calvin` | 34 instructions, chained | Franka | yes | **yes** | scored by chain length |
| `evaluator_metaworld` | the 50 | Sawyer | by task list | no | cheapest breadth per hour |
| `evaluator_rlbench` | ~100 tasks | Franka | yes | no | + COLOSSEUM's 14 factors |
| `evaluator_bench` | **a real machine** | whatever is bolted down | **no** | no | person resets and judges |

All twelve share one trial loop, `gantry.rollout.ClosedLoop`. A suite supplies
three methods — begin a scene, advance one action, close — and inherits chunk
arithmetic, horizons, milestone indexing, schema inference, provenance and the
success rate. That sharing is not tidiness: the four pre-existing evaluators had
each written their own loop and had begun to disagree about whether an observation
is recorded before or after the step that follows it, which is a silently
corrupted export rather than a bug with a symptom.

## Costs nothing to add: manifests over the gym bridge

These speak `reset`/`step` and need **no plugin at all** — an evaluator entry in a
manifest naming a factory and a success reader is the whole integration.

| suite | what it adds | reader |
| --- | --- | --- |
| `gym-aloha` | bimanual ALOHA, 14-DoF | `success_from_info("success")` |
| `gym-pusht` | 2-D contact-rich, very fast | `terminated_is_success()` |
| Gymnasium-Robotics Fetch | reach/push/slide, sparse reward | `success_from_info("is_success")` |
| Franka Kitchen | 7 subtasks, multi-goal | `milestones_from_info("completed_tasks")` |
| `robosuite` extras | wipe, door, nut-assembly | via `evaluator_robosuite` |

Worth saying plainly because it is the part people re-implement: if a suite is
gym-shaped and you know how it reports success, adding it is a JSON file. Only
write a plugin when the suite's *construction* is genuinely different — a scene
identity that is not a seed, a body that can be swapped, an outcome that arrives
at the end.

## Costs nothing to add: task packs on existing evaluators

Not new backends. New *scenes* for backends already here, which is often the higher
value per hour.

- **MimicGen D0 → D1 → D2** on `evaluator_robosuite`. The single best-value item on
  this page. Same task, three graded levels of initial-state randomisation, which
  is the narrow-vs-wide experiment this project ran by hand — except pre-built,
  graded, and citable. If only one thing gets done next, this.
- **robomimic square / transport / tool_hang** on `evaluator_robosuite`. Already
  converted locally; needs manifests and rubrics.
- **LIBERO-90** on `evaluator_libero`. One argument; ninety more tasks.
- **COLOSSEUM magnitudes** on `evaluator_rlbench`. The factors are wired; the graded
  intensity per factor is not yet exposed.

## Deliberately not added

| suite | why not |
| --- | --- |
| Isaac Lab / Isaac Sim | Strategically the right home for a GR00T-based benchmark, and the heaviest dependency on this page — Omniverse, a specific driver stack, tens of gigabytes. Worth doing when there is a machine dedicated to it, not before. |
| BEHAVIOR-1K / OmniGibson | 1 000 household activities and genuine mobile manipulation. Same Omniverse cost, and its tasks are far longer than anything a current VLA completes, so the measurement would be zero everywhere for a while. |
| Habitat | Navigation. A different problem; the manipulation contracts here would not fit and pretending otherwise would distort them. |
| CLIPort / Ravens | Top-down pick-and-place with a different action space (pixel affordances). Would need its own action semantics for little new signal. |
| VLABench, GemBench, ARNOLD | Newer, narrower, less independently reproduced. Reconsider once results from them are widely replicated. |
| Isaac Gym (legacy) | Superseded by Isaac Lab; adding it would be inheriting a migration. |

## Two honest limitations

**The simulator bindings have not been executed.** Every new plugin's *adapter
logic* is tested — against fakes that copy the real APIs where they are easy to get
wrong: device tensors that refuse `np.asarray`, a success flag that flickers, an
instruction that arrives as a list of paraphrases, a nested observation dict. What
has not happened is a run against the actual simulator, because none of them
install on this laptop. Each plugin carries one test marked `skip` that does
exactly that, so the first machine with the dependency present will find out
immediately. Until then, treat the sim-side binding of these five as
plausible-and-unverified, and the adapter logic as tested.

**ManiSkill's parallelism is not available.** Its headline feature is thousands of
environments stepping together on a GPU, and this harness cannot use it: the policy
contract is one observation in, one chunk out. Making it real means a batched
policy contract — a genuine piece of work on the policy plane, not a flag here. The
descriptor says `num_envs=1` and `batched="no"` so nobody reads "GPU-parallel
simulator" as a throughput claim this project can make good on. It is the single
highest-leverage change available for trial counts, and it is not done.

## Adding one

`gantry.rollout.ClosedLoop` plus `World`. Four methods and a descriptor:

```python
class MySuite(ClosedLoop):
    def descriptor(self): ...       # what it can honestly report
    def action(self): ...           # what a policy must emit
    def task_for(self, name=""): ...  # the scenes it offers
    def world_for(self, scene): ...   # a world in that scene's condition
```

The `World` is `begin(scene) -> observation`, `advance(action) -> Step`, `close()`.
Add `verdict(trial)` if the outcome only exists once the attempt is over.

Three rules the existing twelve follow, each learned from something that went wrong:

1. **Do not import the simulator at module load.** Optional extra, imported inside
   the factory, `ConfigError` with the install line when it is missing. This is
   what lets a task list be planned against on a laptop.
2. **Read the action width from the live environment.** Every suite here changes it
   with the control mode or the body, and every assumed width becomes a run of
   truncated commands rather than an error.
3. **Declare only what is true.** `seedable=False` on hardware, `outcomes=False`
   until the footage is labelled, `stage_events=False` where a funnel would have one
   rung. Conformance checks declarations against an actual record — it caught a
   stage-event off-by-one in the shared loop during this round, and a suite claiming
   a funnel it does not emit.
