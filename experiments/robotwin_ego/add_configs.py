"""Add the three ablation arms to openpi's own config list.

Ordinary entries in their list rather than a fork, for the same reason as the
first ego config: what makes a run reproducible is that it goes through their
trainer, their weight loader and their norm stats, unchanged.

The three differ in exactly one thing — which dataset they read. Same model,
same steps, same batch size, same prompt, same everything else. If they differed
in two things, the comparison would be measuring their sum.

The prompt is fixed rather than varied
--------------------------------------
RoboTwin generates a different sentence per scene, and its demonstrations carry
none at all. Training on one wording and evaluating on ten others would test
language generalisation, which is not the question here. So every arm trains and
evaluates on the same sentence, and language is held constant rather than left
to vary alongside the thing being measured.
"""

from pathlib import Path

CONFIG = Path("/home/ubuntu/openpi/src/openpi/training/config.py")
PROMPT = "pick up both bottles"
STEPS = 3000

TEMPLATE = '''
    TrainConfig(
        name="pi05_{arm}",
        model=pi0_config.Pi0Config(pi05=True, action_horizon=10, discrete_state_input=False),
        data=LeRobotAlohaDataConfig(
            repo_id="gantry/{arm}",
            # Absolute end-effector poses in RoboTwin's world frame, sixteen wide.
            # Not joint angles, so neither of openpi's Aloha conversions applies:
            # adapt_to_pi remaps a real Aloha rig's joint and gripper conventions,
            # and use_delta_joint_actions would turn absolute poses into
            # velocities still labelled as positions.
            use_delta_joint_actions=False,
            adapt_to_pi=False,
            default_prompt="{prompt}",
            # Fixed, not from the task string: RoboTwin varies its sentence per
            # scene and its demonstrations carry none. Holding language constant
            # keeps this an ablation on data rather than on wording.
            base_config=DataConfig(prompt_from_task=False),
            repack_transforms=_transforms.Group(
                inputs=[
                    _transforms.RepackTransform(
                        {{
                            "images": {{"cam_high": "observation.images.head"}},
                            "state": "observation.state",
                            "actions": "action",
                        }}
                    )
                ]
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "gs://openpi-assets/checkpoints/pi05_base/params"
        ),
        num_train_steps={steps},
        batch_size=16,
        freeze_filter=pi0_config.Pi0Config(
            pi05=True, action_horizon=10, paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ).get_freeze_filter(),
        ema_decay=None,
    ),
'''


def main() -> None:
    text = CONFIG.read_text()
    arms = ["rt_base", "rt_ego", "rt_shuffled"]

    already = [arm for arm in arms if f'name="pi05_{arm}"' in text]
    if already:
        print(f"already present: {already}")
        return

    block = "".join(
        TEMPLATE.format(arm=arm, prompt=PROMPT, steps=STEPS) for arm in arms
    )
    # Insert immediately before the pi05_ego entry, which is the last thing in
    # the list and the one these are modelled on.
    anchor = "# --- gantry ego: appended TrainConfig"
    if anchor not in text:
        raise SystemExit("cannot find the gantry block; the file has changed")
    header = (
        "# --- gantry RoboTwin ablation: base / ego / shuffled --------------------\n"
        "# Three arms that differ in exactly one thing: which dataset they read.\n"
        "#   rt_base      RoboTwin's own demonstrations\n"
        "#   rt_ego       the same, plus the ego data\n"
        "#   rt_shuffled  the same, plus the ego data with its actions detached\n"
        "# base can do the task, so the comparison has something to move. ego and\n"
        "# shuffled are the same size, so the contrast between them is the part\n"
        "# attributable to the correspondence rather than to the extra frames.\n"
    )
    CONFIG.write_text(text.replace(anchor, header + block.rstrip() + "\n" + anchor, 1))
    print(f"added {arms} at {STEPS} steps, prompt {PROMPT!r}")

    # Prove they parse and resolve before anything spends an hour on the GPU.
    import subprocess

    for arm in arms:
        out = subprocess.run(
            ["python", "-c", f"import openpi.training.config as c; c.get_config('pi05_{arm}')"],
            cwd="/home/ubuntu/openpi", capture_output=True, text=True,
        )
        print(f"  pi05_{arm}: {'ok' if out.returncode == 0 else out.stderr.strip()[-200:]}")


if __name__ == "__main__":
    main()
