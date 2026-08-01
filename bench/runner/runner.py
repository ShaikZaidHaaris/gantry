"""Turn a dataset into rollouts, with openpi and RoboTwin. One job spec in.

This is the *how* behind Gate 3. The gate says what it needs — these arms, this
many scenes — and knows nothing about trainers, checkpoints, policy servers or
simulators. This file makes all four of those choices, which is exactly why it
lives here on the machine that has them and not in the gate.

The stages, and why each is its own step
----------------------------------------
1. build the shuffled control as a dataset on disk, because openpi trains from
   LeRobot directories and not from Python objects
2. register both arms in openpi's own config list, because what makes a run
   reproducible is that it goes through their trainer and their norm stats
   unchanged rather than a fork
3. train, serve, evaluate — one arm at a time
4. write run records the gate can read

One arm at a time, and the checkpoint is deleted after its evaluation. Not
tidiness: a checkpoint here is 8.5 GB against 13 GB free, so two arms do not
coexist. The previous hand-run of this pipeline died mid-write with ENOSPC and
lost a training run, which is the expensive way to learn the disk is full.

Progress is appended to a file the gate tails. The trainer and the simulator
both write freely to stdout and neither format is ours; a file this script owns
is a contract, and scraping their logs would be a guess that breaks whenever
they reformat.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

HOME = Path.home()
OPENPI = HOME / "openpi"
ROBOTWIN = HOME / "RoboTwin"
EGORUN = HOME / "egorun"
LEROBOT = HOME / ".cache/huggingface/lerobot/gantry"
CHECKPOINTS = OPENPI / "checkpoints"
PORT = 8000

#: Training steps. Read from the job so a proof run and a paid run are the same
#: code path with different numbers -- a separate "test mode" would be the one
#: thing never exercised before it mattered.
DEFAULT_STEPS = int(os.environ.get("BENCH_TRAIN_STEPS", 3000))

#: Free bytes below which we refuse to start a training run rather than
#: discover it mid-write. A checkpoint is about 8.5 GB.
NEED_BYTES = 11 * 1024**3


def emit(progress: Path, phase: str, current=None, total=None, note: str = "") -> None:
    with progress.open("a") as out:
        out.write(json.dumps({"phase": phase, "current": current, "total": total, "note": note}) + "\n")
    print(f"[runner] {phase} {current or ''}/{total or ''} {note}", flush=True)


#: Where uv lives. A worker started by a service manager has a much barer PATH
#: than an interactive shell, and the trainer is invoked through `uv run` --
#: which failed with exit 127, a code that says "command not found" and reads,
#: three layers up, as "training failed".
TOOLS = f"{HOME}/.local/bin"


def run(cmd: list[str] | str, cwd: Path | None = None, env: dict | None = None) -> int:
    shell = isinstance(cmd, str)
    print(f"[runner] $ {cmd if shell else ' '.join(cmd)}", flush=True)
    merged = {
        **os.environ,
        "PATH": f"{TOOLS}:{os.environ.get('PATH', '')}",
        "WANDB_MODE": "disabled",
        **(env or {}),
    }
    return subprocess.call(cmd, cwd=str(cwd) if cwd else None, shell=shell, env=merged)


def stop_server() -> None:
    """Stop any policy server, without stopping ourselves.

    ``pkill -f serve_policy.py`` matches the command line of the very shell
    running it, so it kills its own process group and takes the runner with it.
    That is not a hypothetical -- it ended a run here after the training had
    already succeeded, and it ended four SSH sessions earlier in this project
    for the same reason. The bracket keeps the pattern from matching the text of
    the command that contains it.
    """
    subprocess.call("pkill -f '[s]erve_policy.py' || true", shell=True)
    time.sleep(10)


def free_bytes(path: Path = Path("/")) -> int:
    stat = os.statvfs(path)
    return stat.f_bavail * stat.f_frsize


def derangement(count: int, seed: int = 20250801) -> list[int]:
    """A permutation with no fixed point. Every clip gets somebody else's actions.

    A plain shuffle leaves some clips matched with themselves, and those are not
    controls at all — they are training data sitting inside the control arm,
    quietly making it look better and the contributor's data look worse by
    comparison.
    """
    import numpy as np

    rng = np.random.default_rng(seed)
    order = list(range(count))
    for _ in range(1000):
        rng.shuffle(order)
        if all(i != p for p, i in enumerate(order)):
            return order
    return [(i + 1) % count for i in range(count)]


def _retime(target: Path, lengths: list[int], actions: list | None = None) -> None:
    """Cut every clip to a given length, optionally replacing its actions."""
    import numpy as np
    import pandas as pd

    parquets = sorted(target.rglob("data/**/*.parquet"))
    for index, path in enumerate(parquets):
        frame = pd.read_parquet(path).iloc[: lengths[index]].copy()
        if actions is not None:
            frame["action"] = list(actions[index][: lengths[index]])
        for column in ("frame_index", "index"):
            if column in frame:
                frame[column] = np.arange(lengths[index])
        frame.to_parquet(path, index=False)

    meta = target / "meta"
    episodes = [json.loads(l) for l in (meta / "episodes.jsonl").read_text().splitlines() if l.strip()]
    for record, keep in zip(episodes, lengths):
        record["length"] = keep
    (meta / "episodes.jsonl").write_text("".join(json.dumps(r) + "\n" for r in episodes))
    info = json.loads((meta / "info.json").read_text())
    info["total_frames"] = int(sum(lengths))
    (meta / "info.json").write_text(json.dumps(info, indent=2))


def build_arms(source: Path, real: Path, control: Path, progress: Path) -> None:
    """Both arms, cut to the same lengths. The only difference is the pairing.

    Built by copying and rewriting one column, not by re-encoding. Re-encoding
    was the obvious approach and is wrong twice over: the writer refuses to
    carry video, and a lossy re-encode would make the control differ from the
    treatment in a *second* way.

    Both arms are truncated identically, and that is the part worth being
    careful about. Donating clip B's actions to clip A means cutting the pair to
    the shorter of the two — and doing that to the control alone left it with
    17% fewer frames than the treatment on this dataset. The control would then
    have been worse for two reasons, one of them the thing being measured and
    one of them a shorter training set, and nothing downstream could separate
    them. So the treatment keeps its own actions and takes the same cut.

    The videos keep their extra tail frames. Those are never indexed, because
    the episode length says they are not there — cheaper than a re-encode and
    exactly as correct.
    """
    import pandas as pd

    emit(progress, "building both arms", note=f"{real.name} and {control.name}")
    for target in (real, control):
        shutil.rmtree(target, ignore_errors=True)
        shutil.copytree(source, target)

    parquets = sorted(source.rglob("data/**/*.parquet"))
    donors = [pd.read_parquet(p, columns=["action"])["action"].to_numpy() for p in parquets]
    order = derangement(len(parquets))
    lengths = [min(len(donors[i]), len(donors[order[i]])) for i in range(len(parquets))]

    _retime(real, lengths)
    _retime(control, lengths, [donors[order[i]] for i in range(len(parquets))])
    emit(progress, "building both arms",
         note=f"{len(lengths)} clips, {sum(lengths)} frames each")


#: openpi's config list is Python source, so a new dataset means a new entry in
#: it. Written from the same template the hand-run ablation used, because what
#: makes a run reproducible is that it goes through their trainer, their weight
#: loader and their norm stats unchanged rather than through a fork.
CONFIG_PY = OPENPI / "src/openpi/training/config.py"
ANCHOR = "# --- gantry ego: appended TrainConfig"

TEMPLATE = """
    TrainConfig(
        name="pi05_{arm}",
        model=pi0_config.Pi0Config(pi05=True, action_horizon=10, discrete_state_input=False),
        data=LeRobotAlohaDataConfig(
            repo_id="gantry/{arm}",
            use_delta_joint_actions=False,
            adapt_to_pi=False,
            default_prompt="{prompt}",
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
            pi05=True, action_horizon=10, discrete_state_input=False,
            paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora",
        ).get_freeze_filter(),
        ema_decay=None,
    ),
"""


def register(arm: str, steps: int, prompt: str, progress: Path) -> None:
    """Add this arm to openpi's own config list, if it is not already there.

    Written here rather than by ``add_configs.py``, which takes no arguments and
    is hardcoded to the three arms of the hand-run ablation. A submission needs
    its own entry, named after itself.
    """
    emit(progress, "registering the arm", note=arm)
    text = CONFIG_PY.read_text()
    if f'name="pi05_{arm}"' in text:
        return
    if ANCHOR not in text:
        raise RuntimeError(
            "openpi's config file no longer has the anchor this inserts before; "
            "it has been edited or upgraded and this needs looking at"
        )
    block = TEMPLATE.format(arm=arm, prompt=prompt, steps=steps)
    CONFIG_PY.write_text(text.replace(ANCHOR, block.rstrip() + "\n" + ANCHOR, 1))

    # Prove it parses and resolves before anything spends an hour on the GPU.
    check = subprocess.run(
        [str(OPENPI / ".venv/bin/python"), "-c",
         f"import openpi.training.config as c; c.get_config('pi05_{arm}')"],
        cwd=str(OPENPI), capture_output=True, text=True,
    )
    if check.returncode != 0:
        raise RuntimeError(f"the config for {arm} does not resolve: {check.stderr.strip()[-400:]}")


def norm_stats(arm: str, progress: Path) -> None:
    """Compute this dataset's normalisation statistics.

    Its own stage because it is its own failure. The trainer refuses without
    them and says so clearly, but three layers up that arrives as "training
    failed (exit 1)" — a message about the wrong thing, pointing at the
    expensive step rather than the cheap one that was skipped.

    Per dataset, not per model: the statistics describe the contributor's
    actions, so the treatment arm and its shuffled control get different ones,
    computed from what each actually contains.
    """
    emit(progress, "computing normalisation statistics", note=arm)
    code = run(f"uv run scripts/compute_norm_stats.py --config-name=pi05_{arm}", cwd=OPENPI)
    if code != 0:
        raise RuntimeError(f"computing norm stats for {arm} failed (exit {code})")


def train(arm: str, steps: int, progress: Path) -> Path:
    """Train one arm. Returns the checkpoint directory."""
    out = CHECKPOINTS / f"pi05_{arm}" / "bench" / str(steps - 1)
    if out.exists():
        emit(progress, "training", note=f"{arm} already trained")
        return out
    if free_bytes() < NEED_BYTES:
        raise RuntimeError(
            f"only {free_bytes() / 1024**3:.1f} GB free; a checkpoint needs about 8.5 GB "
            "and a run that fills the disk mid-write loses the whole training"
        )
    emit(progress, "training", total=steps, note=arm)
    # Kept, not streamed. The gate shows the last few lines of a failure, and a
    # trainer's last few lines are its stack trace -- which says what broke and
    # never why. The log is where the why is, and it has to outlive the run.
    log = HOME / f"bench_train_{arm}.log"
    code = run(
        f"XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi05_{arm} "
        f"--exp-name=bench --overwrite --no-wandb-enabled > {log} 2>&1",
        cwd=OPENPI,
    )
    if code != 0 or not out.exists():
        why = ""
        if log.exists():
            lines = [l for l in log.read_text().splitlines() if l.strip()]
            why = " / ".join(lines[-6:])
        raise RuntimeError(f"training {arm} failed (exit {code}). {why}"[:1200])
    return out


def serve(arm: str, checkpoint: Path, progress: Path) -> subprocess.Popen:
    emit(progress, "starting the policy server", note=arm)
    stop_server()
    for _ in range(40):
        used = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True).stdout.strip()
        if used.isdigit() and int(used) < 2000:
            break
        time.sleep(10)
    log = HOME / f"bench_serve_{arm}.log"
    # Same environment as every other command here. This built its own Popen and
    # so missed the PATH that `run` sets, which meant `uv` was not found, the
    # server never came up, and the failure arrived twenty minutes later as
    # "the policy server never came up" — a timeout standing in for a typo.
    process = subprocess.Popen(
        f"XLA_PYTHON_CLIENT_MEM_FRACTION=0.35 uv run scripts/serve_policy.py --port {PORT} "
        f"policy:checkpoint --policy.config=pi05_{arm} --policy.dir={checkpoint} "
        f"> {log} 2>&1",
        cwd=str(OPENPI),
        shell=True,
        env={**os.environ, "PATH": f"{TOOLS}:{os.environ.get('PATH', '')}", "WANDB_MODE": "disabled"},
    )
    import socket
    for _ in range(120):
        try:
            socket.create_connection(("127.0.0.1", PORT), 2).close()
            time.sleep(20)
            return process
        except OSError:
            time.sleep(10)
    tail = ""
    if log.exists():
        lines = [l for l in log.read_text().splitlines() if l.strip()]
        tail = " / ".join(lines[-5:])
    raise RuntimeError(f"the policy server for {arm} never came up. {tail}"[:800])


def evaluate(arm: str, task: str, scenes: int, progress: Path) -> Path:
    emit(progress, "evaluating", total=scenes, note=f"{arm} · {task}")
    code = run(
        [str(ROBOTWIN / ".venv/bin/python"), "-u", str(EGORUN / "run_ablation.py"),
         task, str(scenes), str(PORT), arm],
        cwd=ROBOTWIN,
    )
    record = EGORUN / f"abl_run_{arm}_{task}.json"
    if code != 0 or not record.exists():
        raise RuntimeError(f"evaluating {arm} failed (exit {code}); no record at {record}")
    return record


def main() -> int:
    job = json.loads(Path(sys.argv[1]).read_text())
    out = Path(job["out"])
    progress = Path(job["progress"])
    scenes = int(job.get("trials") or 50)
    steps = int(job.get("steps") or DEFAULT_STEPS)
    task = job.get("task") or "pick_dual_bottles"
    prompt = job.get("prompt") or "pick up both bottles"
    stamp = out.parent.name.replace("-", "_")[:24] or "sub"

    treatment, control = job["arms"][0], job["arms"][1]
    source = Path(job["dataset"])

    # The contributor's clips, where the trainer expects to find them.
    real_repo = f"bench_{stamp}"
    control_repo = f"bench_{stamp}_shuf"
    real_dir, control_dir = LEROBOT / real_repo, LEROBOT / control_repo
    if not (real_dir.exists() and control_dir.exists()):
        build_arms(source, real_dir, control_dir, progress)

    written: dict[str, str] = {}
    for arm_name, repo in ((treatment, real_repo), (control, control_repo)):
        register(repo, steps, prompt, progress)
        norm_stats(repo, progress)
        checkpoint = train(repo, steps, progress)
        server = serve(repo, checkpoint, progress)
        try:
            record = evaluate(repo, task, scenes, progress)
        finally:
            stop_server()
            server.poll()
        target = out / f"{repo}.json"
        shutil.copy(record, target)
        written[arm_name] = target.name
        # One arm at a time: 8.5 GB of checkpoint against 13 GB free, so the
        # second arm cannot be trained until the first one's is gone.
        emit(progress, "clearing the checkpoint", note=repo)
        shutil.rmtree(checkpoint.parent, ignore_errors=True)

    baseline = EGORUN / f"abl_run_rt_base100_{task}.json"
    if baseline.exists():
        shutil.copy(baseline, out / "baseline.json")
        written[job.get("baseline", "baseline")] = "baseline.json"

    (out / "arms.json").write_text(json.dumps(written, indent=2))
    emit(progress, "done", note=f"{len(written)} arm(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
