"""The matrix: every checkpoint, on every body, at every task.

Ordered checkpoint-outermost because one L4 holds one model and loading is
minutes; everything inside a checkpoint is free by comparison.

Resumable. Each cell writes its own JSON as it finishes, and a cell that
already has one is skipped, so this survives a disconnect without losing the
hours before it.
"""
import json, os, pathlib, subprocess, sys, time, traceback

# Baseline checkpoints. Each entry names a model path and the embodiment tag
# to serve it with. "base" is the pretrained model with its libero_sim
# projector — the closest matching pretrained embodiment, since the lift
# datasets have modality byte-for-byte identical to libero_sim. Any non-Panda
# number can then be read as "our fine-tune adds N points over what pretraining
# already knew", which is the thing a lab actually asks about a fine-tune.
BASELINES = {
    "base": {
        "model": "/home/ubuntu/models/GR00T-N1.7-3B",
        "tag": "libero_sim",
        "modality": None,
    },
}

sys.path.insert(0, "/home/ubuntu")

from observe_arm import native_state_spec, observer_for
from gantry.contracts.evaluator import Protocol
from gantry.errors import GantryError
from gantry_embodiment_declared import from_file
from gantry_evaluator_robosuite import RobosuiteEvaluator
from gantry_policy_gr00t import Endpoint, Gr00tPolicy
from gantry_retargeter_gripper import calibration_from, state_spec_for
from gantry_tasks_declared import DeclaredTasks

ROOT = pathlib.Path("/home/ubuntu/sweep")
ROOT.mkdir(exist_ok=True)
EMB = pathlib.Path("/home/ubuntu/gantry/manifests/embodiments")
PORT = 5555

CHECKPOINTS = tuple(os.environ.get("CKPTS", "ph,mh,mg").split(","))
ARMS = tuple(os.environ.get("ARMS", "panda,sawyer,iiwa,kinova3,jaco,ur5e").split(","))
TRIALS = int(os.environ.get("TRIALS", "10"))
EXECUTE = int(os.environ.get("EXECUTE", "16"))
#: Cut the horizon short to prove every cell is reachable without paying for a
#: measurement. Recorded in each cell, because 0/1 at eight steps is not a
#: capability result and must never be read as one.
HORIZON = int(os.environ.get("HORIZON", "0")) or None
ONLY = tuple(t for t in os.environ.get("TASKS", "").split(",") if t)

tasks = DeclaredTasks("/home/ubuntu/gantry/manifests/tasks")
TASKS = ONLY or tasks.names()
TRAINED_ON = from_file(str(EMB / "panda.json"))   # the body the checkpoints saw


def serve(split):
    # Two shapes of checkpoint: a fine-tuned one under ft_<split>, or a
    # named baseline (the pretrained model with one of its own projectors).
    # Both go through the same server on the same wire, so downstream code
    # cannot tell which is which — that is what makes the comparison paired.
    if split in BASELINES:
        b = BASELINES[split]
        model_path = b["model"]
        tag = b["tag"]
        modality = b.get("modality")
    else:
        if not os.path.exists(f"/home/ubuntu/ft_{split}/model.safetensors.index.json"):
            print(f"  {split}: no checkpoint", flush=True)
            return False
        model_path = f"/home/ubuntu/ft_{split}"
        tag = "new_embodiment"
        modality = "/home/ubuntu/lift_modality.py"
    subprocess.run(["pkill", "-f", "[r]un_gr00t_server"], check=False)
    time.sleep(4)
    cmd = ["setsid", "/home/ubuntu/Isaac-GR00T-benchmark/.venv/bin/python",
           "/home/ubuntu/Isaac-GR00T-benchmark/gr00t/eval/run_gr00t_server.py",
           "--model-path", model_path,
           "--embodiment-tag", tag,
           "--port", str(PORT)]
    if modality:
        cmd += ["--modality-config-path", modality]
    subprocess.Popen(
        cmd,
        stdout=open(f"/home/ubuntu/serve_{split}.log", "w"), stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL,
        env={"GR00T_VIDEO_BACKEND": "pyav", "PATH": "/usr/bin:/bin", "HOME": "/home/ubuntu"})
    for _ in range(150):
        time.sleep(5)
        if f":{PORT}" in subprocess.run(["ss", "-ltn"], capture_output=True, text=True).stdout:
            time.sleep(8)
            return True
    return False


def cell(split, arm_name, task_name, policy):
    out = ROOT / f"{split}__{arm_name}__{task_name}.json"
    if out.exists():
        return json.loads(out.read_text()), True
    arm = from_file(str(EMB / f"{arm_name}.json"))
    task = tasks.task(task_name)
    record = {"checkpoint": split, "arm": arm_name, "task": task_name,
              "trials": TRIALS, "execute": EXECUTE,
              "horizon": HORIZON or task.horizon}
    if HORIZON:
        record["smoke"] = (
            f"horizon cut from {task.horizon} to {HORIZON} and {TRIALS} trial(s): "
            "this cell proves the path runs, it does not measure the policy")
    started = time.time()
    try:
        try:
            observe, retargeter = observer_for(arm, TRAINED_ON)
        except GantryError as cannot_read:
            # The policy cannot read this body. Before reporting that, ask the
            # world whether it would host it at all: if it would not, that is
            # the more fundamental reason and the one worth recording first.
            # A body nobody can drive because the task wants one arm and it has
            # two is a different fact from a body whose hand cannot be mapped.
            try:
                RobosuiteEvaluator.for_task(task, trials=1, embodiment=arm).env
            except GantryError as cannot_host:
                record["also"] = str(cannot_read).splitlines()[0][:160]
                raise cannot_host from None
            raise
        record["retargeter"] = retargeter.name
        # What the conversion cost, recorded in the cell rather than inferred
        # from the retargeter's name by whoever reads this months from now.
        native = native_state_spec(arm)
        record["retargeter_losses"] = list(
            retargeter.losses(native, state_spec_for(native, calibration_from(TRAINED_ON))))
        ev = RobosuiteEvaluator.for_task(
            task, trials=TRIALS, embodiment=arm, observe=observe, use_image_obs=True,
            observations=("robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"),
            # The two cameras the checkpoints were trained to read, at the size
            # their processor expects.
            world={"camera_names": ["agentview", "robot0_eye_in_hand"],
                   "camera_heights": 256, "camera_widths": 256})
        spec = ev.task_for(horizon=HORIZON)
        run = ev.evaluate(policy, spec, Protocol(execute=EXECUTE))
        solved = [bool(e.labels.success) for e in run.episodes]
        errors = [e.labels.annotations.get("error") for e in run.episodes]
        m = run.metrics.get("success_rate")
        record.update(
            status="ran", solved=sum(solved), scored=len([s for s in solved if s is not None]),
            success_rate=None if m is None else round(m.value, 4),
            ci=None if m is None or not m.ci else [round(v, 4) for v in m.ci],
            n=None if m is None else m.n,
            method=None if m is None else m.method,
            errors=[e for e in errors if e][:3],
            n_errors=len([e for e in errors if e]))
        ev.close()
    except Exception as error:
        record.update(status="refused" if isinstance(error, GantryError) else "failed",
                      reason=f"{type(error).__name__}: {str(error).splitlines()[0][:200]}")
        if not isinstance(error, GantryError):
            record["trace"] = traceback.format_exc()[-600:]
    record["seconds"] = round(time.time() - started, 1)
    out.write_text(json.dumps(record, indent=2))
    return record, False


total = len(CHECKPOINTS) * len(ARMS) * len(TASKS)
done = 0
print(f"matrix: {len(CHECKPOINTS)} checkpoints x {len(ARMS)} arms x {len(TASKS)} tasks "
      f"= {total} cells, {TRIALS} trials each, execute={EXECUTE}\n", flush=True)
for split in CHECKPOINTS:
    pending = [(a, t) for a in ARMS for t in TASKS
               if not (ROOT / f"{split}__{a}__{t}.json").exists()]
    if not pending:
        print(f"[{split}] all cells already recorded", flush=True)
        done += len(ARMS) * len(TASKS)
        continue
    print(f"[{split}] serving ({len(pending)} cells to run)", flush=True)
    if not serve(split):
        print(f"[{split}] server never bound — skipping", flush=True)
        continue
    policy = Gr00tPolicy("/home/ubuntu/lift_lerobot/ph",
                         Endpoint(port=PORT, timeout_ms=180000), name=f"gr00t-{split}")
    for arm_name in ARMS:
        for task_name in TASKS:
            rec, cached = cell(split, arm_name, task_name, policy)
            done += 1
            tag = ("cached" if cached else
                   f"{rec.get('solved','-')}/{rec.get('scored','-')}" if rec["status"] == "ran"
                   else rec["status"])
            print(f"  [{done:3}/{total}] {split:3} {arm_name:8} {task_name:20} "
                  f"{tag:12} {rec.get('seconds','')}s", flush=True)
subprocess.run(["pkill", "-f", "[r]un_gr00t_server"], check=False)
print("\nsweep complete", flush=True)
