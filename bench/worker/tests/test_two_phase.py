"""The robot test trains in two phases, and the second one is the same for both arms.

What is pinned here is the shape of the job the gate asks for and the sequence
the runner performs, because both encode the thing that makes a verdict
readable: the contributor's clips are pretraining, the benchmark's own
demonstrations are a finetune every arm receives identically, and only what each
arm brought to that finetune differs.

The training itself is not exercised. It needs a GPU, openpi and ninety minutes.
What can be checked cheaply is that nobody has quietly changed which dataset an
arm trains on, in which order, or from which weights, and those are the parts
that would silently turn the run back into two arms trained on the upload alone.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gates import robot  # noqa: E402


def test_the_gate_asks_for_both_phases(monkeypatch, tmp_path):
    """The job names the demonstrations and both budgets, or the runner cannot
    do anything but the old single-phase run."""
    monkeypatch.setattr(robot, "DEMONSTRATIONS", "/somewhere/rt_base")
    monkeypatch.setattr(robot, "PRETRAIN_STEPS", 3000)
    monkeypatch.setattr(robot, "FINETUNE_STEPS", 1500)
    monkeypatch.setattr(robot, "RUNNER", "")  # stop before spawning anything

    captured = {}

    def spy(job, folder, report):
        captured.update(job)
        return False, "not run"

    monkeypatch.setattr(robot, "produce", spy)
    robot.run(tmp_path / "x.zip", tmp_path, params={"trials": 50, "task": "pick_dual_bottles"})

    assert captured["finetune"]["dataset"] == "/somewhere/rt_base"
    assert captured["finetune"]["steps"] == 1500
    assert captured["pretrain"]["steps"] == 3000
    # Both arms, and the control is not optional: without it the run is a
    # ranking rather than an attribution.
    assert captured["arms"] == [robot.TREATMENT, robot.CONTROL]


def test_budgets_are_steps_not_epochs():
    """A step count is the only budget that does not grow with the upload.

    With epochs, 1,000 clips buys ten times the gradient steps of 100 and the
    run measures compute rather than data.
    """
    assert isinstance(robot.PRETRAIN_STEPS, int)
    assert isinstance(robot.FINETUNE_STEPS, int)


def test_no_demonstrations_configured_still_produces_a_job():
    """A benchmark with no demonstrations of its own is not an error, it is a
    single-phase run, and the runner treats an absent finetune that way."""
    assert robot.DEMONSTRATIONS == "" or Path(robot.DEMONSTRATIONS).name


class TestRunnerSequence:
    """What train_arm actually does, without a GPU.

    The training is ninety minutes on hardware this suite does not have, but the
    sequencing is the part that carries the meaning and the part a later edit
    could quietly invert: which dataset each phase trains on, which weights it
    starts from, and that phase one's checkpoint is released once phase two has
    read it.
    """

    @staticmethod
    def _trace(finetune):
        import types

        sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "runner"))
        import runner

        calls = []
        real = (runner.register, runner.norm_stats, runner.train, runner.emit, runner.shutil)
        runner.register = (
            lambda arm, steps, prompt, progress, *, repo="", weights=runner.BASE_WEIGHTS:
            calls.append(("register", arm, steps, repo or arm, weights))
        )
        runner.norm_stats = lambda arm, progress: None
        runner.train = lambda arm, steps, progress: (
            calls.append(("train", arm, steps)) or Path(f"/ckpt/{arm}/bench/{steps - 1}")
        )
        runner.emit = lambda *a, **k: None
        runner.shutil = types.SimpleNamespace(
            rmtree=lambda p, ignore_errors=False: calls.append(("delete", str(p)))
        )
        try:
            served = runner.train_arm("arm", "pick up both bottles", 3000, finetune, Path("/dev/null"))
        finally:
            runner.register, runner.norm_stats, runner.train, runner.emit, runner.shutil = real
        return calls, served

    def test_phase_one_trains_on_the_upload_from_the_released_weights(self):
        calls, _ = self._trace({"repo": "rt_base", "steps": 1500})
        first = next(c for c in calls if c[0] == "register")
        assert first[3] == "arm", "phase one must train on the contributor's clips"
        assert first[4].startswith("gs://"), "phase one starts from openpi's released weights"

    def test_phase_two_trains_on_the_benchmark_from_phase_one(self):
        """The two facts that make the comparison mean anything."""
        calls, _ = self._trace({"repo": "rt_base", "steps": 1500})
        second = [c for c in calls if c[0] == "register"][1]
        assert second[3] == "rt_base", "phase two must train on the benchmark's demonstrations"
        assert second[4].endswith("/params"), "phase two starts from phase one's checkpoint"
        assert "_pre" in second[4], "and specifically from the pretraining run"
        assert second[2] == 1500, "phase two uses the finetune budget, not the pretraining one"

    def test_the_pretraining_checkpoint_is_released(self):
        """Two at 8.5 GB is 17 GB, and only one is ever needed again."""
        calls, served = self._trace({"repo": "rt_base", "steps": 1500})
        assert any(c[0] == "delete" for c in calls)
        assert "_pre" not in str(served), "the served checkpoint is phase two's"

    def test_no_demonstrations_is_a_single_phase_run(self):
        """Not an error. A benchmark without demonstrations of its own still runs."""
        calls, served = self._trace({})
        assert [c[0] for c in calls] == ["register", "train"]
        assert calls[0][3] == "arm"
        assert "_pre" not in str(served)
