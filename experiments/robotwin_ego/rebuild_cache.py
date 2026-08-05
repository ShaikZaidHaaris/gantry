"""Rebuild the seed cache from what the first arm was actually asked to do.

The cache carried seeds but not sentences, so the second arm reused it, skipped
the screen that produces the sentences, and refused for want of a prompt. The
sentences are recoverable: the first arm recorded, per episode, the instruction
the environment held while it was being scored.

Taking them from there rather than re-screening makes the pairing exact instead
of merely probable. Re-screening would regenerate them from the same seeded
draw and *should* agree, but "should" is doing work there — whether RoboTwin's
description generator is itself deterministic is not something this needs to
depend on.
"""

import json
import sys
from pathlib import Path

from gantry.store import read_run

TASK = sys.argv[1] if len(sys.argv) > 1 else "pick_dual_bottles"
SCENES = int(sys.argv[2]) if len(sys.argv) > 2 else 10
HERE = Path("/home/ubuntu/egorun")

record = read_run(HERE / f"robotwin_run_ego_{TASK}.json")
cache = HERE / f"robotwin_seeds_{TASK}_{SCENES}.json"
payload = json.loads(cache.read_text())

instructions = {}
for episode, seed in zip(record.episodes, payload["seeds"]):
    sentence = episode.labels.annotations.get("instruction_given")
    if not sentence:
        raise SystemExit(f"episode for seed {seed} recorded no instruction")
    instructions[str(seed)] = sentence

payload["instructions"] = instructions
cache.write_text(json.dumps(payload, indent=2))
print(f"rebuilt {cache} with {len(instructions)} sentences")
for seed, sentence in instructions.items():
    print(f"  {seed:>3}  {sentence}")
