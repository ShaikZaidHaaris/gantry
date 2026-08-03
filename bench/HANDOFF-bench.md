# Handoff — gantry-closed-loop/bench

Last updated 2026-08-03. The previous version of this file was written from a
session that picked up this work by mistake; everything in it is now either done
or answered, and the answers are below rather than the questions.

**What this is:** the FastAPI + React + worker bench app, *not* the static
RoboBench site.

## Live right now

    https://gantry.gurasees.com

Served by Caddy on the L40S box, TLS from Let's Encrypt, renewed by Caddy
itself. `gurasees.com/gantry.html` is a GitHub Pages redirect pointing at it.

| | |
|---|---|
| host | `i-0edd3c6d798336706`, `35.172.172.202` (Elastic IP, survives stop/start) |
| units | `caddy`, `gantry-api`, `gantry-worker`, all enabled, verified across a reboot |
| identity | `edge` mode, shared secret set by Caddy, forged headers refused |
| checkpoints | `/opt/dlami/nvme/openpi-checkpoints`, 391 GB free |
| gate 3 | **armed** (`BENCH_RUNNER` set) |

Health: `systemctl is-active caddy gantry-api gantry-worker` and
`curl -s https://gantry.gurasees.com/api/me`. Want `"mode": "edge"`.
See `deploy/PUBLIC-URL.md` and `deploy/IDENTITY.md`.

## The six threads from the old handoff, closed

1. **The dangling `import re`** — finished, not reverted. `_why()` in
   `runner/runner.py` now finds the line that names the exception instead of
   tailing the log, and `train()` and `serve()` both use it. This mattered: the
   old message showed `ptrain_step(...)` and a row of `^^^^` carets, and the
   actual cause was four lines above, pushed out by the footer JAX prints
   *after* its own traceback.

2. **The leaderboard** — not broken. `/api/compare?benchmark=<key>` returns a
   full structure; `entries` is empty because publishing is opt-in and nobody
   has published. Three submissions have a passed g3 and are eligible;
   `listed = 0`. There is nothing to fix unless you want to publish something.
   Note the endpoint **requires** `?benchmark=`, and without it returns 422,
   which reads like a server error if you are only looking at the status code.

3. **"What we could not judge"** — removed from `web/src/components/DataReport.tsx`.
   Built and deployed.

4. **"What this does not say"** — now a compact `.what-list` in
   `web/src/components/Verdict.tsx` plus the rule in `web/src/lib/tokens.css`.
   Built and deployed.

5. **The training crash** — root-caused, and it was neither of the two
   hypotheses. Not OOM at `batch_size=16`, and not a dataset shape mismatch. The
   real exception was

       jaxlib.xla_extension.XlaRuntimeError: UNKNOWN: CUDNN_STATUS_NOT_SUPPORTED
       in .../cuda_dnn.cc(4988): 'engine_config' CUDNN_BACKEND_ENGINECFG_DESCRIPTOR

   **It does not reproduce.** The same command, config and arm now trains
   cleanly (`Step 0: grad_norm=1.6593, loss=0.2741`, exit 0). Treat it as
   transient until it recurs. See the open thread below for the one real oddity
   found while chasing it.

6. **`/admin/submissions`** — still not built, still optional. `email` remains
   PII in the clear: `SELECT id,name,email,status,listed FROM submissions`.

## Open threads

- **The cuDNN load is split across two versions.** The process maps
  `libcudnn.so.9` from openpi's venv (9.5.1.17, what uv resolved) and
  `libcudnn_graph.so.9.10.2` from `/usr/local/cuda-12.8/lib`, because
  `LD_LIBRARY_PATH` puts the system CUDA first. That is a real mismatch and a
  plausible cause of the crash above, but **it is not proven**: synthetic
  attention and conv probes pass under both library orders, and the real
  training step now passes under the mixed one too. If the crash recurs, put the
  venv's `nvidia/cudnn/lib` first in `LD_LIBRARY_PATH` for the trainer and see
  whether it stops.

- **Gate 3 is free and public.** `cost_cents: 0`, roughly 1.6 hours of L40S per
  arm across several arms, triggerable by any visitor. Two visitors uploaded on
  3 August. Gating it to one org is a small change to the `start`/`retry` routes
  if the bill starts to matter.

- **Root disk is 98% full**, 12 GB free of 485 GB. Checkpoints no longer land
  there, but `openpi/checkpoints` still holds five 8.5 GB checkpoints (42.5 GB)
  left behind by runs that were killed mid-flight. Kept deliberately pending a
  decision on which are still wanted.

- **The ephemeral NVMe is not persisted anywhere.** Not in fstab; its mount unit
  is one systemd inferred from `/proc/self/mountinfo`, and it is rebuilt at boot
  by `/opt/aws/dlami/bin/nvme_ephemeral_drives.sh`. `runner.py` refuses to train
  if `BENCH_CHECKPOINTS` resolves onto the root device, so a boot where that
  script does not run fails loudly instead of filling `/`.

## Watch-outs

- Publishing is **opt-in** on purpose. Do not list-all without consent.
- `deploy.sh` copies from the working tree, not from git. Check what you have
  before deploying.
- The runner writes 8.5 GB per checkpoint and deletes each after its evaluation.
  Two never coexist by design.
