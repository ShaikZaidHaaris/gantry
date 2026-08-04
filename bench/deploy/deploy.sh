#!/usr/bin/env bash
# Put the bench on a host and keep it there.
#
# Usage:  bench/deploy/deploy.sh ubuntu@HOST /path/to/key.pem
#
# What this does, and what it deliberately does not
# -------------------------------------------------
# Builds the frontend, copies the pipeline, the API, the built bundle and the
# worker to the host, makes the virtualenv they run in, and installs both as
# systemd units. It does *not* open a port: the API binds to loopback and
# something else is responsible for exposing it (see "Reaching it" below). A
# deploy script that punches a hole in a firewall is a deploy script that does it
# on a day nobody was watching.
#
# The pipeline is copied, not assumed. An earlier version of this script only
# sent bench/ and left the units pointing at a virtualenv it never created, which
# worked on the one host that already had a checkout and would have failed on
# every other. What is deployed has to be what is in the repository.
#
# Why systemd and not nohup
# -------------------------
# Every earlier attempt backgrounded the worker with nohup, and every time it
# died quietly with the shell that launched it. A worker that is gone looks
# exactly like a queue that is slow -- the submissions sit at "queued" and
# nothing anywhere says why. Restart=always is the difference between an outage
# and a blip.
set -euo pipefail

HOST="${1:?usage: deploy.sh user@host /path/to/key.pem}"
KEY="${2:?usage: deploy.sh user@host /path/to/key.pem}"
# Keepalives, because the alternative is not a slow deploy but a stuck one. The
# virtualenv stage is minutes of remote pip with no output, which is exactly the
# window an idle NAT or a dropped link closes the connection in. Without these,
# ssh waits on a socket nobody is on the other end of and the script sits at
# "==> virtualenv" forever: the host is idle, nothing is running, and there is no
# error anywhere to read. Four missed probes at 15s gives up after about a
# minute, which is far longer than any pause this script legitimately has.
#
# ConnectTimeout covers the other end of it: an unreachable host should say so
# rather than hang on the first hop.
SSH_OPTS=(-o StrictHostKeyChecking=no -o ServerAliveInterval=15 -o ServerAliveCountMax=4 -o ConnectTimeout=15)
SSH=(ssh -i "$KEY" "${SSH_OPTS[@]}" "$HOST")
RSYNC_E="ssh -i $KEY ${SSH_OPTS[*]}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
REMOTE=/home/ubuntu/gantry_bench

# rsync is not on a stock Windows box, and Git Bash does not ship it. Rather
# than require it, fall back to tar over the ssh connection we already have.
# The difference that matters is --delete: rsync removes files the source no
# longer has, and tar does not, so the fallback clears the destination itself
# where the caller asked for it.
HAVE_RSYNC=1; command -v rsync >/dev/null 2>&1 || HAVE_RSYNC=0
[ "$HAVE_RSYNC" = 1 ] || echo "  note: rsync not found, copying with tar over ssh"

#: send <local-dir> <remote-dir> [--delete] [exclude...]
send() {
  local src="$1" dst="$2"; shift 2
  local wipe=0; [ "${1:-}" = "--delete" ] && { wipe=1; shift; }
  local ex=() tarex=()
  for e in "$@"; do ex+=(--exclude "$e"); tarex+=(--exclude="$e"); done
  # `${a[@]+"${a[@]}"}` rather than `"${a[@]}"`. Under `set -u`, bash 3.2 calls
  # an empty array unbound and aborts, and 3.2 is what macOS ships as /bin/bash.
  # Every call with no excludes -- the samples directory, the built bundle --
  # therefore killed the deploy at the first copy with "ex[@]: unbound
  # variable", on any Mac. The idiom expands to nothing when the array is empty
  # and is correct on bash 4 too.
  if [ "$HAVE_RSYNC" = 1 ]; then
    local flags=(-az -e "$RSYNC_E" ${ex[@]+"${ex[@]}"})
    [ "$wipe" = 1 ] && flags+=(--delete)
    rsync "${flags[@]}" "$src/" "$HOST:$dst/"
  else
    [ "$wipe" = 1 ] && "${SSH[@]}" "rm -rf '$dst'"
    tar -czf - -C "$src" ${tarex[@]+"${tarex[@]}"} . | "${SSH[@]}" "mkdir -p '$dst' && tar -xzf - -C '$dst'"
  fi
}

# A running Vite dev server holds node_modules/@esbuild/*/esbuild.exe open, and
# `npm ci` starts by deleting node_modules. On Windows that unlink fails with
# EPERM, npm aborts partway, and what it leaves behind is a tree with the dev
# dependencies removed -- so the very next line dies with "tsc is not
# recognized". The deploy stops after one line of output and the diagnosis is
# nowhere near the symptom. Check first and say so.
#
# Scoped to *this* checkout's binary, which the first version was not: it asked
# whether any process named esbuild or node existed anywhere on the machine, so
# an unrelated site open in another window, or Adobe's bundled node, refused the
# deploy and sent you looking for a dev server you did not have running. A guard
# that cries wolf gets bypassed, and this one is worth keeping.
#
# `node` is not consulted at all. The dev server is a node process but the file
# npm cannot unlink is esbuild.exe, so that is the handle to look for.
#
# The comparison deliberately contains no regex and no backslash escaping. The
# version before this normalised separators with `-replace '\\','/'`, and bash
# collapsed those backslashes on the way into powershell, leaving the invalid
# pattern `\`. Every process then threw inside Where-Object, the pipeline
# produced nothing, and a guard that can never fire looks exactly like a guard
# that found nothing wrong. The path is converted to backslashes on this side
# instead, and passed as a literal single-quoted string.
#
# stderr is not discarded, for the same reason: hiding it is what let the broken
# version look like it was working.
if [ -d "$ROOT/bench/web/node_modules" ]; then
  # `|| true` because `pwd -W` is a Git Bash extension and every other shell
  # rejects it. Under `set -e` a bare assignment carries the substitution's exit
  # status, so on macOS and Linux this line ended the script here: no output at
  # all, not even "==> building the frontend", and exit 1. A deploy that prints
  # nothing looks like a deploy that did nothing, which is exactly what it was.
  # An empty value is the correct answer anyway, and means "not Windows" to the
  # check below.
  MODULES_WIN="$( (cd "$ROOT/bench/web/node_modules" && pwd -W) 2>/dev/null || true )"
  MODULES_WIN="${MODULES_WIN//\//\\}"   # C:/x/y -> C:\x\y, without tr's backslash warning
  if [ -n "$MODULES_WIN" ] && command -v powershell.exe >/dev/null 2>&1; then
    LOCK_OUT="$(powershell.exe -NoProfile -Command "
      \$want = '$MODULES_WIN'
      Get-Process -Name esbuild -ErrorAction SilentlyContinue |
        Where-Object { \$_.Path -and \$_.Path.StartsWith(\$want, 'OrdinalIgnoreCase') } |
        Select-Object -First 1 -ExpandProperty Path" 2>&1 | tr -d '\r' | tr -s '\n' '\n')"
    case "$LOCK_OUT" in
      "")
        : ;;                                  # nothing holding it
      *esbuild.exe*)
        echo "  a dev server is holding this checkout's esbuild, and npm ci would corrupt node_modules:" >&2
        echo "    $LOCK_OUT" >&2
        echo "  stop it first:  bench/.demo/run-local.sh stop   (and close any 'npm run dev')" >&2
        exit 1 ;;
      *)
        # Said out loud rather than treated as "all clear", which is the failure
        # this check has already had once.
        echo "  note: could not check for a running dev server, continuing anyway:" >&2
        echo "    $LOCK_OUT" >&2 ;;
    esac
  fi
fi

echo "==> building the frontend"
(cd "$ROOT/bench/web" && npm ci --silent && npm run build)

echo "==> copying"
# The pipeline first: the gates import gantry and its feedback plugins, so the
# worker cannot run without them.
send "$ROOT/src"     /home/ubuntu/gantry/src     --delete '__pycache__' '*.egg-info'
send "$ROOT/plugins" /home/ubuntu/gantry/plugins --delete '__pycache__' '*.egg-info' '.venv'
if [ "$HAVE_RSYNC" = 1 ]; then
  rsync -az -e "$RSYNC_E" "$ROOT/pyproject.toml" "$HOST:/home/ubuntu/gantry/pyproject.toml"
else
  "${SSH[@]}" "cat > /home/ubuntu/gantry/pyproject.toml" < "$ROOT/pyproject.toml"
fi
# The samples, so a host can hand out something to upload. Skipped when
# absent rather than failing: a checkout without them is still deployable.
if [ -d "$ROOT/samples" ]; then send "$ROOT/samples" /home/ubuntu/gantry/samples; fi
"${SSH[@]}" "mkdir -p $REMOTE/api $REMOTE/web/dist $REMOTE/worker $REMOTE/data /home/ubuntu/bench_work"
send "$ROOT/bench/api" "$REMOTE/api" --delete '__pycache__' 'data'
send "$ROOT/bench/web/dist" "$REMOTE/web/dist" --delete
send "$ROOT/bench/worker" "$REMOTE/worker" --delete '__pycache__'
send "$ROOT/bench/deploy" "$REMOTE/deploy"
# The runner, to the path its own run.sh names. Left out until now, so the only
# copy on the deployed host was one somebody placed by hand: the exact thing the
# note at the top of this file warns about, and it had already drifted from the
# repository. It is copied whether or not BENCH_RUNNER is set, because shipping
# it is not the same as switching it on, and the gate reads the variable.
send "$ROOT/bench/runner" /home/ubuntu/gantry_runner --delete '__pycache__'
"${SSH[@]}" "chmod +x /home/ubuntu/gantry_runner/run.sh"

echo "==> virtualenv"
"${SSH[@]}" bash -s <<'REMOTE_VENV'
set -e
cd /home/ubuntu/gantry
if [ ! -x .venv/bin/python ]; then
  python3 -m venv .venv
  echo "  made a fresh .venv"
fi
.venv/bin/pip install -q --upgrade pip
.venv/bin/pip install -q -e .
# Every feedback plugin, because a worker with fewer installed returns a shorter
# report on the same footage and a short report reads as a clean one.
for p in plugins/*/; do
  [ -f "$p/pyproject.toml" ] && .venv/bin/pip install -q -e "$p" || true
done
.venv/bin/pip install -q fastapi "uvicorn[standard]" sqlalchemy python-multipart
.venv/bin/python -c "
from importlib.metadata import entry_points
names = sorted(e.name for e in entry_points(group='gantry.feedback'))
print(f'  {len(names)} feedback checks installed')"
REMOTE_VENV

echo "==> environment"
# Generated on the host and never in the repo. The worker token is the only
# thing standing between the job queue and anything that can route to the port.
"${SSH[@]}" bash -s <<'REMOTE_ENV'
set -e
ENV=/home/ubuntu/gantry_bench/env
if [ ! -f "$ENV" ]; then
  TOKEN=$(python3 -c "import secrets;print(secrets.token_hex(16))")
  {
    echo "BENCH_DATA=/home/ubuntu/gantry_bench/data"
    # Said, not guessed. The API's fallback derives this from its own location,
    # which is right in a source tree and wrong here, because the deployed
    # layout has one fewer directory above the app.
    echo "BENCH_SAMPLES=/home/ubuntu/gantry/samples"
    echo "BENCH_WORKER_TOKEN=$TOKEN"
    # Every feedback module this host has. A worker with fewer installed returns
    # a shorter report on the same footage, and a shorter report reads as a
    # cleaner dataset -- so the gate names anything missing rather than being
    # quietly thinner.
    echo "BENCH_REQUIRED_MODULES=$(/home/ubuntu/gantry/.venv/bin/python -c "
from importlib.metadata import entry_points
print(','.join(sorted(e.name for e in entry_points(group='gantry.feedback'))))")"
    # Identity. The salt is generated here rather than left unset, because an
    # unset salt is regenerated per process: every visitor becomes a new org on
    # restart and loses their submissions, with nothing on screen looking wrong.
    #
    # The edge secret is deliberately NOT generated. It has to equal the value a
    # Cloudflare Transform Rule attaches, and inventing one here would produce a
    # host that looks configured, quietly reports "direct" mode, and puts every
    # visitor in one org. Set it by hand on both sides -- see deploy/IDENTITY.md.
    echo "BENCH_IP_SALT=$(python3 -c "import secrets;print(secrets.token_hex(16))")"
    # On by default, because this script's own deployment is exactly the case
    # it describes: uvicorn binds to 127.0.0.1 and a local cloudflared dials it,
    # so nothing outside the host can reach the origin to forge a header. Left
    # off, every visitor through a quick tunnel collapses into one org and can
    # read each other's submissions -- silently, since the product looks fine.
    #
    # Turn this OFF if you ever bind the API to 0.0.0.0. The flag is an
    # assertion about the binding that the app cannot verify for itself.
    echo "BENCH_TRUST_TUNNEL=1"
    # Preferred over the tunnel flag when you have a named tunnel and a zone:
    # a secret only your edge knows beats an argument about the host.
    echo "# BENCH_TRUST_HEADER=x-bench-edge"
    echo "# BENCH_TRUST_SECRET=    <- must match the Cloudflare Transform Rule"
    echo "# BENCH_CLIENT_IP=cf-connecting-ip"
  } > "$ENV"
  chmod 600 "$ENV"
  echo "  wrote a fresh $ENV"
else
  # Keep every value already there, and add only keys that are missing.
  #
  # This branch used to do nothing at all, and "nothing" is wrong in a specific
  # and silent way: a host deployed before a required key existed never gets it,
  # so whatever that key controls is off with no sign on any screen. That is
  # exactly what happened with identity. The box ran without BENCH_TRUST_TUNNEL,
  # fell back to trusting no forwarding header, and put every visitor in one org
  # able to read each other's uploads, while the deploy reported success.
  #
  # Only ever appends. A value an operator set by hand is theirs, and a deploy
  # script that overwrites secrets is one that rotates them on a day nobody
  # meant to.
  echo "  keeping the existing $ENV"
  grep -q "^BENCH_WORKER_TOKEN=" "$ENV" || echo "    WARNING: no BENCH_WORKER_TOKEN, the job queue is open"
fi

# Backfill keys a newer version of this script introduced, on both paths.
#
# Without this, an already-deployed host never receives a newly required
# setting: the file exists, the fresh-file block above is skipped, and the
# deploy reports success while the feature that needed the setting is quietly
# off. That is how BENCH_SAMPLES came to be missing on a host whose sample files
# were present, and how identity ran with every visitor sharing one org.
#
# One helper covering both branches, rather than the two near-identical ones a
# merge left here. On a fresh file every line below is already present and each
# is a no-op, which is the property that lets this run unconditionally.
#
# Only ever adds. A value already in the file is the operator's, including
# secrets this script must not regenerate.
add_if_missing() {
  grep -q "^$1=" "$ENV" || { echo "$1=$2" >> "$ENV"; echo "  added $1"; }
}
add_if_missing BENCH_DATA /home/ubuntu/gantry_bench/data
add_if_missing BENCH_SAMPLES /home/ubuntu/gantry/samples
add_if_missing BENCH_IP_SALT "$(python3 -c 'import secrets;print(secrets.token_hex(16))')"
add_if_missing BENCH_TRUST_TUNNEL 1
# BENCH_RUNNER is deliberately NOT backfilled. Setting it arms gate 3, which is
# hours of GPU per submission and free to whoever clicks it, so switching it on
# is a decision about a host and its bill, not a side effect of deploying.
REMOTE_ENV

echo "==> services"
"${SSH[@]}" bash -s <<'REMOTE_SVC'
set -e
sudo cp /home/ubuntu/gantry_bench/deploy/gantry-api.service /etc/systemd/system/
sudo cp /home/ubuntu/gantry_bench/deploy/gantry-worker.service /etc/systemd/system/
sudo systemctl daemon-reload
# `enable --now` starts a stopped unit and does nothing at all to a running
# one. On every deploy after the first that means the new code sits on disk
# while the old code keeps serving from memory: the script reports success,
# the health checks pass, and nothing you changed is live. Restart, always.
sudo systemctl enable gantry-api gantry-worker
sudo systemctl restart gantry-api gantry-worker
# The tunnel unit is installed by deploy/tunnel-setup.sh, not by this script,
# because it needs a hostname and a Cloudflare login that only a human has. But
# once it exists, a deploy has to refresh it, or an edit to the unit in this
# repository never reaches the host it is deployed on.
if [ -f /etc/systemd/system/gantry-tunnel.service ]; then
    sudo cp /home/ubuntu/gantry_bench/deploy/gantry-tunnel.service /etc/systemd/system/
    sudo systemctl daemon-reload
    sudo systemctl restart gantry-tunnel
fi
sleep 6
systemctl is-active gantry-api gantry-worker
# Started *after* this deploy, or the restart did not take.
echo "  api up since: $(systemctl show -p ActiveEnterTimestamp --value gantry-api)"
curl -s -o /dev/null -m 5 -w "  api: %{http_code}\n" http://127.0.0.1:8090/api/me
curl -s -o /dev/null -m 5 -w "  spa: %{http_code}\n" http://127.0.0.1:8090/
REMOTE_SVC

cat <<'NOTE'

==> done. Reaching it:

  If deploy/tunnel-setup.sh has been run on this host, it is already reachable
  at its own hostname and there is nothing to do here. Check with

      systemctl is-active gantry-tunnel

  Otherwise, for a URL that survives a restart, see deploy/PUBLIC-URL.md. Two
  options there: a hostname of your own with Caddy for TLS, or a Cloudflare
  named tunnel that opens no inbound port at all.

  For a throwaway link right now:

      cloudflared tunnel --url http://127.0.0.1:8090

  which prints a https://<name>.trycloudflare.com URL. Ephemeral in both senses:
  the name changes every time cloudflared starts, and as a bare process it does
  not come back after a reboot. Fine for a demo you are watching, wrong for a
  link you send anybody.

  The alternative is opening a port in the host's firewall, which this script
  will not do for you.

==> before anyone outside your team gets the link:

  Every visiting address gets its own org, so two visitors cannot see each
  other's submissions. Confirm which mode this host is in:

      curl -s http://127.0.0.1:8090/api/me | python3 -m json.tool

  Want "mode": "tunnel" behind cloudflared, or "edge" behind a named tunnel
  with a shared secret. "direct" means forwarding headers are being ignored and
  every visitor is sharing one org right now. See deploy/IDENTITY.md.

  This partitions visitors; it does not authenticate them. Anyone behind the
  same NAT is one visitor, and a changed address is a new one.

==> training

  The runner is now copied to /home/ubuntu/gantry_runner, but the robot test
  only *produces* rollouts when BENCH_RUNNER points at it:

      echo BENCH_RUNNER=/home/ubuntu/gantry_runner/run.sh >> gantry_bench/env
      sudo systemctl restart gantry-worker

  Left unset, the gate reads rollouts already on disk and otherwise reports
  plainly that it cannot make them. That is the right default for anything
  shared: gate 3 is free to the visitor, takes about 1.6 hours of GPU per arm,
  and runs several arms, so a stray click on a public URL is most of a day of
  GPU that nobody is billed for.

  Before switching it on, check the disk. A checkpoint is 8.5 GB, the runner
  refuses to start below 11 GB free, and it deletes each checkpoint after its
  evaluation precisely because two cannot coexist.
NOTE
