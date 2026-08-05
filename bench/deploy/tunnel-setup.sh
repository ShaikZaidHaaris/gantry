#!/usr/bin/env bash
# Put the bench behind a Cloudflare named tunnel, on a stable hostname.
#
# Runs ON THE HOST, after `cloudflared tunnel login` has been done once.
#
#     ./tunnel-setup.sh bench.yourdomain.com
#
# Why this is a script and not a paragraph in a runbook
# -----------------------------------------------------
# The dangerous step is retiring the quick tunnel, because that is the URL
# currently serving real traffic. Done by hand the tempting order is "kill the
# old one, start the new one", which leaves the site down for as long as the
# next step takes to debug. This script cuts over only after the new hostname
# has answered over the public internet, so a failure anywhere leaves the old
# URL still serving and nothing to roll back.
#
# Everything here is idempotent. Re-running it after a partial failure picks up
# the tunnel and the DNS record that already exist rather than erroring on them.
set -euo pipefail

HOSTNAME_ARG="${1:-}"
TUNNEL_NAME="${TUNNEL_NAME:-gantry}"
ORIGIN="${ORIGIN:-http://127.0.0.1:8090}"
CF_DIR="$HOME/.cloudflared"

if [[ -z "$HOSTNAME_ARG" ]]; then
    echo "usage: $0 <hostname>        e.g. $0 bench.yourdomain.com" >&2
    exit 2
fi

say() { printf '==> %s\n' "$*"; }

# ---------------------------------------------------------------- preflight
command -v cloudflared >/dev/null || {
    echo "cloudflared is not installed" >&2
    exit 1
}

if [[ ! -f "$CF_DIR/cert.pem" ]]; then
    cat >&2 <<EOF
No $CF_DIR/cert.pem, so this host is not authorised against your Cloudflare
account and cannot create a tunnel or a DNS record.

Run:

    cloudflared tunnel login

It prints a URL. This box has no browser, so open that URL on your laptop and
pick the zone. Then run this script again.
EOF
    exit 1
fi

# The origin has to be up before the cutover, or the verification below cannot
# distinguish "tunnel is broken" from "there is nothing behind it".
curl -fsS -o /dev/null -m 5 "$ORIGIN/api/me" || {
    echo "the API is not answering on $ORIGIN; start gantry-api first" >&2
    exit 1
}

# ----------------------------------------------------------------- tunnel
# `tunnel create` errors if the name is taken, which on a re-run is the normal
# case rather than a problem, so look before creating.
uuid_for() {
    cloudflared tunnel list --output json 2>/dev/null |
        python3 -c "
import json,sys
name = sys.argv[1]
try:
    rows = json.load(sys.stdin)
except Exception:
    sys.exit(0)
for row in rows:
    if row.get('name') == name and not row.get('deleted_at'):
        print(row['id'])
        break
" "$TUNNEL_NAME"
}

UUID="$(uuid_for || true)"
if [[ -n "$UUID" ]]; then
    say "reusing existing tunnel '$TUNNEL_NAME' ($UUID)"
else
    say "creating tunnel '$TUNNEL_NAME'"
    cloudflared tunnel create "$TUNNEL_NAME" >/dev/null
    UUID="$(uuid_for || true)"
    [[ -n "$UUID" ]] || {
        echo "created the tunnel but could not read its id back" >&2
        exit 1
    }
    say "created ($UUID)"
fi

CREDS="$CF_DIR/$UUID.json"
[[ -f "$CREDS" ]] || {
    echo "no credentials file at $CREDS" >&2
    echo "the tunnel exists in your account but not on this host; delete it in" >&2
    echo "the dashboard and re-run, or copy the credentials file across." >&2
    exit 1
}
chmod 600 "$CREDS"

# ----------------------------------------------------------------- config
# The final rule must have no hostname. cloudflared refuses to start without a
# catch-all, and the failure message does not make that obvious.
say "writing $CF_DIR/config.yml"
cat >"$CF_DIR/config.yml" <<EOF
tunnel: $UUID
credentials-file: $CREDS

ingress:
  - hostname: $HOSTNAME_ARG
    service: $ORIGIN
  - service: http_status:404
EOF

cloudflared --config "$CF_DIR/config.yml" tunnel ingress validate

# -------------------------------------------------------------------- dns
# Creates a proxied CNAME to <uuid>.cfargotunnel.com, a target that only
# resolves inside Cloudflare's network. Re-running against an existing record
# is fine with --overwrite-dns; older builds lack the flag, hence the fallback.
say "routing $HOSTNAME_ARG"
if ! out="$(cloudflared tunnel route dns --overwrite-dns "$TUNNEL_NAME" "$HOSTNAME_ARG" 2>&1)"; then
    # Older builds have no --overwrite-dns. Fall back, and treat "it is already
    # there" as success rather than as the failure it is reported as.
    if ! out="$(cloudflared tunnel route dns "$TUNNEL_NAME" "$HOSTNAME_ARG" 2>&1)"; then
        if grep -qiE "already exists|record with that host" <<<"$out"; then
            say "the DNS record is already in place"
        else
            printf '%s\n' "$out" >&2
            exit 1
        fi
    fi
fi

# ---------------------------------------------------------------- service
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
say "installing gantry-tunnel.service"
sudo cp "$HERE/gantry-tunnel.service" /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable gantry-tunnel >/dev/null
# Restart rather than `enable --now`, which does nothing to an already-running
# unit and would leave the old config live after a re-run.
sudo systemctl restart gantry-tunnel

# ----------------------------------------------------------------- verify
# Over the public internet, not over loopback. Loopback proves the origin is up,
# which we already knew; the question is whether Cloudflare can reach it.
say "waiting for https://$HOSTNAME_ARG to answer"
ok=""
for attempt in $(seq 1 30); do
    code="$(curl -s -o /dev/null -m 8 -w '%{http_code}' "https://$HOSTNAME_ARG/api/me" || true)"
    if [[ "$code" == "200" ]]; then
        ok=1
        say "answering (attempt $attempt)"
        break
    fi
    sleep 10
done

if [[ -z "$ok" ]]; then
    cat >&2 <<EOF

The new hostname did not answer, so nothing has been cut over and the existing
quick tunnel is still serving. Look at:

    systemctl status gantry-tunnel
    journalctl -u gantry-tunnel -n 50 --no-pager

EOF
    exit 1
fi

# ------------------------------------------------------------- retire old
# Only now, and matched on --url so it cannot hit the named tunnel this script
# just started.
QUICK="$(pgrep -f 'cloudflared tunnel .*--url' || true)"
if [[ -n "$QUICK" ]]; then
    say "retiring the quick tunnel (pid $QUICK)"
    kill $QUICK || true
else
    say "no quick tunnel running"
fi

cat <<EOF

==> done.

    https://$HOSTNAME_ARG

  Same address after every restart, and the unit is enabled, so a reboot brings
  it back without anyone logging in.

==> two things worth doing now

  Confirm it really survives, which is the entire point:

      sudo reboot          # then load the URL again

  And upgrade identity from an argument about the host to a shared secret the
  internet cannot guess. A named tunnel has a zone, so it can carry a Transform
  Rule. See deploy/IDENTITY.md sections 1 and 2, and keep BENCH_TRUST_TUNNEL=1
  set afterwards so a deleted rule degrades safely.

EOF
