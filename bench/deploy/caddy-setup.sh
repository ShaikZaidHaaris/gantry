#!/usr/bin/env bash
# Serve the bench over HTTPS on your own hostname, without a tunnel.
#
# Runs ON THE HOST.
#
#     ./caddy-setup.sh bench.yourdomain.com
#
# Caddy terminates TLS with a Let's Encrypt certificate it obtains and renews by
# itself, and proxies to the API on loopback. The API keeps binding to
# 127.0.0.1, so what is reachable from the internet is Caddy, not the origin.
#
# What this does NOT do
# ---------------------
# Open the ports. Caddy needs inbound 80 and 443, and that is a change to the
# instance's security group, made deliberately by a person who knows what else
# is on the host. A deploy script that edits a firewall is one that does it on a
# day nobody was watching. It checks and tells you instead.
#
# Idempotent. Re-running reuses the existing secret rather than generating a new
# one, which matters: rotating it mid-flight would make every request fail the
# edge check at once, and the app would fall back to treating every visitor as
# the same org until the API restarted.
set -euo pipefail

HOSTNAME_ARG="${1:-}"
ORIGIN_PORT="${ORIGIN_PORT:-8090}"
ENV_FILE="${ENV_FILE:-/home/ubuntu/gantry_bench/env}"
TRUST_HEADER="x-bench-edge"

if [[ -z "$HOSTNAME_ARG" ]]; then
    echo "usage: $0 <hostname>        e.g. $0 bench.yourdomain.com" >&2
    exit 2
fi

say() { printf '==> %s\n' "$*"; }
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ---------------------------------------------------------------- preflight
curl -fsS -o /dev/null -m 5 "http://127.0.0.1:$ORIGIN_PORT/api/me" || {
    echo "the API is not answering on 127.0.0.1:$ORIGIN_PORT; start gantry-api first" >&2
    exit 1
}

[[ -f "$ENV_FILE" ]] || {
    echo "no env file at $ENV_FILE; run deploy.sh first" >&2
    exit 1
}

# Does the name point here? Getting this wrong is the single most common way to
# spend an hour reading Caddy logs: the certificate request fails, and the
# reason is a DNS record that was never added or has not propagated.
say "checking DNS"
MY_IP="$(curl -fsS -m 10 https://checkip.amazonaws.com | tr -d '[:space:]' || true)"
RESOLVED="$(getent ahostsv4 "$HOSTNAME_ARG" | awk '{print $1; exit}' || true)"
if [[ -z "$RESOLVED" ]]; then
    cat >&2 <<EOF
$HOSTNAME_ARG does not resolve to anything yet.

Add an A record at your registrar pointing it at this host:

    type  A
    name  ${HOSTNAME_ARG%%.*}
    value $MY_IP

then run this again. A new record is usually visible within a few minutes.
EOF
    exit 1
fi
if [[ "$RESOLVED" != "$MY_IP" ]]; then
    cat >&2 <<EOF
$HOSTNAME_ARG resolves to $RESOLVED, but this host is $MY_IP.

Certificate issuance will fail, because Let's Encrypt proves you control the
name by connecting to wherever it points. Fix the A record, wait for the old
value's TTL to expire, and run this again.
EOF
    exit 1
fi
say "$HOSTNAME_ARG -> $MY_IP, correct"

# ------------------------------------------------------------------ caddy
if ! command -v caddy >/dev/null; then
    say "installing caddy"
    sudo apt-get install -y -qq debian-keyring debian-archive-keyring apt-transport-https curl >/dev/null
    curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/gpg.key' |
        sudo gpg --batch --yes --dearmor -o /usr/share/keyrings/caddy-stable-archive-keyring.gpg
    curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/debian.deb.txt' |
        sudo tee /etc/apt/sources.list.d/caddy-stable.list >/dev/null
    sudo apt-get update -qq
    sudo apt-get install -y -qq caddy
else
    say "caddy already installed ($(caddy version | head -1))"
fi

# ----------------------------------------------------------------- secret
# Reused if present. Never printed: it is the whole of the edge proof, and a
# secret echoed into a terminal is a secret in a scrollback buffer and a
# transcript.
if grep -q "^BENCH_TRUST_SECRET=" "$ENV_FILE"; then
    say "reusing the existing BENCH_TRUST_SECRET"
    SECRET="$(grep "^BENCH_TRUST_SECRET=" "$ENV_FILE" | head -1 | cut -d= -f2-)"
else
    say "generating BENCH_TRUST_SECRET"
    SECRET="$(openssl rand -hex 32)"
    printf 'BENCH_TRUST_SECRET=%s\n' "$SECRET" >>"$ENV_FILE"
fi

set_env() {
    local key="$1" value="$2"
    if grep -q "^$key=" "$ENV_FILE"; then
        sed -i "s|^$key=.*|$key=$value|" "$ENV_FILE"
    else
        printf '%s=%s\n' "$key" "$value" >>"$ENV_FILE"
    fi
}

set_env BENCH_TRUST_HEADER "$TRUST_HEADER"
# Deliberately a single-value header set by Caddy from {remote_host}, not the
# X-Forwarded-For chain. The chain resolver takes the left-most public entry,
# which a client can prepend to; this header is written by the proxy on every
# request and overwrites whatever arrived.
set_env BENCH_CLIENT_IP "x-real-ip"
# Left set on purpose. The origin still binds loopback, so it is still true, and
# it means a broken Caddyfile degrades to visitors sharing one org rather than
# to visitors able to pick each other's identities.
set_env BENCH_TRUST_TUNNEL "1"
chmod 600 "$ENV_FILE"

# --------------------------------------------------------------- caddyfile
say "writing /etc/caddy/Caddyfile"
TMP="$(mktemp)"
trap 'rm -f "$TMP"' EXIT
sed -e "s|__HOSTNAME__|$HOSTNAME_ARG|g" \
    -e "s|__TRUST_HEADER__|$TRUST_HEADER|g" \
    -e "s|__SECRET__|$SECRET|g" \
    "$HERE/Caddyfile.template" >"$TMP"
sudo install -o root -g caddy -m 640 "$TMP" /etc/caddy/Caddyfile
sudo caddy validate --config /etc/caddy/Caddyfile >/dev/null

say "restarting caddy and the api"
sudo systemctl enable caddy >/dev/null
sudo systemctl restart caddy
# The API has to re-read the env file to see the secret. Without this it keeps
# running in the old mode and every check below reports the old answer.
sudo systemctl restart gantry-api

# ----------------------------------------------------------------- verify
say "waiting for a certificate and a first response"
ok=""
for attempt in $(seq 1 20); do
    code="$(curl -s -o /dev/null -m 10 -w '%{http_code}' "https://$HOSTNAME_ARG/api/me" || true)"
    if [[ "$code" == "200" ]]; then
        ok=1
        say "answering (attempt $attempt)"
        break
    fi
    sleep 10
done

if [[ -z "$ok" ]]; then
    cat >&2 <<EOF

No answer on https://$HOSTNAME_ARG yet.

The usual cause is that inbound 80 and 443 are not open in this instance's
security group, so Let's Encrypt cannot reach Caddy to validate the name and no
certificate is ever issued. This script will not edit your firewall. Check:

    sudo journalctl -u caddy -n 50 --no-pager

EOF
    exit 1
fi

MODE="$(curl -s -m 10 "https://$HOSTNAME_ARG/api/me" |
    python3 -c 'import json,sys; print(json.load(sys.stdin)["identity"]["mode"])' 2>/dev/null || echo "?")"

cat <<EOF

==> done.

    https://$HOSTNAME_ARG          identity mode: $MODE

  Want "edge". "direct" or "tunnel" means the header is not arriving and every
  visitor is sharing one org; check that BENCH_TRUST_SECRET in $ENV_FILE
  matches the value in /etc/caddy/Caddyfile.

==> two things worth doing now

  Confirm it survives a restart, which is the point of the exercise:

      sudo reboot          # then load the URL again

  And confirm a client cannot choose its own identity. From your laptop:

      curl -s https://$HOSTNAME_ARG/api/me -H 'x-real-ip: 93.184.216.34' | python3 -m json.tool

  The org id must be *your* org, not one derived from 93.184.216.34. Caddy
  overwrites that header on every request, so a forged one never reaches the
  API. If it does change, the Caddyfile is appending where it should be
  setting, and anyone can read anyone's submissions.

EOF
