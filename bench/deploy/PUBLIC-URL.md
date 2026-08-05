# A URL that survives a restart

The bench was reachable at `https://martha-educational-unified-honolulu.trycloudflare.com`.
Two things were wrong with that, and the ugly name was the smaller one.

A **quick tunnel** (`cloudflared tunnel --url ...`) mints a fresh random hostname
every time the process starts. It has no account, no zone and no DNS record, so
there is nothing for the name to be stable against. On the deployed host it was
also running as a bare process rather than a service, which compounds it: a
reboot, an OOM kill or a closed shell took the site down and, on the next manual
start, brought it back at a *different* address. Every link anybody had been sent
was dead, silently, with no error anywhere.

Two ways to fix it properly. Both keep the API bound to `127.0.0.1`, so in
neither case is the origin itself reachable from the internet.

| | Caddy on your own hostname | Cloudflare named tunnel |
|---|---|---|
| DNS | one A record at your registrar | nameservers must move to Cloudflare |
| Firewall | inbound 80 and 443 | nothing opened |
| Instance stop/start | breaks unless an Elastic IP is attached | unaffected, the connector dials out |
| Identity mode | `edge` immediately | `tunnel`, or `edge` after a Transform Rule |
| Certificate | Let's Encrypt, obtained and renewed by Caddy | terminated by Cloudflare |

The first is fewer moving parts and does not involve a third party in your DNS.
The second is better if the instance is stopped often, because a tunnel does not
care what the public IP is.

---

## Caddy on your own hostname

### 1. A stable address for the box

Attach an **Elastic IP** to the instance first, before creating the DNS record.
An EC2 public IP is released when the instance stops, so without one the A record
points at somebody else's machine after your next stop/start. Doing it first also
saves editing DNS twice, since attaching an EIP replaces the auto-assigned
address.

### 2. Open 80 and 443

In the instance's security group, inbound from `0.0.0.0/0`. Port 443 serves the
site; port 80 is how Let's Encrypt validates that you control the name, and
redirects to HTTPS afterwards.

`caddy-setup.sh` does not do this for you. A deploy script that edits a firewall
is one that does it on a day nobody was watching.

### 3. One DNS record

At your registrar:

    type   A
    name   bench
    value  <the Elastic IP>

Nothing else. No CNAME, no nameserver change, no TXT.

### 4. Run it

```sh
./caddy-setup.sh bench.yourdomain.com     # on the host
```

It checks the name resolves to this machine before touching anything, because a
missing or stale A record is the single most common way to spend an hour reading
certificate logs. Then it installs Caddy, generates the shared secret, writes it
to both the API's env file and the Caddyfile so the two cannot drift, restarts
both services and verifies over the public internet.

Re-running is safe. It reuses the existing secret rather than generating a new
one, which matters: rotating it mid-flight fails every request's edge check at
once, and until the API restarts every visitor collapses into a single org.

### The part that is easy to get wrong

Identity here is the access control, so the proxy's header handling is a
permission check.

Caddy's default, and nginx's, is to **append** the peer to whatever
`X-Forwarded-For` the client sent. The origin resolves a chain by taking the
left-most public entry. That is the right read behind Cloudflare, which owns the
header and rewrites it, and an impersonation bug behind an appending proxy: a
client sending `X-Forwarded-For: 93.184.216.34` produces the chain
`93.184.216.34, <their real address>`, the origin believes the first entry, and
they are now reading that visitor's submissions.

A resolver cannot fix this by looking at the chain, because only the proxy knows
which entries it added. So `Caddyfile.template` overwrites rather than appends,
and `BENCH_CLIENT_IP` points at a single-value header written from
`{remote_host}` on every request. The chain is then never consulted at all.

Both halves are pinned by tests in `api/tests/test_identity.py`:
`test_the_chain_resolver_believes_whatever_is_left_most` asserts the hazard,
`test_a_proxy_set_single_value_header_beats_a_forged_chain` asserts the fix.

Confirm it on the live site, from your laptop:

```sh
curl -s https://bench.yourdomain.com/api/me -H 'x-real-ip: 93.184.216.34' | python3 -m json.tool
```

The org id must be your own. If it changes, the Caddyfile is appending where it
should be setting.

---

## Cloudflare named tunnel

Opens no port and is immune to the public IP changing. The cost is that the zone
has to be on Cloudflare, because a tunnel route is a CNAME to
`<uuid>.cfargotunnel.com` and that target only resolves inside Cloudflare's
network. Registration can stay wherever you bought it; only the nameservers move.

Add the site at Cloudflare, point your registrar's nameservers at the two it
gives you, and wait for the zone to go active. Check the imported DNS records
before flipping, particularly `MX` and the `TXT` records behind SPF and DKIM:
those break email silently, and a scan does not always catch them.

Then on the host:

```sh
cloudflared tunnel login          # prints a URL; open it on a machine with a browser
./tunnel-setup.sh bench.yourdomain.com
```

`tunnel-setup.sh` creates the tunnel, writes the config, routes the DNS record,
installs the systemd unit, and retires the quick tunnel **only after** the new
hostname answers. That order matters: a failure anywhere leaves the old URL
serving rather than leaving nothing serving.

Identity does not change on cutover. The origin still binds loopback and the
connector still dials it locally, so `BENCH_TRUST_TUNNEL=1` stays true and
visitors stay separated exactly as before. Upgrading to `edge` is a Transform
Rule, described in [IDENTITY.md](IDENTITY.md) sections 1 and 2.

**Leave `BENCH_TRUST_TUNNEL=1` set afterwards.** Both statements are true at once
and `client_ip()` accepts either, so if the rule is ever deleted the loopback
path keeps working instead of every visitor silently collapsing into one org.

---

## What `/api/me` will not tell you

`describe()` reports `"mode": "edge"` as soon as `BENCH_TRUST_HEADER` and
`BENCH_TRUST_SECRET` are set **at the origin**. It does not, and cannot, check
that the header is actually arriving. A Transform Rule that was never created, or
a Caddyfile with a mismatched secret, still reads as `"edge"`.

What proves the secret is live is a request that bypasses the proxy, described in
IDENTITY.md section 4. On both deployments here the origin binds loopback, so
there is no way to make that request from outside at all, which is the stronger
property. Worth remembering that it is the binding rather than the header check
doing that work.

---

## Checking either one

```sh
systemctl is-active gantry-api gantry-worker
curl -s https://bench.yourdomain.com/api/me | python3 -m json.tool
```

Then the step that is easy to skip and is the entire point of this document:

```sh
sudo reboot
```

Wait, and load the same URL. If it comes back on its own at the same address, the
problem this document exists for is fixed. If it does not, a unit is not enabled
and you have a stable name in front of a service that still needs a human after
every restart.
