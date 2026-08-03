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

A **named tunnel** fixes both. It belongs to your Cloudflare account, it is bound
to a hostname in a zone you control, and it is the same hostname after every
restart. It still opens no inbound port: the connector dials out to Cloudflare
and dials the origin over loopback, so the security group keeps only port 22 and
the origin has no public ingress at all.

The cost is a domain. Cloudflare Registrar sells at wholesale with no markup and
no first-year discount games, which for a `.com` is roughly ten dollars a year.
Registering it there rather than elsewhere also skips a step: the zone is created
for you, so there are no nameservers to change and nothing to wait for.

---

## What you do

Two things need a human: paying for a domain, and an OAuth login in a browser.
Everything after that is scripted.

### 1. Register the domain

At <https://dash.cloudflare.com> under Domain Registration. If you already own one
elsewhere, add it as a zone instead and point its nameservers at the pair
Cloudflare gives you, then wait for it to go active before continuing.

### 2. Authorise the host

On the box:

```sh
cloudflared tunnel login
```

It prints a URL. The box has no browser, so copy that URL to your laptop, open it,
and pick the zone. That writes `~/.cloudflared/cert.pem` on the host, which is the
credential letting it create tunnels and DNS records in your account. It is not a
secret this repository ever sees.

### 3. Everything else

```sh
bench/deploy/tunnel-setup.sh bench.yourdomain.com
```

which creates the tunnel, writes the config, routes the DNS record, installs the
systemd unit, and retires the quick tunnel **only after** the new hostname answers.
That order matters: verify first, cut over second, so a failure leaves the old URL
serving rather than nothing at all.

---

## What the script does, for when it needs doing by hand

```sh
cloudflared tunnel create gantry
```

Prints a UUID and writes `~/.cloudflared/<UUID>.json`. That file is the tunnel's
credential. It is per host and per tunnel, and it does not belong in git.

`~/.cloudflared/config.yml`:

```yaml
tunnel: <UUID>
credentials-file: /home/ubuntu/.cloudflared/<UUID>.json

ingress:
  - hostname: bench.yourdomain.com
    service: http://127.0.0.1:8090
  - service: http_status:404
```

The trailing catch-all rule is mandatory; cloudflared refuses to start without a
final rule that has no hostname.

```sh
cloudflared tunnel route dns gantry bench.yourdomain.com
```

Creates a proxied CNAME to `<UUID>.cfargotunnel.com`. That target only resolves
inside Cloudflare's network, which is why this route needs the zone to be on
Cloudflare and why a free subdomain from an outside provider cannot be pointed at
a tunnel.

Then the unit:

```sh
sudo cp /home/ubuntu/gantry_bench/deploy/gantry-tunnel.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable gantry-tunnel
sudo systemctl restart gantry-tunnel
```

`enable --now` starts a stopped unit and does nothing whatsoever to a running one,
which is how a previous deploy left new code on disk and old code serving. Restart,
always.

---

## Identity, before and after

Nothing about identity changes on the day you cut over, and that is deliberate.

The origin still binds `127.0.0.1:8090` and the connector still dials it from the
same machine, so `BENCH_TRUST_TUNNEL=1` remains a true assertion and visitors stay
separated exactly as they are now. You get the stable hostname without touching
the part that decides who can read whose uploads.

The upgrade is separate and can happen whenever. A named tunnel has a zone, so it
can carry a Transform Rule, so it can attach a shared secret. That replaces an
argument about the host with a value the public internet cannot guess. Steps are
in [IDENTITY.md](IDENTITY.md) sections 1 and 2.

**Leave `BENCH_TRUST_TUNNEL=1` set after you add the secret.** Both statements are
true at once, and `client_ip()` accepts either. If the Transform Rule is ever
deleted or edited, the loopback path keeps working and visitors stay separated,
instead of every one of them silently collapsing into a single shared org.

### One thing `/api/me` will not tell you

`describe()` reports `"mode": "edge"` as soon as `BENCH_TRUST_HEADER` and
`BENCH_TRUST_SECRET` are set **at the origin**. It does not, and cannot, check
that the header is actually arriving. So a Transform Rule that was never created,
or was created with a typo in the value, still reads as `"edge"`.

The proof that the secret is live is the direct-origin test in IDENTITY.md section
4: hit the origin from somewhere that bypasses Cloudflare with a forged
`cf-connecting-ip` and confirm the org id you get back is not the one that address
really gets. On this deployment there is no way to reach the origin at all from
outside, which is the stronger property, and it is worth remembering that it is
the binding rather than the header check doing that work.

---

## Checking it

```sh
systemctl is-active gantry-api gantry-worker gantry-tunnel
curl -s https://bench.yourdomain.com/api/me | python3 -m json.tool
```

Then the part that is easy to skip and is the whole point of this document:

```sh
sudo reboot
```

Wait, and load the same URL. If it comes back on its own at the same address, the
problem this document exists for is fixed. If it does not, the unit is not enabled
and you have a stable name in front of a service that still needs a human after
every restart.
