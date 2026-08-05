# One visitor, one set of submissions

The bench gives every visiting IP address its own org. Submissions are scoped to
that org, so two visitors cannot see each other's uploads, and a result reaches
the shared leaderboard only when its owner publishes it.

This document is the part that lives outside the code. **The isolation is only
real if Cloudflare and the origin are configured to agree**, and a
misconfiguration fails in one of two ways, both of which are worth being able to
name before you deploy.

---

## The trap

Behind Cloudflare, the visitor's real address does not arrive on the socket. The
TCP peer is a Cloudflare machine; the real address arrives in a header,
`CF-Connecting-IP`.

A header is a thing the client writes.

So an origin that trusts `CF-Connecting-IP` unconditionally lets anybody who can
reach it **directly**, bypassing Cloudflare, which on AWS is usually possible
whatever your DNS says, set that header to any value and read that visitor's
submissions. Identity here *is* the access control, so believing a forged header
is a full authorization bypass, not a cosmetic bug.

The origin therefore trusts a forwarding header only when the request carries
proof it came through your edge: a shared secret Cloudflare attaches and the
public internet cannot guess.

## The two failure modes

| symptom | cause | how it looks |
|---|---|---|
| **Everyone shares one org** | secret not configured, or not matching | every visitor sees every submission; `/api/me` reports `"mode": "direct"` |
| **Anyone can impersonate anyone** | origin trusts the header without proof | silent; nobody notices until someone reads another person's uploads |

The first is loud and harmless. The second is silent and serious. The code fails
toward the first on purpose: with no secret configured it ignores forwarding
headers entirely.

---

## Which mode do you need?

Three, and the right one depends on how the origin is reachable.

| your setup | mode | what earns the trust |
|---|---|---|
| `cloudflared tunnel --url http://127.0.0.1:8090` (quick tunnel) | **tunnel** | the origin binds to loopback, so nothing outside can reach it to forge a header |
| Caddy or a named tunnel ([PUBLIC-URL.md](PUBLIC-URL.md)), or Cloudflare in front of a public origin | **edge** | a shared secret only your edge knows |
| laptop | **direct** | nothing; every visitor is one org, which is correct locally |

**A quick tunnel cannot use the secret**, because a secret is attached by a
Transform Rule and a Transform Rule needs a zone, which a quick tunnel has none
of. That is why tunnel mode exists. Without it, the whole `trycloudflare.com`
deployment silently runs with every visitor sharing one org.

Tunnel mode rests on a weaker argument than the secret: a fact about the host
rather than a value only your edge knows. It is checked as far as it can be:
a request arriving from anything but a loopback peer is not trusted even with
the flag set. But the flag itself is your assertion that the API binds to
loopback. **Turn it off if you ever bind to `0.0.0.0`.**

    BENCH_TRUST_TUNNEL=1

`deploy.sh` writes this into a fresh env file, since the deployment it produces
is exactly this shape.

## 1. Origin environment

Add to `/home/ubuntu/gantry_bench/env`:

```sh
BENCH_TRUST_HEADER=x-bench-edge
BENCH_TRUST_SECRET=<64 hex chars, generated once, never committed>
BENCH_CLIENT_IP=cf-connecting-ip
BENCH_IP_SALT=<32 hex chars, generated once, never committed>
```

Generate both:

```sh
openssl rand -hex 32   # BENCH_TRUST_SECRET
openssl rand -hex 16   # BENCH_IP_SALT
```

`BENCH_IP_SALT` matters more than it looks. Addresses are hashed before storage,
the database never holds an IP, and an unsalted SHA-256 of a 32-bit value is a
rainbow table, not a hash. **If it is unset, one is generated per process**,
which means every visitor becomes a new org on restart and loses their
submissions. Set it, and keep it: changing it later orphans every existing org.

## 2. Cloudflare: attach the secret

Rules → Transform Rules → **Modify Request Header** → Add:

| field | value |
|---|---|
| Header name | `x-bench-edge` |
| Value | the same string as `BENCH_TRUST_SECRET` |
| Expression | `true` (all requests to this hostname) |

Cloudflare sets `CF-Connecting-IP` itself; you do not configure that.

## 3. AWS: make the origin unreachable except through Cloudflare

The secret stops a direct caller *impersonating* someone. It does not stop them
reaching the origin at all, and there is no reason to let them. In the security
group for the instance or load balancer, allow 443 **only** from
[Cloudflare's published ranges](https://www.cloudflare.com/ips/), and remove
`0.0.0.0/0`.

Better still, run a Cloudflare Tunnel and give the origin no public ingress at
all. Then the secret is defence in depth rather than the only thing standing
between a stranger and somebody else's data.

## 4. Check it

```sh
curl -s https://your-host/api/me | jq .identity
```

Expected:

```json
{ "mode": "edge", "client_ip_header": "cf-connecting-ip",
  "salt_is_ephemeral": false, "warning": null }
```

`"mode": "direct"` means the secret is not matching and **every visitor is
sharing one org right now**. `"salt_is_ephemeral": true` means orgs will not
survive a restart.

Read `"edge"` narrowly, though. `describe()` reports it as soon as the two
variables are set **at the origin**; it has no way to check that the header is
arriving. A Transform Rule that was never created, or created with a typo in the
value, still reads as `"edge"` here. Which is why the next check exists.

Then confirm the guard actually holds, from a machine that can reach the origin
directly:

```sh
curl -s http://<origin-ip>/api/me -H 'cf-connecting-ip: 1.2.3.4' | jq .org.id
```

That org id must **not** match the one a real browser from `1.2.3.4` gets. If it
does, the secret is missing at the origin and the header is being believed.

---

## What this is not

IP is a partition, not an identity. Three consequences, none of them bugs, all of
them worth deciding about rather than discovering:

- **Shared addresses share submissions.** A lab, a university, an office, a
  corporate VPN or a mobile carrier behind CGNAT is *one visitor* here. Everyone
  on it sees the same list. If your users are colleagues on one network, this
  model does not separate them.
- **A changed address is a new visitor.** A reconnected router, a laptop moving
  from wifi to tethering, or IPv6 privacy extensions rotating, all arrive as
  somebody new, and cannot reach what they uploaded an hour earlier. Nothing is
  deleted; it is simply no longer theirs.
- **Nobody is authenticated.** There are no accounts and no passwords. Two people
  who share an address are indistinguishable, and always will be under this
  scheme.

If any of those start to matter, the fix is a signed cookie issued on first visit
(survives address changes, separates people behind one NAT) or real accounts.
Both slot into the same seam: `viewer()` in `app/main.py` is the only place that
decides who is asking.
