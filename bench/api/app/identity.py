"""Who is asking, when the answer is "whoever owns this IP".

The dev seam this replaces read an ``x-demo-user`` header and created an org for
whatever it found. That is exactly right on a laptop and a full authorization
bypass the moment the port is reachable, because in this product **identity is
the access control**: every submission is scoped to an org, so anyone who can
choose their identity can read anyone's uploads.

The same trap is waiting in the IP version, and it is worth being precise about
it. Behind Cloudflare the client's real address arrives in a *header*
(``CF-Connecting-IP``), because the TCP peer is a Cloudflare edge machine. A
header is a thing the client writes. So an origin that trusts
``CF-Connecting-IP`` unconditionally lets anyone who can reach it directly --
bypassing Cloudflare entirely -- claim any address they like and read that
address's submissions.

Hence one rule, enforced below: **a forwarding header is trusted only when the
request is proven to have come through our own edge.** Proof is a shared secret
that Cloudflare attaches and the public internet cannot guess. Without that
proof configured, this module refuses to read forwarding headers at all and
falls back to the TCP peer -- which behind a proxy is the proxy, meaning every
user collapses into one org. That failure is loud, harmless and obvious in
testing; the opposite failure is silent and hands out other people's data.

Configuration:

    BENCH_TRUST_HEADER   name of the header Cloudflare attaches, e.g. x-bench-edge
    BENCH_TRUST_SECRET   its expected value; a long random string
    BENCH_CLIENT_IP      header carrying the real client IP (default cf-connecting-ip)
    BENCH_IP_SALT        salt for hashing addresses at rest

All four are read per request, not at import -- see the note on the accessors.
With none of them set the module runs in direct mode, which is correct for a
laptop and reported by :func:`describe`.
"""

from __future__ import annotations

import hashlib
import hmac
import ipaddress
import os
import secrets

#: Read on every call, deliberately, rather than once at import.
#:
#: Import-time configuration is an ordering trap: anything that imports this
#: module before the environment is set -- a test file that is not the first
#: one collected, an app server that loads the package and then configures it,
#: an interactive session -- silently gets direct mode and no isolation between
#: visitors. That failure is invisible until somebody sees another person's
#: uploads, which is far too late to learn about an ordering dependency. A
#: dictionary lookup per request is not a cost worth that risk.
#:
#: The salt is the one exception: generated once per process when unset, so
#: that a deployment which forgets to set it at least stays consistent while it
#: runs, rather than reshuffling every visitor's identity mid-session.
_EPHEMERAL_SALT = secrets.token_hex(16)


def trust_header() -> str:
    """Name of the header our edge attaches. Cloudflare sets it via Transform Rule."""
    return os.environ.get("BENCH_TRUST_HEADER", "").lower()


def trust_secret() -> str:
    """Its expected value. Nothing but our edge can guess it."""
    return os.environ.get("BENCH_TRUST_SECRET", "")


def client_ip_header() -> str:
    """Where the real address lives once the request is trusted.

    Cloudflare's own name for it by default; ``true-client-ip`` on Enterprise,
    and an X-Forwarded-For chain behind a plain ALB.
    """
    return os.environ.get("BENCH_CLIENT_IP", "cf-connecting-ip").lower()


def ip_salt() -> str:
    """Salt for hashing addresses at rest.

    An IP is personal data in most of the world and this product needs only to
    tell visitors apart, so the address is never stored. The salt stops the
    hash being a rainbow table of the whole IPv4 space, which an unsalted
    SHA-256 of a 32-bit value certainly is.
    """
    return os.environ.get("BENCH_IP_SALT") or _EPHEMERAL_SALT


def trusts_edge() -> bool:
    return bool(trust_header() and trust_secret())


def _from_edge(headers) -> bool:
    """Did this request come through our own edge?

    Compared with :func:`hmac.compare_digest` rather than ``==``: the secret is
    checked on every request, and a byte-by-byte comparison that returns early
    leaks its length and content to anyone willing to time it.
    """
    if not trusts_edge():
        return False
    got = headers.get(trust_header(), "")
    return bool(got) and hmac.compare_digest(got, trust_secret())


def _first_public(chain: str) -> str | None:
    """The left-most address in an X-Forwarded-For chain that a client could own.

    Read left to right because the client's own address is left-most and each
    proxy appends. Private and loopback entries are skipped: a client behind its
    own NAT may have prepended ``10.x``, and taking that would put every such
    visitor in one org.
    """
    for part in (p.strip() for p in chain.split(",")):
        if not part:
            continue
        try:
            addr = ipaddress.ip_address(part)
        except ValueError:
            continue
        if addr.is_private or addr.is_loopback or addr.is_link_local:
            continue
        return str(addr)
    return None


def client_ip(request) -> str:
    """The caller's address, or the TCP peer when nothing can be proven.

    Never raises. A request with no resolvable address at all returns
    ``"unknown"``, which is a real org like any other -- everyone unresolvable
    shares it. That is a deliberate, visible degradation rather than a 500 on a
    path the user cannot fix.
    """
    headers = request.headers
    if _from_edge(headers):
        stated = headers.get(client_ip_header(), "").strip()
        if stated:
            try:
                return str(ipaddress.ip_address(stated))
            except ValueError:
                pass
        chain = headers.get("x-forwarded-for", "")
        if chain:
            found = _first_public(chain)
            if found:
                return found
    peer = getattr(request.client, "host", None)
    return peer or "unknown"


def org_key(ip: str) -> str:
    """A stable, non-reversible handle for one address.

    The database never sees the address itself. Two requests from the same
    address produce the same key for as long as the salt lives, and nothing in
    the store can be turned back into someone's IP.
    """
    digest = hashlib.sha256(f"{ip_salt()}:{ip}".encode()).hexdigest()
    return f"ip_{digest[:32]}"


def label(ip: str) -> str:
    """What to call this visitor on screen.

    Deliberately not the address. Showing a user their own IP invites them to
    read it as an account name, and showing it anywhere a second person can see
    would publish it. The last four of the hash is enough to tell two orgs apart
    when debugging and says nothing about anyone.
    """
    return f"Visitor {org_key(ip)[-4:]}"


def describe() -> dict:
    """What this deployment actually does, for /api/me and the logs.

    Reported rather than assumed, because the difference between "every visitor
    is separate" and "every visitor is the same org" is invisible from the
    outside and is exactly what a misconfigured proxy silently causes.
    """
    return {
        "mode": "edge" if trusts_edge() else "direct",
        "client_ip_header": client_ip_header() if trusts_edge() else None,
        "salt_is_ephemeral": not os.environ.get("BENCH_IP_SALT"),
        "warning": None
        if trusts_edge()
        else (
            "No BENCH_TRUST_HEADER/BENCH_TRUST_SECRET set, so forwarding headers "
            "are ignored and identity is the TCP peer. Behind a proxy that makes "
            "every visitor one org. Correct for localhost, wrong in production."
        ),
    }
