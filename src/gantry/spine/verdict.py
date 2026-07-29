"""Structured yes/no answers.

Nothing in Gantry returns a bare bool for a compatibility question. Every
check returns a Verdict carrying machine-readable reasons, because refusal
messages are a product surface: a resolver that says "incompatible" is
useless, one that says "rate 30 Hz vs 20 Hz, needs a resampler" is a
diagnosis. Reasons are also attached when a check passes, so callers can
surface notes ("units differ by a constant factor") without failing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping

from ..errors import GantryError


@dataclass(frozen=True)
class Reason:
    """One machine-readable explanation.

    ``code`` is a dotted, stable identifier that other components may branch
    on (the resolver maps codes to candidate adapters). ``message`` is for
    humans. ``detail`` carries whatever structured payload the code implies.
    """

    code: str
    message: str
    hint: str | None = None
    detail: Mapping[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        text = f"[{self.code}] {self.message}"
        return f"{text} ({self.hint})" if self.hint else text


@dataclass(frozen=True)
class Verdict:
    """The outcome of a check, with its reasons."""

    ok: bool
    reasons: tuple[Reason, ...] = ()

    @classmethod
    def yes(cls, *reasons: Reason) -> "Verdict":
        return cls(True, tuple(reasons))

    @classmethod
    def no(
        cls,
        code: str,
        message: str,
        hint: str | None = None,
        **detail: Any,
    ) -> "Verdict":
        return cls(False, (Reason(code, message, hint, detail),))

    @classmethod
    def note(cls, code: str, message: str, hint: str | None = None, **detail: Any) -> "Verdict":
        """A passing verdict that still carries something worth reporting."""
        return cls(True, (Reason(code, message, hint, detail),))

    @classmethod
    def all(cls, verdicts: Iterable["Verdict"]) -> "Verdict":
        """Conjunction: ok only if every verdict is ok. Keeps every reason."""
        result = cls.yes()
        for verdict in verdicts:
            result = result & verdict
        return result

    def __and__(self, other: "Verdict") -> "Verdict":
        return Verdict(self.ok and other.ok, self.reasons + other.reasons)

    def __bool__(self) -> bool:
        return self.ok

    def codes(self) -> tuple[str, ...]:
        return tuple(reason.code for reason in self.reasons)

    def because(self, code: str) -> tuple[Reason, ...]:
        """Every reason with this exact code."""
        return tuple(reason for reason in self.reasons if reason.code == code)

    def explain(self, indent: str = "  ") -> str:
        """Multi-line human rendering. Used verbatim in CLI refusals."""
        head = "ok" if self.ok else "refused"
        if not self.reasons:
            return head
        lines = [head] + [f"{indent}{reason}" for reason in self.reasons]
        return "\n".join(lines)

    def raise_if_refused(self, context: str = "") -> None:
        if not self.ok:
            prefix = f"{context}\n" if context else ""
            raise IncompatibleError(prefix + self.explain(), self)


class IncompatibleError(GantryError):
    """Raised when a refusal is escalated to an exception.

    A ``GantryError`` because that is what it is. The runner, and every sweep
    written against it, catches ``GantryError`` to record a component refusing
    as a refusal — so an escalated refusal outside that hierarchy is caught by
    nothing and reads as a crash. The distinction it used to draw, that a
    refusal is not a failure, is carried by the verdict it holds rather than
    by its base class.

    Recoverable, and deliberately: a refusal is one component declining one
    question it cannot answer meaningfully, and the rest of the run is
    unaffected. A feedback module that will not analyse a single cohort should
    fail only itself; halting there would let one module's honest "no" discard
    every other module's finished analysis.
    """

    halts = False
    recoverable = True

    def __init__(self, message: str, verdict: Verdict):
        super().__init__(message)
        self.verdict = verdict
