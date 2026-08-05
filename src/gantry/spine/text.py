"""Turning counts and identifiers into sentences a person would write.

This project's output is prose. A finding is a sentence somebody reads and acts
on, so the difference between "7 check(s) had nothing to read" and "7 checks had
nothing to read" is not cosmetic: the first tells the reader they are looking at
a template with a number substituted in, and everything around it starts reading
like machine output too.

Small on purpose. There is no attempt at general English pluralisation, because
the vocabulary here is small and known, and a library that gets "analyses" right
would still be a dependency in the one package that promises to have none.
"""

from __future__ import annotations

import re

#: Words whose plural is not "add an s". Extended when something needs it, not
#: in anticipation.
IRREGULAR = {
    "is": "are",
    "was": "were",
    "has": "have",
    "does": "do",
    "this": "these",
    "it": "they",
    "analysis": "analyses",
}


def plural(count: int, singular: str, many: str | None = None) -> str:
    """``plural(1, "clip")`` is "clip"; ``plural(3, "clip")`` is "clips"."""
    if abs(count) == 1:
        return singular
    if many is not None:
        return many
    if singular in IRREGULAR:
        return IRREGULAR[singular]
    if singular.endswith(("s", "x", "z", "ch", "sh")):
        return singular + "es"
    if singular.endswith("y") and singular[-2:-1] not in "aeiou":
        return singular[:-1] + "ies"
    return singular + "s"


def count_of(count: int, singular: str, many: str | None = None) -> str:
    """``count_of(3, "clip")`` is "3 clips". The common case, spelled once."""
    return f"{count} {plural(count, singular, many)}"


def readable(identifier: str) -> str:
    """A machine name as a person would say it.

    ``pick_dual_bottles`` becomes "Pick dual bottles";
    ``observation.images.head`` becomes "Observation images head".

    For display only. The identifier itself stays whatever it was, because it is
    what a report cites and what an API returns, and rewriting it would break
    the one thing it is for.
    """
    words = re.split(r"[._\-\s]+", identifier.strip())
    words = [w for w in words if w]
    if not words:
        return identifier
    joined = " ".join(words)
    return joined[0].upper() + joined[1:]


#: A dotted code, as the pipeline writes them: ``capture.hands_offscreen``.
CODE = re.compile(r"\b[a-z][a-z0-9]*(?:\.[a-z0-9_]+)+\b")


def without_codes(text: str) -> str:
    """The same sentence with the machine codes taken out of it.

    Modules name their own checks when they explain themselves, which is right
    for a log and wrong for a page somebody is reading. The codes are not
    deleted from the record, only from the sentence: the gate keeps the original
    alongside, so a lab that wants to cite ``capture.hands_offscreen`` still can.

    Handles the three shapes these appear in: a bracketed prefix, a parenthesised
    list, and a bare mention mid-sentence.
    """
    text = re.sub(r"\[\s*" + CODE.pattern + r"\s*\]\s*", "", text)
    # A parenthesis containing nothing but codes, commas and "and N more".
    def _drop(match: re.Match[str]) -> str:
        inner = match.group(1)
        stripped = CODE.sub("", inner)
        return "" if re.fullmatch(r"[\s,]*(and \d+ more)?[\s,]*", stripped) else match.group(0)

    text = re.sub(r"\s*\(([^()]*)\)", _drop, text)
    text = CODE.sub("", text)
    return re.sub(r"\s+([.,;:])", r"\1", re.sub(r"\s{2,}", " ", text)).strip()
