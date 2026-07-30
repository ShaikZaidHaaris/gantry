"""Worked ``ask`` callables for real model APIs.

Kept apart from the scorer, and each importing its SDK only when called, because
the plugin's argument is about the method rather than about a vendor. Somebody
with their own inference stack writes their own three-line function and never
touches this file; these exist so that nobody has to write it in order to try
the thing once.

Every one of them is the same shape: ``(prompt, frames) -> text``. That is the
entire integration surface.
"""

from __future__ import annotations

import base64
from typing import Any, Callable, Sequence

from gantry.errors import ConfigError


def anthropic_wire(
    model: str = "claude-sonnet-5",
    *,
    max_tokens: int = 512,
    temperature: float = 0.0,
    client: Any = None,
) -> Callable[[str, Sequence[bytes]], str]:
    """Ask a Claude model. Frames go as PNG image blocks, in order."""

    def ask(prompt: str, frames: Sequence[bytes]) -> str:
        nonlocal client
        if client is None:
            try:
                import anthropic
            except ImportError as error:
                raise ConfigError("pip install 'gantry-scorer-vlm[anthropic]'") from error
            client = anthropic.Anthropic()
        content: list[dict[str, Any]] = [
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/png",
                    "data": base64.b64encode(frame).decode(),
                },
            }
            for frame in frames
        ]
        content.append({"type": "text", "text": prompt})
        reply = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[{"role": "user", "content": content}],
        )
        return "".join(
            block.text for block in reply.content if getattr(block, "type", "") == "text"
        )

    return ask


def openai_wire(
    model: str = "gpt-4o",
    *,
    max_tokens: int = 512,
    temperature: float = 0.0,
    client: Any = None,
) -> Callable[[str, Sequence[bytes]], str]:
    """Ask an OpenAI model. Frames go as data-URI images, in order."""

    def ask(prompt: str, frames: Sequence[bytes]) -> str:
        nonlocal client
        if client is None:
            try:
                import openai
            except ImportError as error:
                raise ConfigError("pip install 'gantry-scorer-vlm[openai]'") from error
            client = openai.OpenAI()
        content: list[dict[str, Any]] = [
            {
                "type": "image_url",
                "image_url": {"url": "data:image/png;base64," + base64.b64encode(frame).decode()},
            }
            for frame in frames
        ]
        content.append({"type": "text", "text": prompt})
        reply = client.chat.completions.create(
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[{"role": "user", "content": content}],
        )
        return reply.choices[0].message.content or ""

    return ask


def cached(
    ask: Callable[..., str], store: dict[str, str]
) -> Callable[[str, Sequence[bytes], str], str]:
    """Wrap an ``ask`` so each trial is only ever paid for once.

    Useful when a scoring pass is interrupted, and useful when a rubric changes
    for one criterion and the rest should not be re-billed. ``store`` is
    whatever mapping the caller wants to persist.
    """

    def wrapped(prompt: str, frames: Sequence[bytes], trial: str = "") -> str:
        key = trial or str(len(store))
        if key not in store:
            try:
                store[key] = ask(prompt, frames, trial)  # type: ignore[call-arg]
            except TypeError:
                store[key] = ask(prompt, frames)
        return store[key]

    return wrapped
