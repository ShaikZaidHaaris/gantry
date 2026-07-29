# Integration tests

These import plugins. That is the whole difference between this directory and
the one above it.

`tests/` proves a claim the project makes out loud: **core installs and works
with nothing else present**. A test that imports `gantry_adapters_core` cannot
prove that, and worse, it silently breaks the proof — the core CI job installs
core alone, so a plugin import there is a collection error, not a skipped test.
That is exactly what happened, and it is the same failure the plugin isolation
check exists to catch, one level up.

So: anything needing a plugin lives here. The core job ignores this directory;
the integration job runs it with everything installed.
