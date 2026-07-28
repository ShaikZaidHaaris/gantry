# Writing a plugin

A plugin is an ordinary Python package. It depends on `gantry-core`, implements
one contract, declares an entry point, and passes the conformance kit for its
plane. Nothing in Gantry needs to know your plugin exists in order to run it —
only in order to *resolve* it, which it does from your descriptor.

The kits are the specification. `gantry checks <plane>` prints every check for
any plane, and each one corresponds to one sentence in that contract's
docstring. If the prose and the kit ever disagree, the prose is the spec and the
kit is the bug.

## The shape

```
my-plugin/
  pyproject.toml
  src/my_plugin/__init__.py
  tests/test_conformance.py
```

```toml
[project]
name = "gantry-connector-myformat"
dependencies = ["gantry-core"]

[project.entry-points."gantry.connectors"]
myformat = "my_plugin:MyConnector"
```

Groups, one per plane: `gantry.connectors`, `gantry.embodiments`,
`gantry.policies`, `gantry.evaluators`, `gantry.feedback`, `gantry.adapters`,
`gantry.retargeters`.

Installing the package is all it takes to make it appear in `gantry list`.

## The one test that matters

```python
from gantry.conformance import check_connector

def test_conforms():
    verdict = check_connector(MyConnector("fixture.dat"), strict=True)
    assert verdict.ok, verdict.explain()
```

This is the same kit the first-party plugins run, and you inherit new checks
when you upgrade. There is deliberately no trusted tier.

## Declaring things honestly

Everything below is a claim the resolver plans from. A wrong claim does not
produce an error — it produces a run that looks fine.

**Capabilities are load-bearing.** A connector saying `stage_events=True` will be
handed a funnel diagnosis. One saying `False` will have that analysis refused on
its behalf. The kit checks both directions: overstating is caught, and so is
understating, because the second quietly denies people analyses your data could
support.

**Describe your channels.** Units, frames, dimension labels, and meaning. A
channel with none of these can only be matched by name and width, which is
matching by coincidence. `strict=True` turns that into a failure, and you should
run strict.

**Say what a transform costs.** An adapter or retargeter that discards
information and declares no loss is the most dangerous thing you can publish
here: it produces plausible output, and the run's provenance tells every later
reader that nothing was given up. Both kits check this specifically.

**Declare isolation if you need it.** If your dependencies conflict with
anything, set `isolation="venv"` or `"container"` in your descriptor. Gantry will
run you in your own interpreter rather than importing your stack into someone
else's — which is what you were asking for.

## Testing against the fixtures

`gantry.fixtures` ships generated episodes with planted defects and a ground
truth. Use them rather than inventing data:

```python
from gantry.fixtures import make_clean, make_defective, make_duration_confound
```

If you are writing a feedback module, `make_duration_confound()` is the important
one. It is a suite where nothing is wrong and everything varies in length, and a
module that reports a finding on it is measuring duration. Getting that suite to
stay silent is harder and more valuable than getting the defective suites to
speak.

## Versioning

Pin the contract you implement — `"connector@1.0"` — and the resolver checks the
major matches before anything runs. Contracts are frozen at v1 and a break
requires a major bump; `tests/test_frozen_v1.py` in core is what makes that
deliberate rather than accidental.

## What not to do

**Do not coerce.** If the data does not fit, refuse. A connector that guesses
units, a policy that pads an action vector, a feedback module that degrades to a
narrower analysis without saying so — these are the failure this framework
exists to prevent, and they all look like helpfulness.

**Do not report a number without its n.** Use `Measurement`. A bare float has no
sample size, no interval, and no way to be argued with.

**Do not average across provenance.** `RunSet.comparable(holding=[...])` exists
because a simulated arm and a real humanoid in one mean is a category error, and
it is one that reads as a result.
