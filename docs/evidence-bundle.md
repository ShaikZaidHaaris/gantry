# The evidence bundle

Every submission's verdict can leave the bench as a bundle of tables:

```
GET /api/submissions/{id}/evidence        ->  gantry_evidence_{id}.zip
```

readable exactly where the report page is readable — your own submissions,
the seeded worked examples, anything published to the leaderboard — and also
offered as a download button on the report page itself.

## Why it exists

A verdict you cannot interrogate is a claim. The gates already record their
working — per-clip signal checks, the ladder's rollout counts, every finding
and measurement — but until now that working was readable by the product's
screens and by nothing else. The bundle is the same working in a shape any
query engine reads, so "which held-out clips beat their shuffled control by
the least" is a `SELECT`, not a support request.

The intended first reader is [Bagel](https://github.com/Extelligence-ai/bagel),
the talk-to-your-robotics-data MCP server: point it at an unzipped bundle and
it serves each table as a topic, with the owning gate's verdict quoted in the
topic description. Anything else that reads CSV — DuckDB, pandas, a
spreadsheet — works the same day.

## The contract

One `manifest.json` plus one CSV per table, zipped flat. The manifest is the
bundle's word on its own contents:

- `magic: "GANTRY_EVIDENCE"` and `format_version: 1` — what a reader sniffs.
- the submission header, the dataset's detected shape, and each gate's
  one-line verdict.
- `tables`: for every table present, its file, its row count, and a
  per-column type map (`string · int · float · bool · timestamp · json`).
  CSV has no types; the manifest is where they live, and a reader casts
  instead of sniffing.

The tables, all optional — a gate that has not run has no table, because an
empty file would read as "ran and found nothing", which is a different claim:

| table | one row per | from |
|---|---|---|
| `gates` | gate of the gauntlet | status, verdict, timing, cost |
| `findings` | finding raised | code, severity, summary, prescription |
| `measures` | quantity measured | value, n, CI bounds, units, method |
| `abstained` | module that declined | with its reason, never dropped |
| `signal_pairs` | held-out clip | G2's fit error, yours vs actions-detached |
| `ladder` | rung × arm | wins, n, rate, CI; unmeasured is null, not zero |
| `ladder_vs_baseline` | rung | paired scenes, test detail as JSON |
| `events` | timeline entry | the one genuinely temporal table |
| `coach` | piece of advice | `fix` rows carry the finding code they answer |

## What it deliberately never contains

- **Nothing computed.** No margins, deltas or rankings are derived on the way
  out. The exporter copies what the gates recorded; the arithmetic belongs to
  whoever queries the bundle, where it can be audited.
- **No contact address.** The bundle is built with `owner=False`
  unconditionally: an export is made to leave, and the uploader's email is
  not part of the evidence.
- **Nothing the report page would not show.** It is built from the same dict
  the page renders, so the bundle can never say more than the screen does.

## Reading it with Bagel

```bash
unzip gantry_evidence_<id>.zip -d verdict/
# then, with Bagel's MCP server connected to Claude (or any MCP client):
#   "Describe the data source at ./verdict"
#   "Which held-out clips barely beat the shuffled control?"
#   "Where does the ladder break, my arm vs baseline?"
```

Bagel resolves the directory by the manifest's magic, types every column from
the manifest's map, and answers with DuckDB SQL it shows you — the same
auditability the bench practices, one tool downstream.

Code: `bench/api/app/evidence.py` (the exporter and the schema),
`bench/api/tests/test_evidence.py` (the fidelity and reach tests).
