# Bench UI — design language

Audience: research labs. The reference class is Stripe Dashboard / Linear /
Vercel — quiet, dense, professional tools — not a themed "instrument panel".

## Principles

1. **Summary before evidence.** Every page opens with one sentence and at most
   four numbers. Detail exists behind a disclosure, never alongside the summary.
2. **One finding, one row.** The diagnosis is a list grouped by check, with the
   cohorts as chips on the row — never one card per (finding × cohort). Twenty
   cards saying the same sentence four ways is noise wearing a border.
3. **Tables for records, prose for verdicts.** Runs, datasets, rungs are rows
   with right-aligned tabular numerals. Sentences are reserved for what the
   numbers do not say.
4. **The pipeline is the narrative.** The overview is a stepper — ingest →
   convert → split → train → evaluate → verdict — because that is the order a
   lab thinks in and the order the work actually happened.
5. **Colour is status, nothing else.** One brand blue, used for interaction and
   the current step. Red/amber/green only ever mean severity or outcome. The
   canvas is near-white; hierarchy comes from type weight and spacing, not hue.
6. **Nothing decorative.** No serif display, no philosophy sidebar, no
   animation beyond 150 ms of ease. The house rules survive as behaviour
   (intervals everywhere, abstentions rendered) — not as copy on the wall.

## Tokens

    canvas   #F7F8FA      surface #FFFFFF     border #E4E7EC / #D0D5DD
    text     #101828      secondary #475467   tertiary #98A2B3
    accent   #2456D6      (interaction, active step, CI marker)
    danger   #B42318/#FEF3F2   warn #B54708/#FFFAEB   ok #067647/#ECFDF3
    type     system sans 13.5/1.55; mono ui-monospace 12 for numerals and codes
    radius   8; rows 40px; container 1024px; spacing on an 8px grid
