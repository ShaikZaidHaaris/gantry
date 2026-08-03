/** The free data report, what Gate 1 found in somebody's footage.
 *
 *  This is the first thing a contributor gets back, and for most uploads it is
 *  the only thing they will ever read closely. It is laid out as an argument in
 *  three steps rather than as a dump of everything the modules returned:
 *
 *    1. what to fix        the deliverable, ranked, each with a fix
 *    2. also noted         real but not urgent, folded away
 *    3. what we measured   the numbers, each with its n
 *
 *  Nothing here computes. Every number was written by a module upstream, and
 *  the page's whole job is to order it.
 */

import { useState } from "react";
import type { Finding, Gate, Measure } from "../api/types";
import { FindingRow, readable } from "./ui";

/** Ranked. Findings arrive per-module, so an ordering has to be imposed here or
 *  the list reads in alphabetical module order, which is no order at all. */
const RANK: Record<string, number> = { strong: 0, moderate: 1, weak: 2, info: 3 };

function byRank(a: Finding, b: Finding) {
  return (RANK[a.severity] ?? 9) - (RANK[b.severity] ?? 9);
}

function Section({
  title,
  hint,
  children,
}: {
  title: string;
  hint: string;
  children: React.ReactNode;
}) {
  return (
    <section className="report-step">
      <div className="report-step-head">
        <div>
          <h3>{title}</h3>
          <p>{hint}</p>
        </div>
      </div>
      {children}
    </section>
  );
}

function Fold({ label, children }: { label: string; children: React.ReactNode }) {
  const [open, setOpen] = useState(false);
  return (
    <div className="fold">
      <button type="button" className="fold-head" onClick={() => setOpen(!open)} aria-expanded={open}>
        <span className="chev" data-open={open}>
          ▸
        </span>
        {label}
      </button>
      {open && <div className="fold-body">{children}</div>}
    </div>
  );
}

/** A measured rate, drawn as its interval rather than as a number alone.
 *  Two clips in six and two hundred in six hundred are the same 33% and not the
 *  same evidence; the bar is the only place that difference is visible. */
function MeasureBar({ measure }: { measure: Measure }) {
  const pct = (x: number) => `${(Math.max(0, Math.min(1, x)) * 100).toFixed(0)}%`;
  const [lo, hi] = measure.ci ?? [measure.value, measure.value];
  return (
    <div className="measure-bar" title={measure.method ?? undefined}>
      <span className="track">
        <span className="ci" style={{ left: pct(lo), right: `${100 - Math.min(100, hi * 100)}%` }} />
        <span className="dot" style={{ left: pct(measure.value) }} />
      </span>
    </div>
  );
}

function isRate(measure: Measure) {
  return measure.ci !== null && measure.value >= 0 && measure.value <= 1;
}

/** Modules key their measurements by cohort, because most of them compare two
 *  or more. A data report has one cohort, so the prefix is the same string on
 *  every row and carries nothing. Dropped for display, and the rest made
 *  readable: `frozen[dimensions 0-6]` is a lookup key, not a label. The full
 *  key stays on the row as its title, since that is what a lab quoting a number
 *  back at us would cite. */
function shortKey(key: string): string {
  const cut = key.indexOf(".");
  const rest = cut < 0 ? key : key.slice(cut + 1);
  const bracket = rest.match(/^([a-z_]+)\[(.+)\]$/);
  if (bracket) return `${readable(bracket[1])}: ${bracket[2]}`;
  return readable(rest);
}

export function DataReport({ gate }: { gate: Gate }) {
  const findings = [...gate.findings].sort(byRank);
  const fix = findings.filter((f) => f.severity === "strong" || f.severity === "moderate");
  const noted = findings.filter((f) => f.severity === "weak" || f.severity === "info");
  const measures = Object.entries(gate.measures ?? {});

  return (
    <div className="report">
      {/* The count is in the title because this is the answer people came back
          for, and "What to fix" alone reads the same whether there are two
          things or eleven. */}
      <Section
        title={fix.length ? `What to fix (${fix.length})` : "Nothing to fix"}
        hint={
          fix.length
            ? "Ordered by how much they matter. Each one comes from a measurement you can check."
            : "None of the checks suggest changing how you filmed this."
        }
      >
        {fix.length > 0 ? (
          <div className="card pad">
            {fix.map((f) => (
              <FindingRow key={f.code + f.module} finding={f} />
            ))}
          </div>
        ) : (
          <div className="note">
            Every check that had something to read came back clean, which is not the same
            as the data being good.
          </div>
        )}
      </Section>

      {noted.length > 0 && (
        <Section
          title="Also noted"
          hint="Real, but not worth acting on yet. Folded away to keep the list short."
        >
          <Fold label={`${noted.length} further observation${noted.length === 1 ? "" : "s"}`}>
            <div className="card pad">
              {noted.map((f) => (
                <FindingRow key={f.code + f.module} finding={f} />
              ))}
            </div>
          </Fold>
        </Section>
      )}

      {measures.length > 0 && (
        <Section
          title="What we measured"
          hint="Each rate shows how many clips it came from, with a 95% interval."
        >
          <div className="card">
            <div className="table-lite">
              {measures.map(([key, measure]) => (
                <div className="mrow" key={key} title={key}>
                  <span className="mkey">{shortKey(key)}</span>
                  <span className="mval mono">
                    {isRate(measure)
                      ? `${(measure.value * 100).toFixed(0)}%`
                      : measure.value.toFixed(measure.value % 1 === 0 ? 0 : 3)}
                  </span>
                  {isRate(measure) ? <MeasureBar measure={measure} /> : <span />}
                  <span className="mn mono">{measure.n === null ? "" : `n=${measure.n}`}</span>
                </div>
              ))}
            </div>
          </div>
        </Section>
      )}

    </div>
  );
}
