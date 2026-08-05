/** The free data report, what Gate 1 found in somebody's footage.
 *
 *  This is the first thing a contributor gets back, and for most uploads it is
 *  the only thing they will ever read closely. It is laid out as an argument in
 *  two steps rather than as a dump of everything the modules returned:
 *
 *    1. what to fix        the deliverable, ranked, each with a fix
 *    2. also noted         real but not urgent, folded away
 *
 *  Nothing here computes. Every number was written by a module upstream, and
 *  the page's whole job is to order it.
 */

import { useState } from "react";
import type { Finding, Gate } from "../api/types";
import { FindingRow } from "./ui";

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




export function DataReport({
  gate,
  notes = {},
}: {
  gate: Gate;
  notes?: Record<string, { say?: string; do?: string }>;
}) {
  const findings = [...gate.findings].sort(byRank);
  const fix = findings.filter((f) => f.severity === "strong" || f.severity === "moderate");
  const noted = findings.filter((f) => f.severity === "weak" || f.severity === "info");

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
              <FindingRow key={f.code + f.module} finding={f} note={notes[f.code] ?? {}} />
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
                <FindingRow key={f.code + f.module} finding={f} note={notes[f.code] ?? {}} />
              ))}
            </div>
          </Fold>
        </Section>
      )}

      {/* "What we measured" was here: the raw measure table with intervals.
          Removed at the owner's call, alongside the verdict page's "What this
          does not say". The findings above each cite their measurement, and
          the full numbers stay in the gate record for anyone who asks. */}
    </div>
  );
}
