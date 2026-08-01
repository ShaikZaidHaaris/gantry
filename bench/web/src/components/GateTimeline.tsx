/** The hero component: the gauntlet as a vertical timeline.
 *
 *  Done gates collapse to one line. The running gate is open and live. Gates
 *  ahead of the front are dimmed and say only what they will ask and what they
 *  cost -- nothing pretends to have a result before it has one.
 */

import type { Gate } from "../api/types";
import { StatusPill, duration } from "./ui";

const MARK: Record<string, string> = {
  passed: "✓",
  refused: "✕",
  abstained: "?",
  failed: "!",
};

function price(cents: number): string {
  return cents === 0 ? "free" : `$${(cents / 100).toFixed(0)}`;
}

export function GateTimeline({ gates, currentGate }: { gates: Gate[]; currentGate: string }) {
  const reachedIndex = gates.findIndex((g) => g.key === currentGate);
  return (
    <div className="card">
      {gates.map((gate, index) => {
        const future = gate.status === "queued" && (reachedIndex < 0 || index > reachedIndex);
        // Counted here, read below. The timeline is for "where is this up to";
        // eleven findings inline turn it into the wall of text the report
        // section exists to replace.
        const blocking = gate.findings.filter(
          (f) => f.severity === "strong" || f.severity === "moderate",
        ).length;
        const noted = gate.findings.length - blocking;
        return (
          <div key={gate.key} className={`gate ${gate.status} ${future ? "future" : ""}`}>
            <div className="mark">{MARK[gate.status] ?? index + 1}</div>
            <div>
              <div className="name">{gate.name}</div>
              <div className="question">{gate.question}</div>

              {gate.status === "running" && (
                <div className="result">Working on it — this takes {gate.eta}.</div>
              )}

              {gate.verdict?.summary && gate.status !== "running" && (
                <div className="result">{gate.verdict.summary}</div>
              )}

              {future && (
                <div className="result">
                  Not started · {price(gate.cost_cents)} · {gate.eta}
                </div>
              )}

              {gate.findings.length > 0 && (
                <div className="tally">
                  {blocking > 0 && (
                    <span className="chip strong">
                      {blocking} to fix
                    </span>
                  )}
                  {noted > 0 && <span className="chip">{noted} noted</span>}
                  {gate.abstained?.length > 0 && (
                    <span className="chip warn">{gate.abstained.length} not judged</span>
                  )}
                </div>
              )}
            </div>
            <div className="meta">
              <StatusPill status={gate.status} />
              {gate.finished_at && (
                <div style={{ marginTop: 6 }}>{duration(gate.started_at, gate.finished_at)}</div>
              )}
            </div>
          </div>
        );
      })}
    </div>
  );
}
