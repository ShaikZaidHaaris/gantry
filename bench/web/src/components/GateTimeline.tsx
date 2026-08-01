/** The hero component: the gauntlet as a vertical timeline.
 *
 *  Done gates collapse to one line. The running gate is open and live. Gates
 *  ahead of the front are dimmed and say only what they will ask and what they
 *  cost -- nothing pretends to have a result before it has one.
 *
 *  The running gate is the whole reason this component exists. Intake finishes
 *  in under a second, but the signal check runs for ten minutes and the robot
 *  test for hours, and a bar that has not moved for four hours is
 *  indistinguishable from one that has died. So a running gate shows what it is
 *  doing, how far through it is when that is knowable, and an honest
 *  indeterminate bar when it is not.
 */

import type { Gate, Progress } from "../api/types";
import { Elapsed, StatusPill, duration } from "./ui";

const MARK: Record<string, string> = {
  passed: "✓",
  refused: "✕",
  abstained: "?",
  failed: "!",
};

function price(cents: number): string {
  return cents === 0 ? "free" : `$${(cents / 100).toFixed(0)}`;
}

/** Live position, falling back to whatever the record last stored.
 *
 *  The fallback is what makes a reload during a long run survivable: the stream
 *  has nothing to replay for a page that just opened, so without it a user who
 *  refreshes an hour into the robot test sees an empty bar and concludes it
 *  restarted. */
function Running({ gate, live }: { gate: Gate; live: Progress | null }) {
  const stored = "phase" in gate.progress ? (gate.progress as Progress) : null;
  const at = live && (!live.gate || live.gate === gate.key) ? live : stored;

  if (!at) {
    return <div className="result">Working on it — this takes {gate.eta}.</div>;
  }

  const known = typeof at.current === "number" && typeof at.total === "number" && at.total > 0;
  const fraction = known ? Math.min(1, (at.current as number) / (at.total as number)) : 0;

  return (
    <div className="running">
      <div className="phase">
        <span className="what">{at.phase}</span>
        {at.note && <span className="note-i">{at.note}</span>}
        <span className="spacer" />
        <span className="mono count">
          {known ? `${at.current} / ${at.total}` : `takes ${gate.eta}`}
        </span>
      </div>
      <div className={`bar ${known ? "" : "indeterminate"}`}>
        <i style={known ? { width: `${fraction * 100}%` } : undefined} />
      </div>
    </div>
  );
}

export function GateTimeline({
  gates,
  currentGate,
  live,
}: {
  gates: Gate[];
  currentGate: string;
  live?: Progress | null;
}) {
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

              {gate.status === "running" && <Running gate={gate} live={live ?? null} />}

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
                  {blocking > 0 && <span className="chip strong">{blocking} to fix</span>}
                  {noted > 0 && <span className="chip">{noted} noted</span>}
                  {gate.abstained?.length > 0 && (
                    <span className="chip warn">{gate.abstained.length} not judged</span>
                  )}
                </div>
              )}
            </div>
            <div className="meta">
              <StatusPill status={gate.status} />
              <div style={{ marginTop: 6 }}>
                {gate.finished_at ? (
                  duration(gate.started_at, gate.finished_at)
                ) : gate.status === "running" ? (
                  <Elapsed since={gate.started_at} />
                ) : (
                  ""
                )}
              </div>
            </div>
          </div>
        );
      })}
    </div>
  );
}
