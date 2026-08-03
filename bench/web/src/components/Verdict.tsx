/** The verdict, what the robot test concluded, and what it refused to.
 *
 *  This is the screen a lab lead screenshots into Slack, so it is built as a
 *  document rather than a dashboard: one statement, the evidence under it, and
 *  the limits under that. Three sections, read top to bottom, and nothing on
 *  the page is computed here, every rate, interval and p-value was produced by
 *  the gate.
 *
 *  The third section is not a disclaimer. "What this does not say" carries the
 *  refusals the gate made, no control arm, rungs nobody measured, a comparison
 *  the sample size cannot support, and those are the difference between a
 *  result and a number. A verdict page that shows only the first two sections
 *  is the thing this product exists not to be.
 */

import type { Cell, Gate, LadderRow, Paired } from "../api/types";
import { sentence } from "./ui";

const NOT_MEASURED = "not measured";

function pct(x: number): string {
  return `${(x * 100).toFixed(0)}%`;
}

/** The funnel bar. Width is the rate; the lighter span behind it is the 95%
 *  interval, so four scenes out of fifty never reads as firmly as forty. */
function Bar({ cell }: { cell: Cell }) {
  if (!cell.measured) return <span className="rung-bar empty">{NOT_MEASURED}</span>;
  const [lo, hi] = cell.ci;
  return (
    <span className="rung-bar">
      <span className="ci" style={{ left: pct(lo), right: `${100 - hi * 100}%` }} />
      <span className="fill" style={{ width: pct(cell.rate) }} />
    </span>
  );
}

function Cellbox({ cell }: { cell: Cell }) {
  if (!cell.measured) {
    return (
      <span className="rung-num none" title="this arm reported no events for this rung">
        -
      </span>
    );
  }
  return (
    <span className="rung-num">
      <b>{pct(cell.rate)}</b>
      <span className="of">
        {cell.wins}/{cell.n}
      </span>
    </span>
  );
}

function Ladder({ rows, arms }: { rows: LadderRow[]; arms: string[] }) {
  const treatment = arms[0];
  const others = arms.slice(1);
  return (
    <div className="card scroll-x">
      <div className="ladder">
        <div className="lrow head">
          <span>Rung</span>
          <span>{treatment}</span>
          <span />
          {others.map((name) => (
            <span key={name}>{name}</span>
          ))}
          <span className="right">Paired</span>
        </div>
        {rows.map((row) => {
          const paired = others
            .map((name) => row[`vs ${name}`] as Paired | undefined)
            .find(Boolean);
          return (
            <div className="lrow" key={row.rung}>
              <span className="rung-name">{row.rung}</span>
              <Cellbox cell={row.arms[treatment]} />
              <Bar cell={row.arms[treatment]} />
              {others.map((name) => (
                <Cellbox key={name} cell={row.arms[name]} />
              ))}
              <span className="right mono paired">
                {!paired || !paired.measured ? (
                  <span className="none" title="one of the arms never measured this rung">
                    not comparable
                  </span>
                ) : (
                  <>
                    <span className={paired.separated ? "sep" : ""}>
                      p={paired.p_value.toFixed(3)}
                    </span>
                    <span className="disagree">
                      {paired.a_only}↑ {paired.b_only}↓
                    </span>
                  </>
                )}
              </span>
            </div>
          );
        })}
      </div>
    </div>
  );
}

export function Verdict({ gate }: { gate: Gate }) {
  const detail = gate.detail ?? {};
  const rows = detail.ladder ?? [];
  const arms = detail.arms ?? [];
  const limits = gate.findings.filter((f) => f.severity === "strong");
  const notes = gate.findings.filter((f) => f.severity !== "strong");

  if (!rows.length) {
    return (
      <div className="note warn">
        <b>{gate.verdict?.summary ?? "the robot test did not produce a ladder"}</b>
      </div>
    );
  }

  return (
    <div className="report">
      <section className="report-step">
        <div className="statement">
          <span className="eyebrow">Verdict</span>
          <p>{sentence(gate.verdict?.summary ?? "")}</p>
        </div>
      </section>

      <section className="report-step">
        <div className="report-step-head">
          <div>
            <h3>How far it got</h3>
            <p>
              Every rung comes from the same rollouts, so none of this costs extra to measure.
              The bar behind each rate is its 95% interval. <b>Paired</b> counts only the scenes
              where the two arms disagreed, since a scene they both won or both lost tells you
              nothing about which is better.
            </p>
          </div>
        </div>
        <Ladder rows={rows} arms={arms} />
      </section>

      <section className="report-step">
        <div className="report-step-head">
          <div>
            <h3>What this does not say</h3>
            <p>
              What this run could not settle. Worth reading alongside the numbers above.
            </p>
          </div>
        </div>
        <div className="card pad">
          {limits.length === 0 && notes.length === 0 ? (
            <div className="what">Nothing was held back. Every arm and every rung was measured.</div>
          ) : (
            <ul className="what-list">
              {[...limits, ...notes].map((f) => (
                <li key={f.code}>{sentence(f.summary)}</li>
              ))}
            </ul>
          )}
        </div>
      </section>
    </div>
  );
}
