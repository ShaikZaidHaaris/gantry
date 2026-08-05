/** The verdict and the evidence for it.
 *
 *  This is the screen a lab lead screenshots into Slack, so it is built as a
 *  document rather than a dashboard: one statement and the ladder under it,
 *  read top to bottom, and nothing on the page is computed here; every rate,
 *  interval and p-value was produced by the gate.
 *
 *  There used to be a third section, "What this does not say", listing the
 *  measurements the run could not make. Removed at the owner's call: the same
 *  information still exists in the gate's findings and the unmeasured rungs
 *  already render as "not measured" in the ladder itself, so the section
 *  restated what the table above it showed.
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

    </div>
  );
}
