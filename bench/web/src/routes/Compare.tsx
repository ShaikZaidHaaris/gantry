/** The leaderboard — every submission on one benchmark, ranked and paired.
 *
 *  This is the screen a lab lead buys, and the one most likely to be misread,
 *  so two things are designed against that.
 *
 *  **The letter, not the rank.** At the sample sizes a robot evaluation can
 *  afford, the ordering is mostly noise. Entries that nothing separates share a
 *  letter, and that is the column to read: "we came third" means nothing if
 *  third and second are the same letter. A ranked table without this invites
 *  exactly that reading.
 *
 *  **The rung is a control, not a footnote.** Success is the coarsest measure
 *  and the least able to separate anything; the rungs below it are measured on
 *  the same rollouts at no extra cost and routinely separate when success
 *  cannot. Putting them behind a switch, rather than in an appendix, is the
 *  difference between a leaderboard that can answer a question and one that
 *  reports a single number.
 */

import { useState } from "react";
import { Link } from "react-router-dom";
import { useCompare } from "../api/client";
import { Empty, ErrorNote, Skeleton } from "../components/ui";

function pct(x: number): string {
  return `${(x * 100).toFixed(0)}%`;
}

export function Compare() {
  const [rung, setRung] = useState("solved");
  const { data, isPending, error } = useCompare("pick_dual_bottles", rung);

  if (isPending) {
    return (
      <div className="page">
        <div className="card">
          <Skeleton rows={4} />
        </div>
      </div>
    );
  }
  if (error) {
    return (
      <div className="page">
        <ErrorNote error={error} />
      </div>
    );
  }
  if (!data) return null;

  const best = Math.max(0.0001, ...data.entries.map((e) => e.ci[1]));

  return (
    <div className="page">
      <div className="page-head">
        <div>
          <h1>Leaderboard</h1>
          <p>
            {data.benchmark.name} · {data.benchmark.simulator}
            {data.baseline && (
              <>
                {" "}
                · today's model solves {data.baseline.wins} of {data.baseline.n}
              </>
            )}
          </p>
        </div>
      </div>

      {data.entries.length === 0 ? (
        <Empty title="Nothing to rank yet">
          Submissions show up here once their robot test has run.
        </Empty>
      ) : (
        <>
          <div className="rungbar">
            <span className="lab">Rank on</span>
            <div className="segmented">
              {data.rungs.map((name) => (
                <button
                  key={name}
                  type="button"
                  className={name === rung ? "on" : ""}
                  onClick={() => setRung(name)}
                >
                  {name}
                </button>
              ))}
            </div>
          </div>

          <div className="card scroll-x">
            <div className="board">
              <div className="brow head">
                <span>Group</span>
                <span>Submission</span>
                <span className="right">Scenes</span>
                <span className="right">Rate</span>
                <span>95% interval</span>
              </div>
              {data.entries.map((entry) => (
                <div className="brow" key={entry.id}>
                  <span className="group" title="entries sharing a letter are not separated">
                    {entry.group}
                  </span>
                  <Link className="bname" to={`/submissions/${entry.id}`}>
                    {entry.name}
                  </Link>
                  <span className="right mono dim">
                    {entry.wins}/{entry.n}
                  </span>
                  <span className="right mono rate">{pct(entry.rate)}</span>
                  <span className="interval">
                    <span
                      className="span"
                      style={{
                        left: `${(entry.ci[0] / best) * 100}%`,
                        right: `${100 - (entry.ci[1] / best) * 100}%`,
                      }}
                    />
                    <span className="point" style={{ left: `${(entry.rate / best) * 100}%` }} />
                  </span>
                </div>
              ))}
            </div>
          </div>

          <h2>
            Head to head
            <span className="h2-sub">only the scenes where they disagreed tell you anything</span>
          </h2>
          <div className="card pad">
            {data.pairs.length === 0 ? (
              <div className="what">Only one submission so far, so there is nothing to compare it against.</div>
            ) : (
              data.pairs.map((pair) => {
                const left = data.entries.find((e) => e.id === pair.left);
                const right = data.entries.find((e) => e.id === pair.right);
                return (
                  <div className="headtohead" key={pair.left + pair.right}>
                    <div className="who">
                      {left?.name} <span className="v">vs</span> {right?.name}
                    </div>
                    <div className="numbers mono">
                      <span>{pair.shared_scenes} shared scenes</span>
                      <span className="dim">{pair.agreed} agreed</span>
                      <span>
                        {pair.left_only} / {pair.right_only} disagreements
                      </span>
                      <span className={pair.separated ? "sep" : "dim"}>
                        p={pair.p_value.toFixed(4)}
                      </span>
                    </div>
                    <div className="reading">
                      {pair.separated ? (
                        <>
                          Separated on <b>{rung}</b>. Of the {pair.left_only + pair.right_only}{" "}
                          scenes where they differed, {Math.max(pair.left_only, pair.right_only)} went
                          the same way.
                        </>
                      ) : (
                        <>
                          Not separated on <b>{rung}</b>. {pair.agreed} of {pair.shared_scenes}{" "}
                          scenes came out the same for both, and{" "}
                          {pair.left_only + pair.right_only} disagreements is too few to tell them
                          apart. That says more about how many scenes were run than about the
                          datasets.
                        </>
                      )}
                    </div>
                  </div>
                );
              })
            )}
          </div>

          <p className="board-foot">
            Each pair is compared scene by scene, using only the scenes both submissions ran.
            Scenes they both won or both lost still count in the rates, but they are left out of
            the test, because they say nothing about which is better. Leaving them in is why
            comparing the two rates on their own needs several times as many scenes to see the
            same difference.
          </p>
        </>
      )}
    </div>
  );
}
