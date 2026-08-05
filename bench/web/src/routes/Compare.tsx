/** The leaderboard, every submission on one benchmark, ranked and paired.
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
 *  **One rung: solved.** There was a picker here offering every intermediate
 *  rung -- lifted, moved, reached and their variants -- on the argument that
 *  the finer rungs separate entries when success cannot. True, and removed at
 *  the owner's call: seven unexplained switches on the most public screen read
 *  as complexity, not power. The board now ranks on solved, full stop. The
 *  per-rung rates still exist on every submission's own verdict page, and the
 *  API still takes a rung parameter, so the picker is one state variable away
 *  if it ever earns its place back.
 */

import { useState } from "react";
import { Link } from "react-router-dom";
import { useCompare } from "../api/client";
import { Empty, ErrorNote, Skeleton } from "../components/ui";

function pct(x: number): string {
  return `${(x * 100).toFixed(0)}%`;
}

export function Compare() {
  const rung = "solved";
  const [left, setLeft] = useState<string | null>(null);
  const [right, setRight] = useState<string | null>(null);
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

  // Which two are being compared. Defaults to the top two, because that is the
  // question somebody arriving at a leaderboard already has, and every other
  // pair is one selection away. Every pair is computed by the server, so
  // changing the choice costs nothing and never refetches.
  const a = left ?? data.entries[0]?.id ?? "";
  const b = right ?? data.entries.find((e) => e.id !== a)?.id ?? "";
  const chosen =
    a && b
      ? data.pairs.find(
          (p) => (p.left === a && p.right === b) || (p.left === b && p.right === a),
        )
      : undefined;
  const named = (id: string) => data.entries.find((e) => e.id === id)?.name ?? id;

  /** The pair is stored as the server ordered it, which is by rate. Read it back
   *  in the order the reader picked, or "3 went the other way" points at the
   *  wrong submission. */
  const flipped = chosen ? chosen.left !== a : false;
  const forA = chosen && chosen.shared_scenes > 0 ? (flipped ? chosen.right_only : chosen.left_only) : 0;
  const forB = chosen && chosen.shared_scenes > 0 ? (flipped ? chosen.left_only : chosen.right_only) : 0;

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
        <Empty
          title="Nothing to rank yet"
          action={
            <Link className="btn primary" to="/submissions/new">
              Submit a dataset
            </Link>
          }
        >
          A dataset appears here once it has been through the robot test, the
          last check, where a policy is actually trained on your data and run on
          the robot. The three checks before it tell you whether that is worth
          doing, and none of them need one.
        </Empty>
      ) : (
        <>
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

          {data.entries.length < 2 ? (
            <div className="card pad">
              <div className="what">
                Only one submission so far, so there is nothing to compare it against.
              </div>
            </div>
          ) : (
            <div className="card pad">
              <div className="picker">
                <select value={a} onChange={(e) => setLeft(e.target.value)} aria-label="First submission">
                  {data.entries.map((entry) => (
                    <option key={entry.id} value={entry.id} disabled={entry.id === b}>
                      {entry.name}
                    </option>
                  ))}
                </select>
                <button
                  type="button"
                  className="swap"
                  title="Swap them"
                  onClick={() => {
                    setLeft(b);
                    setRight(a);
                  }}
                >
                  against
                </button>
                <select value={b} onChange={(e) => setRight(e.target.value)} aria-label="Second submission">
                  {data.entries.map((entry) => (
                    <option key={entry.id} value={entry.id} disabled={entry.id === a}>
                      {entry.name}
                    </option>
                  ))}
                </select>
              </div>

              {!chosen || chosen.shared_scenes === 0 ? (
                <div className="what">
                  These two have no scenes in common on this rung, so there is nothing to
                  compare. That happens when one of them was never measured on it.
                </div>
              ) : (
                // The wrapper is what the `.numbers` and `.reading` rules hang
                // off. Rendering them bare dropped the spacing and ran the
                // figures together into one string.
                <div className="headtohead bare">
                  <div className="numbers mono">
                    <span>{chosen.shared_scenes} shared scenes</span>
                    <span className="dim">{chosen.agreed} agreed</span>
                    <span>
                      {forA} / {forB} disagreements
                    </span>
                    <span className={chosen.separated ? "sep" : "dim"}>
                      p={chosen.p_value.toFixed(4)}
                    </span>
                  </div>
                  <div className="reading">
                    {chosen.separated ? (
                      <>
                        Separated on <b>{rung}</b>. Of the {forA + forB} scenes where they
                        differed, {Math.max(forA, forB)} went to{" "}
                        <b>{named(forA >= forB ? a : b)}</b>.
                      </>
                    ) : (
                      <>
                        Not separated on <b>{rung}</b>. {chosen.agreed} of{" "}
                        {chosen.shared_scenes} scenes came out the same for both, and{" "}
                        {forA + forB} disagreements is too few to tell them apart. That says
                        more about how many scenes were run than about the datasets.
                      </>
                    )}
                  </div>
                </div>
              )}
            </div>
          )}

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
