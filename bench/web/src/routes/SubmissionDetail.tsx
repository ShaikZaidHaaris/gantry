/** The hero screen: one submission, its gauntlet, live. */

import { Link, useParams } from "react-router-dom";
import { useRetryGate, useStartGate, useSubmission, useSubmissionEvents } from "../api/client";
import { BudgetPanel } from "../components/BudgetPanel";
import { AnswerBanner } from "../components/Answer";
import { Publish } from "../components/Publish";
import { DataReport } from "../components/DataReport";
import { Verdict } from "../components/Verdict";
import { Versions } from "../components/Versions";
import { GateTimeline } from "../components/GateTimeline";
import { ErrorNote, Skeleton, StatusPill, ago, bytes, readable, submissionStatus } from "../components/ui";

/** What an event was, in words. The log stores `gate.finished` because that is
 *  a stable key to branch on; a person reading their own submission's history
 *  should see what happened, not the name of the row. */
const HAPPENED: Record<string, string> = {
  "submission.created": "Submission created",
  "dataset.uploaded": "Dataset uploaded",
  "schema.confirmed": "Channel meanings confirmed",
  "gate.queued": "Queued",
  "gate.started": "Started",
  "gate.finished": "Finished",
  "gate.retried": "Started again",
};

function happened(kind: string): string {
  return HAPPENED[kind] ?? readable(kind);
}

/** Whether to offer the robot-test configurator at all.
 *
 *  It prices a run in scenes, hours and dollars and then offers to start one,
 *  which is an operator's decision rather than a contributor's. A visitor who
 *  has just been told their data is worth training on does not need a $39
 *  compute estimate in front of the answer, and cannot act on it anyway.
 *
 *  Kept behind a flag rather than deleted, because ops still runs from it. Set
 *  VITE_SHOW_RUN_PANEL=1 at build time to get it back.
 */
const SHOW_RUN_PANEL = import.meta.env.VITE_SHOW_RUN_PANEL === "1";

const GATE_NAMES: Record<string, string> = {
  g0: "Intake",
  g1: "Data report",
  g2: "Signal check",
  g3: "Robot test",
};

function gateName(key: string): string {
  return GATE_NAMES[key] ?? key;
}

/** 1,284 as 1,284 and 17,371 as 17.4K: the tile contract's auto-compact.
 *  Proportional figures on purpose; tabular-nums is for columns, and a lone
 *  display-size number set tabular reads loose. */
function compact(n: number | undefined): string {
  if (n === undefined || n === null) return "-";
  if (n < 10000) return n.toLocaleString("en-US");
  if (n < 1e6) return `${(n / 1e3).toFixed(1).replace(/\.0$/, "")}K`;
  return `${(n / 1e6).toFixed(1).replace(/\.0$/, "")}M`;
}

/** The newest event of one kind, for the states the event log narrates --
 *  a fetch in flight, a fetch that failed -- where the gates have nothing to
 *  say because the gates do not exist yet. */
function lastEvent(sub: { events?: { kind: string; [k: string]: unknown }[] }, kind: string) {
  return [...(sub.events ?? [])].reverse().find((e) => e.kind === kind);
}

export function SubmissionDetail() {
  const { id } = useParams();
  const { data, isPending, error } = useSubmission(id);
  const live = useSubmissionEvents(id);
  const start = useStartGate(id ?? "");
  const retry = useRetryGate(id ?? "");

  if (isPending) {
    return (
      <div className="page">
        <div className="card">
          <Skeleton rows={5} />
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

  const detected = data.dataset?.detected;
  const report = data.gates.find((g) => g.key === "g1");
  // The gate whose price is a choice, if it has not run. Not simply the next
  // paid gate: the signal check is fixed work at a fixed price, and putting a
  // trial slider on it would offer a decision that does not exist. The budget
  // question also belongs here rather than in the upload wizard, nobody should
  // pick a trial count before knowing their data is readable, and a price
  // quoted before intake is a price for work that may never happen.
  const sized = data.gates.find((g) => g.sized && g.status === "queued");
  const robot = data.gates.find((g) => g.key === "g3");

  //: Whether anything is still moving. `queued` counts: a gate waiting for a
  //: worker is a gate this page will change on its own, and the timeline is the
  //: only place that says so.
  const stillRunning = data.gates.some((g) => g.status === "queued" || g.status === "running");

  //: May this reader act on this page, or only read it? Demos and other
  //: people's published results both render the report without the controls;
  //: a Run-again button that can only answer 404 is worse than no button.
  const mine = data.mine !== false && !data.demo;

  //: Rendered above the findings while a check is live and below them after,
  //: rather than duplicated, so the two positions cannot drift apart.
  const progress = (
    <>
      <h2>Progress</h2>
      <GateTimeline
        gates={data.gates}
        currentGate={data.current_gate}
        live={live}
        onStart={mine ? (gate) => start.mutate({ gate }) : undefined}
        starting={start.isPending}
        onRetry={mine ? (gate) => retry.mutate(gate) : undefined}
        retrying={retry.isPending}
      />
    </>
  );

  return (
    <div className="page">
      <div className="page-head">
        <div>
          <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
            <h1>{data.name}</h1>
            <StatusPill {...submissionStatus(data)} />
          </div>
          <p>
            {data.benchmark?.name} · {data.benchmark?.simulator}
            {data.dataset && ` · v${data.dataset.version} · ${bytes(data.dataset.bytes)}`} ·
            created {ago(data.created_at)}
          </p>
        </div>
        <div className="spacer" />
        {/* A sample is not one of yours, so "All submissions" would be a lie
            about where you came from and where this sits. */}
        <Link className="btn" to="/">
          {data.demo ? "Back to the start" : "All submissions"}
        </Link>
      </div>

      {/* Said once, at the top, before any of the numbers. Somebody who arrived
          here from the front page needs to know these are not their results
          before they read a verdict, not after. */}
      {data.demo && (
        <div className="demo-note">
          <b>A worked example.</b>
          <span>
            This is a finished run from the experiment this benchmark was built against,
            kept here so you can see what comes out before uploading anything. It belongs
            to nobody and cannot be changed. Your own uploads are private to you.
          </span>
        </div>
      )}

      {/* A fetch in flight is the whole page: there are no gates yet, so
          nothing else here has anything to say. The event stream flips this
          card into the normal pipeline view the moment the archive lands. */}
      {data.status === "fetching" && (
        <div className="note">
          <b>
            Pulling {String(lastEvent(data, "fetch.queued")?.repo ?? "your dataset")} from
            Hugging Face.
          </b>{" "}
          The download runs on our side and this page updates itself; the first check
          starts the moment the archive lands. Big datasets take a few minutes.
        </div>
      )}
      {data.status === "draft" && lastEvent(data, "fetch.failed") && (
        <div className="note danger">
          <b>The fetch did not land.</b>{" "}
          {String(lastEvent(data, "fetch.failed")?.reason ?? "")} You can paste a
          corrected link from the new-submission page, or upload the archive directly.
        </div>
      )}

      {/* The answer first. Everything below it is the reasoning behind the
          answer or reference material about the upload, and both were being
          read before the result they explain. */}
      <AnswerBanner submission={data} />

      {/* The takeaway, directly under the answer: what to do about it. It is
          the one thing on this page a model wrote rather than a gate measured,
          so it is labelled as generated and the sections below stay the ground
          truth it must be checked against. */}
      {(data.coach?.points?.length ?? 0) > 0 && (
        <>
          <h2>How to improve this dataset</h2>
          <div className="card pad">
            <ul className="coach-list">
              {data.coach!.points!.slice(0, 3).map((point, i) =>
                typeof point === "string" ? (
                  <li key={i}>{point}</li>
                ) : (
                  <li key={i}>
                    <b>{point.title}</b>
                    {point.detail && <p>{point.detail}</p>}
                  </li>
                ),
              )}
            </ul>
          </div>
        </>
      )}

      {/* Progress leads while there is progress to watch, and gets out of the
          way once there is not.
          While a check is running, the timeline is the page: it is the only
          thing changing and the only reason to keep the tab open. The moment
          everything has settled it becomes the least interesting thing here,
          and leaving it on top pushed the findings, which is what the visitor
          came back for, below the fold behind a list of ticks. */}
      {stillRunning && progress}

      {detected?.episodes ? (
        <>
        <h2>What we found in the upload</h2>
        {/* Stat tiles, per the house dataviz rules: label in sentence case,
            value semibold in the same sans as everything else, proportional
            figures, compacted past four digits. No chart, because these are
            identities and magnitudes with nothing to compare against. */}
        <div className="stats" style={{ marginBottom: 18 }}>
          <div className="stat">
            <span className="stat-label">Episodes</span>
            <span className="stat-value">{compact(detected.episodes)}</span>
          </div>
          <div className="stat">
            <span className="stat-label">Frames</span>
            <span className="stat-value">{compact(detected.frames)}</span>
          </div>
          <div className="stat">
            <span className="stat-label">Frame rate</span>
            <span className="stat-value">
              {detected.fps}
              <span className="stat-unit">fps</span>
            </span>
          </div>
          <div className="stat">
            <span className="stat-label">Videos</span>
            <span className="stat-value">{compact(detected.videos)}</span>
          </div>
          <div className="stat">
            <span className="stat-label">Channels</span>
            <span className="stat-value">{(detected.channels ?? []).length}</span>
          </div>
          {detected.tasks && detected.tasks.length > 0 && (
            <div className="stat">
              <span className="stat-label">Distinct instructions</span>
              <span className="stat-value">{detected.tasks.length}</span>
            </div>
          )}
        </div>
        </>
      ) : null}

      {/* The report appears only once the gate that produces it has finished.
          Nothing on this page pretends to have a result before it has one. */}
      {report && report.status !== "queued" && report.status !== "running" && (
        <>
          <h2>
            Data report
            <span className="h2-sub">
              what your footage is like, before anything is trained on it
            </span>
          </h2>
          <DataReport gate={report} notes={data.coach?.fixes ?? {}} />
        </>
      )}

      {/* `passed` only. The robot test returns just "failed" or "passed"
          (grep '"status":' in robot.py), so the `abstained` arm this used to
          carry was unreachable. A failed run has no ladder to draw, and the
          timeline already says it broke on our side and should not be charged
          for. Inventing a verdict section for it would show an empty table
          where a result belongs. */}
      {robot && robot.status === "passed" && (
        <>
          <h2>
            Robot test
            <span className="h2-sub">closed-loop, on scenes the policy has never seen</span>
            <Link className="h2-link" to={`${data.demo ? "/samples" : "/submissions"}/${data.id}/verdict`}>
              Open as a document →
            </Link>
          </h2>
          <Verdict gate={robot} />
          {mine && <Publish submission={data} />}
        </>
      )}

      {SHOW_RUN_PANEL && mine && sized && report?.status === "passed" && (
        <>
          <h2>
            What to run next
            <span className="h2-sub">what a budget can conclude, before it is spent</span>
          </h2>
          <BudgetPanel
            benchmark={data.benchmark?.key}
            gateName={sized.name}
            gateKey={sized.key}
            submissionId={data.id}
          />
        </>
      )}

      {/* Finished: the timeline is now history, so it sits under the findings
          it produced rather than in front of them. */}
      {!stillRunning && progress}

      {/* Below the result, not above it. Re-uploading is something you do
          *because* of an answer, so the control for it follows the answer
          rather than sitting between the reader and it. */}
      {id && mine && <Versions id={id} />}

      {data.events && data.events.length > 0 && (
        <>
          <h2>Activity</h2>
          <div className="card pad">
            {data.events
              .slice()
              .reverse()
              .map((event, i) => (
                <div
                  key={i}
                  style={{
                    display: "flex",
                    gap: 12,
                    padding: "5px 0",
                    fontSize: 12.5,
                    color: "var(--text-2)",
                  }}
                >
                  <span className="mono" style={{ color: "var(--text-3)", minWidth: 84 }}>
                    {ago(String(event.ts))}
                  </span>
                  {/* What happened and to which stage, and nothing else. The
                      verdict text used to be repeated here from the stored
                      event payload, which duplicated the timeline above and,
                      because events are history, went on showing whatever the
                      wording used to be. One of them was a training progress
                      bar. A log of what happened does not need to re-state the
                      result. */}
                  <span>{happened(String(event.kind))}</span>
                  <span style={{ color: "var(--text-3)" }}>
                    {"gate" in event ? gateName(String(event.gate)) : ""}
                  </span>
                </div>
              ))}
          </div>
        </>
      )}
    </div>
  );
}
