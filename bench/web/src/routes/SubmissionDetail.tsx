/** The hero screen: one submission, its gauntlet, live. */

import { Link, useParams } from "react-router-dom";
import { useRetryGate, useStartGate, useSubmission, useSubmissionEvents } from "../api/client";
import { BudgetPanel } from "../components/BudgetPanel";
import { DataReport } from "../components/DataReport";
import { Verdict } from "../components/Verdict";
import { GateTimeline } from "../components/GateTimeline";
import { ErrorNote, Skeleton, StatusPill, ago, bytes, submissionStatus } from "../components/ui";

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
  // question also belongs here rather than in the upload wizard — nobody should
  // pick a trial count before knowing their data is readable, and a price
  // quoted before intake is a price for work that may never happen.
  const sized = data.gates.find((g) => g.sized && g.status === "queued");
  const robot = data.gates.find((g) => g.key === "g3");

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
        <Link className="btn" to="/">
          All submissions
        </Link>
      </div>

      {detected?.episodes ? (
        <div className="card pad" style={{ marginBottom: 18 }}>
          <div className="kv">
            <span className="k">Episodes</span>
            <span className="v">{detected.episodes}</span>
            <span className="k">Frames</span>
            <span className="v">{detected.frames}</span>
            <span className="k">Frame rate</span>
            <span className="v">{detected.fps} fps</span>
            <span className="k">Videos</span>
            <span className="v">{detected.videos}</span>
            <span className="k">Channels</span>
            <span className="v">
              {detected.channels?.map((c) => `${c.name}${c.width ? ` (${c.width})` : ""}`).join("  ·  ")}
            </span>
            {detected.tasks && detected.tasks.length > 0 && (
              <>
                <span className="k">Instructions</span>
                <span className="v">
                  {detected.tasks.length === 1
                    ? `1 — “${detected.tasks[0]}”`
                    : `${detected.tasks.length} distinct`}
                </span>
              </>
            )}
            <span className="k">Provenance sidecar</span>
            <span className="v">{detected.has_sidecar ? "present" : "absent"}</span>
          </div>
        </div>
      ) : null}

      <h2>Progress</h2>
      <GateTimeline
        gates={data.gates}
        currentGate={data.current_gate}
        live={live}
        onStart={(gate) => start.mutate({ gate })}
        starting={start.isPending}
        onRetry={(gate) => retry.mutate(gate)}
        retrying={retry.isPending}
      />

      {/* The report appears only once the gate that produces it has finished.
          Nothing on this page pretends to have a result before it has one. */}
      {report && report.status !== "queued" && report.status !== "running" && (
        <>
          <h2>
            Data report
            <span className="h2-sub">
              free · what your footage is like, before anything is trained on it
            </span>
          </h2>
          <DataReport gate={report} />
        </>
      )}

      {robot && (robot.status === "passed" || robot.status === "abstained") && (
        <>
          <h2>
            Robot test
            <span className="h2-sub">closed-loop, on scenes the policy has never seen</span>
            <Link className="h2-link" to={`/submissions/${data.id}/verdict`}>
              Open as a document →
            </Link>
          </h2>
          <Verdict gate={robot} />
        </>
      )}

      {sized && report?.status === "passed" && (
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
                  <span className="mono">{event.kind}</span>
                  <span>
                    {"summary" in event && event.summary ? String(event.summary) : ""}
                    {"gate" in event && !("summary" in event) ? String(event.gate) : ""}
                  </span>
                </div>
              ))}
          </div>
        </>
      )}
    </div>
  );
}
