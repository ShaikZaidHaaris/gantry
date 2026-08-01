/** The hero screen: one submission, its gauntlet, live. */

import { Link, useParams } from "react-router-dom";
import { useSubmission, useSubmissionEvents } from "../api/client";
import { DataReport } from "../components/DataReport";
import { GateTimeline } from "../components/GateTimeline";
import { ErrorNote, Skeleton, StatusPill, ago, bytes, submissionStatus } from "../components/ui";

export function SubmissionDetail() {
  const { id } = useParams();
  const { data, isPending, error } = useSubmission(id);
  const live = useSubmissionEvents(id);

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
      <GateTimeline gates={data.gates} currentGate={data.current_gate} live={live} />

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
