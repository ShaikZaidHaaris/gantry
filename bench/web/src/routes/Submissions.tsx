/** Home. Every submission this org has, glanceable without opening anything. */

import { Link } from "react-router-dom";
import { useSubmissions } from "../api/client";
import type { GateStatus, Submission } from "../api/types";
import { Empty, ErrorNote, Skeleton, StatusPill, ago } from "../components/ui";

const COLUMNS = "1.4fr 1fr 150px 130px 90px";

/** What the row's pill should say: the submission's own state, expressed
 *  through the gate it is at, so the table is readable without opening a row. */
function rowStatus(sub: Submission): { status: GateStatus; label: string } {
  const gate = sub.gates.find((g) => g.key === sub.current_gate);
  if (sub.status === "running" && gate) return { status: "running", label: `${gate.name}…` };
  if (sub.status === "refused") return { status: "refused", label: "Refused" };
  if (sub.status === "failed") return { status: "failed", label: "Our error" };
  if (sub.status === "draft") return { status: "queued", label: "Draft — no data yet" };
  if (sub.status === "queued") return { status: "queued", label: "Queued" };
  const done = [...sub.gates].reverse().find((g) => g.status === "passed");
  return { status: "passed", label: done ? `${done.name} passed` : "Ready" };
}

export function Submissions() {
  const { data, isPending, error } = useSubmissions();

  return (
    <div className="page">
      <div className="page-head">
        <div>
          <h1>Submissions</h1>
          <p>
            Each submission is one dataset tested against one benchmark. Free checks run
            first; paid stages are only offered once the free ones say there is something
            worth testing.
          </p>
        </div>
        <div className="spacer" />
        <Link className="btn primary" to="/submissions/new">
          New submission
        </Link>
      </div>

      {error && <ErrorNote error={error} />}

      <div className="table">
        <div className="thead" style={{ gridTemplateColumns: COLUMNS }}>
          <span>Name</span>
          <span>Benchmark</span>
          <span>Status</span>
          <span>Data</span>
          <span>Created</span>
        </div>

        {isPending && <Skeleton rows={3} />}

        {data?.submissions.length === 0 && (
          <Empty
            title="Nothing submitted yet"
            action={
              <Link className="btn primary" to="/submissions/new">
                Submit a dataset
              </Link>
            }
          >
            Upload a LeRobot dataset and we will tell you whether it is readable, what the
            footage is like, and — if you choose to go further — whether it actually makes a
            robot better.
          </Empty>
        )}

        {data?.submissions.map((sub) => {
          const pill = rowStatus(sub);
          const detected = sub.dataset?.detected;
          return (
            <Link
              key={sub.id}
              className="trow"
              style={{ gridTemplateColumns: COLUMNS }}
              to={`/submissions/${sub.id}`}
            >
              <span style={{ fontWeight: 550 }}>{sub.name}</span>
              <span style={{ color: "var(--text-2)" }}>{sub.benchmark?.name ?? "—"}</span>
              <span>
                <StatusPill status={pill.status} label={pill.label} />
              </span>
              <span className="mono">
                {detected?.episodes ? `${detected.episodes} eps · ${detected.frames} fr` : "—"}
              </span>
              <span className="mono" style={{ color: "var(--text-3)" }}>
                {ago(sub.created_at)}
              </span>
            </Link>
          );
        })}
      </div>
    </div>
  );
}
