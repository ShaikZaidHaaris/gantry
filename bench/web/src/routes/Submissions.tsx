/** Home. Every submission this org has, glanceable without opening anything. */

import { Link } from "react-router-dom";
import { useSubmissions } from "../api/client";
import { Hero, TaskStrip } from "../components/Hero";
import { HowItWorks } from "../components/HowItWorks";
import { Empty, ErrorNote, Skeleton, StatusPill, ago, submissionStatus } from "../components/ui";

const COLUMNS = "1.4fr 1fr 150px 130px 90px";

export function Submissions() {
  const { data, isPending, error } = useSubmissions();

  return (
    <div className="page">
      {/* Full on a first visit, compact once there is work to read: the
          picture is the front door, and a front door does not need to be as
          wide on the hundredth entry as on the first. */}
      <Hero compact={(data?.submissions.length ?? 0) > 0} />

      {/* Shown to somebody who has not submitted anything yet, alongside the
          full hero. Once there is a list to read, the argument has been made
          and the space belongs to their own results. */}
      {(data?.submissions.length ?? 0) === 0 && <TaskStrip />}

      <div className="page-head">
        <div>
          <h2 style={{ margin: 0 }}>Submissions</h2>
          <p style={{ color: "var(--text-2)", marginTop: 4 }}>
            One dataset, checked four ways, against one benchmark.
          </p>
        </div>
        <div className="spacer" />
        <Link className="btn primary" to="/submissions/new">
          New submission
        </Link>
      </div>

      {/* Open for somebody with nothing here yet, who needs the whole thing
          explained; folded away once they have submissions, because by then
          they have read it and the list is what they came for. */}
      <HowItWorks defaultOpen={data?.submissions.length === 0} />

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
            footage is like, and, if you choose to go further, whether it actually makes a
            robot better.
          </Empty>
        )}

        {data?.submissions.map((sub) => {
          const pill = submissionStatus(sub);
          const detected = sub.dataset?.detected;
          return (
            <Link
              key={sub.id}
              className="trow"
              style={{ gridTemplateColumns: COLUMNS }}
              to={`/submissions/${sub.id}`}
            >
              <span style={{ fontWeight: 550 }}>{sub.name}</span>
              <span style={{ color: "var(--text-2)" }}>{sub.benchmark?.name ?? "-"}</span>
              <span>
                <StatusPill status={pill.status} label={pill.label} />
              </span>
              <span className="mono">
                {detected?.episodes ? `${detected.episodes} eps · ${detected.frames} fr` : "-"}
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
