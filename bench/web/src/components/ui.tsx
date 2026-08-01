/** The parts every screen is built from. Small on purpose. */

import { useEffect, useState } from "react";
import type { ReactNode } from "react";
import type { Finding, GateStatus, Severity, Submission } from "../api/types";

/** One vocabulary for state, product-wide. "failed" is *our* machinery, and is
 *  worded so it never reads as a judgement on the user's data. */
const PILL_LABEL: Record<GateStatus, string> = {
  queued: "Queued",
  running: "Running",
  passed: "Passed",
  refused: "Refused",
  abstained: "Can't tell",
  failed: "Our error",
};

export function StatusPill({ status, label }: { status: GateStatus; label?: string }) {
  return (
    <span className={`pill ${status}`}>
      <span className="dot" />
      {label ?? PILL_LABEL[status]}
    </span>
  );
}

/** What a submission's pill says, everywhere it appears.
 *
 *  Here rather than per screen. The detail page had its own shortened version
 *  that treated anything not running and not refused as "Ready", so a
 *  submission whose worker died showed a green Ready beside a gate reading "Our
 *  error". One vocabulary means the two cannot disagree.
 *
 *  Note what each word protects. "Refused" is a judgement on somebody's data;
 *  "Our error" is a judgement on our machine. A submission is never told the
 *  first when the second is true. */
export function submissionStatus(sub: Submission): { status: GateStatus; label: string } {
  const gate = sub.gates.find((g) => g.key === sub.current_gate);
  if (sub.status === "running") return { status: "running", label: gate ? `${gate.name}…` : "Running" };
  if (sub.status === "refused") return { status: "refused", label: "Refused" };
  if (sub.status === "abstained") return { status: "abstained", label: "Can't tell" };
  if (sub.status === "failed") return { status: "failed", label: "Our error" };
  if (sub.status === "draft") return { status: "queued", label: "Draft — no data yet" };
  if (sub.status === "queued") return { status: "queued", label: "Queued" };
  const done = [...sub.gates].reverse().find((g) => g.status === "passed");
  return { status: "passed", label: done ? `${done.name} passed` : "Ready" };
}

export function Empty({
  title,
  children,
  action,
}: {
  title: string;
  children: ReactNode;
  action?: ReactNode;
}) {
  return (
    <div className="empty">
      <h3>{title}</h3>
      <p>{children}</p>
      {action}
    </div>
  );
}

export function Skeleton({ rows = 3 }: { rows?: number }) {
  return (
    <div style={{ padding: 16, display: "grid", gap: 12 }}>
      {Array.from({ length: rows }).map((_, i) => (
        <div key={i} className="skeleton" style={{ width: `${90 - i * 12}%` }} />
      ))}
    </div>
  );
}

export function ErrorNote({ error }: { error: unknown }) {
  const message = error instanceof Error ? error.message : String(error);
  return (
    <div className="note warn">
      <b>Something went wrong on our side.</b>
      <div style={{ marginTop: 4 }}>{message}</div>
    </div>
  );
}

/** A finding: sentence first, code second, fix in its own block. Labs get the
 *  precision; contributors get the English. */
export function FindingRow({ finding }: { finding: Finding }) {
  return (
    <div className="finding">
      <span className={`sev ${finding.severity satisfies Severity}`} />
      <div style={{ flex: 1 }}>
        <div className="code">
          {finding.code}
          {/* Which module said it. A reader who disagrees needs to know what to
              go and read, and a code with no author behind it is unarguable. */}
          {finding.module && <span className="by">via {finding.module}</span>}
        </div>
        <div className="what">{finding.summary}</div>
        {finding.prescription && (
          <div className="fix">
            <b>What to do</b>
            {finding.prescription}
          </div>
        )}
      </div>
    </div>
  );
}

export function bytes(n: number): string {
  if (n < 1024) return `${n} B`;
  if (n < 1024 ** 2) return `${(n / 1024).toFixed(0)} KB`;
  if (n < 1024 ** 3) return `${(n / 1024 ** 2).toFixed(1)} MB`;
  return `${(n / 1024 ** 3).toFixed(2)} GB`;
}

export function ago(iso: string): string {
  if (!iso) return "—";
  const seconds = (Date.now() - new Date(iso).getTime()) / 1000;
  if (seconds < 60) return "just now";
  if (seconds < 3600) return `${Math.round(seconds / 60)}m ago`;
  if (seconds < 86400) return `${Math.round(seconds / 3600)}h ago`;
  return `${Math.round(seconds / 86400)}d ago`;
}

/** A clock that runs while a gate does.
 *
 *  Its own component with its own interval, rather than a string computed
 *  during render. A string only updates when something else re-renders the
 *  page, so a gate whose stage lasts twenty minutes would show the same figure
 *  for twenty minutes — which is precisely the "is this alive?" question the
 *  running gate exists to answer. */
export function Elapsed({ since }: { since: string }) {
  const [, tick] = useState(0);
  useEffect(() => {
    const timer = setInterval(() => tick((n) => n + 1), 1000);
    return () => clearInterval(timer);
  }, []);
  if (!since) return null;
  const seconds = Math.max(0, (Date.now() - new Date(since).getTime()) / 1000);
  if (seconds < 60) return <>{Math.floor(seconds)}s</>;
  const minutes = Math.floor(seconds / 60);
  if (minutes < 60) return <>{`${minutes}m ${Math.floor(seconds % 60)}s`}</>;
  return <>{`${Math.floor(minutes / 60)}h ${minutes % 60}m`}</>;
}

export function duration(from: string, to: string): string {
  if (!from || !to) return "";
  const seconds = (new Date(to).getTime() - new Date(from).getTime()) / 1000;
  if (seconds < 1) return "<1s";
  if (seconds < 60) return `${Math.round(seconds)}s`;
  if (seconds < 3600) return `${Math.round(seconds / 60)}m`;
  return `${(seconds / 3600).toFixed(1)}h`;
}
