/** The parts every screen is built from. Small on purpose. */

import type { ReactNode } from "react";
import type { Finding, GateStatus, Severity } from "../api/types";

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
        <div className="code">{finding.code}</div>
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

export function duration(from: string, to: string): string {
  if (!from || !to) return "";
  const seconds = (new Date(to).getTime() - new Date(from).getTime()) / 1000;
  if (seconds < 1) return "<1s";
  if (seconds < 60) return `${Math.round(seconds)}s`;
  if (seconds < 3600) return `${Math.round(seconds / 60)}m`;
  return `${(seconds / 3600).toFixed(1)}h`;
}
