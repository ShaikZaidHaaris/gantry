/** The parts every screen is built from. Small on purpose. */

import { useEffect, useState } from "react";
import type { ReactNode } from "react";
import type { Finding, GateStatus, Submission } from "../api/types";

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
  // "Running" only when a gate really is. The server moves `current_gate` on to
  // the next gate as soon as the previous one passes, so a submission whose
  // checks have all finished sits at `running` pointing at something merely
  // queued -- and the pill then said "Signal check…" beside a page reporting
  // that the signal check had passed.
  if (sub.status === "running" && gate?.status === "running")
    return { status: "running", label: `${gate.name}…` };
  if (sub.status === "running" && !gate) return { status: "running", label: "Running" };
  // A fetch has no gate to point at, so it names itself. Running-coloured
  // because our machinery is working; nothing about the data has been judged.
  if (sub.status === "fetching")
    return { status: "running", label: "Fetching from Hugging Face…" };
  if (sub.status === "refused") return { status: "refused", label: "Refused" };
  if (sub.status === "abstained") return { status: "abstained", label: "Can't tell" };
  if (sub.status === "failed") return { status: "failed", label: "Our error" };
  if (sub.status === "draft") return { status: "queued", label: "Draft, no data yet" };
  if (sub.status === "queued") return { status: "queued", label: "Queued" };
  const done = [...sub.gates].reverse().find((g) => g.status === "passed");
  return { status: "passed", label: done ? `${done.name} passed` : "Ready" };
}

/** Evidence, folded away until asked for.
 *
 *  The pattern the screens are built on: the plain answer is always visible and
 *  the figures that justify it are one click below it. Shared rather than
 *  per-screen because "how we know" has to look and behave identically
 *  everywhere -- a disclosure that opens differently on two pages reads as two
 *  different kinds of thing.
 *
 *  Closed by default, and that is the whole point. Nothing is deleted to make
 *  room for plain language; the p-value, the interval and the n are always one
 *  click away, and a reader who wants them never has to trust a paraphrase. */
export function Fold({
  label = "How we know",
  defaultOpen = false,
  children,
}: {
  label?: string;
  defaultOpen?: boolean;
  children: ReactNode;
}) {
  const [open, setOpen] = useState(defaultOpen);
  return (
    <div className="fold">
      <button
        type="button"
        className="fold-head"
        onClick={() => setOpen(!open)}
        aria-expanded={open}
      >
        <span className="chev" data-open={open}>
          ▸
        </span>
        {label}
      </button>
      {open && <div className="fold-body">{children}</div>}
    </div>
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

/** One finding: what was found, then what to do about it. Prose, not chrome.
 *
 *  The first version of this printed the machine code as a heading, the module
 *  name beside it, a severity dot, and the prescription inside a tinted
 *  "WHAT TO DO" box, a card inside a card. All of that was the system talking
 *  about itself. The code still exists for anyone who needs to cite it (hover),
 *  and severity is carried by the *grouping* on every screen that uses this,  *  "what to fix" versus "also noted", which says more than a colored dot did.
 */
export function FindingRow({
  finding,
  note,
}: {
  finding: Finding;
  note?: { say?: string; detail?: string; do?: string };
}) {
  // `note` is the generated pair for this finding: `say` restates the
  // observation and replaces the gate's summary line, `do` is the action under
  // it. When the caller passes notes at all, the canned prescription prose
  // stays hidden even for findings the model had nothing to say about; the
  // gate's summary is the fallback headline, never the fallback advice. The
  // machine code stays on hover either way, since the generated line is the
  // one most worth being able to check.
  const advised = note !== undefined;
  return (
    <div className="finding" title={finding.code}>
      <div className="what">{sentence(note?.say || finding.summary)}</div>
      {advised
        ? (note.detail || note.do) && (
            <p className="fix">{sentence(note.detail || note.do || "")}</p>
          )
        : finding.prescription && <p className="fix">{finding.prescription}</p>}
    </div>
  );
}

/** A machine name as a person would say it: `pick_dual_bottles` becomes
 *  "Pick dual bottles". Display only. The identifier stays whatever it was,
 *  because it is what a report cites and what the API returns. */
export function readable(identifier: string): string {
  const words = identifier.split(/[._\-\s]+/).filter(Boolean);
  if (!words.length) return identifier;
  const joined = words.join(" ");
  return joined.charAt(0).toUpperCase() + joined.slice(1);
}

/** Gate summaries arrive lowercase (they read as clause fragments in logs);
 *  people read them as sentences. */
export function sentence(text: string): string {
  return text ? text.charAt(0).toUpperCase() + text.slice(1) : text;
}

export function bytes(n: number): string {
  if (n < 1024) return `${n} B`;
  if (n < 1024 ** 2) return `${(n / 1024).toFixed(0)} KB`;
  if (n < 1024 ** 3) return `${(n / 1024 ** 2).toFixed(1)} MB`;
  // Trailing zeros dropped at GB, because the upload ceiling is a round number
  // and "up to 1.00 GB" reads like a rounding artifact rather than a rule.
  return `${(n / 1024 ** 3).toFixed(2).replace(/\.00$/, "")} GB`;
}

export function ago(iso: string): string {
  if (!iso) return "-";
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
 *  for twenty minutes, which is precisely the "is this alive?" question the
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
