/** The shapes the API returns. Kept in one file so the UI cannot drift from
 *  the server by guessing; when the API grows an OpenAPI schema these become
 *  generated. */

export type GateStatus =
  | "queued"
  | "running"
  | "passed"
  | "refused"
  | "abstained"
  | "failed";

export type Severity = "strong" | "moderate" | "weak" | "info";

export interface Finding {
  code: string;
  severity: Severity;
  summary: string;
  prescription: string | null;
  /** Which module said it. Shown, because a reader who disagrees needs to know
   *  what to go and read. */
  module?: string;
  /** Whatever the finding named — usually the specific clips. */
  evidence?: Record<string, unknown>;
}

/** A number with its own n and interval. Never a bare float: a rate of 1.0 from
 *  four clips and from four hundred are the same float and not the same claim. */
export interface Measure {
  value: number;
  n: number | null;
  ci: [number, number] | null;
  units: string | null;
  method: string | null;
  module?: string;
}

/** A module that declined, and why. Kept and shown: a report that silently
 *  omits what it could not judge reads as a clean bill of health. */
export interface Abstention {
  module: string;
  reason: string;
}

export interface Gate {
  key: "g0" | "g1" | "g2" | "g3";
  name: string;
  question: string;
  eta: string;
  cost_cents: number;
  status: GateStatus;
  verdict: { summary?: string };
  findings: Finding[];
  measures: Record<string, Measure>;
  abstained: Abstention[];
  started_at: string;
  finished_at: string;
}

export interface Channel {
  name: string;
  dtype: string;
  shape: number[] | null;
  width: number | null;
}

export interface Detected {
  episodes?: number;
  frames?: number;
  fps?: number;
  robot_type?: string;
  channels?: Channel[];
  videos?: number;
  has_stats?: boolean;
  has_sidecar?: boolean;
  tasks?: string[];
}

export interface Submission {
  id: string;
  name: string;
  status: string;
  current_gate: string;
  created_at: string;
  benchmark: { key: string; name: string; simulator: string } | null;
  gates: Gate[];
  dataset: {
    version: number;
    bytes: number;
    detected: Detected;
    meaning: Record<string, string>;
  } | null;
  events?: { ts: string; kind: string; [k: string]: unknown }[];
}

export interface Benchmark {
  key: string;
  name: string;
  task: string;
  embodiment: string;
  simulator: string;
  reference: { baseline?: { wins: number; n: number }; expert?: number; note?: string };
}
