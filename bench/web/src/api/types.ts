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
