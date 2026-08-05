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
  /** Whatever the finding named, usually the specific clips. */
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
  /** The machine codes behind the sentence, kept so a lab can cite one. Shown
   *  on hover, never in the prose. */
  codes?: string[];
}

/** Where a running gate is up to.
 *
 *  `total` is optional on purpose: a stage that does not know how much work is
 *  left says so and gets an indeterminate bar. Inventing a denominator to make
 *  the bar move is a lie the user cannot check. */
export interface Progress {
  gate?: string;
  phase: string;
  current?: number;
  total?: number;
  note?: string;
}

/** One arm's rate on one rung, or a statement that it was never measured.
 *
 *  `measured: false` is not zero. An arm evaluated before the stage events
 *  existed reports nothing on every rung, and rendering that as 0/100 turns an
 *  absence of measurement into a total loss, which is exactly how a fabricated
 *  clean sweep gets published. */
export type Cell =
  | { measured: true; wins: number; n: number; rate: number; ci: [number, number]; unmeasured: number }
  | { measured: false; n: 0; unmeasured: number };

export type Paired =
  | { rung: string; measured: false; scenes: 0 }
  | {
      rung: string;
      measured: true;
      scenes: number;
      a_only: number;
      b_only: number;
      agreed: number;
      p_value: number;
      separated: boolean;
    };

export interface LadderRow {
  rung: string;
  arms: Record<string, Cell>;
  [comparison: string]: unknown;
}

export interface RobotDetail {
  ladder?: LadderRow[];
  order?: string[];
  arms?: string[];
  objects_tracked?: number;
  has_control?: boolean;
  has_baseline?: boolean;
  unreported?: Record<string, string[]>;
  alpha?: number;
}

export interface Gate {
  key: "g0" | "g1" | "g2" | "g3";
  name: string;
  question: string;
  eta: string;
  cost_cents: number;
  /** Whether this gate's price is a choice. Only the robot test runs as many
   *  scenes as you buy; the rest are fixed work, and a trial slider for them
   *  would be a control over nothing. */
  sized: boolean;
  /** Whether the product will offer to run this again. True only for our own
   *  failures, never for a refusal, which is a judgement on the data and must
   *  not be re-rollable until the answer is liked. */
  retryable: boolean;
  status: GateStatus;
  verdict: { summary?: string };
  findings: Finding[];
  measures: Record<string, Measure>;
  abstained: Abstention[];
  /** Present only while running. A finished gate's last position is noise. */
  progress: Progress | Record<string, never>;
  /** The gate's own working, per-scene pairs, p-values, which arms ran.
   *  What a lab asks for when it wants to check the arithmetic itself. */
  detail: RobotDetail & Record<string, unknown>;
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
  /** On the shared leaderboard. False until its owner publishes it. */
  listed: boolean;
  /** A seeded worked example rather than somebody's upload. Read-only: it
   *  belongs to nobody, so every control that would change it is refused by
   *  the server and hidden here. */
  demo?: boolean;
  /** Whether the reader owns this submission. False on a published result
   *  somebody else is reading: they get the report, never the controls. */
  mine?: boolean;
  /** Where to reach the uploader if a check breaks. Empty when not given. */
  email: string;
  /** Generated advice on improving the dataset, {} until the worker writes it.
   *  Interpretation of the measurements, not a measurement itself. `fixes` is
   *  one short line per finding code, shown in place of the canned prose. */
  coach?: { points?: string[]; fixes?: Record<string, string>; model?: string };
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

/** What a budget can and cannot see, computed by the pipeline's own sizing.
 *
 *  `detects` is the smallest effect this many trials can be relied on to
 *  separate. It is null when the answer is "none", a budget too small to
 *  detect anything at this baseline. Null is not zero and must not render as
 *  a reassuring number. */
export interface Plan {
  benchmark: string;
  trials: number;
  magnitude: number;
  baseline: {
    rate: number;
    measured: boolean;
    wins: number | null;
    n: number | null;
    note: string;
  };
  detects: number | null;
  needed: number;
  ok: boolean;
  reasons: { code: string; summary: string; hint: string; detail: Record<string, unknown> }[];
  cost: {
    seconds: number;
    cents: number;
    arms_trained: number;
    arms_evaluated: number;
    measured_on: string;
  };
}

/** One submission's place on a leaderboard.
 *
 *  `group` is the compact letter display: entries sharing a letter are not
 *  separated by the paired test. It is the column to read, at these sample
 *  sizes the ordering is mostly noise, and a rank without this reads as a
 *  total ordering that the evidence does not support. */
export interface Entry {
  id: string;
  name: string;
  wins: number;
  n: number;
  rate: number;
  ci: [number, number];
  group: string;
}

export interface Pair {
  left: string;
  right: string;
  shared_scenes: number;
  left_only: number;
  right_only: number;
  agreed: number;
  p_value: number;
  separated: boolean;
}

export interface Leaderboard {
  benchmark: { key: string; name: string; simulator: string };
  rung: string;
  rungs: string[];
  baseline: { wins: number; n: number } | null;
  entries: Entry[];
  pairs: Pair[];
}

export interface VersionChange {
  from: number;
  to: number;
  fixed: { code: string; summary: string }[];
  new: { code: string; summary: string }[];
  remaining: { code: string; summary: string }[];
  clips: (number | null)[];
  frames: (number | null)[];
}

export interface VersionRow {
  version: number;
  bytes: number;
  created_at: string;
  detected: Detected;
  gates: Record<string, GateStatus>;
  findings: Finding[];
  verdicts: Record<string, string>;
}

export interface Versions {
  submission: string;
  current: number;
  versions: VersionRow[];
  changes: VersionChange[];
}
