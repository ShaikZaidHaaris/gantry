/** Every check's result, said the way a person would say it.
 *
 *  The gates report themselves accurately and unreadably. "your data beat its
 *  own shuffled control on 6 of 6 held-out clips (p=0.031), cutting prediction
 *  error by 8%" is four statistical ideas in one line, and the reader has to
 *  hold all four before learning the thing they came for, which is that their
 *  footage is worth training on.
 *
 *  So each result gets a headline sentence, and the sentence the gate wrote
 *  becomes the evidence under it. NOTHING IS DELETED. Every figure stays one
 *  click away, and the headline is written so that opening the fold can only
 *  ever confirm it -- a plain sentence that overstates the statistic it hides
 *  would be worse than the statistic.
 *
 *  The audience knows robotics and does not necessarily know statistics, so
 *  "episode", "gripper" and "trajectory" are used freely, and "held-out",
 *  "control arm", "p-value" and "abstained" are always explained.
 *
 *  The distinction this file exists to protect:
 *
 *    passed     we measured it and it is good
 *    refused    we measured it and it is wrong -- your data
 *    abstained  we measured and could not tell -- NOT a no
 *    failed     our machinery broke -- says nothing about your data
 *
 *  Three of those are commonly flattened into "error". Doing that here would
 *  destroy the only reason to trust the tool, so each gets its own verb.
 */

import type { Gate, GateStatus } from "../api/types";

export interface Outcome {
  /** The sentence a person reads. Never contains a number. */
  headline: string;
  /** One clause of context. May be empty. */
  detail: string;
}

/** Headlines per gate, per outcome.
 *
 *  Keyed by gate rather than written generically, because "passed" means
 *  something different every time: readable, well-filmed, learnable, better.
 *  A generic "Passed" would throw that away, and it is the whole content. */
const OUTCOMES: Record<string, Partial<Record<GateStatus, Outcome>>> = {
  g0: {
    passed: {
      headline: "We can read your dataset",
      detail: "The episodes, the joint positions and the video all opened.",
    },
    refused: {
      headline: "We can't read this dataset",
      detail: "Something is missing or malformed. The points below say what.",
    },
  },
  // g1 and g3 return only "failed" or "passed" -- grep '"status":' in
  // report.py and robot.py -- and both return "passed" while carrying
  // findings. So "passed" here means "this check produced its report", NOT
  // "your data is fine". Saying otherwise puts a claim in the user's head that
  // the gate never made, and does it above the findings that contradict it.
  g1: {
    passed: {
      headline: "We produced a report on this footage",
      detail:
        "This check describes your footage; it never passes or fails it. Read " +
        "the report below, including the checks that had nothing to go on.",
    },
  },
  g2: {
    passed: {
      headline: "Your footage predicts the robot's movements",
      detail:
        "There is a real relationship between what the camera saw and what the " +
        "arms did, which is the thing a policy learns.",
    },
    refused: {
      headline: "Your footage and your actions don't line up",
      detail:
        "A scrambled copy of your own data predicted just as well, which usually " +
        "means the actions are offset from the frames they belong to.",
    },
    abstained: {
      headline: "We couldn't tell whether the footage predicts the movements",
      detail:
        "Not a verdict against your data. There simply were not enough episodes " +
        "to separate it from a scrambled copy of itself.",
    },
  },
  g3: {
    passed: {
      headline: "The robot test ran",
      detail:
        "It reports how far the robot got at each stage, against a control. Read " +
        "it with the limits listed underneath, because some of them decide what this " +
        "result may be called.",
    },
    failed: {
      headline: "The robot test did not produce a result",
      detail:
        "A fault on our side. Your data was not judged, and this run should not " +
        "be charged for.",
    },
  },
};

/** What every status means when the gate itself has nothing specific to say. */
const GENERIC: Partial<Record<GateStatus, Outcome>> = {
  passed: { headline: "This check passed", detail: "" },
  refused: { headline: "This check found a problem", detail: "" },
  abstained: {
    headline: "This check couldn't reach a conclusion",
    detail: "It ran, but the result was not clear enough to call either way.",
  },
  failed: {
    headline: "This check broke on our side",
    detail:
      "A fault in our machinery, not a judgement on your data. Running it " +
      "again is usually all it takes.",
  },
  queued: { headline: "Not started yet", detail: "" },
  running: { headline: "Running now", detail: "" },
};

export function outcomeFor(gate: Gate): Outcome {
  return (
    OUTCOMES[gate.key]?.[gate.status] ??
    GENERIC[gate.status] ?? { headline: gate.name, detail: "" }
  );
}

/** The chips under a result, in words rather than counts alone.
 *
 *  "1 noted" and "2 not judged" were accurate and told a reader nothing about
 *  whether either mattered. The severity split is what matters: things you must
 *  fix, things worth knowing, and checks that never ran at all -- and that last
 *  group is the one people mistake for a pass. */
export type Tally = {
  text: string;
  kind: "strong" | "note" | "warn";
  /** Which list this chip is counting, so the chip can open it. A count the
   *  reader cannot expand is a claim they have to go looking for. */
  shows: "fix" | "noted" | "abstained";
};

export function tallies(gate: Gate): Tally[] {
  const blocking = gate.findings.filter(
    (f) => f.severity === "strong" || f.severity === "moderate",
  ).length;
  const noted = gate.findings.length - blocking;
  const unjudged = gate.abstained?.length ?? 0;
  const out: Tally[] = [];
  if (blocking)
    out.push({ text: `${blocking} to fix before this can pass`, kind: "strong", shows: "fix" });
  if (noted)
    out.push({
      text: `${noted} thing${noted === 1 ? "" : "s"} worth knowing`,
      kind: "note",
      shows: "noted",
    });
  if (unjudged)
    out.push({
      text: `${unjudged} check${unjudged === 1 ? "" : "s"} had nothing to go on`,
      kind: "warn",
      shows: "abstained",
    });
  return out;
}

/** The findings or abstentions a chip stands for. */
export function behind(gate: Gate, shows: Tally["shows"]) {
  const blocking = (f: { severity: string }) =>
    f.severity === "strong" || f.severity === "moderate";
  if (shows === "fix") return { findings: gate.findings.filter(blocking), abstained: [] };
  if (shows === "noted") return { findings: gate.findings.filter((f) => !blocking(f)), abstained: [] };
  return { findings: [], abstained: gate.abstained ?? [] };
}

/** Statistical terms, explained once, where they are used.
 *
 *  Deliberately a lookup rather than prose in the components: the same term has
 *  to be explained identically everywhere it appears, and two screens quietly
 *  disagreeing about what "held-out" means is worse than not explaining it. */
export const GLOSSARY: Record<string, string> = {
  "held-out":
    "Episodes deliberately kept back from the fitting, so the check is scored on " +
    "footage it has never seen.",
  "shuffled control":
    "A copy of your own data with the actions attached to the wrong episodes. If " +
    "it scores as well as the real thing, the footage was not carrying the signal.",
  "p-value":
    "How often a result this strong would turn up by luck alone. 0.03 means about " +
    "three times in a hundred.",
  interval:
    "The range the true value is likely to sit in. A wide range means not enough " +
    "episodes to pin it down.",
  abstained:
    "The check ran and could not reach a conclusion. That is different from " +
    "failing it.",
};
