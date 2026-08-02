/** The answer to the question the page is actually open to settle.
 *
 *  A submission detail page used to open with the name, then a grid of
 *  episode/frame/fps counts, then versions, and only then the timeline whose
 *  entries carry the result. Every one of those is worth showing, and none of
 *  them is what the person came for. They came to find out whether their data
 *  is any good, and that answer was four sections down and phrased as a
 *  statistic.
 *
 *  So it is said once, in a sentence, at the top -- and the sentence has to
 *  keep the distinctions the rest of the product is careful about:
 *
 *    refused   we read it and something is wrong with the data
 *    abstained we read it and could not tell; that is not a no
 *    failed    our machinery broke; this says nothing about the data
 *
 *  Collapsing those into "error" would be the single most damaging thing this
 *  rewrite could do, because the whole reason to trust the tool is that it
 *  distinguishes "your data is bad" from "we could not measure it".
 */

import { Link } from "react-router-dom";
import type { Finding, Gate, Submission } from "../api/types";

type Tone = "passed" | "refused" | "abstained" | "failed" | "running";

interface Answer {
  tone: Tone;
  headline: string;
  detail: string;
  /** What to actually do about it. Omitted when there is nothing to do but wait. */
  next?: string;
}

/** The findings a user is being asked to act on, worst first. */
function blocking(gate: Gate | undefined): Finding[] {
  return (gate?.findings ?? []).filter(
    (f) => f.severity === "strong" || f.severity === "moderate",
  );
}

function answerFor(sub: Submission): Answer {
  const gates = sub.gates;
  const at = gates.find((g) => g.key === sub.current_gate);

  // Only when a check is *actually* running. A submission sits at `running`
  // with its pointer on a gate that is merely queued -- the server advances the
  // pointer the moment the previous gate passes -- and reporting that as
  // "checking your data" tells someone whose checks have all finished that we
  // are still working. Ask the gate, not the submission.
  if (sub.status === "running" && at?.status === "running") {
    return {
      tone: "running",
      headline: `Checking your data: ${at.name.toLowerCase()}`,
      detail: at.question ?? "This page updates on its own as each check finishes.",
    };
  }

  if (sub.status === "failed") {
    return {
      tone: "failed",
      headline: "Something broke on our side",
      detail:
        "This is a fault in our machinery, not a judgement on your data. Nothing " +
        "here says your dataset is wrong; we simply did not get an answer.",
      next: "Run the check again. If it breaks a second time, the fault is ours to fix.",
    };
  }

  if (sub.status === "refused") {
    const stopped = gates.find((g) => g.status === "refused");
    const problems = blocking(stopped);
    return {
      tone: "refused",
      headline: "We can't use this dataset yet",
      detail:
        problems.length === 1
          ? problems[0].summary
          : `${problems.length} things need fixing before the checks can run.`,
      next:
        problems.find((f) => f.prescription)?.prescription ??
        "Fix the points listed below and upload the dataset again.",
    };
  }

  if (sub.status === "abstained") {
    const unsure = gates.find((g) => g.status === "abstained");
    return {
      tone: "abstained",
      headline: "We couldn't tell either way",
      detail:
        unsure?.verdict?.summary ??
        "The check ran, but the result was not clear enough to call. That is not " +
          "a verdict against your data. It means this much data could not settle it.",
      // The gate's own prescription, never a guess. Only some abstentions are
      // cured by more episodes: the signal check also abstains when the clips
      // share no camera, and telling that user to film more of the same is
      // advice that cannot work.
      next:
        unsure?.findings.find((f) => f.prescription)?.prescription ??
        "The reason below says whether more clips would help.",
    };
  }

  const passed = gates.filter((g) => g.status === "passed");
  const noted = passed.flatMap((g) => g.findings).length;
  const nextUp = gates.find((g) => g.status === "queued");

  if (passed.length === 0) {
    return {
      tone: "running",
      headline: "Waiting to start",
      detail: "Your upload is queued. The first check begins in a moment.",
    };
  }

  // "Nothing is blocking it", not "your data is good". Two of the four checks
  // describe rather than judge -- the data report returns "passed" once it has
  // produced a report, and the robot test once it has a ladder to draw -- so
  // counting passes as approvals claims a verdict nobody gave. What is true is
  // that nothing has stopped it yet, and that is what a person needs to know.
  return {
    tone: "passed",
    headline:
      passed.length === gates.length
        ? "All four checks have run, and nothing is blocking your data"
        : `${passed.length} of ${gates.length} checks done, nothing blocking so far`,
    detail: noted
      ? `${noted} thing${noted === 1 ? "" : "s"} worth reading about below, none of them stop you.`
      : "Nothing has come back needing a fix.",
    next: nextUp ? `Next: ${nextUp.name}, ${nextUp.question.toLowerCase()}` : undefined,
  };
}

export function AnswerBanner({ submission }: { submission: Submission }) {
  const { tone, headline, detail, next } = answerFor(submission);
  return (
    <div className={`answer ${tone}`}>
      <h2>{headline}</h2>
      <p>{detail}</p>
      {next && <p className="next">{next}</p>}
    </div>
  );
}

/** The same answer, one line, for a row in the submissions list. */
export function answerLine(sub: Submission): string {
  return answerFor(sub).headline;
}

export function AnswerLink({ submission }: { submission: Submission }) {
  return (
    <Link className="answer-link" to={`/submissions/${submission.id}`}>
      {answerLine(submission)}
    </Link>
  );
}
