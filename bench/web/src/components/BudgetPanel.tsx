/** What a budget buys, before it is spent.
 *
 *  A contributor choosing a trial count is choosing what the run is able to
 *  conclude, and almost nobody in this space presents it that way. The failure
 *  it prevents is specific and common: pick fifty trials because fifty sounds
 *  reasonable, get a null result, and read it as "the data did not help" — when
 *  fifty trials at a twelve percent baseline could only ever have separated a
 *  thirty-three point difference. The null was about the budget, not the data.
 *
 *  So the headline here is not the price. It is the sentence "this budget can
 *  detect a change of at least X", with the price underneath it.
 *
 *  Nothing on this component computes. Every figure is returned by the
 *  pipeline's own sizing against the baseline this benchmark has actually
 *  recorded, and a benchmark with no recorded baseline says so rather than
 *  being given a flattering one.
 */

import { useState } from "react";
import { usePlan } from "../api/client";
import type { Plan } from "../api/types";
import { Skeleton } from "./ui";

/** Trial counts worth offering. Not a continuous slider: the arithmetic is
 *  lumpy and a smooth control invites fine-tuning a number whose precision is
 *  imaginary. */
const CHOICES = [20, 50, 100, 210, 400, 800];

/** The effect the sizing is asked about, in success-rate points. A run is
 *  planned to detect *something*; "any improvement" is not an answer, because
 *  any positive noise satisfies it. */
const EFFECTS = [0.05, 0.08, 0.15, 0.3];

function money(cents: number): string {
  return cents < 100 ? `$${(cents / 100).toFixed(2)}` : `$${Math.round(cents / 100)}`;
}

function hours(seconds: number): string {
  const h = seconds / 3600;
  return h < 1 ? `${Math.round(seconds / 60)} min` : `${h.toFixed(1)} h`;
}

function Verdict({ plan }: { plan: Plan }) {
  if (plan.detects === null) {
    return (
      <div className="verdict short">
        <b>This budget cannot detect anything.</b>
        <span>
          At a {(plan.baseline.rate * 100).toFixed(0)}% baseline there is no effect size{" "}
          {plan.trials} trials can reliably separate. Whatever comes back, it will not be
          evidence either way.
        </span>
      </div>
    );
  }
  return (
    <div className={`verdict ${plan.ok ? "ok" : "short"}`}>
      <b>
        Detects a change of {(plan.detects * 100).toFixed(1)} points or more
      </b>
      <span>
        {plan.ok
          ? `Enough to answer the ${(plan.magnitude * 100).toFixed(0)}-point question you picked.`
          : `Not enough for the ${(plan.magnitude * 100).toFixed(0)}-point question you picked — that needs ${plan.needed} trials.`}
      </span>
    </div>
  );
}

export function BudgetPanel({
  benchmark,
  gateName,
}: {
  benchmark: string | undefined;
  gateName: string;
}) {
  const [trials, setTrials] = useState(210);
  const [magnitude, setMagnitude] = useState(0.08);
  const { data: plan, isPending } = usePlan(benchmark, trials, magnitude);

  if (isPending || !plan) {
    return (
      <div className="card">
        <Skeleton rows={4} />
      </div>
    );
  }

  return (
    <div className="card pad budget">
      <div className="budget-head">
        <div>
          <h3>{gateName}</h3>
          <p>
            {plan.baseline.measured ? (
              <>
                Today's model solves {plan.baseline.wins} of {plan.baseline.n} scenes.
                Your data is measured against that, on the same scenes.
              </>
            ) : (
              <>
                Nothing has been recorded for this benchmark yet, so the sizing below assumes
                the noisiest case rather than a measured rate.
              </>
            )}
          </p>
        </div>
      </div>

      <div className="budget-controls">
        <label className="field">
          <span className="lab">How many scenes to run</span>
          <div className="segmented">
            {CHOICES.map((n) => (
              <button
                key={n}
                type="button"
                className={n === trials ? "on" : ""}
                onClick={() => setTrials(n)}
              >
                {n}
              </button>
            ))}
          </div>
        </label>

        <label className="field">
          <span className="lab">The difference you care about</span>
          <div className="segmented">
            {EFFECTS.map((m) => (
              <button
                key={m}
                type="button"
                className={m === magnitude ? "on" : ""}
                onClick={() => setMagnitude(m)}
              >
                +{(m * 100).toFixed(0)} pts
              </button>
            ))}
          </div>
        </label>
      </div>

      <Verdict plan={plan} />

      <div className="budget-facts">
        <div>
          <span className="k">Price</span>
          <span className="v big">{money(plan.cost.cents)}</span>
        </div>
        <div>
          <span className="k">Takes about</span>
          <span className="v big">{hours(plan.cost.seconds)}</span>
        </div>
        <div>
          <span className="k">Arms</span>
          <span className="v">
            {plan.cost.arms_trained} trained, {plan.cost.arms_evaluated} evaluated
          </span>
        </div>
        <div>
          <span className="k">For the question you picked</span>
          <span className="v">{plan.needed} trials</span>
        </div>
      </div>

      {plan.reasons.length > 0 && (
        <div className="budget-reasons">
          {plan.reasons.map((r) => (
            <div className="note warn" key={r.code}>
              <b>{r.summary}</b>
              {r.hint && <div style={{ marginTop: 4 }}>{r.hint}</div>}
            </div>
          ))}
        </div>
      )}

      <p className="budget-foot">
        Your data and its own shuffled control are both trained and both evaluated, on the
        same scenes. The shuffle is the comparison that matters: fine-tuning on anything
        moves a model, so beating the baseline shows only that something changed. Beating
        the shuffle shows the change came from your footage.
        {plan.cost.measured_on && <> Times measured on {plan.cost.measured_on}.</>}
      </p>
    </div>
  );
}
