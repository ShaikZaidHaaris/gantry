/** Publishing a result to the shared leaderboard: opt-in, and reversible.
 *
 *  Off until its owner turns it on. That default is the feature, not a
 *  precaution around it -- a benchmark that published by default would be
 *  collecting results people did not choose to stand behind, and the numbers on
 *  it would mean less as a consequence.
 *
 *  Withdrawing is offered as plainly as publishing. A one-way door makes the
 *  decision unaskable in practice: people decline rather than risk a number they
 *  cannot take back, and the board ends up emptier and less honest than one that
 *  lets them change their mind.
 *
 *  Only appears once the robot test has produced a result, because that is the
 *  only thing the board ranks. The server enforces the same rule; showing the
 *  control earlier would offer a choice that returns an error.
 */

import { useState } from "react";
import { useSetListed } from "../api/client";
import type { Gate, Submission } from "../api/types";
import { ErrorNote } from "./ui";

export function Publish({ submission }: { submission: Submission }) {
  const robot = submission.gates.find((g: Gate) => g.key === "g3");
  const ranked = robot?.status === "passed";
  const setListed = useSetListed(submission.id);
  const [error, setError] = useState<unknown>(null);

  if (!ranked) return null;

  const on = submission.listed;

  return (
    <div className={`publish ${on ? "on" : ""}`}>
      <div>
        <b>{on ? "On the leaderboard" : "Not on the leaderboard"}</b>
        <p>
          {on ? (
            <>
              Anyone comparing this benchmark can see this result and the name you
              gave it. Nothing else about you is shown: not your address, and no
              identifier derived from it.
            </>
          ) : (
            <>
              This result is yours alone. You can see it on the leaderboard to
              judge where it stands; nobody else can, until you publish it.
            </>
          )}
        </p>
        {error != null && <ErrorNote error={error} />}
      </div>
      <button
        type="button"
        className={`btn ${on ? "" : "primary"}`}
        disabled={setListed.isPending}
        onClick={() => {
          setError(null);
          setListed.mutate(!on, { onError: setError });
        }}
      >
        {setListed.isPending ? "Saving…" : on ? "Take it down" : "Publish to leaderboard"}
      </button>
    </div>
  );
}
