/** What this product does, said once, where a new person lands.
 *
 *  The page used to carry two sentences of subtitle that assumed you already
 *  knew what a submission, a benchmark and a gate were -- and promised that
 *  "paid stages" would follow, which is a claim about pricing this product has
 *  not made. Somebody arriving for the first time needed the four checks explained,
 *  the file format stated, and the four possible answers spelled out, and none
 *  of that was anywhere on the screen.
 *
 *  Every number here is read off the implementation rather than remembered:
 *  the required channels are intake's NEEDS_ACTION and NEEDS_STATE, the ten-clip
 *  floor is signal.py's `smallest_conclusive() + FIT_FLOOR` (6 + 4), and the
 *  timings are the API's own `eta` strings. A how-to that drifts from the code
 *  is worse than none, because it is believed.
 */

import { Link } from "react-router-dom";
import { useSamples } from "../api/client";
import { Fold } from "./ui";

/** Two ways in: look at a finished one, or take the file and run it yourself.
 *
 *  Looking comes first, and is the primary control. Somebody deciding whether
 *  this is worth their dataset wants to see what comes out the other end, and
 *  before this the only way to find out was to upload something, which is the
 *  commitment the verdict is supposed to help them make. The download is still
 *  here, one line down, for anyone who would rather run it than read it.
 *
 *  The two results are seeded worked examples, readable by anyone and owned by
 *  nobody. They are the product's own screen over real numbers, not a mock: the
 *  gates, findings and activity log are exactly what the run produced.
 */
/** Mebibytes, labelled MB, because that is what the rest of this product and
 *  samples/README.md already say for these same two files. Two numbers for one
 *  file reads as a mistake even when both are defensible. */
function megabytes(bytes: number): string {
  return `${(bytes / 1024 / 1024).toFixed(1)} MB`;
}

const HAND: Record<string, string> = {
  two_handed: "Both hands tracked",
  one_handed: "One hand mostly absent",
};

function SampleCard() {
  const { data } = useSamples();
  // An offer is worth showing if *either* half of it works: a result to read, or
  // a file to take. Filtering on the file alone hid both datasets outright on a
  // deployment whose samples directory had not been copied, when the results
  // they link to were sitting right there and fine.
  const offers = (data?.samples ?? []).filter((s) => s.available || s.result);
  if (offers.length === 0) return null;

  return (
    <div className="sample">
      <div>
        <b>Two real datasets, and what each one scored</b>
        <p>
          Training sets from an experiment that actually ran:{" "}
          <code>pick_dual_bottles</code> on an aloha-agilex, 58 clips each, in the
          layout above. Both share the same 50 RoboTwin demonstrations; what differs
          is the egocentric human footage added on top, in one case where both hands
          were tracked and in the other where one was mostly absent.{" "}
          <b>Open either result</b> to see all four checks through to the feedback,
          or take the file and run it yourself.
        </p>
        <div className="sample-offers">
          {offers.map((s) => (
            <div className="sample-offer" key={s.key}>
              <div className="sample-what">
                <b>{HAND[s.key] ?? s.what}</b>
                {/* The size is read off the file. When it is not there, the
                    clip count still is, and inventing a number to fill the gap
                    is how the missing file stayed hidden last time. */}
                <span>58 clips{s.available ? ` · ${megabytes(s.bytes)}` : ""}</span>
              </div>
              {/* Only rendered when the server says a result was seeded. A
                  deployment without the fixture offers the download alone
                  rather than a link that 404s. */}
              {s.result ? (
                <Link className="btn primary" to={`/samples/${s.result}`}>
                  See the live result
                </Link>
              ) : (
                <span className="sample-none">no result seeded here</span>
              )}
              {s.available && (
                <a className="sample-dl" href={`/api/samples/${s.key}`}>
                  Download the dataset
                </a>
              )}
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

interface Check {
  n: string;
  name: string;
  asks: string;
  does: string;
  takes: string;
}

const CHECKS: Check[] = [
  {
    n: "1",
    name: "Intake",
    asks: "Can we read this at all?",
    does:
      "Opens the archive and looks for the pieces every later check needs: the " +
      "episode index, an 'action' channel, an 'observation.state' channel, and at " +
      "least one camera. It also opens a sample of the videos to check they decode. " +
      "This is the only check that can stop you outright.",
    takes: "seconds",
  },
  {
    n: "2",
    name: "Data report",
    asks: "What is this footage like?",
    does:
      "Describes the recording: how the arms move, how long episodes run, how much " +
      "the footage varies. It never passes or fails your data; it produces a report " +
      "and tells you which of its checks had nothing to work with.",
    takes: "about a minute",
  },
  {
    n: "3",
    name: "Signal check",
    asks: "Is there anything learnable here?",
    does:
      "The one that can say no. It fits a small probe on your clips, then fits the " +
      "same probe on a copy of your data with the actions attached to the wrong " +
      "episodes, and compares the two on footage neither has seen. If your real data " +
      "does not beat its own scrambled copy, the footage is not carrying the signal a " +
      "policy would need. Needs at least 10 episodes to conclude anything.",
    takes: "about ten minutes",
  },
  {
    n: "4",
    name: "Robot test",
    asks: "Does the robot actually get better?",
    does:
      "Trains a policy on your data and runs it on the robot, against a control " +
      "trained the same way on scrambled data, on scenes neither has seen. This is " +
      "the only check that touches a robot, and the only one that needs one.",
    takes: "a few hours",
  },
];

const ANSWERS: { word: string; tone: string; means: string }[] = [
  {
    word: "Passed",
    tone: "passed",
    means:
      "The check ran and produced its result. For intake that means readable; for " +
      "the signal check it means your footage predicts the movements. It does not " +
      "always mean 'good'. Read what the check itself says.",
  },
  {
    word: "Refused",
    tone: "refused",
    means:
      "We read your data and something in it is wrong. This is a judgement about " +
      "the file, and it always comes with the thing to fix.",
  },
  {
    word: "Can't tell",
    tone: "abstained",
    means:
      "The check ran and could not reach an answer either way. This is not a no. " +
      "Usually it means there was not enough data to separate your result from " +
      "chance. The check says whether more would help.",
  },
  {
    word: "Our error",
    tone: "failed",
    means:
      "Our machinery broke. Your data was never judged, and nothing here counts " +
      "for or against it. Running it again is usually all it takes.",
  },
];

export function HowItWorks({ defaultOpen = false }: { defaultOpen?: boolean }) {
  return (
    <div className="howto">
      <Fold label="How this works" defaultOpen={defaultOpen}>
        <div className="howto-body">
          <section>
            <h3>What you upload</h3>
            <p>
              <b>Annotated video</b>, not video on its own. Every frame of footage has to
              carry the numbers that go with it: the command that was issued at that frame,
              and where the arms actually were. Video with no annotations cannot be checked
              here and cannot be trained on, because there is nothing for a policy to copy.
            </p>
            <p>
              In practice that means a <b>.zip of a LeRobot v2 recording</b>, the layout{" "}
              <code>lerobot</code> writes: a <code>meta/</code> folder with{" "}
              <code>info.json</code> and <code>episodes.jsonl</code>, one parquet file per
              episode holding the per-frame numbers, and your camera streams as{" "}
              <code>.mp4</code> under <code>videos/</code>.
            </p>
            <p>
              The two annotation channels are required by name: <code>action</code> for the
              command issued at each frame, and <code>observation.state</code> for where the
              arms were. At least one camera has to be present, and the same camera has to
              be in every episode, because a stream only some episodes carry cannot be checked
              across the set.
            </p>

            <SampleCard />
          </section>

          <section>
            <h3>What happens to it</h3>
            <p>
              Four checks run in order. Each one only starts if the one before it got
              through, so a dataset that cannot be opened never reaches the ten minutes of
              probe-fitting.
            </p>
            <ol className="checks">
              {CHECKS.map((c) => (
                <li key={c.n}>
                  <div className="checks-head">
                    <span className="checks-n">{c.n}</span>
                    <b>{c.name}</b>
                    <span className="checks-asks">{c.asks}</span>
                    <span className="spacer" />
                    <span className="checks-takes">{c.takes}</span>
                  </div>
                  <p>{c.does}</p>
                </li>
              ))}
            </ol>
            <p className="howto-note">
              The first two run on an ordinary machine. The signal check wants a GPU to
              be quick about it, and the robot test needs a simulator.
            </p>
          </section>

          <section>
            <h3>What the answers mean</h3>
            <p>
              Four different things, kept deliberately distinct. The difference between
              “your data is wrong” and “we could not measure it” is the whole reason to
              trust the result.
            </p>
            <dl className="answers">
              {ANSWERS.map((a) => (
                <div key={a.word} className={`answers-row ${a.tone}`}>
                  <dt>{a.word}</dt>
                  <dd>{a.means}</dd>
                </div>
              ))}
            </dl>
          </section>

          <section>
            <h3>If something needs fixing</h3>
            <p>
              Fix it and upload again from the same submission. Each upload is a new
              version, every check runs again on it, and the old version keeps its
              results, so you can see what your change actually did rather than
              remembering what the numbers used to be.
            </p>
          </section>
        </div>
      </Fold>
    </div>
  );
}
