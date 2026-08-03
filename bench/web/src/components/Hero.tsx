/** The front door: what this is, in one line, over a picture of the thing.
 *
 *  Photography rather than an illustration, and a real rig rather than a
 *  render, because the claim this product makes is that it measures what
 *  actually happens on hardware. A page about real-robot measurement that
 *  showed only diagrams would be arguing against itself.
 *
 *  Sized to sit above the working table rather than replace it: a full-viewport
 *  hero is right for a project page somebody visits once, and this screen is
 *  opened repeatedly by people who came to read a result.
 *
 *  It used to hide the photograph entirely once you had a submission, on the
 *  theory that a returning visitor did not need it. That was wrong twice over.
 *  It made the picture vanish the moment anybody used the product, which reads
 *  as a broken image rather than a considered choice, and it meant the page a
 *  person shows somebody else is the one without the photograph on it.
 */

import { Link } from "react-router-dom";
import { useSamples } from "../api/client";

export function Hero() {
  // The second button used to be the leaderboard, which is opt-in and therefore
  // empty until somebody publishes: a first-time visitor pressed it and got a
  // page with nothing on it, which reads as a broken product rather than an
  // honest default. A finished worked example answers the question they
  // actually have, which is what one of these reports looks like.
  //
  // The id comes from the API rather than being written here, because a
  // deployment whose fixture failed to seed reports `result: null`, and a link
  // to a page that is not there is worse than the leaderboard was.
  const { data } = useSamples();
  const example = data?.samples.find((s) => s.result)?.result ?? null;

  return (
    <section className="hero">
      <div className="hero-copy">
        <p>
          Upload a robot dataset and it runs four checks. Can we read the file, what
          the footage actually contains, whether there is any signal a policy could
          learn from, and whether a policy trained on it does better than one trained
          on shuffled data.
        </p>
        <div className="hero-cta">
          <Link className="btn primary" to="/submissions/new">
            Upload a dataset
          </Link>
          <Link className="btn" to={example ? `/samples/${example}` : "/compare"}>
            {example ? "Example results" : "See the leaderboard"}
          </Link>
        </div>
      </div>

      <figure className="hero-shot">
        <img
          src="/hero-rig.jpg"
          alt="A dual-arm robot at a work table in a lab"
          width={900}
          height={506}
          loading="eager"
        />
      </figure>
    </section>
  );
}

/** Scenes with their working area written down.
 *
 *  Chosen over a plain contact sheet of clips because of what is annotated on
 *  it: 35cm by 40cm, plus or minus 45 degrees. These are setups somebody
 *  measured and then wrote down, which is the same claim this product makes
 *  about datasets -- that a result means something only alongside the
 *  conditions it was produced under. A grid of pretty robot footage would have
 *  been decoration; a grid of *specified* footage is the argument.
 */
export function TaskStrip() {
  return (
    <figure className="strip">
      <img
        src="/task-grid.jpg"
        alt="Six robot workspaces, each annotated with its measured working area and rotation range"
        width={1600}
        height={1529}
        loading="lazy"
      />
    </figure>
  );
}
