/** The front door: what this is, in one line, over a picture of the thing.
 *
 *  Photography rather than an illustration, and a real rig rather than a
 *  render, because the claim this product makes is that it measures what
 *  actually happens on hardware. A page about real-robot measurement that
 *  showed only diagrams would be arguing against itself.
 *
 *  Sized to sit above the working table rather than replace it. A full-viewport
 *  hero is right for a project page somebody visits once; this screen is opened
 *  every day by people who came to read a result, and burying the list under a
 *  screenful of image would cost them a scroll every time.
 */

import { Link } from "react-router-dom";

export function Hero({ compact = false }: { compact?: boolean }) {
  return (
    <section className={`hero ${compact ? "compact" : ""}`}>
      <div className="hero-copy">
        <p>
          Upload a robot dataset. Four checks read it, in order, and stop at the
          first one that can answer. You find out whether it is readable, what the
          footage is like, whether it carries any learnable signal at all, and
          finally whether a policy trained on it actually does better.
        </p>
        <div className="hero-cta">
          <Link className="btn primary" to="/submissions/new">
            Upload a dataset
          </Link>
          <Link className="btn" to="/compare">
            See the leaderboard
          </Link>
        </div>
      </div>

      {!compact && (
        <figure className="hero-shot">
          <img
            src="/hero-rig.jpg"
            alt="A dual-arm robot at a work table in a lab"
            width={900}
            height={506}
            loading="eager"
          />
        </figure>
      )}
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
