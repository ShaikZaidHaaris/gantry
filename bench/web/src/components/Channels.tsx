/** What is in the upload, named the way a person would say it.
 *
 *  This row answers one question: did you find my camera and my actions. That
 *  is worth confirming, because an upload whose camera was not picked up is the
 *  most common thing that goes wrong and the least visible.
 *
 *  So it shows the channels that carry data and nothing else. `frame_index`,
 *  `episode_index`, `index`, `task_index` and `timestamp` are how the format
 *  keeps its rows in order. Every LeRobot dataset has them, they are identical
 *  in all of them, and listing five columns of width one in front of the three
 *  that matter buries the answer.
 *
 *  The exact strings are not decoration either: they are the schema, and
 *  somebody working out why a channel was not picked up needs the name that is
 *  really there. So the readable name is what you see and the real one is on
 *  hover.
 */

import type { Channel } from "../api/types";

/** Columns the format needs and nobody uploads on purpose. */
const BOOKKEEPING = new Set(["frame_index", "episode_index", "index", "task_index", "timestamp"]);

function words(text: string): string {
  const out = text.split(/[._\-\s]+/).filter(Boolean).join(" ");
  return out.charAt(0).toUpperCase() + out.slice(1);
}

/** `observation.images.head` reads as "Head camera". */
export function channelName(raw: string): string {
  const camera = raw.match(/^observation\.images?\.(.+)$/);
  if (camera) {
    // `cam_high` is "High camera", not "Cam high camera". The prefix is how the
    // format namespaces cameras, and saying it twice is how the label ends up
    // sounding like the column name it replaced.
    const rest = camera[1].replace(/^(cam|camera)[._-]?/, "") || camera[1];
    return `${words(rest)} camera`;
  }
  if (raw === "observation.state") return "Robot state";
  const observed = raw.match(/^observation\.(.+)$/);
  if (observed) return words(observed[1]);
  return words(raw);
}

export function Channels({ channels }: { channels: Channel[] }) {
  const main = channels.filter((c) => !BOOKKEEPING.has(c.name));

  return (
    <span className="channels">
      {main.map((c) => (
        <span className="chan" key={c.name} title={c.name}>
          {channelName(c.name)}
          {c.width && c.width > 1 && <span className="w">{c.width}</span>}
        </span>
      ))}
      {/* The format's own row-ordering columns are not shown at all. They are
          not a finding, they are not a choice the contributor made, and a count
          of them is one more number to read past. */}
    </span>
  );
}
