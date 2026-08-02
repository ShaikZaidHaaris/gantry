/** What changed between two uploads.
 *
 *  The product loop is submit, fix, resubmit, see the change, and the last step
 *  is the one that makes the first three worth doing. Without it a contributor
 *  who refilms has two reports and no way to tell whether refilming worked.
 *
 *  What this deliberately does not do is declare a winner. Two versions were
 *  measured on their own runs, so "v2 is better" is a comparison between two
 *  experiments; its honest home is the leaderboard, where it is paired scene by
 *  scene. Here the job is narrower and more useful: say which findings went
 *  away, which arrived, and which are still there.
 *
 *  A finding that is *gone* is the headline, because that is the thing the
 *  contributor did. A finding that is still there is second, because it is what
 *  to do next. New findings come last and are not styled as failure, a
 *  refilmed corpus that surfaces something previously hidden behind a bigger
 *  problem is progress, not regression.
 */

import { useRef, useState } from "react";
import { useQueryClient } from "@tanstack/react-query";
import { keys, uploadDataset, useVersions } from "../api/client";
import type { VersionChange } from "../api/types";
import { Skeleton, sentence } from "./ui";

/** Upload a new version of the same dataset.
 *
 *  On the submission rather than behind "new submission", because a refilm is
 *  the same question asked again, putting it in the wizard would produce two
 *  unrelated rows and lose the only thing that makes resubmitting worth doing,
 *  which is the comparison with what came before. */
export function Resubmit({ id }: { id: string }) {
  const input = useRef<HTMLInputElement>(null);
  const client = useQueryClient();
  const [progress, setProgress] = useState<number | null>(null);
  const [failed, setFailed] = useState<string | null>(null);

  async function send(file: File) {
    setFailed(null);
    setProgress(0);
    try {
      await uploadDataset(id, file, setProgress);
      await client.invalidateQueries({ queryKey: keys.submission(id) });
      await client.invalidateQueries({ queryKey: ["versions", id] });
    } catch (error) {
      setFailed(error instanceof Error ? error.message : String(error));
    } finally {
      setProgress(null);
    }
  }

  return (
    <div className="resubmit">
      <input
        ref={input}
        type="file"
        accept=".zip"
        hidden
        onChange={(event) => {
          const file = event.target.files?.[0];
          if (file) void send(file);
          event.target.value = "";
        }}
      />
      <button
        type="button"
        className="btn"
        onClick={() => input.current?.click()}
        disabled={progress !== null}
      >
        {progress === null ? "Upload a new version" : `Uploading ${(progress * 100).toFixed(0)}%`}
      </button>
      <span className="resubmit-note">
        Every check runs again on the new upload. The old version keeps its results, so you can
        see what your changes did.
      </span>
      {progress !== null && (
        <div className="bar" style={{ marginTop: 10 }}>
          <i style={{ width: `${progress * 100}%` }} />
        </div>
      )}
      {failed && (
        <div className="note warn" style={{ marginTop: 10 }}>
          <b>That upload did not go through.</b>
          <div style={{ marginTop: 4 }}>{failed}</div>
        </div>
      )}
    </div>
  );
}

function delta(before: number | null, after: number | null): string | null {
  if (before === null || after === null || before === after) return null;
  const change = after - before;
  return `${change > 0 ? "+" : ""}${change}`;
}

function Change({ change }: { change: VersionChange }) {
  const clips = delta(change.clips[0] ?? null, change.clips[1] ?? null);
  const frames = delta(change.frames[0] ?? null, change.frames[1] ?? null);
  return (
    <div className="card pad change">
      <div className="change-head">
        <span className="vpair">
          v{change.from} <span className="arrow">→</span> v{change.to}
        </span>
        <span className="dim mono">
          {change.clips[1] ?? "—"} clips{clips && ` (${clips})`} ·{" "}
          {change.frames[1] ?? "—"} frames{frames && ` (${frames})`}
        </span>
      </div>

      {change.fixed.length > 0 && (
        <div className="change-group fixed">
          <b>
            {change.fixed.length} gone
          </b>
          {change.fixed.map((f) => (
            <div className="line" key={f.code} title={f.code}>
              {sentence(f.summary)}
            </div>
          ))}
        </div>
      )}

      {change.remaining.length > 0 && (
        <div className="change-group remaining">
          <b>{change.remaining.length} still there</b>
          {change.remaining.map((f) => (
            <div className="line" key={f.code} title={f.code}>
              {sentence(f.summary)}
            </div>
          ))}
        </div>
      )}

      {change.new.length > 0 && (
        <div className="change-group appeared">
          <b>{change.new.length} new</b>
          {change.new.map((f) => (
            <div className="line" key={f.code} title={f.code}>
              {sentence(f.summary)}
            </div>
          ))}
          <p className="aside">
            A new finding is not always a step back. Fixing the biggest problem tends to reveal
            whatever was sitting behind it.
          </p>
        </div>
      )}

      {change.fixed.length === 0 && change.new.length === 0 && (
        <div className="what">
          The checks found the same things as last time, even though the upload changed.
        </div>
      )}
    </div>
  );
}

export function Versions({ id }: { id: string }) {
  const { data, isPending } = useVersions(id);

  if (isPending) {
    return (
      <div className="card">
        <Skeleton rows={3} />
      </div>
    );
  }
  // The upload control shows from the first version; the comparison only once
  // there is something to compare against.
  if (!data) return null;
  if (data.versions.length < 2) {
    return (
      <>
        <h2>
          Versions
          <span className="h2-sub">Refilm, upload again, and see what changed</span>
        </h2>
        <div className="card pad">
          <Resubmit id={id} />
        </div>
      </>
    );
  }

  return (
    <>
      <h2>
        Since your last upload
        <span className="h2-sub">
          {data.versions.length} versions · showing v{data.current}
        </span>
      </h2>
      <div className="changes">
        <div className="card pad">
          <Resubmit id={id} />
        </div>
        {[...data.changes].reverse().map((change) => (
          <Change key={`${change.from}-${change.to}`} change={change} />
        ))}
      </div>
    </>
  );
}
