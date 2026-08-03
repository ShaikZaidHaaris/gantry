/** One form, one button, one trip to the server.
 *
 *  This was a two-step wizard: name it, press Continue, then upload. The split
 *  bought nothing. Neither half was a decision the other depended on, the first
 *  step held one text field and a line of read-only text, and the step
 *  indicator above them was more chrome than either step had content.
 *
 *  Worse, pressing Continue created a real submission server-side. Anybody who
 *  stopped there, which is most people who open a form to see what it wants,
 *  left an empty record behind with no dataset attached and no way to tell it
 *  apart from an upload that failed. Nothing is created now until there is a
 *  file to attach, and the create and the upload happen back to back from one
 *  press.
 */

import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { useBenchmarks, useCreateSubmission, useMe, uploadDataset } from "../api/client";
import { ErrorNote, bytes } from "../components/ui";

/** The one benchmark. Fixed while there is only one, because a select with a single
 *  option is a decision the product does not have. The value still travels
 *  with the submission; nobody is asked to pick it. */
const BENCHMARK = "pick_dual_bottles";

export function NewSubmission() {
  const navigate = useNavigate();
  const benchmarks = useBenchmarks();
  const create = useCreateSubmission();

  const [name, setName] = useState("");
  const [email, setEmail] = useState("");
  const [file, setFile] = useState<File | null>(null);
  const [progress, setProgress] = useState(0);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<unknown>(null);
  const [over, setOver] = useState(false);

  const chosen = benchmarks.data?.benchmarks.find((b) => b.key === BENCHMARK);

  // The server's number, not a copy of it. A limit written into this screen
  // separately is one that eventually disagrees with the check, and the visitor
  // finds out by waiting for an upload that was never going to be accepted.
  const limit = useMe().data?.max_upload_bytes ?? 1024 ** 3;
  const tooBig = file !== null && file.size > limit;

  async function send() {
    if (!file || tooBig) return;
    setBusy(true);
    setError(null);
    try {
      const sub = await create.mutateAsync({
        name: name.trim() || "untitled",
        benchmark: BENCHMARK,
        email: email.trim(),
      });
      await uploadDataset(sub.id, file, setProgress);
      // Intake is queued the instant the bytes land; the submission page is
      // where it is watched, so go there rather than inventing a third state.
      navigate(`/submissions/${sub.id}`);
    } catch (err) {
      setError(err);
      setBusy(false);
    }
  }

  return (
    <div className="page" style={{ maxWidth: 720 }}>
      <div className="page-head">
        <div>
          <h1>New submission</h1>
          <p>Name it, attach it, and the first check starts as soon as the bytes land.</p>
        </div>
      </div>

      {error ? <ErrorNote error={error} /> : null}

      <div className="card pad">
        <label className="field">
          <span className="lab">Name this submission</span>
          <input
            type="text"
            value={name}
            placeholder="kitchen-ego-v1"
            onChange={(e) => setName(e.target.value)}
            disabled={busy}
          />
          <span className="hint">
            Use a name you will recognise later. Resubmissions keep the history together.
          </span>
        </label>

        {/* Optional, and it says so, because a required field here would be an
            account by another name. But the reason to fill it in is specific
            rather than "stay in touch": the robot test runs for hours with
            nobody watching, and identity on this site is your IP address, so a
            changed network is otherwise a submission you cannot get back to. */}
        <label className="field">
          <span className="lab">
            Email <span className="opt">optional</span>
          </span>
          <input
            type="email"
            value={email}
            placeholder="you@lab.edu"
            onChange={(e) => setEmail(e.target.value)}
            disabled={busy}
          />
          <span className="hint">
            Only so we can reach you if a check breaks on our side, hours in, with
            nobody watching. It is also the way back to this submission if your network
            changes. You are identified here by address, and a new one looks like a
            new person.
          </span>
        </label>

        <div className="field">
          <span className="lab">Tested against</span>
          <div className="hint" style={{ marginTop: 0 }}>
            {chosen
              ? `${chosen.name} · ${chosen.simulator} · ${chosen.embodiment}`
              : "Pick two bottles (dual-arm)"}
          </div>
        </div>

        <label
          className={`drop ${over ? "over" : ""}`}
          onDragOver={(e) => {
            e.preventDefault();
            setOver(true);
          }}
          onDragLeave={() => setOver(false)}
          onDrop={(e) => {
            e.preventDefault();
            setOver(false);
            const dropped = e.dataTransfer.files[0];
            if (dropped) setFile(dropped);
          }}
        >
          <b>{file ? file.name : "Drop your dataset (.zip)"}</b>
          {file
            ? `${bytes(file.size)}${tooBig ? `, over the ${bytes(limit)} limit` : ""}`
            : `LeRobot v2 export with meta/, data/ and videos/ · up to ${bytes(limit)}`}
          <input
            type="file"
            accept=".zip"
            style={{ display: "none" }}
            disabled={busy}
            onChange={(e) => setFile(e.target.files?.[0] ?? null)}
          />
        </label>

        {busy && (
          <div style={{ marginTop: 16 }}>
            <div className="bar">
              <i style={{ width: `${Math.round(progress * 100)}%` }} />
            </div>
            <div className="mono" style={{ marginTop: 6, color: "var(--text-3)" }}>
              {Math.round(progress * 100)}% uploaded
            </div>
          </div>
        )}

        <div style={{ marginTop: 18 }}>
          <button className="btn primary" onClick={send} disabled={!file || busy || tooBig}>
            {busy ? "Uploading…" : "Upload and run intake"}
          </button>
        </div>

        {tooBig && (
          <div className="note danger" style={{ marginTop: 18 }}>
            <b>This archive is {bytes(file.size)}, over the {bytes(limit)} limit.</b> Trim
            the export, or upload a subset of the episodes: the checks read the same
            things either way.
          </div>
        )}

        <div className="note" style={{ marginTop: 18 }}>
          <b>Up to {bytes(limit)} per upload.</b> Larger archives are refused before they
          finish sending, so trim the export or send a subset of the episodes rather than
          waiting on one that cannot be accepted.
        </div>

        <div className="note" style={{ marginTop: 18 }}>
          <b>Nothing is created until you upload.</b> The first check reads the archive
          and reports what is in it; you see that before anything further is offered.
        </div>
      </div>
    </div>
  );
}
