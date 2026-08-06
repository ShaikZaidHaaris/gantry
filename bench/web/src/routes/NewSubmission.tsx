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
import { fetchFromHub, useBenchmarks, useCreateSubmission, useMe, uploadDataset } from "../api/client";
import { ErrorNote, bytes } from "../components/ui";

/** Enough shape-checking to catch a pasted profile link or a stray sentence
 *  before a round trip; the server stays the authority on what a repo id is. */
function looksLikeRepo(raw: string): boolean {
  const text = raw
    .trim()
    .replace(/^https?:\/\/(www\.)?huggingface\.co\//, "")
    .replace(/^hf\.co\//, "")
    .replace(/^datasets\//, "")
    .split(/[?#]/)[0]
    .replace(/\/$/, "");
  const parts = text.split("/");
  return parts.length >= 2 && parts[0].length > 0 && parts[1].length > 0 && !/\s/.test(text);
}

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
  const [repo, setRepo] = useState("");
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

  // Two ways in, one at a time. A pasted link wins over a chosen file because
  // pasting is the later, more deliberate act -- and sending both would be
  // asking the server to guess which one the visitor meant.
  const viaHub = repo.trim().length > 0;
  const repoOk = viaHub && looksLikeRepo(repo);
  const ready = viaHub ? repoOk : file !== null && !tooBig;

  async function send() {
    if (!ready) return;
    setBusy(true);
    setError(null);
    try {
      const sub = await create.mutateAsync({
        name: name.trim() || (viaHub ? repo.trim().split("/").slice(-1)[0] : "untitled"),
        benchmark: BENCHMARK,
        email: email.trim(),
      });
      if (viaHub) {
        await fetchFromHub(sub.id, repo.trim());
      } else {
        await uploadDataset(sub.id, file!, setProgress);
      }
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

        {/* The path most people can actually take. Nobody browses with a
            LeRobot archive on the machine in front of them; everybody's
            dataset is already on the Hub, one paste away. So the link comes
            first and the upload is the fallback, not the other way round. */}
        <label className="field">
          <span className="lab">Already on Hugging Face?</span>
          <input
            type="text"
            value={repo}
            placeholder="huggingface.co/datasets/you/your-dataset"
            onChange={(e) => setRepo(e.target.value)}
            disabled={busy}
          />
          <span className="hint">
            Paste the dataset page's link, or just the id like <code>you/your-dataset</code>.
            We pull it on our side, so there is no upload and no {bytes(limit)} cap. Public
            LeRobot v2 datasets only.
          </span>
        </label>

        {viaHub && !repoOk && (
          <div className="note" style={{ marginTop: 6 }}>
            That does not look like a dataset link yet. It should end in{" "}
            <code>owner/name</code>, like <code>huggingface.co/datasets/lerobot/pusht</code>.
          </div>
        )}

        {!viaHub && (
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
          <b>{file ? file.name : "Or drop your dataset (.zip)"}</b>
          {file
            ? `${bytes(file.size)}${tooBig ? `, over the ${bytes(limit)} limit` : ""}`
            : `A LeRobot v2 export, or your own video with a clips.json · up to ${bytes(limit)}`}
          <input
            type="file"
            accept=".zip"
            style={{ display: "none" }}
            disabled={busy}
            onChange={(e) => setFile(e.target.files?.[0] ?? null)}
          />
        </label>
        )}

        {/* Said here rather than only inside "How this works". Somebody who
            came to upload footage of a person has already decided to try, and
            finding out on this screen that it is accepted is the difference
            between uploading and leaving. There is no control to set: which of
            the two an archive is, and whether it carries poses, is answered by
            what is in it, and a switch that disagreed with the file would only
            be a way to be wrong. */}
        <p className="drop-note">
          Both shapes give the same answer, so send whichever you have.{" "}
          <b>A robot recording</b> is used as it is. <b>Video of a person</b> needs a{" "}
          <code>clips.json</code> saying what each clip shows and where it was filmed, and
          we work out the arm movements from the hands, which takes a few minutes.{" "}
          <b>If you already track hands</b>, include a <code>poses/</code> folder and we
          use yours instead of estimating. The result records which it was.
        </p>

        {busy && !viaHub && (
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
          <button className="btn primary" onClick={send} disabled={!ready || busy}>
            {busy
              ? viaHub
                ? "Queueing the fetch…"
                : "Uploading…"
              : viaHub
                ? "Fetch it and run intake"
                : "Upload and run intake"}
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
