/** The wizard. One decision per step, and the upload starts working for the
 *  user the moment it lands rather than after they finish clicking. */

import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { useBenchmarks, useCreateSubmission, uploadDataset } from "../api/client";
import { ErrorNote, bytes } from "../components/ui";

const STEPS = ["Name & benchmark", "Upload", "Intake"];

export function NewSubmission() {
  const navigate = useNavigate();
  const benchmarks = useBenchmarks();
  const create = useCreateSubmission();

  const [step, setStep] = useState(0);
  const [name, setName] = useState("");
  const [benchmark, setBenchmark] = useState("pick_dual_bottles");
  const [submissionId, setSubmissionId] = useState<string | null>(null);
  const [file, setFile] = useState<File | null>(null);
  const [progress, setProgress] = useState(0);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<unknown>(null);
  const [over, setOver] = useState(false);

  const chosen = benchmarks.data?.benchmarks.find((b) => b.key === benchmark);

  async function startSubmission() {
    setError(null);
    try {
      const sub = await create.mutateAsync({ name: name.trim() || "untitled", benchmark });
      setSubmissionId(sub.id);
      setStep(1);
    } catch (err) {
      setError(err);
    }
  }

  async function send() {
    if (!file || !submissionId) return;
    setBusy(true);
    setError(null);
    try {
      await uploadDataset(submissionId, file, setProgress);
      // Intake is queued the instant the bytes land; the submission page is
      // where it is watched, so go there rather than inventing a third state.
      navigate(`/submissions/${submissionId}`);
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
          <p>Two steps to a free readability check and a report on your footage.</p>
        </div>
      </div>

      <div className="steps">
        {STEPS.map((label, index) => (
          <div key={label} className={`s ${index === step ? "on" : index < step ? "done" : ""}`}>
            {index + 1}. {label}
          </div>
        ))}
      </div>

      {error ? <ErrorNote error={error} /> : null}

      {step === 0 && (
        <div className="card pad">
          <label className="field">
            <span className="lab">Name this submission</span>
            <input
              type="text"
              value={name}
              placeholder="kitchen-ego-v1"
              onChange={(e) => setName(e.target.value)}
            />
            <span className="hint">
              Use a name you can recognise later — resubmissions keep the history together.
            </span>
          </label>

          <label className="field">
            <span className="lab">Test against</span>
            <select value={benchmark} onChange={(e) => setBenchmark(e.target.value)}>
              {benchmarks.data?.benchmarks.map((b) => (
                <option key={b.key} value={b.key}>
                  {b.name}
                </option>
              ))}
            </select>
            {chosen && (
              <span className="hint">
                {chosen.simulator} · {chosen.embodiment}
                {chosen.reference?.baseline &&
                  ` · today's baseline solves ${chosen.reference.baseline.wins}/${chosen.reference.baseline.n}` +
                    (chosen.reference.expert
                      ? `, the scripted expert ${Math.round(chosen.reference.expert * 100)}%`
                      : "")}
              </span>
            )}
          </label>

          <button className="btn primary" onClick={startSubmission} disabled={create.isPending}>
            {create.isPending ? "Creating…" : "Continue"}
          </button>
        </div>
      )}

      {step === 1 && (
        <div className="card pad">
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
            {file ? bytes(file.size) : "LeRobot v2 export — meta/, data/ and videos/"}
            <input
              type="file"
              accept=".zip"
              style={{ display: "none" }}
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

          <div style={{ marginTop: 18, display: "flex", gap: 8 }}>
            <button className="btn primary" onClick={send} disabled={!file || busy}>
              {busy ? "Uploading…" : "Upload and run intake"}
            </button>
            <button className="btn" onClick={() => setStep(0)} disabled={busy}>
              Back
            </button>
          </div>

          <div className="note" style={{ marginTop: 18 }}>
            <b>Nothing is charged here.</b> Intake and the data report are free. You will see
            what we found before anything paid is offered.
          </div>
        </div>
      )}
    </div>
  );
}
