/** Server state, in one place.
 *
 *  Every screen reads through these hooks, so cache invalidation happens in one
 *  spot and no component invents its own fetch. The SSE hook is the only thing
 *  that pushes: it listens to the submission's event log and invalidates the
 *  query, so live updates and manual refetches take the identical path.
 */

import { useEffect } from "react";
import {
  useMutation,
  useQuery,
  useQueryClient,
  type UseQueryResult,
} from "@tanstack/react-query";
import { useState } from "react";
import type { Benchmark, Gate, Progress, Submission } from "./types";

async function json<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(path, {
    ...init,
    headers: { "Content-Type": "application/json", ...(init?.headers ?? {}) },
  });
  if (!response.ok) {
    const detail = await response.text().catch(() => "");
    throw new Error(detail || `${response.status} ${response.statusText}`);
  }
  return (await response.json()) as T;
}

export const keys = {
  me: ["me"] as const,
  benchmarks: ["benchmarks"] as const,
  submissions: ["submissions"] as const,
  submission: (id: string) => ["submission", id] as const,
};

export function useMe() {
  return useQuery({
    queryKey: keys.me,
    queryFn: () => json<{ email: string; org: { id: string; name: string } }>("/api/me"),
  });
}

export function useBenchmarks() {
  return useQuery({
    queryKey: keys.benchmarks,
    queryFn: () => json<{ benchmarks: Benchmark[]; gates: Gate[] }>("/api/benchmarks"),
    staleTime: 5 * 60 * 1000,
  });
}

export function useSubmissions() {
  return useQuery({
    queryKey: keys.submissions,
    queryFn: () => json<{ submissions: Submission[] }>("/api/submissions"),
  });
}

export function useSubmission(id: string | undefined): UseQueryResult<Submission> {
  return useQuery({
    queryKey: keys.submission(id ?? ""),
    queryFn: () => json<Submission>(`/api/submissions/${id}`),
    enabled: Boolean(id),
  });
}

export function useCreateSubmission() {
  const client = useQueryClient();
  return useMutation({
    mutationFn: (body: { name: string; benchmark: string }) =>
      json<Submission>("/api/submissions", { method: "POST", body: JSON.stringify(body) }),
    onSuccess: () => client.invalidateQueries({ queryKey: keys.submissions }),
  });
}

/** Upload with progress. XHR rather than fetch because a contributor watching a
 *  gigabyte crawl needs a real percentage, and fetch still cannot report it. */
export function uploadDataset(
  id: string,
  file: File,
  onProgress: (fraction: number) => void,
): Promise<Submission> {
  return new Promise((resolve, reject) => {
    const form = new FormData();
    form.append("file", file);
    const request = new XMLHttpRequest();
    request.open("POST", `/api/submissions/${id}/dataset`);
    request.upload.onprogress = (event) => {
      if (event.lengthComputable) onProgress(event.loaded / event.total);
    };
    request.onload = () =>
      request.status < 400
        ? resolve(JSON.parse(request.responseText))
        : reject(new Error(request.responseText || `upload failed (${request.status})`));
    request.onerror = () => reject(new Error("network error during upload"));
    request.send(form);
  });
}

export function useConfirmMeaning(id: string) {
  const client = useQueryClient();
  return useMutation({
    mutationFn: (body: Record<string, string>) =>
      json<Submission>(`/api/submissions/${id}/meaning`, {
        method: "POST",
        body: JSON.stringify(body),
      }),
    onSuccess: () => client.invalidateQueries({ queryKey: keys.submission(id) }),
  });
}

/** Listen to one submission, live. Two channels, deliberately different.
 *
 *  Default `message` frames are the durable log. They carry *what* happened and
 *  never the state itself; the record is refetched so there is one source of
 *  truth. A page that rendered from the event payload would slowly drift from
 *  the record it claims to show.
 *
 *  `progress` frames are where the running gate is. These are returned rather
 *  than cached, because refetching the whole submission every time a training
 *  step ticks would be absurd — and because progress is the one thing on the
 *  page that is genuinely allowed to be a moment out of date.
 *
 *  The connection is *not* closed on error. EventSource reconnects on its own
 *  and sends `Last-Event-ID`, which the server uses to replay exactly what was
 *  missed. Closing the source is what made a dropped connection permanent, and
 *  a four-hour run is where a connection actually drops. */
export function useSubmissionEvents(id: string | undefined): Progress | null {
  const client = useQueryClient();
  const [progress, setProgress] = useState<Progress | null>(null);

  useEffect(() => {
    if (!id) return;
    setProgress(null);
    const source = new EventSource(`/api/submissions/${id}/events`);
    source.onmessage = () => {
      // Anything durable happened: a gate started, finished, was queued. The
      // position it was last at no longer means anything.
      setProgress(null);
      client.invalidateQueries({ queryKey: keys.submission(id) });
      client.invalidateQueries({ queryKey: keys.submissions });
    };
    source.addEventListener("progress", (event) => {
      setProgress(JSON.parse((event as MessageEvent).data) as Progress);
    });
    return () => source.close();
  }, [id, client]);

  return progress;
}
