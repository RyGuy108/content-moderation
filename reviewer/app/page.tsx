"use client";

import { useCallback, useEffect, useRef, useState, type ReactElement } from "react";
import type { Item, Reason, Stats } from "./types";

const API = process.env.NEXT_PUBLIC_MODPIPE_API ?? "http://localhost:8000";
const LABELS = ["sexual", "minor_sexual", "gore", "hate", "self_harm", "real_person", "trademark",
  "nsfw_exposed", "protected_face", "logo"];

function useReviewer(): [string, (v: string) => void] {
  const [name, setName] = useState("");
  useEffect(() => {
    try { setName(localStorage.getItem("reviewer") ?? ""); } catch { /* private mode */ }
  }, []);
  const set = (v: string) => {
    setName(v);
    try { localStorage.setItem("reviewer", v); } catch { /* ignore */ }
  };
  return [name, set];
}

/** Draws detector evidence (NSFW part boxes, matched face, logo quad) over the image. */
function Evidence({ src, reasons }: { src: string; reasons: Reason[] }) {
  const [size, setSize] = useState<{ w: number; h: number } | null>(null);
  const shapes: ReactElement[] = [];
  reasons.forEach((r, i) => {
    const ev = r.evidence;
    if (!ev) return;
    ev.parts?.forEach((p, j) => shapes.push(
      <rect key={`p${i}${j}`} x={p.box[0]} y={p.box[1]} width={p.box[2]} height={p.box[3]} className="ev ev-nsfw" />));
    if (ev.box) shapes.push(
      <rect key={`f${i}`} x={ev.box[0]} y={ev.box[1]} width={ev.box[2]} height={ev.box[3]} className="ev ev-face" />);
    if (ev.quad) shapes.push(
      <polygon key={`q${i}`} points={ev.quad.map((p) => p.join(",")).join(" ")} className="ev ev-logo" />);
  });
  return (
    <div className="media">
      {/* eslint-disable-next-line @next/next/no-img-element */}
      <img src={src} alt="content under review" onLoad={(e) =>
        setSize({ w: e.currentTarget.naturalWidth, h: e.currentTarget.naturalHeight })} />
      {size && <svg viewBox={`0 0 ${size.w} ${size.h}`} preserveAspectRatio="none">{shapes}</svg>}
    </div>
  );
}

export default function ReviewPage() {
  const [reviewer, setReviewer] = useReviewer();
  const [item, setItem] = useState<Item | null>(null);
  const [empty, setEmpty] = useState(false);
  const [labels, setLabels] = useState<Set<string>>(new Set());
  const [note, setNote] = useState("");
  const [stats, setStats] = useState<Stats | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const shown = useRef<number>(0);

  const refreshStats = useCallback(async () => {
    try { setStats(await (await fetch(`${API}/v1/review/stats`)).json()); } catch { /* api down */ }
  }, []);

  const next = useCallback(async () => {
    if (!reviewer) return;
    setBusy(true); setError(null);
    try {
      const r = await fetch(`${API}/v1/review/claim?reviewer=${encodeURIComponent(reviewer)}`, { method: "POST" });
      const body = await r.json();
      if (body.empty) { setItem(null); setEmpty(true); }
      else {
        setItem(body); setEmpty(false); shown.current = Date.now();
        setLabels(new Set((body as Item).check.reasons.filter((x) => LABELS.includes(x.label)).map((x) => x.label)));
        setNote("");
      }
    } catch (e) { setError(String(e)); }
    setBusy(false);
    refreshStats();
  }, [reviewer, refreshStats]);

  const resolve = useCallback(async (decision: "allow" | "block") => {
    if (!item || busy) return;
    setBusy(true);
    const r = await fetch(`${API}/v1/review/${item.id}`, {
      method: "POST", headers: { "content-type": "application/json" },
      body: JSON.stringify({ reviewer, decision, labels: decision === "block" ? [...labels] : [], note: note || null }),
    });
    setBusy(false);
    if (r.status === 409) setError("Your lease on this item expired and someone else may have it. Loading the next one.");
    next();
  }, [item, busy, reviewer, labels, note, next]);

  // a = allow, b = block, n = skip to next
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if ((e.target as HTMLElement).tagName === "TEXTAREA" || (e.target as HTMLElement).tagName === "INPUT") return;
      if (e.key === "a") resolve("allow");
      if (e.key === "b") resolve("block");
      if (e.key === "n") next();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [resolve, next]);

  useEffect(() => { refreshStats(); const t = setInterval(refreshStats, 5000); return () => clearInterval(t); }, [refreshStats]);

  const chk = item?.check;
  const media = chk?.input.media?.[0];
  return (
    <main>
      <header>
        <h1>Review queue</h1>
        {stats && (
          <div className="stats">
            <span><b>{stats.pending}</b> pending</span>
            <span><b>{stats.claimed}</b> in review</span>
            <span><b>{stats.resolved}</b> resolved</span>
            {stats.p50_time_to_decision_s != null &&
              <span>p50 time to decision <b>{Math.round(stats.p50_time_to_decision_s)} s</b></span>}
          </div>
        )}
      </header>

      {!reviewer || (!item && !empty) ? (
        <section className="card start">
          <label>Your name
            <input value={reviewer} onChange={(e) => setReviewer(e.target.value)} placeholder="e.g. ryan" />
          </label>
          <button disabled={!reviewer || busy} onClick={next}>Start reviewing</button>
        </section>
      ) : empty ? (
        <section className="card start"><p>Queue is empty.</p><button onClick={next}>Check again</button></section>
      ) : chk && (
        <section className="item">
          <div className="left">
            {media && (chk.subject === "video"
              ? <video src={`${API}/v1/media/${media}`} controls muted loop />
              : <Evidence src={`${API}/v1/media/${media}`} reasons={chk.reasons} />)}
            {chk.input.prompt && <blockquote>{chk.input.prompt}</blockquote>}
          </div>
          <div className="right card">
            <div className="meta">
              <span className={`pill ${chk.action}`}>{chk.action}</span>
              <span>{chk.stage === "pre" ? "before generation" : "generated output"}</span>
              <span>tenant {chk.tenant}</span>
              <span>priority {item!.priority}</span>
            </div>
            <h2>Why it was flagged</h2>
            <ul className="reasons">
              {chk.reasons.map((r, i) => (
                <li key={i}>
                  <span className="lbl">{r.label}</span>
                  {r.score != null && <span className="bar"><i style={{ width: `${Math.round(r.score * 100)}%` }} /></span>}
                  <span className="detail">
                    {r.evidence?.identity && `matches ${r.evidence.identity} (d=${r.evidence.distance})`}
                    {r.evidence?.logo && `mark "${r.evidence.logo}", ${r.evidence.inliers} matched points`}
                    {r.evidence?.mentions && `mentions ${r.evidence.mentions.join(", ")}`}
                    {r.flags && `obfuscated: ${r.flags.join(", ")}`}
                    {r.evidence?.t != null && ` at ${r.evidence.t}s`}
                  </span>
                </li>
              ))}
            </ul>
            <h2>Correct labels</h2>
            <div className="labels">
              {LABELS.map((l) => (
                <label key={l} className={labels.has(l) ? "on" : ""}>
                  <input type="checkbox" checked={labels.has(l)} onChange={() => {
                    const s = new Set(labels); if (s.has(l)) s.delete(l); else s.add(l); setLabels(s);
                  }} />{l}
                </label>
              ))}
            </div>
            <textarea value={note} onChange={(e) => setNote(e.target.value)} placeholder="Note (optional)" />
            <div className="actions">
              <button className="allow" disabled={busy} onClick={() => resolve("allow")}>Allow <kbd>a</kbd></button>
              <button className="block" disabled={busy} onClick={() => resolve("block")}>Block <kbd>b</kbd></button>
            </div>
            {error && <p className="error">{error}</p>}
          </div>
        </section>
      )}
    </main>
  );
}
