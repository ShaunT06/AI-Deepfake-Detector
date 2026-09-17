"use client";

import { useRef, useState } from "react";
import { ApiError, MAX_UPLOAD_BYTES, predict, type PredictResponse } from "@/lib/api";

type Status = "idle" | "loading" | "error" | "done";

export default function Home() {
  const [file, setFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [status, setStatus] = useState<Status>("idle");
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<PredictResponse | null>(null);
  const [tab, setTab] = useState<"heatmap" | "report">("heatmap");
  const inputRef = useRef<HTMLInputElement>(null);

  function onSelectFile(f: File | null) {
    setResult(null);
    setError(null);
    setStatus("idle");
    if (!f) {
      setFile(null);
      setPreviewUrl(null);
      return;
    }
    if (f.size > MAX_UPLOAD_BYTES) {
      setError(`File is ${(f.size / 1024 / 1024).toFixed(1)}MB — please upload an image under 3.5MB.`);
      setFile(null);
      setPreviewUrl(null);
      return;
    }
    setFile(f);
    setPreviewUrl(URL.createObjectURL(f));
  }

  async function runAnalysis() {
    if (!file) return;
    setStatus("loading");
    setError(null);
    try {
      const res = await predict(file);
      setResult(res);
      setStatus("done");
      setTab("heatmap");
    } catch (e) {
      setError(e instanceof ApiError ? e.message : "Something went wrong. Please try again.");
      setStatus("error");
    }
  }

  const verdictIsFake = result?.is_fake ?? false;
  const verdictColor = verdictIsFake ? "var(--fake)" : "var(--real)";
  const confPct = result ? Math.round((verdictIsFake ? result.fake_prob : result.real_prob) * 100) : 0;

  return (
    <main className="mx-auto w-full max-w-4xl px-6 pb-16">
      {/* Hero */}
      <section className="pt-16 pb-12 text-center">
        <span
          className="inline-block rounded-full px-4 py-1.5 font-mono text-[0.65rem] tracking-[0.2em] uppercase"
          style={{ color: "var(--accent-2)", border: "1px solid #631bff55", background: "#631bff12" }}
        >
          🔬 Powered by Deep Learning + GradCAM
        </span>
        <h1
          className="mt-6 bg-gradient-to-br from-white via-[35%] via-[var(--accent-2)] to-[var(--pink)] bg-clip-text text-6xl font-extrabold tracking-tight text-transparent"
          style={{ fontFamily: "var(--font-syne)" }}
        >
          DeepScan
        </h1>
        <p className="mx-auto mt-4 max-w-lg text-[1.05rem] leading-relaxed" style={{ color: "var(--muted)" }}>
          Upload any face image and our AI will detect deepfake manipulation with pixel-level heatmap
          visualization.
        </p>
      </section>

      {/* Upload */}
      <section
        className="mb-8 rounded-[20px] p-8"
        style={{
          background: "linear-gradient(135deg, #0e0c1a 0%, #110d1f 100%)",
          border: "1px solid #631bff30",
        }}
      >
        <div className="mb-1 flex items-center gap-2 text-lg font-bold" style={{ fontFamily: "var(--font-syne)" }}>
          📁 Upload Image
        </div>
        <div className="mb-5 font-mono text-[0.82rem]" style={{ color: "var(--muted)" }}>
          JPG · PNG · WEBP · Max 3.5MB
        </div>

        <label
          className="flex cursor-pointer flex-col items-center justify-center rounded-2xl border-[1.5px] border-dashed p-8 text-center transition-colors"
          style={{ borderColor: "#631bff40", background: "#0a0814" }}
          onDragOver={(e) => e.preventDefault()}
          onDrop={(e) => {
            e.preventDefault();
            onSelectFile(e.dataTransfer.files?.[0] ?? null);
          }}
        >
          <input
            ref={inputRef}
            type="file"
            accept="image/jpeg,image/png,image/webp"
            className="hidden"
            onChange={(e) => onSelectFile(e.target.files?.[0] ?? null)}
          />
          {previewUrl ? (
            // eslint-disable-next-line @next/next/no-img-element
            <img src={previewUrl} alt="Selected upload" className="max-h-64 rounded-lg" />
          ) : (
            <span style={{ color: "var(--muted)" }}>Drop your image here or click to browse</span>
          )}
        </label>

        {file && (
          <div className="mt-2 font-mono text-[0.65rem]" style={{ color: "#3d3a50" }}>
            {file.name} · {(file.size / 1024).toFixed(0)}KB
          </div>
        )}

        {error && (
          <div
            className="mt-4 rounded-xl px-4 py-3 font-mono text-[0.8rem]"
            style={{ background: "#631bff12", border: "1px solid #631bff30", color: "var(--accent-2)" }}
          >
            {error}
          </div>
        )}

        <button
          disabled={!file || status === "loading"}
          onClick={runAnalysis}
          className="mt-6 w-full rounded-xl py-3.5 font-bold tracking-wide text-white transition-transform disabled:opacity-40"
          style={{
            fontFamily: "var(--font-syne)",
            background: "linear-gradient(135deg, #631bff, #8b5cf6)",
            boxShadow: "0 4px 24px #631bff44",
          }}
        >
          {status === "loading" ? "Running inference + GradCAM…" : "🔬  Run DeepFake Analysis"}
        </button>
      </section>

      {/* Result */}
      {result && (
        <>
          <section
            className="mb-6 rounded-[20px] p-8"
            style={{
              background: verdictIsFake
                ? "linear-gradient(135deg, #1f0a10 0%, #1a0c0c 100%)"
                : "linear-gradient(135deg, #0a1f14 0%, #0c1a10 100%)",
              border: `1px solid ${verdictIsFake ? "#ef444435" : "#22c55e35"}`,
            }}
          >
            <div
              className="mb-1 font-mono text-[0.65rem] tracking-[0.2em] uppercase"
              style={{ color: verdictIsFake ? "#f87171" : "#4ade80" }}
            >
              {verdictIsFake ? "⚠ Manipulation Detected" : "✓ Authenticity Confirmed"}
            </div>
            <div className="mb-1 text-4xl font-extrabold" style={{ fontFamily: "var(--font-syne)", color: verdictColor }}>
              {verdictIsFake ? "DEEPFAKE" : "AUTHENTIC"}
            </div>
            <div className="mb-6 text-sm" style={{ color: "var(--muted)" }}>
              {verdictIsFake
                ? "This image shows strong signs of AI-generated or manipulated content."
                : "No significant manipulation artifacts detected. Likely an authentic photograph."}
            </div>

            <div className="mb-6">
              <div className="mb-2 flex items-center justify-between">
                <span className="font-mono text-[0.7rem] tracking-[0.1em] uppercase" style={{ color: "var(--muted)" }}>
                  Confidence
                </span>
                <span className="text-lg font-bold" style={{ fontFamily: "var(--font-syne)", color: verdictColor }}>
                  {confPct}%
                </span>
              </div>
              <div className="h-2 overflow-hidden rounded-full" style={{ background: "#ffffff10" }}>
                <div
                  className="h-full rounded-full"
                  style={{
                    width: `${confPct}%`,
                    background: verdictIsFake
                      ? "linear-gradient(90deg, #991b1b, #ef4444, #f87171)"
                      : "linear-gradient(90deg, #16a34a, #22c55e, #4ade80)",
                  }}
                />
              </div>
            </div>

            <div className="flex gap-8 font-mono text-[0.72rem]" style={{ color: "var(--muted)" }}>
              <span>
                FAKE <b style={{ color: "#ef4444" }}>{(result.fake_prob * 100).toFixed(1)}%</b>
              </span>
              <span>
                REAL <b style={{ color: "#22c55e" }}>{(result.real_prob * 100).toFixed(1)}%</b>
              </span>
            </div>

            {!result.labels_verified && (
              <div className="mt-5 rounded-xl px-4 py-3 font-mono text-[0.75rem]" style={{ background: "#63160010", border: "1px solid #f9731640", color: "#f97316" }}>
                ⚠ This checkpoint&rsquo;s Real/Fake label order is unverified — see the project README before
                trusting this verdict.
              </div>
            )}
          </section>

          {/* Metrics */}
          <section className="mb-6 grid grid-cols-2 gap-4 sm:grid-cols-3">
            {[
              ["Fake Prob", result.fake_prob.toFixed(2)],
              ["Real Prob", result.real_prob.toFixed(2)],
              ["Confidence", Math.max(result.fake_prob, result.real_prob).toFixed(2)],
              ["Verdict", verdictIsFake ? "FAKE" : "REAL"],
              ["Model", result.backbone],
              ["Input Size", `${result.img_size}×${result.img_size}`],
            ].map(([label, value]) => (
              <div key={label} className="rounded-2xl p-4 text-center" style={{ background: "var(--panel)", border: "1px solid var(--panel-border)" }}>
                <div className="text-xl font-extrabold" style={{ fontFamily: "var(--font-syne)", color: "var(--accent-2)" }}>
                  {value}
                </div>
                <div className="font-mono text-[0.6rem] tracking-[0.12em] uppercase" style={{ color: "var(--muted)" }}>
                  {label}
                </div>
              </div>
            ))}
          </section>

          {/* Tabs */}
          <section className="mb-10">
            <div className="mb-4 flex gap-1 rounded-xl p-1" style={{ background: "var(--panel)" }}>
              {(["heatmap", "report"] as const).map((t) => (
                <button
                  key={t}
                  onClick={() => setTab(t)}
                  className="flex-1 rounded-lg py-2 font-mono text-[0.75rem] tracking-[0.08em] uppercase transition-colors"
                  style={
                    tab === t
                      ? { background: "#631bff25", color: "var(--accent-2)" }
                      : { color: "var(--muted)" }
                  }
                >
                  {t === "heatmap" ? "🌡 GradCAM Heatmap" : "📋 Analysis Report"}
                </button>
              ))}
            </div>

            {tab === "heatmap" ? (
              <div>
                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <div className="mb-2 text-sm font-bold" style={{ fontFamily: "var(--font-syne)" }}>
                      Original
                    </div>
                    {/* eslint-disable-next-line @next/next/no-img-element */}
                    <img src={previewUrl ?? ""} alt="Original" className="w-full rounded-xl" />
                  </div>
                  <div>
                    <div className="mb-2 text-sm font-bold" style={{ fontFamily: "var(--font-syne)" }}>
                      GradCAM Overlay
                    </div>
                    {/* eslint-disable-next-line @next/next/no-img-element */}
                    <img
                      src={`data:image/png;base64,${result.gradcam_image_base64}`}
                      alt="GradCAM overlay"
                      className="w-full rounded-xl"
                    />
                  </div>
                </div>
                <div className="mt-4 font-mono text-[0.8rem]" style={{ color: "var(--muted)" }}>
                  🔴 Red / warm areas = regions driving the predicted verdict · 🟢 Green / cool = low activation
                </div>
              </div>
            ) : (
              <div className="rounded-2xl p-6" style={{ background: "var(--panel)", border: "1px solid var(--panel-border)" }}>
                <div className="space-y-3 font-mono text-[0.8rem]" style={{ color: "var(--muted)" }}>
                  <p>
                    <b style={{ color: "var(--accent-2)" }}>Architecture:</b> {result.backbone}
                  </p>
                  <p>
                    <b style={{ color: "var(--accent-2)" }}>Input size:</b> {result.img_size}×{result.img_size} RGB
                  </p>
                  <p>
                    <b style={{ color: "var(--accent-2)" }}>Labels:</b>{" "}
                    {result.labels_verified ? "verified" : "UNVERIFIED — see README"}
                  </p>
                  <p className="pt-2">
                    {verdictIsFake
                      ? "GradCAM heatmap (see other tab) shows which regions drove this verdict."
                      : "Probability distribution favors the real class."}
                  </p>
                </div>
              </div>
            )}
          </section>
        </>
      )}

      {/* How it works */}
      <section className="mt-4 rounded-[20px] p-8" style={{ background: "var(--panel)", border: "1px solid var(--panel-border)" }}>
        <div className="mb-1 text-lg font-bold" style={{ fontFamily: "var(--font-syne)" }}>
          ⚙ How It Works
        </div>
        <div className="mb-6 font-mono text-[0.8rem]" style={{ color: "var(--muted)" }}>
          Three-stage pipeline from pixels to prediction
        </div>
        <div className="grid gap-6 sm:grid-cols-3">
          {[
            ["01", "Feature Extraction", "A convolutional backbone encodes spatial and texture features through progressive layers."],
            ["02", "Classification", "A custom classifier head outputs probability scores for REAL and FAKE classes."],
            ["03", "GradCAM Viz", "Gradient-weighted Class Activation Mapping highlights which image regions most influenced the verdict."],
          ].map(([num, title, desc]) => (
            <div key={num} className="text-center">
              <div
                className="mb-2 bg-gradient-to-br from-[var(--accent)] to-[var(--accent-2)] bg-clip-text text-4xl font-extrabold text-transparent"
                style={{ fontFamily: "var(--font-syne)" }}
              >
                {num}
              </div>
              <div className="mb-1 text-sm font-bold" style={{ fontFamily: "var(--font-syne)" }}>
                {title}
              </div>
              <div className="text-[0.78rem] leading-relaxed" style={{ color: "var(--muted)" }}>
                {desc}
              </div>
            </div>
          ))}
        </div>
      </section>

      <footer className="pt-12 pb-4 text-center font-mono text-[0.65rem] tracking-[0.1em]" style={{ color: "#3d3a50" }}>
        DeepScan · GradCAM · Built with Next.js
      </footer>
    </main>
  );
}
