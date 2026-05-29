import { useState } from "react";
import type { Analysis, Citation } from "../lib/api";

function CitationChip({ c }: { c: Citation }) {
  const [open, setOpen] = useState(false);
  return (
    <div className="w-full">
      <button
        onClick={() => setOpen((o) => !o)}
        className={`inline-flex items-center gap-2 rounded-lg border px-3 py-1.5 text-sm font-medium transition ${
          c.exists
            ? "border-emerald-500/40 bg-emerald-500/10 text-emerald-300 hover:bg-emerald-500/20"
            : "border-red-500/50 bg-red-500/10 text-red-300 hover:bg-red-500/20"
        }`}
      >
        <span>{c.exists ? "✓" : "✗"}</span>
        <span className="font-mono">{c.label}</span>
        {c.heading && <span className="text-zinc-400 font-normal">· {c.heading}</span>}
        <span className="text-zinc-500">{open ? "▲" : "▼"}</span>
      </button>
      {open && (
        <div className="mt-2 rounded-lg border border-zinc-700 bg-zinc-900/80 p-3 text-sm">
          {c.exists ? (
            <>
              <div className="mb-1 font-semibold text-zinc-200">
                {c.label} — {c.heading}
              </div>
              <p className="text-zinc-400 leading-relaxed">{c.snippet}</p>
              <a
                href={c.url}
                target="_blank"
                rel="noreferrer"
                className="mt-2 inline-block text-amber-400 hover:underline"
              >
                Read full text on eCFR ↗
              </a>
            </>
          ) : (
            <div className="text-red-300">
              ⚠ This section does not exist in the OSHA corpus — likely a hallucinated
              citation.
            </div>
          )}
        </div>
      )}
    </div>
  );
}

export default function AnswerCard({
  title,
  data,
  accent,
}: {
  title: string;
  data: Analysis;
  accent: "amber" | "zinc";
}) {
  const prose = data.text.replace(/\n?Citation:.*$/s, "").trim();
  const allValid = data.n_citations > 0 && data.n_valid === data.n_citations;
  return (
    <div className="card p-5 flex flex-col gap-4">
      <div className="flex items-center justify-between">
        <h3
          className={`text-sm font-semibold uppercase tracking-wide ${
            accent === "amber" ? "text-amber-400" : "text-zinc-400"
          }`}
        >
          {title}
        </h3>
        <span
          className={`rounded-full px-2.5 py-0.5 text-xs font-medium ${
            data.n_citations === 0
              ? "bg-zinc-700/50 text-zinc-300"
              : allValid
              ? "bg-emerald-500/15 text-emerald-300"
              : "bg-red-500/15 text-red-300"
          }`}
        >
          {data.n_valid}/{data.n_citations} citations verified
        </span>
      </div>
      <p className="whitespace-pre-wrap leading-relaxed text-zinc-100">{prose}</p>
      <div className="flex flex-col gap-2">
        {data.citations.length === 0 && (
          <span className="text-sm text-zinc-500">No citation produced.</span>
        )}
        {data.citations.map((c) => (
          <CitationChip key={c.section} c={c} />
        ))}
      </div>
    </div>
  );
}
