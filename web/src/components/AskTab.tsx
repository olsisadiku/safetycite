import { useState } from "react";
import { api, type AskResponse, type DomainInfo } from "../lib/api";
import AnswerCard from "./AnswerCard";

const EXAMPLES = [
  "A worker is on a flat roof 15 feet up with no guardrails or harness. What standard applies?",
  "A maintenance tech is servicing a conveyor and must control hazardous energy. Which OSHA standard?",
  "An employee is hospitalized from a work injury. How soon must we report it to OSHA?",
  "Employees will enter a 7-foot trench in clay soil. What protective-system requirement applies?",
];

export default function AskTab({ domains }: { domains: DomainInfo[] }) {
  const [question, setQuestion] = useState("");
  const [domain, setDomain] = useState("auto");
  const [compare, setCompare] = useState(true);
  const [loading, setLoading] = useState(false);
  const [res, setRes] = useState<AskResponse | null>(null);
  const [err, setErr] = useState("");

  async function ask() {
    if (!question.trim()) return;
    setLoading(true);
    setErr("");
    setRes(null);
    try {
      setRes(await api.ask({ question, domain, compare_base: compare }));
    } catch (e: any) {
      setErr(String(e.message || e));
    } finally {
      setLoading(false);
    }
  }

  const domainLabel = (k: string) => domains.find((d) => d.key === k)?.label ?? k;

  return (
    <div className="flex flex-col gap-5">
      <div className="card p-5 flex flex-col gap-4">
        <textarea
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) ask();
          }}
          placeholder="Describe a workplace safety situation or ask an OSHA question…"
          rows={3}
          className="w-full resize-none rounded-lg border border-zinc-700 bg-zinc-900/70 p-3 text-zinc-100 outline-none focus:border-amber-500"
        />
        <div className="flex flex-wrap items-center gap-3">
          <select
            value={domain}
            onChange={(e) => setDomain(e.target.value)}
            className="rounded-lg border border-zinc-700 bg-zinc-900 px-3 py-2 text-sm"
          >
            <option value="auto">Domain: Auto-route</option>
            {domains.map((d) => (
              <option key={d.key} value={d.key}>
                {d.label} (29 CFR {d.cfr_part})
              </option>
            ))}
          </select>
          <label className="flex items-center gap-2 text-sm text-zinc-300">
            <input
              type="checkbox"
              checked={compare}
              onChange={(e) => setCompare(e.target.checked)}
              className="accent-amber-500"
            />
            Compare base vs fine-tuned
          </label>
          <button
            onClick={ask}
            disabled={loading || !question.trim()}
            className="ml-auto rounded-lg bg-amber-500 px-5 py-2 text-sm font-semibold text-zinc-950 transition hover:bg-amber-400 disabled:opacity-40"
          >
            {loading ? "Thinking…" : "Ask SafetyCite"}
          </button>
        </div>
        <div className="flex flex-wrap gap-2">
          {EXAMPLES.map((ex) => (
            <button
              key={ex}
              onClick={() => setQuestion(ex)}
              className="rounded-full border border-zinc-700 px-3 py-1 text-xs text-zinc-400 hover:border-amber-500 hover:text-amber-300"
            >
              {ex.length > 52 ? ex.slice(0, 52) + "…" : ex}
            </button>
          ))}
        </div>
      </div>

      {err && (
        <div className="card border-red-500/40 p-4 text-red-300">Error: {err}</div>
      )}

      {res && (
        <div className="flex flex-col gap-4">
          <div className="flex flex-wrap items-center gap-2 text-sm">
            <span className="rounded-md bg-zinc-800 px-2.5 py-1 text-zinc-300">
              {res.routing.auto ? "Auto-routed →" : "Domain →"}{" "}
              <span className="font-semibold text-amber-300">
                {domainLabel(res.routing.domain)}
              </span>
              {res.routing.auto && res.routing.confidence > 0 && (
                <span className="text-zinc-500">
                  {" "}
                  (confidence {(res.routing.confidence * 100).toFixed(0)}%)
                </span>
              )}
            </span>
            <span
              className={`rounded-md px-2.5 py-1 ${
                res.adapter.present
                  ? "bg-amber-500/15 text-amber-300"
                  : "bg-zinc-800 text-zinc-400"
              }`}
            >
              {res.adapter.label}
            </span>
            <span className="rounded-md bg-zinc-800 px-2.5 py-1 text-zinc-400">
              backend: {res.backend}
            </span>
          </div>

          {res.base ? (
            <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
              <AnswerCard title="Base model (no adapter)" data={res.base} accent="zinc" />
              <AnswerCard
                title={res.adapter.present ? "Fine-tuned adapter" : "Routed model"}
                data={res.answer}
                accent="amber"
              />
            </div>
          ) : (
            <AnswerCard title="SafetyCite answer" data={res.answer} accent="amber" />
          )}
        </div>
      )}
    </div>
  );
}
