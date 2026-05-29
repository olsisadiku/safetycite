import { useEffect, useState } from "react";
import { api, type DomainInfo, type EvalReport } from "../lib/api";

function Bar({
  label,
  base,
  adapter,
  invert,
}: {
  label: string;
  base?: number;
  adapter?: number;
  invert?: boolean;
}) {
  const pct = (v?: number) => Math.max(0, Math.min(100, (v ?? 0) * 100));
  // "invert" metrics (hallucination) are better when lower — colour accordingly.
  const better = adapter != null && base != null && (invert ? adapter < base : adapter > base);
  return (
    <div className="flex flex-col gap-1">
      <div className="flex justify-between text-xs text-zinc-400">
        <span>{label}</span>
        <span>
          base {(base ?? 0).toFixed(2)} →{" "}
          <span className={better ? "text-emerald-400" : "text-zinc-300"}>
            {(adapter ?? 0).toFixed(2)}
          </span>
        </span>
      </div>
      <div className="relative h-4 w-full overflow-hidden rounded bg-zinc-800">
        <div
          className="absolute inset-y-0 left-0 rounded bg-zinc-600"
          style={{ width: `${pct(base)}%` }}
        />
        <div
          className="absolute inset-y-0 left-0 rounded bg-amber-500/80"
          style={{ width: `${pct(adapter)}%` }}
        />
      </div>
    </div>
  );
}

function DomainEval({ d }: { d: DomainInfo }) {
  const [rep, setRep] = useState<EvalReport | null>(null);
  useEffect(() => {
    api.evalReport(d.key).then(setRep).catch(() => setRep(null));
  }, [d.key]);

  return (
    <div className="card p-5 flex flex-col gap-4">
      <div className="flex items-center justify-between">
        <h3 className="text-lg font-semibold">{d.label}</h3>
        <span className="font-mono text-xs text-zinc-500">29 CFR {d.cfr_part}</span>
      </div>
      {!rep?.available ? (
        <div className="text-sm text-zinc-500">
          No report yet. Run{" "}
          <code className="rounded bg-zinc-900 px-1.5 py-0.5 text-zinc-400">
            safetycite eval {d.key}
          </code>
        </div>
      ) : (
        <>
          <div className="text-xs text-zinc-500">
            {rep.n} test items · adapter: {rep.adapter_method ?? "none"}
          </div>
          <Bar label="Citation F1 (↑)" base={rep.base?.citation_f1} adapter={rep.adapter?.citation_f1} />
          <Bar label="Exact match (↑)" base={rep.base?.exact_match} adapter={rep.adapter?.exact_match} />
          <Bar label="Format (↑)" base={rep.base?.format} adapter={rep.adapter?.format} />
          <Bar
            label="Hallucination rate (↓)"
            base={rep.base?.hallucination_rate}
            adapter={rep.adapter?.hallucination_rate}
            invert
          />
          <div className="mt-1 flex gap-3 text-xs text-zinc-500">
            <span className="inline-flex items-center gap-1">
              <span className="inline-block h-2 w-3 rounded bg-zinc-600" /> base
            </span>
            <span className="inline-flex items-center gap-1">
              <span className="inline-block h-2 w-3 rounded bg-amber-500/80" /> fine-tuned
            </span>
          </div>
        </>
      )}
    </div>
  );
}

export default function EvalTab({ domains }: { domains: DomainInfo[] }) {
  return (
    <div className="grid grid-cols-1 gap-4 md:grid-cols-3">
      {domains.map((d) => (
        <DomainEval key={d.key} d={d} />
      ))}
    </div>
  );
}
