import type { DomainInfo } from "../lib/api";

export default function AdaptersTab({ domains }: { domains: DomainInfo[] }) {
  return (
    <div className="grid grid-cols-1 gap-4 md:grid-cols-3">
      {domains.map((d) => (
        <div key={d.key} className="card p-5 flex flex-col gap-3">
          <div className="flex items-center justify-between">
            <h3 className="text-lg font-semibold text-zinc-100">{d.label}</h3>
            <span className="font-mono text-xs text-zinc-500">29 CFR {d.cfr_part}</span>
          </div>
          <p className="text-sm text-zinc-400">{d.blurb}</p>
          <div className="flex items-center gap-2 text-xs">
            <span className="rounded-md bg-zinc-800 px-2 py-1 text-zinc-300">
              {d.sections} sections
            </span>
            {d.has_adapter ? (
              <span className="rounded-md bg-amber-500/15 px-2 py-1 font-medium text-amber-300">
                adapter: {d.adapter_method?.toUpperCase()}
              </span>
            ) : (
              <span className="rounded-md bg-zinc-800 px-2 py-1 text-zinc-500">
                no adapter yet
              </span>
            )}
          </div>
          {d.has_adapter && d.adapter_metrics?.final_mean_reward != null && (
            <div className="text-xs text-zinc-500">
              final mean reward: {d.adapter_metrics.final_mean_reward.toFixed(3)}
            </div>
          )}
          {!d.has_adapter && (
            <code className="rounded bg-zinc-900 px-2 py-1 text-xs text-zinc-400">
              safetycite sft {d.key}
            </code>
          )}
          <a
            href={d.cfr_url}
            target="_blank"
            rel="noreferrer"
            className="mt-auto text-sm text-amber-400 hover:underline"
          >
            Browse 29 CFR {d.cfr_part} ↗
          </a>
        </div>
      ))}
    </div>
  );
}
