import { useEffect, useState, type ReactNode } from "react";
import { api, type BackendInfo, type DomainInfo } from "./lib/api";
import AskTab from "./components/AskTab";
import AdaptersTab from "./components/AdaptersTab";
import EvalTab from "./components/EvalTab";

type Tab = "ask" | "adapters" | "eval";

export default function App() {
  const [tab, setTab] = useState<Tab>("ask");
  const [backend, setBackend] = useState<BackendInfo | null>(null);
  const [domains, setDomains] = useState<DomainInfo[]>([]);
  const [err, setErr] = useState("");

  useEffect(() => {
    api.backend().then(setBackend).catch((e) => setErr(String(e.message || e)));
    api.domains().then(setDomains).catch((e) => setErr(String(e.message || e)));
  }, []);

  const tabs: { id: Tab; label: string }[] = [
    { id: "ask", label: "Ask" },
    { id: "adapters", label: "Adapters" },
    { id: "eval", label: "Evaluation" },
  ];

  return (
    <div className="min-h-full">
      <div className="hazard-stripe h-1.5 w-full" />
      <header className="border-b border-zinc-800 bg-zinc-950/60 backdrop-blur">
        <div className="mx-auto flex max-w-5xl flex-wrap items-center gap-3 px-5 py-4">
          <div className="flex items-center gap-3">
            <div className="flex h-9 w-9 items-center justify-center rounded-lg bg-amber-500 text-lg font-black text-zinc-950">
              §
            </div>
            <div>
              <h1 className="text-lg font-bold leading-tight">SafetyCite</h1>
              <p className="text-xs text-zinc-500">OSHA Compliance Copilot · trained with MinT</p>
            </div>
          </div>
          <div className="ml-auto flex flex-wrap items-center gap-2 text-xs">
            {backend && (
              <>
                <Badge>backend: {backend.backend}</Badge>
                {backend.base_model && <Badge>{backend.base_model}</Badge>}
                {backend.device && <Badge>{backend.device.toUpperCase()}</Badge>}
                <Badge>{backend.corpus_sections} CFR sections</Badge>
                <span className="flex items-center gap-1 text-emerald-400">
                  <span className="h-2 w-2 rounded-full bg-emerald-400" /> live
                </span>
              </>
            )}
          </div>
        </div>
        <div className="mx-auto flex max-w-5xl gap-1 px-5">
          {tabs.map((t) => (
            <button
              key={t.id}
              onClick={() => setTab(t.id)}
              className={`-mb-px border-b-2 px-4 py-2 text-sm font-medium transition ${
                tab === t.id
                  ? "border-amber-500 text-amber-400"
                  : "border-transparent text-zinc-400 hover:text-zinc-200"
              }`}
            >
              {t.label}
            </button>
          ))}
        </div>
      </header>

      <main className="mx-auto max-w-5xl px-5 py-6">
        {err && (
          <div className="card mb-4 border-red-500/40 p-4 text-red-300">
            Can't reach the API ({err}). Is the server running on :8000?
          </div>
        )}
        {tab === "ask" && <AskTab domains={domains} />}
        {tab === "adapters" && <AdaptersTab domains={domains} />}
        {tab === "eval" && <EvalTab domains={domains} />}
      </main>

      <footer className="mx-auto max-w-5xl px-5 py-8 text-center text-xs text-zinc-600">
        Citations verified against the live eCFR corpus · not legal advice.
      </footer>
    </div>
  );
}

function Badge({ children }: { children: ReactNode }) {
  return (
    <span className="rounded-md border border-zinc-700 bg-zinc-900 px-2 py-1 text-zinc-400">
      {children}
    </span>
  );
}
