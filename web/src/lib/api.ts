export type Citation = {
  section: string;
  label: string;
  exists: boolean;
  heading: string | null;
  snippet: string | null;
  url: string;
};

export type Analysis = {
  text: string;
  citations: Citation[];
  n_citations: number;
  n_valid: number;
};

export type Routing = {
  domain: string;
  confidence: number;
  scores: Record<string, number>;
  auto: boolean;
};

export type AskResponse = {
  question: string;
  backend: string;
  routing: Routing;
  adapter: {
    present: boolean;
    method: string | null;
    label: string;
    notes: string;
    metrics: Record<string, number>;
  };
  answer: Analysis;
  base: Analysis | null;
};

export type DomainInfo = {
  key: string;
  label: string;
  blurb: string;
  cfr_part: string;
  cfr_url: string;
  sections: number;
  has_adapter: boolean;
  adapter_method: string | null;
  adapter_metrics: Record<string, number>;
};

export type BackendInfo = {
  backend: string;
  base_model?: string;
  device?: string;
  corpus_sections: number;
};

export type EvalReport = {
  domain: string;
  available: boolean;
  n?: number;
  adapter_present?: boolean;
  adapter_method?: string | null;
  base?: Record<string, number>;
  adapter?: Record<string, number>;
  examples?: any[];
};

async function j<T>(r: Response): Promise<T> {
  if (!r.ok) throw new Error(`${r.status} ${await r.text()}`);
  return r.json();
}

export const api = {
  backend: () => fetch("/api/backend").then(j<BackendInfo>),
  domains: () => fetch("/api/domains").then(j<DomainInfo[]>),
  evalReport: (d: string) => fetch(`/api/eval/${d}`).then(j<EvalReport>),
  ask: (body: { question: string; domain: string; compare_base: boolean; max_new_tokens?: number }) =>
    fetch("/api/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    }).then(j<AskResponse>),
};
