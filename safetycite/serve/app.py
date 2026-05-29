"""FastAPI app: the live OSHA copilot API + static UI host.

Every answer is a real model generation. Citations in the answer are parsed and
verified against the eCFR corpus so the UI can render the actual regulatory text and
flag any hallucinated section. Set `compare_base=true` to also run the un-adapted base
model for an honest side-by-side.
"""

from __future__ import annotations

import json

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool

from safetycite.backends import SamplingParams, find_adapter, get_backend, list_adapters
from safetycite.config import DATASETS_DIR, DOMAINS, WEB_DIST, get_domain, settings
from safetycite.corpus.index import CorpusIndex
from safetycite.data.render import render_messages
from safetycite.rewards.citation import extract_citations
from safetycite.serve.router import resolve_domain

app = FastAPI(title="SafetyCite", version="0.1.0")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)

# --- lazy singletons --------------------------------------------------------
_STATE: dict = {}


def _index() -> CorpusIndex:
    if "index" not in _STATE:
        _STATE["index"] = CorpusIndex.load()
    return _STATE["index"]


def _backend():
    if "backend" not in _STATE:
        _STATE["backend"] = get_backend()
    return _STATE["backend"]


def _sampler(adapter):
    cache = _STATE.setdefault("samplers", {})
    key = adapter.path if adapter else "__base__"
    if key not in cache:
        cache[key] = _backend().sampler(adapter)
    return cache[key]


# --- response shaping -------------------------------------------------------
def _analyze(text: str) -> dict:
    idx = _index()
    citations = []
    for sid in extract_citations(text):
        rec = idx.lookup(sid)
        citations.append(
            {
                "section": sid,
                "label": f"29 CFR §{sid}",
                "exists": rec is not None,
                "heading": rec.heading if rec else None,
                "snippet": rec.snippet(600) if rec else None,
                "url": rec.url if rec else "https://www.ecfr.gov/current/title-29",
            }
        )
    return {
        "text": text,
        "citations": citations,
        "n_citations": len(citations),
        "n_valid": sum(1 for c in citations if c["exists"]),
    }


# --- schemas ----------------------------------------------------------------
class AskRequest(BaseModel):
    question: str
    domain: str = "auto"
    compare_base: bool = False
    max_new_tokens: int = 256


# --- endpoints --------------------------------------------------------------
@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.get("/api/backend")
def backend_info():
    return {**_backend().info(), "corpus_sections": len(_index())}


@app.get("/api/domains")
def domains():
    be = _backend()
    out = []
    for key, d in DOMAINS.items():
        ad = find_adapter(key, be.name)
        out.append(
            {
                "key": key,
                "label": d.label,
                "blurb": d.blurb,
                "cfr_part": d.cfr_part,
                "cfr_url": d.cfr_url,
                "sections": len(_index().sections_for_domain(key)),
                "has_adapter": ad is not None,
                "adapter_method": ad.method if ad else None,
                "adapter_metrics": ad.metrics if ad else {},
            }
        )
    return out


@app.get("/api/adapters")
def adapters():
    return [a.to_dict() for a in list_adapters(_backend().name)]


@app.get("/api/corpus/{section}")
def corpus_section(section: str):
    rec = _index().lookup(section)
    if not rec:
        raise HTTPException(404, f"Section {section} not found in corpus")
    return {
        "section": rec.section_id,
        "heading": rec.heading,
        "text": rec.text,
        "part": rec.part,
        "domain": rec.domain,
        "url": rec.url,
    }


@app.get("/api/eval/{domain}")
def eval_results(domain: str):
    if domain not in DOMAINS:
        raise HTTPException(404, f"Unknown domain {domain}")
    path = DATASETS_DIR / domain / f"eval_{_backend().name}.json"
    if not path.exists():
        return {"domain": domain, "available": False}
    return {"domain": domain, "available": True, **json.loads(path.read_text())}


@app.post("/api/ask")
async def ask(req: AskRequest):
    if not req.question.strip():
        raise HTTPException(400, "Empty question")
    be = _backend()
    route = resolve_domain(req.question, req.domain)
    domain = route.domain
    get_domain(domain)  # validate

    adapter = find_adapter(domain, be.name)
    params = SamplingParams(max_new_tokens=req.max_new_tokens)
    messages = render_messages(req.question, domain)

    text = await run_in_threadpool(_sampler(adapter).generate, messages, params)
    answer = _analyze(text)

    base = None
    if req.compare_base:
        btext = await run_in_threadpool(_sampler(None).generate, messages, params)
        base = _analyze(btext)

    return {
        "question": req.question,
        "backend": be.name,
        "routing": route.as_dict(),
        "adapter": {
            "present": adapter is not None,
            "method": adapter.method if adapter else None,
            "label": (f"{DOMAINS[domain].label} · {adapter.method.upper()}" if adapter else f"{DOMAINS[domain].label} · base (no adapter yet)"),
            "notes": adapter.notes if adapter else "",
            "metrics": adapter.metrics if adapter else {},
        },
        "answer": answer,
        "base": base,
    }


# --- static UI --------------------------------------------------------------
if WEB_DIST.exists():
    app.mount("/", StaticFiles(directory=str(WEB_DIST), html=True), name="ui")
else:

    @app.get("/")
    def root():
        return {
            "service": "SafetyCite",
            "ui": "not built — run `cd web && bun install && bun run build`, or use the Vite dev server",
            "backend": settings.backend,
            "docs": "/docs",
        }
