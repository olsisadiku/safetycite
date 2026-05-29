"""Run a domain's held-out test set through base + adapter and aggregate the
verifiable-reward metrics. Saves JSON the UI dashboard reads.
"""

from __future__ import annotations

import json

from safetycite.backends import SamplingParams, find_adapter, get_backend
from safetycite.config import DATASETS_DIR
from safetycite.corpus.index import CorpusIndex
from safetycite.data.render import render_messages
from safetycite.data.schema import read_jsonl
from safetycite.rewards.composite import composite_reward

_METRIC_KEYS = (
    "reward",
    "citation_f1",
    "precision",
    "recall",
    "format",
    "hallucination_rate",
    "exact_match",
)


def _metrics_over(sampler, examples, index, max_new_tokens):
    agg = {k: 0.0 for k in _METRIC_KEYS}
    rows = []
    for ex in examples:
        text = sampler.generate(
            render_messages(ex.question, ex.domain), SamplingParams(max_new_tokens=max_new_tokens)
        )
        m = composite_reward(text, ex.gold_citations, index=index).as_metrics()
        for k in _METRIC_KEYS:
            agg[k] += m[k]
        rows.append(
            {"id": ex.id, "question": ex.question, "gold": ex.gold_citations, "output": text, **m}
        )
    n = max(len(examples), 1)
    return {k: agg[k] / n for k in _METRIC_KEYS}, rows


def evaluate_domain(
    domain: str,
    backend_name: str | None = None,
    *,
    compare_base: bool = True,
    max_new_tokens: int = 256,
    limit: int | None = None,
    save: bool = True,
) -> dict:
    be = get_backend(backend_name)
    index = CorpusIndex.load()
    test = read_jsonl(DATASETS_DIR / domain / "test.jsonl")
    if limit:
        test = test[:limit]
    if not test:
        raise RuntimeError(f"No test set for '{domain}'. Run `safetycite build` first.")

    adapter = find_adapter(domain, be.name)
    result = {
        "domain": domain,
        "backend": be.name,
        "n": len(test),
        "adapter_present": adapter is not None,
        "adapter_method": adapter.method if adapter else None,
    }
    a_metrics, a_rows = _metrics_over(be.sampler(adapter), test, index, max_new_tokens)
    result["adapter"] = a_metrics
    if compare_base:
        b_metrics, _ = _metrics_over(be.sampler(None), test, index, max_new_tokens)
        result["base"] = b_metrics
    result["examples"] = a_rows[:8]

    if save:
        path = DATASETS_DIR / domain / f"eval_{be.name}.json"
        path.write_text(json.dumps(result, indent=2))
        result["saved_to"] = str(path)
    return result
