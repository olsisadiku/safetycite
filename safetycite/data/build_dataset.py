"""Build per-domain Q&A datasets from the corpus + seed scenarios.

Two generators:
  * heading-based — one (templated) Q&A per corpus section, for breadth.
  * scenario      — hand-curated realistic scenarios from seed/{domain}.jsonl.

Every example's gold citations are validated against the corpus index, so we never
train or evaluate on a citation that can't be verified.
"""

from __future__ import annotations

import json
import random
import re
from pathlib import Path

from safetycite.config import DATASETS_DIR, DOMAINS, get_domain
from safetycite.corpus.index import CorpusIndex, normalize_section
from safetycite.data.schema import QAExample, write_jsonl

SEED_DIR = Path(__file__).resolve().parent / "seed"

_HEADING_TEMPLATES = (
    "What does OSHA require regarding {h}?",
    "Which OSHA standard covers {h}?",
    "Under 29 CFR {part}, what are the requirements for {h}?",
)


def summarize(text: str, max_chars: int = 320) -> str:
    """First substantive sentence(s) of a section, stripped of enumeration labels."""
    t = " ".join((text or "").split())
    sentences = re.split(r"(?<=[.;])\s+", t)
    picked: list[str] = []
    total = 0
    label_re = re.compile(r"^\((?:[a-z0-9ivx]{1,4})\)\s*")
    for s in sentences:
        s_clean = s.strip()
        while label_re.match(s_clean):
            s_clean = label_re.sub("", s_clean).strip()
        if len(s_clean.split()) < 6:
            continue
        picked.append(s_clean)
        total += len(s_clean)
        if total >= max_chars or len(picked) >= 2:
            break
    out = " ".join(picked) if picked else t[:max_chars]
    return out[:max_chars].rstrip()


def heading_examples(domain: str, index: CorpusIndex) -> list[QAExample]:
    out: list[QAExample] = []
    d = get_domain(domain)
    for rec in index.sections_for_domain(domain):
        h = rec.heading.rstrip(".").strip()
        if not h:
            continue
        h_l = h[0].lower() + h[1:]
        ans = summarize(rec.text)
        for ti, tmpl in enumerate(_HEADING_TEMPLATES):
            q = tmpl.format(h=h_l, part=d.cfr_part)
            out.append(
                QAExample(
                    id=f"{domain}-h-{rec.section_id}-{ti}",
                    domain=domain,
                    question=q,
                    gold_citations=[rec.section_id],
                    reference_answer=ans,
                    source_section=rec.section_id,
                    kind="heading",
                )
            )
    return out


def scenario_examples(domain: str, index: CorpusIndex) -> tuple[list[QAExample], list[str]]:
    """Load seeds, validate gold sections against the corpus. Returns (examples, warnings)."""
    path = SEED_DIR / f"{domain}.jsonl"
    out: list[QAExample] = []
    warnings: list[str] = []
    raw = []
    if path.exists():
        for line in path.read_text().splitlines():
            line = line.strip()
            if line:
                raw.append(json.loads(line))
    for i, ex in enumerate(raw):
        gold = [normalize_section(g) or g for g in ex["gold_citations"]]
        missing = [g for g in gold if not index.exists(g)]
        if missing:
            warnings.append(f"{domain} scenario #{i}: gold {missing} not in corpus — skipped")
            continue
        out.append(
            QAExample(
                id=f"{domain}-s-{i}",
                domain=domain,
                question=ex["question"],
                gold_citations=gold,
                reference_answer=ex["reference_answer"],
                source_section=ex.get("source_section") or gold[0],
                kind="scenario",
            )
        )
    return out, warnings


def _split(rows: list[QAExample], val_frac: float, test_frac: float, rng: random.Random):
    rows = rows[:]
    rng.shuffle(rows)
    n = len(rows)
    n_test = max(1, int(n * test_frac)) if n else 0
    n_val = max(1, int(n * val_frac)) if n else 0
    test, val, train = rows[:n_test], rows[n_test : n_test + n_val], rows[n_test + n_val :]
    return train, val, test


def build_domain(
    domain: str,
    *,
    val_frac: float = 0.1,
    test_frac: float = 0.15,
    seed: int = 0,
) -> dict:
    """Build + write train/val/test for one domain. Splits each kind separately so the
    test set always contains the high-value scenarios."""
    index = CorpusIndex.load([domain])
    if len(index) == 0:
        raise RuntimeError(f"No corpus for '{domain}'. Run `safetycite fetch` first.")
    rng = random.Random(seed)
    headings = heading_examples(domain, index)
    scenarios, warnings = scenario_examples(domain, index)

    h_tr, h_va, h_te = _split(headings, val_frac, test_frac, rng)
    s_tr, s_va, s_te = _split(scenarios, val_frac, test_frac, rng)
    train, val, test = h_tr + s_tr, h_va + s_va, h_te + s_te
    rng.shuffle(train)

    out_dir = DATASETS_DIR / domain
    write_jsonl(out_dir / "train.jsonl", train)
    write_jsonl(out_dir / "val.jsonl", val)
    write_jsonl(out_dir / "test.jsonl", test)
    return {
        "domain": domain,
        "headings": len(headings),
        "scenarios": len(scenarios),
        "train": len(train),
        "val": len(val),
        "test": len(test),
        "warnings": warnings,
    }


def build_all(**kw) -> list[dict]:
    return [build_domain(d, **kw) for d in DOMAINS]
