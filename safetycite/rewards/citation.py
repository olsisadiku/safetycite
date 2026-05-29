"""Citation extraction + F1 scoring.

The model is asked to cite the controlling OSHA standard. We parse every CFR section
reference out of its answer and compare the *set* of cited sections to the gold set.
We use **F1** (not recall) on purpose: rewarding recall alone invites the obvious hack
of dumping many citations, so precision must matter too.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# Catches "1926.501", "§ 1926.501", "29 CFR 1926.501", "[29 CFR §1910.1200]" — the
# leading 19xx guards against matching arbitrary decimals like "1.8 m" or "Form 300".
_CITE_RE = re.compile(r"\b(19\d{2})\.(\d{1,4})\b")


def extract_citations(text: str) -> list[str]:
    """Return canonical section ids in order of first appearance, de-duplicated."""
    seen: dict[str, None] = {}
    for m in _CITE_RE.finditer(text or ""):
        sid = f"{m.group(1)}.{m.group(2)}"
        seen.setdefault(sid, None)
    return list(seen.keys())


def citation_set(text: str) -> set[str]:
    return set(extract_citations(text))


@dataclass
class CitationScore:
    precision: float
    recall: float
    f1: float
    pred: set[str] = field(default_factory=set)
    gold: set[str] = field(default_factory=set)

    @property
    def exact_match(self) -> bool:
        return self.pred == self.gold and bool(self.gold)


def _to_set(x: str | set[str] | list[str]) -> set[str]:
    if isinstance(x, str):
        return citation_set(x)
    return set(x)


def citation_score(pred: str | set[str], gold: str | set[str] | list[str]) -> CitationScore:
    p = _to_set(pred)
    g = _to_set(gold)
    if not g and not p:
        return CitationScore(1.0, 1.0, 1.0, p, g)
    if not p or not g:
        return CitationScore(0.0, 0.0, 0.0, p, g)
    tp = len(p & g)
    precision = tp / len(p)
    recall = tp / len(g)
    f1 = 0.0 if (precision + recall) == 0 else 2 * precision * recall / (precision + recall)
    return CitationScore(precision, recall, f1, p, g)


def citation_f1(pred: str | set[str], gold: str | set[str] | list[str]) -> float:
    return citation_score(pred, gold).f1
