"""Domain router: pick which OSHA adapter should answer a question.

Lightweight keyword scoring over each domain's signature terms. Returns the chosen
domain plus a confidence and the full score breakdown so the UI can show *why* it routed.
"""

from __future__ import annotations

from dataclasses import dataclass

from safetycite.config import DEFAULT_DOMAIN, DOMAINS


@dataclass
class RouteResult:
    domain: str
    confidence: float
    scores: dict[str, int]
    auto: bool = True

    def as_dict(self) -> dict:
        return {
            "domain": self.domain,
            "confidence": round(self.confidence, 3),
            "scores": self.scores,
            "auto": self.auto,
        }


def route(question: str) -> RouteResult:
    q = (question or "").lower()
    scores: dict[str, int] = {}
    for key, dom in DOMAINS.items():
        scores[key] = sum(1 for kw in dom.routing_keywords if kw in q)
    best = max(scores, key=lambda k: scores[k])
    total = sum(scores.values())
    if scores[best] == 0:
        return RouteResult(DEFAULT_DOMAIN, 0.0, scores, auto=True)
    confidence = scores[best] / total if total else 0.0
    return RouteResult(best, confidence, scores, auto=True)


def resolve_domain(question: str, requested: str | None) -> RouteResult:
    """If a specific domain was requested, honour it; otherwise auto-route."""
    if requested and requested != "auto" and requested in DOMAINS:
        scores = {k: 0 for k in DOMAINS}
        scores[requested] = 1
        return RouteResult(requested, 1.0, scores, auto=False)
    return route(question)
