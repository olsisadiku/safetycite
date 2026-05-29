"""Composite reward = citation F1 + format − hallucination penalty.

This single function is both the GRPO reward and the eval metric. It needs the corpus
index so it can detect **hallucinated citations** — sections the model invents that do
not exist in 29 CFR. That anti-hallucination signal is what makes the copilot trustworthy.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from safetycite.config import REWARD_WEIGHTS, RewardWeights
from safetycite.corpus.index import CorpusIndex
from safetycite.rewards.citation import citation_score
from safetycite.rewards.format import format_score


@dataclass
class RewardBreakdown:
    total: float
    citation_f1: float
    precision: float
    recall: float
    fmt: float
    hallucination_rate: float
    exact_match: bool

    def as_metrics(self) -> dict[str, float]:
        return {
            "reward": self.total,
            "citation_f1": self.citation_f1,
            "precision": self.precision,
            "recall": self.recall,
            "format": self.fmt,
            "hallucination_rate": self.hallucination_rate,
            "exact_match": float(self.exact_match),
        }


def _hallucination_rate(pred_sections: set[str], index: CorpusIndex | None) -> float:
    if not pred_sections or index is None:
        return 0.0
    bad = sum(1 for s in pred_sections if not index.exists(s))
    return bad / len(pred_sections)


def composite_reward(
    text: str,
    gold: list[str] | set[str],
    index: CorpusIndex | None = None,
    weights: RewardWeights = REWARD_WEIGHTS,
) -> RewardBreakdown:
    cs = citation_score(text, gold)
    fmt = format_score(text)
    halluc = _hallucination_rate(cs.pred, index)
    total = weights.citation * cs.f1 + weights.fmt * fmt - weights.hallucination * halluc
    return RewardBreakdown(
        total=total,
        citation_f1=cs.f1,
        precision=cs.precision,
        recall=cs.recall,
        fmt=fmt,
        hallucination_rate=halluc,
        exact_match=cs.exact_match,
    )


def make_reward_fn(
    index: CorpusIndex | None = None,
    weights: RewardWeights = REWARD_WEIGHTS,
) -> Callable[[str, list[str] | set[str]], float]:
    """Return a scalar (text, gold) -> reward function for RL backends."""

    def _fn(text: str, gold: list[str] | set[str]) -> float:
        return composite_reward(text, gold, index=index, weights=weights).total

    return _fn
