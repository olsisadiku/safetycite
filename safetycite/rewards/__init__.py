"""Verifiable rewards for OSHA citation accuracy.

The same functions serve as (a) the RL reward signal and (b) the eval metric — no
judge model, fully deterministic and reproducible.
"""

from safetycite.rewards.citation import (
    CitationScore,
    citation_f1,
    citation_set,
    extract_citations,
)
from safetycite.rewards.composite import RewardBreakdown, composite_reward, make_reward_fn
from safetycite.rewards.format import format_score

__all__ = [
    "CitationScore",
    "RewardBreakdown",
    "citation_f1",
    "citation_set",
    "composite_reward",
    "extract_citations",
    "format_score",
    "make_reward_fn",
]
