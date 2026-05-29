"""Format reward.

We teach a fixed answer shape so citations are machine-checkable and the UI can render
them: a plain-language answer followed by a bracketed citation line, e.g.

    Guardrails, safety nets, or personal fall arrest systems are required at 6 feet.
    Citation: [29 CFR §1926.501]

`format_score` rewards (a) at least one well-formed bracketed citation and (b) real prose.
"""

from __future__ import annotations

import re

from safetycite.rewards.citation import _CITE_RE

# A well-formed bracketed citation like [29 CFR §1926.501] (the § and "29 CFR" are optional-ish).
_BRACKET_RE = re.compile(r"\[\s*(?:29\s*CFR\s*)?§?\s*19\d{2}\.\d{1,4}\s*\]")


def _prose_without_citations(text: str) -> str:
    no_cite = _CITE_RE.sub("", text or "")
    no_cite = re.sub(r"(?i)\b(29\s*CFR|citation:?|§)\b", "", no_cite)
    no_cite = re.sub(r"[\[\]§]", "", no_cite)
    return no_cite.strip()


def format_score(text: str) -> float:
    """0.0–1.0. Half for a well-formed bracketed citation, half for substantive prose."""
    text = text or ""
    well_formed = 1.0 if _BRACKET_RE.search(text) else 0.0
    prose = _prose_without_citations(text)
    has_answer = 1.0 if len(prose.split()) >= 8 else 0.0
    return 0.5 * well_formed + 0.5 * has_answer
