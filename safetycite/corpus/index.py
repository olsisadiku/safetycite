"""In-memory index over the OSHA corpus.

This is the single source of truth for **citation verification**: the reward
function and the UI both ask the index "does 29 CFR 1926.501 exist, and what does
it say?". Section ids are normalised to a canonical `PART.SECTION` form so that
`§ 1926.501`, `1926.501(b)(1)`, and `29 CFR 1926.501` all resolve to the same record.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass

from safetycite.config import CORPUS_DIR, DOMAINS, part_to_domain

# Matches a CFR section number like 1926.501 or 1910.1200 (optionally with paragraph tail).
_SECTION_RE = re.compile(r"\b(19\d{2})\.(\d{1,4})\b")


def normalize_section(raw: str) -> str | None:
    """Return canonical 'PART.SECTION' (e.g. '1926.501') or None if not a CFR cite.

    Strips '§', '29 CFR', whitespace, and any trailing paragraph like '(b)(1)'.
    """
    if not raw:
        return None
    m = _SECTION_RE.search(raw)
    if not m:
        return None
    return f"{m.group(1)}.{m.group(2)}"


@dataclass
class SectionRecord:
    section_id: str  # canonical, e.g. "1926.501"
    heading: str  # e.g. "Duty to have fall protection."
    text: str  # full regulatory text (paragraphs joined)
    part: str  # "1926"
    domain: str  # "construction"
    url: str

    @property
    def citation(self) -> str:
        return f"29 CFR §{self.section_id}"

    def snippet(self, n: int = 400) -> str:
        return self.text if len(self.text) <= n else self.text[: n - 1].rstrip() + "…"


class CorpusIndex:
    """Loads per-domain corpus JSON files and indexes them by canonical section id."""

    def __init__(self, records: list[SectionRecord] | None = None):
        self._by_id: dict[str, SectionRecord] = {}
        for rec in records or []:
            self._by_id[rec.section_id] = rec

    # --- construction -------------------------------------------------------
    @classmethod
    def load(cls, domains: list[str] | None = None) -> CorpusIndex:
        """Load corpus JSON for the given domains (default: all that exist on disk)."""
        idx = cls()
        wanted = domains or list(DOMAINS.keys())
        for dom in wanted:
            path = CORPUS_DIR / f"{dom}.json"
            if not path.exists():
                continue
            data = json.loads(path.read_text())
            for row in data:
                rec = SectionRecord(**row)
                idx._by_id[rec.section_id] = rec
        return idx

    def add(self, rec: SectionRecord) -> None:
        self._by_id[rec.section_id] = rec

    # --- lookups ------------------------------------------------------------
    def exists(self, section: str) -> bool:
        norm = normalize_section(section)
        return norm is not None and norm in self._by_id

    def lookup(self, section: str) -> SectionRecord | None:
        norm = normalize_section(section)
        return self._by_id.get(norm) if norm else None

    def all_sections(self) -> list[SectionRecord]:
        return list(self._by_id.values())

    def sections_for_domain(self, domain: str) -> list[SectionRecord]:
        return [r for r in self._by_id.values() if r.domain == domain]

    def __len__(self) -> int:
        return len(self._by_id)

    def __contains__(self, section: str) -> bool:
        return self.exists(section)

    # --- naive keyword search (used for dataset building, NOT for answering) -
    def search(self, query: str, k: int = 5, domain: str | None = None) -> list[SectionRecord]:
        terms = [t for t in re.findall(r"[a-z]+", query.lower()) if len(t) > 2]
        pool = self.sections_for_domain(domain) if domain else self.all_sections()
        scored: list[tuple[int, SectionRecord]] = []
        for rec in pool:
            hay = f"{rec.heading} {rec.text}".lower()
            score = sum(hay.count(t) for t in terms)
            if score:
                scored.append((score, rec))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [r for _, r in scored[:k]]


def save_domain_corpus(domain: str, records: list[SectionRecord]) -> int:
    """Persist a domain's records to data/corpus/{domain}.json. Returns count written."""
    path = CORPUS_DIR / f"{domain}.json"
    path.write_text(json.dumps([asdict(r) for r in records], indent=2, ensure_ascii=False))
    return len(records)


def domain_for_part(part: str) -> str:
    dom = part_to_domain(part)
    if dom is None:
        raise ValueError(f"CFR part {part} is not mapped to a SafetyCite domain.")
    return dom
