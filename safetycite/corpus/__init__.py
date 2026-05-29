"""OSHA regulation corpus: fetch real 29 CFR text from eCFR + index it for verification."""

from safetycite.corpus.index import CorpusIndex, SectionRecord, normalize_section

__all__ = ["CorpusIndex", "SectionRecord", "normalize_section"]
