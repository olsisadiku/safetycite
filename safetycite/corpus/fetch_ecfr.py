"""Fetch real OSHA regulation text from the public eCFR API.

eCFR full-text endpoint returns XML where every section is a
`<DIV8 TYPE="SECTION" N="1926.501">` with a `<HEAD>` and `<P>` paragraphs:

    GET https://www.ecfr.gov/api/versioner/v1/full/{date}/title-29.xml?part=1926[&section=1926.501]

We parse those into `SectionRecord`s. Two modes:
  * curated  — fetch a hand-picked high-value section list per domain (fast, focused)
  * full     — fetch the entire CFR part (complete, larger download)
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET

import httpx

from safetycite.config import DOMAINS, settings
from safetycite.corpus.index import SectionRecord, domain_for_part, normalize_section

ECFR_FULL = "https://www.ecfr.gov/api/versioner/v1/full/{date}/title-29.xml"

# High-value, frequently-cited sections per domain — a focused default corpus.
CURATED: dict[str, list[str]] = {
    "construction": [
        "1926.20", "1926.21", "1926.95", "1926.100", "1926.102", "1926.103",
        "1926.451", "1926.452", "1926.453", "1926.501", "1926.502", "1926.503",
        "1926.651", "1926.652", "1926.760", "1926.1053", "1926.1101", "1926.1153",
    ],
    "general_industry": [
        "1910.23", "1910.28", "1910.95", "1910.132", "1910.133", "1910.134",
        "1910.146", "1910.147", "1910.157", "1910.178", "1910.212", "1910.213",
        "1910.219", "1910.242", "1910.303", "1910.305", "1910.1000", "1910.1030",
        "1910.1200",
    ],
    "recordkeeping": [
        "1904.4", "1904.5", "1904.6", "1904.7", "1904.8", "1904.10", "1904.29",
        "1904.32", "1904.35", "1904.39", "1904.40", "1904.41",
    ],
}

_HEAD_PREFIX = re.compile(r"^§?\s*19\d{2}\.\d+\s*")


def _clean(text: str) -> str:
    return " ".join(text.split())


def _parse_div8(div8: ET.Element, part: str, domain: str) -> SectionRecord | None:
    section_id = normalize_section(div8.get("N") or "")
    if not section_id:
        return None
    head_el = div8.find("HEAD")
    heading_full = _clean("".join(head_el.itertext())) if head_el is not None else ""
    heading = _HEAD_PREFIX.sub("", heading_full).strip()
    if "[reserved]" in heading.lower():
        return None
    paras: list[str] = []
    for el in div8:
        if el.tag == "HEAD":
            continue
        txt = _clean("".join(el.itertext()))
        if txt:
            paras.append(txt)
    text = "\n".join(paras)
    if not text:
        return None
    return SectionRecord(
        section_id=section_id,
        heading=heading or heading_full,
        text=text,
        part=part,
        domain=domain,
        url=f"https://www.ecfr.gov/current/title-29/section-{section_id}",
    )


def _parse_sections(xml_bytes: bytes, part: str, domain: str) -> list[SectionRecord]:
    root = ET.fromstring(xml_bytes)
    out: list[SectionRecord] = []
    for div8 in root.iter("DIV8"):
        if (div8.get("TYPE") or "").upper() != "SECTION":
            continue
        rec = _parse_div8(div8, part, domain)
        if rec:
            out.append(rec)
    return out


def fetch_section(client: httpx.Client, date: str, part: str, section: str) -> SectionRecord | None:
    domain = domain_for_part(part)
    r = client.get(ECFR_FULL.format(date=date), params={"part": part, "section": section})
    r.raise_for_status()
    recs = _parse_sections(r.content, part, domain)
    return recs[0] if recs else None


def fetch_part_full(client: httpx.Client, date: str, part: str) -> list[SectionRecord]:
    domain = domain_for_part(part)
    r = client.get(ECFR_FULL.format(date=date), params={"part": part})
    r.raise_for_status()
    return _parse_sections(r.content, part, domain)


def fetch_domain(
    domain: str,
    *,
    date: str | None = None,
    full: bool = False,
    sections: list[str] | None = None,
    timeout: float = 60.0,
) -> list[SectionRecord]:
    """Fetch a domain's corpus. `full` grabs the whole CFR part; otherwise the
    curated (or explicitly provided) section list."""
    date = date or settings.ecfr_date
    dom = DOMAINS[domain]
    out: list[SectionRecord] = []
    with httpx.Client(timeout=timeout, headers={"User-Agent": "SafetyCite/0.1"}) as client:
        if full:
            out = fetch_part_full(client, date, dom.cfr_part)
        else:
            wanted = sections or CURATED.get(domain, [])
            for sec in wanted:
                rec = None
                for _attempt in range(3):  # eCFR occasionally returns a transient error
                    try:
                        rec = fetch_section(client, date, dom.cfr_part, sec)
                        if rec:
                            break
                    except httpx.HTTPError:
                        rec = None
                if rec:
                    out.append(rec)
    return out
