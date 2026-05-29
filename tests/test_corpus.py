from safetycite.corpus.index import CorpusIndex, SectionRecord, normalize_section


def _rec(sid, heading="H", text="some text here about safety", part="1926", domain="construction"):
    return SectionRecord(sid, heading, text, part, domain, url=f"http://x/{sid}")


def test_normalize_section_variants():
    assert normalize_section("1926.501") == "1926.501"
    assert normalize_section("§ 1926.501") == "1926.501"
    assert normalize_section("29 CFR 1926.501") == "1926.501"
    assert normalize_section("[29 CFR §1926.501(b)(1)]") == "1926.501"
    assert normalize_section("1910.1200") == "1910.1200"
    assert normalize_section("the capital is 1.8 m") is None
    assert normalize_section("Form 300") is None
    assert normalize_section("") is None


def test_index_exists_and_lookup():
    idx = CorpusIndex([_rec("1926.501", "Duty to have fall protection")])
    assert idx.exists("1926.501")
    assert idx.exists("§ 1926.501(b)(1)")  # paragraph tail still resolves
    assert "29 CFR 1926.501" in idx
    assert not idx.exists("9999.99")
    assert idx.lookup("1926.501").heading == "Duty to have fall protection"
    assert idx.lookup("0000.0") is None
    assert len(idx) == 1


def test_search_ranks_by_keyword():
    idx = CorpusIndex(
        [
            _rec("1926.501", "Fall protection", "guardrail safety nets fall arrest at six feet"),
            _rec("1926.451", "Scaffolds", "scaffold platform capacity planking"),
        ]
    )
    hits = idx.search("fall protection guardrail")
    assert hits and hits[0].section_id == "1926.501"
