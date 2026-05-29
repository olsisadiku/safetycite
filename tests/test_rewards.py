from safetycite.corpus.index import CorpusIndex, SectionRecord
from safetycite.rewards import (
    citation_f1,
    composite_reward,
    extract_citations,
    format_score,
    make_reward_fn,
)


def _index():
    recs = [
        SectionRecord("1926.501", "Fall protection", "txt", "1926", "construction", "u"),
        SectionRecord("1926.502", "FP criteria", "txt", "1926", "construction", "u"),
    ]
    return CorpusIndex(recs)


def test_extract_citations_forms():
    assert extract_citations("see [29 CFR §1926.501]") == ["1926.501"]
    assert extract_citations("29 CFR 1910.1200 and §1910.147") == ["1910.1200", "1910.147"]
    assert extract_citations("dups 1926.501 and 1926.501") == ["1926.501"]  # de-duped
    assert extract_citations("no cites at 1.8 m, Form 300") == []


def test_citation_f1():
    assert citation_f1("[29 CFR §1926.501]", ["1926.501"]) == 1.0
    assert citation_f1("nothing here", ["1926.501"]) == 0.0
    assert citation_f1("", []) == 1.0  # both empty
    # partial: predicts one right + one wrong vs one gold -> P=.5 R=1 -> F1=2/3
    f1 = citation_f1("1926.501 and 1926.999", ["1926.501"])
    assert abs(f1 - (2 / 3)) < 1e-9


def test_format_score():
    good = "Guardrails are required above six feet on a surface.\nCitation: [29 CFR §1926.501]"
    assert format_score(good) == 1.0
    assert format_score("[29 CFR §1926.501]") == 0.5  # citation, no prose
    assert format_score("Guardrails are required when working at height above the floor.") == 0.5


def test_composite_rewards_correct_and_penalizes_hallucination():
    idx = _index()
    perfect = "Fall protection is required at six feet.\nCitation: [29 CFR §1926.501]"
    r = composite_reward(perfect, ["1926.501"], index=idx)
    assert r.citation_f1 == 1.0 and r.fmt == 1.0 and r.hallucination_rate == 0.0
    assert r.exact_match

    # invents a non-existent section -> hallucination penalty applies
    halluc = "Some answer about safety at heights here.\nCitation: [29 CFR §1926.999]"
    rh = composite_reward(halluc, ["1926.501"], index=idx)
    assert rh.hallucination_rate == 1.0
    assert rh.total < r.total


def test_make_reward_fn_returns_scalar():
    fn = make_reward_fn(_index())
    val = fn("Answer text about safety.\nCitation: [29 CFR §1926.501]", ["1926.501"])
    assert isinstance(val, float) and val > 0
