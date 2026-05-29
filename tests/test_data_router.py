from safetycite.data.build_dataset import summarize
from safetycite.data.render import render_sft, render_target
from safetycite.data.schema import QAExample
from safetycite.rewards import composite_reward, format_score
from safetycite.serve.router import resolve_domain, route


def _ex(gold):
    return QAExample(
        id="t",
        domain="construction",
        question="What standard applies to fall protection?",
        gold_citations=gold,
        reference_answer="Workers must be protected from falls above six feet.",
        source_section=gold[0],
    )


def test_render_target_is_perfectly_formatted():
    """The gold completion the model is trained on must score 1.0 on format + citation."""
    ex = _ex(["1926.501"])
    target = render_target(ex)
    assert "Citation: [29 CFR §1926.501]" in target
    assert format_score(target) == 1.0
    r = composite_reward(target, ex.gold_citations)
    assert r.citation_f1 == 1.0 and r.fmt == 1.0


def test_render_sft_returns_messages_and_target():
    msgs, target = render_sft(_ex(["1926.501"]))
    assert msgs[0]["role"] == "system" and msgs[1]["role"] == "user"
    assert "1926.501" in target


def test_summarize_strips_enumeration():
    s = summarize("(a) Scope. (1) This section requires employers to provide fall protection systems for workers.")
    assert s.startswith("This section requires") and "(a)" not in s


def test_router_keywords():
    assert route("a worker fell off a scaffold with no guardrail").domain == "construction"
    assert route("lockout tagout of a machine and respirator use").domain == "general_industry"
    assert route("report a fatality and hospitalization to OSHA").domain == "recordkeeping"


def test_resolve_domain_honours_explicit_choice():
    r = resolve_domain("anything at all", "recordkeeping")
    assert r.domain == "recordkeeping" and r.auto is False
    r2 = resolve_domain("a worker fell off a scaffold", "auto")
    assert r2.domain == "construction" and r2.auto is True
