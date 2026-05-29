"""Chat rendering: the single place that defines SafetyCite's answer contract.

Both training (SFT targets / RL prompts) and live inference go through here, so the
format the model is taught is exactly the format the reward measures and the UI parses.
"""

from __future__ import annotations

from safetycite.config import DOMAINS
from safetycite.data.schema import QAExample

SYSTEM_PROMPT = (
    "You are SafetyCite, an OSHA compliance assistant. Answer the worker-safety question "
    "concisely in plain language, then cite the single controlling OSHA standard from "
    "Title 29 of the CFR. Always end with a citation line in exactly this format:\n"
    "Citation: [29 CFR §XXXX.XX]\n"
    "Cite only standards you are confident apply. Do not invent section numbers."
)


def format_citation(section_id: str) -> str:
    return f"[29 CFR §{section_id}]"


def domain_hint(domain: str) -> str:
    d = DOMAINS.get(domain)
    return f" (Context: {d.label}, 29 CFR {d.cfr_part}.)" if d else ""


def render_messages(question: str, domain: str | None = None) -> list[dict]:
    """Chat messages for inference / RL prompt (system + user)."""
    user = question.strip()
    if domain:
        user += domain_hint(domain)
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


def render_target(example: QAExample) -> str:
    """Gold completion in canonical format."""
    cites = ", ".join(format_citation(s) for s in example.gold_citations) or "[29 CFR §____]"
    answer = example.reference_answer.strip()
    return f"{answer}\nCitation: {cites}"


def render_sft(example: QAExample) -> tuple[list[dict], str]:
    """(messages, target_text) for supervised fine-tuning."""
    return render_messages(example.question, example.domain), render_target(example)
