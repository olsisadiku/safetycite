"""Central configuration: paths, settings, the OSHA domain registry, reward weights.

Everything else in the package reads from here so the corpus, datasets, rewards,
training, serving, and UI all agree on domains, models, and where artifacts live.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# --- Paths -------------------------------------------------------------------
PKG_DIR = Path(__file__).resolve().parent
REPO_DIR = PKG_DIR.parent
DATA_DIR = REPO_DIR / "data"
CORPUS_DIR = DATA_DIR / "corpus"
DATASETS_DIR = DATA_DIR / "datasets"
ADAPTERS_DIR = DATA_DIR / "adapters"
WEB_DIST = REPO_DIR / "web" / "dist"

for _d in (CORPUS_DIR, DATASETS_DIR, ADAPTERS_DIR):
    _d.mkdir(parents=True, exist_ok=True)


# --- Domain registry ---------------------------------------------------------
@dataclass(frozen=True)
class TrainConfig:
    """Hyperparameters for one adapter (shared shape across backends)."""

    lora_rank: int = 16
    lora_alpha: int = 32
    sft_lr: float = 2e-4
    sft_epochs: int = 3
    rl_lr: float = 1e-5
    rl_steps: int = 60
    rl_group_size: int = 8  # GRPO samples per prompt
    max_prompt_tokens: int = 768
    max_new_tokens: int = 384


@dataclass(frozen=True)
class Domain:
    """One OSHA regulatory domain → one LoRA adapter."""

    key: str
    label: str
    cfr_part: str  # e.g. "1926"
    blurb: str
    routing_keywords: tuple[str, ...] = ()
    train: TrainConfig = field(default_factory=TrainConfig)

    @property
    def cfr_url(self) -> str:
        return f"https://www.ecfr.gov/current/title-29/part-{self.cfr_part}"


DOMAINS: dict[str, Domain] = {
    "construction": Domain(
        key="construction",
        label="Construction",
        cfr_part="1926",
        blurb="Construction industry safety & health — 29 CFR 1926.",
        routing_keywords=(
            "scaffold", "fall protection", "guardrail", "excavation", "trench",
            "ladder", "crane", "rebar", "leading edge", "roof", "construction",
            "concrete", "demolition", "steel erection", "harness", "lanyard",
        ),
    ),
    "general_industry": Domain(
        key="general_industry",
        label="General Industry",
        cfr_part="1910",
        blurb="General industry safety & health — 29 CFR 1910.",
        routing_keywords=(
            "lockout", "tagout", "loto", "hazard communication", "hazcom", "sds",
            "respirator", "respiratory", "ppe", "machine guarding", "forklift",
            "powered industrial truck", "confined space", "electrical", "bloodborne",
            "hearing", "noise", "permit", "energy control",
        ),
    ),
    "recordkeeping": Domain(
        key="recordkeeping",
        label="Recordkeeping",
        cfr_part="1904",
        blurb="Recording & reporting occupational injuries and illnesses — 29 CFR 1904.",
        routing_keywords=(
            "recordable", "300 log", "form 300", "301", "fatality", "report",
            "hospitalization", "amputation", "first aid", "restricted duty",
            "days away", "recordkeeping", "injury log", "illness log",
        ),
    ),
}

DEFAULT_DOMAIN = "general_industry"


def get_domain(key: str) -> Domain:
    if key not in DOMAINS:
        raise KeyError(f"Unknown domain '{key}'. Known: {', '.join(DOMAINS)}")
    return DOMAINS[key]


def part_to_domain(part: str) -> str | None:
    for d in DOMAINS.values():
        if d.cfr_part == part:
            return d.key
    return None


# --- Reward weights (used by RL and eval alike) ------------------------------
@dataclass(frozen=True)
class RewardWeights:
    citation: float = 1.0  # weight on citation F1
    fmt: float = 0.2  # weight on format correctness
    hallucination: float = 0.5  # penalty per fraction of non-existent citations


REWARD_WEIGHTS = RewardWeights()


# --- Runtime settings (env-driven) -------------------------------------------
class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="SAFETYCITE_", env_file=".env", extra="ignore")

    backend: str = "local"  # "local" | "mint"
    base_model: str = "Qwen/Qwen3-0.6B"
    ecfr_date: str = "2025-01-01"  # eCFR snapshot date for reproducible corpus
    host: str = "0.0.0.0"
    port: int = 8000


settings = Settings()
