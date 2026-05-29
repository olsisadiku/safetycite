"""Backend contract + adapter registry.

A `Backend` can serve a model (`sampler`) and train LoRA adapters (`train_sft`,
`train_rl`). An `AdapterRef` points at a trained adapter (a local dir for the HF
engine, a weights URI for MinT) plus its eval metrics. Adapters live under
`data/adapters/{domain}/{backend}_{method}/` with a `meta.json` describing them.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path

from safetycite.config import ADAPTERS_DIR, TrainConfig


@dataclass
class SamplingParams:
    max_new_tokens: int = 320
    temperature: float = 0.0  # 0 -> greedy (deterministic serving)
    top_p: float = 0.95

    @property
    def do_sample(self) -> bool:
        return self.temperature > 0


@dataclass
class AdapterRef:
    backend: str
    base_model: str
    domain: str
    path: str  # local dir or remote uri ("" => base model, no adapter)
    method: str = "sft"  # "sft" | "rl" | "base"
    metrics: dict = field(default_factory=dict)
    created: str = ""
    notes: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> AdapterRef:
        known = {f for f in cls.__dataclass_fields__}  # noqa: C416
        return cls(**{k: v for k, v in d.items() if k in known})

    def save(self) -> Path:
        p = Path(self.path)
        p.mkdir(parents=True, exist_ok=True)
        meta = p / "meta.json"
        meta.write_text(json.dumps(self.to_dict(), indent=2))
        return meta


def adapter_dir(domain: str, backend: str, method: str) -> Path:
    return ADAPTERS_DIR / domain / f"{backend}_{method}"


def load_adapter_ref(directory: str | Path) -> AdapterRef | None:
    meta = Path(directory) / "meta.json"
    if not meta.exists():
        return None
    return AdapterRef.from_dict(json.loads(meta.read_text()))


def list_adapters(backend: str | None = None) -> list[AdapterRef]:
    out: list[AdapterRef] = []
    if not ADAPTERS_DIR.exists():
        return out
    for domain_dir in sorted(ADAPTERS_DIR.iterdir()):
        if not domain_dir.is_dir():
            continue
        for ad in sorted(domain_dir.iterdir()):
            ref = load_adapter_ref(ad)
            if ref and (backend is None or ref.backend == backend):
                out.append(ref)
    return out


def find_adapter(
    domain: str, backend: str, method: str | None = None
) -> AdapterRef | None:
    """Best adapter for a domain+backend. Prefers RL > SFT unless a method is forced."""
    candidates = [
        r for r in list_adapters(backend) if r.domain == domain and (method is None or r.method == method)
    ]
    if not candidates:
        return None
    order = {"rl": 0, "sft": 1, "base": 2}
    candidates.sort(key=lambda r: order.get(r.method, 9))
    return candidates[0]


class Sampler(ABC):
    """Live text generation from a (possibly adapter-augmented) model."""

    @abstractmethod
    def generate(self, messages: list[dict], params: SamplingParams | None = None) -> str: ...

    def generate_batch(
        self, batch: list[list[dict]], params: SamplingParams | None = None
    ) -> list[str]:
        return [self.generate(m, params) for m in batch]


class Backend(ABC):
    name: str = "base"

    def info(self) -> dict:
        return {"backend": self.name}

    @abstractmethod
    def sampler(self, adapter: AdapterRef | None = None) -> Sampler:
        """Return a live sampler for the base model (adapter=None) or a trained adapter."""

    @abstractmethod
    def train_sft(
        self, domain: str, train, val, cfg: TrainConfig | None = None
    ) -> AdapterRef: ...

    @abstractmethod
    def train_rl(
        self,
        domain: str,
        train,
        val,
        reward_fn: Callable[[str, list[str]], float],
        cfg: TrainConfig | None = None,
    ) -> AdapterRef: ...
