"""Swappable training/serving engines behind one interface.

All engines are **live** (real model inference): `local` (Hugging Face on MPS/CUDA/CPU)
and `mint` (remote Tinker-compatible GPUs). Same datasets + rewards drive both.
"""

from __future__ import annotations

from safetycite.backends.base import (
    AdapterRef,
    Backend,
    Sampler,
    SamplingParams,
    find_adapter,
    list_adapters,
)
from safetycite.config import settings


def get_backend(name: str | None = None) -> Backend:
    name = (name or settings.backend).lower()
    if name == "local":
        from safetycite.backends.local_backend import LocalBackend

        return LocalBackend()
    if name == "mint":
        from safetycite.backends.mint_backend import MintBackend

        return MintBackend()
    raise ValueError(f"Unknown backend '{name}'. Use 'local' or 'mint'.")


__all__ = [
    "AdapterRef",
    "Backend",
    "Sampler",
    "SamplingParams",
    "find_adapter",
    "get_backend",
    "list_adapters",
]
