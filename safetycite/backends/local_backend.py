"""Live local engine: real Qwen3 via Hugging Face on MPS / CUDA / CPU.

Inference shares ONE base model in memory; LoRA adapters are loaded by name and
switched per request (or disabled to get the true base model) — so "base vs
fine-tuned" comparison is cheap and honest. Training is a real LoRA SFT loop.

torch/transformers/peft are imported lazily so the rest of the package stays light.
"""

from __future__ import annotations

import contextlib
from collections.abc import Callable

from safetycite.backends.base import (
    AdapterRef,
    Backend,
    Sampler,
    SamplingParams,
    adapter_dir,
)
from safetycite.config import TrainConfig, settings
from safetycite.data.render import render_messages, render_target

# Qwen3 LoRA target modules
_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

_ENGINES: dict[str, _Engine] = {}


def _device() -> str:
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _dtype(device: str):
    import torch

    return torch.bfloat16 if device == "cuda" else torch.float32


class _Engine:
    """Holds one base model + tokenizer and any loaded adapters."""

    def __init__(self, model_name: str):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.model_name = model_name
        self.device = _device()
        self.dtype = _dtype(self.device)
        self.tok = AutoTokenizer.from_pretrained(model_name)
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        self.base = AutoModelForCausalLM.from_pretrained(model_name, dtype=self.dtype)
        self.base.to(self.device)
        self.base.eval()
        self.peft = None  # becomes a PeftModel once an adapter is attached
        self._adapters: dict[str, str] = {}  # name -> path
        self._torch = torch

    def _key(self, ref: AdapterRef) -> str:
        return f"{ref.domain}_{ref.method}"

    def ensure_adapter(self, ref: AdapterRef) -> str:
        from peft import PeftModel

        name = self._key(ref)
        if name in self._adapters:
            return name
        if self.peft is None:
            self.peft = PeftModel.from_pretrained(self.base, ref.path, adapter_name=name)
            self.peft.to(self.device)
            self.peft.eval()
        else:
            self.peft.load_adapter(ref.path, adapter_name=name)
        self._adapters[name] = ref.path
        return name

    def _chat_text(self, messages: list[dict]) -> str:
        return self.tok.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,  # keep Qwen3 out of long reasoning traces
        )

    def generate(
        self, messages: list[dict], adapter_name: str | None, params: SamplingParams
    ) -> str:
        torch = self._torch
        model = self.peft if self.peft is not None else self.base
        text = self._chat_text(messages)
        inputs = self.tok(text, return_tensors="pt").to(self.device)
        in_len = inputs["input_ids"].shape[1]

        if self.peft is not None and adapter_name is not None:
            self.peft.set_adapter(adapter_name)
            ctx = contextlib.nullcontext()
        elif self.peft is not None:  # want the true base => disable all adapters
            ctx = self.peft.disable_adapter()
        else:
            ctx = contextlib.nullcontext()

        gen_kwargs = dict(
            max_new_tokens=params.max_new_tokens,
            do_sample=params.do_sample,
            pad_token_id=self.tok.pad_token_id,
        )
        if params.do_sample:
            gen_kwargs.update(temperature=params.temperature, top_p=params.top_p)

        with torch.no_grad(), ctx:
            out = model.generate(**inputs, **gen_kwargs)
        new_tokens = out[0][in_len:]
        decoded = self.tok.decode(new_tokens, skip_special_tokens=True).strip()
        return _strip_think(decoded)


def _strip_think(text: str) -> str:
    import re

    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    # If a stray opening tag remains, drop everything up to its close or keep as-is.
    return text


def _engine(model_name: str) -> _Engine:
    if model_name not in _ENGINES:
        _ENGINES[model_name] = _Engine(model_name)
    return _ENGINES[model_name]


class LocalSampler(Sampler):
    def __init__(self, model_name: str, adapter: AdapterRef | None):
        self.engine = _engine(model_name)
        self.adapter_name = self.engine.ensure_adapter(adapter) if adapter and adapter.path else None

    def generate(self, messages: list[dict], params: SamplingParams | None = None) -> str:
        return self.engine.generate(messages, self.adapter_name, params or SamplingParams())


class LocalBackend(Backend):
    name = "local"

    def __init__(self, base_model: str | None = None):
        self.base_model = base_model or settings.base_model

    def info(self) -> dict:
        return {"backend": "local", "base_model": self.base_model, "device": _device()}

    def sampler(self, adapter: AdapterRef | None = None) -> Sampler:
        model_name = adapter.base_model if adapter else self.base_model
        return LocalSampler(model_name, adapter)

    # --- training -----------------------------------------------------------
    def train_sft(self, domain, train, val, cfg: TrainConfig | None = None) -> AdapterRef:
        from safetycite.backends.local_train import train_sft_local

        return train_sft_local(self.base_model, domain, train, cfg or TrainConfig())

    def train_rl(
        self,
        domain,
        train,
        val,
        reward_fn: Callable[[str, list[str]], float],
        cfg: TrainConfig | None = None,
    ) -> AdapterRef:
        from safetycite.backends.local_train import train_rl_local

        return train_rl_local(self.base_model, domain, train, reward_fn, cfg or TrainConfig())


def _adapter_path(domain: str, method: str) -> str:
    return str(adapter_dir(domain, "local", method))


# expose for trainer module
_render_messages = render_messages
_render_target = render_target
