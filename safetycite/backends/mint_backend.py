"""Remote MinT engine (Tinker-compatible).

MinT runs training/inference on remote GPUs; the `tinker` SDK (==0.6.3, sk-mint-* keys)
is a thin client. We reuse the HF tokenizer to turn prompts/targets into token ids and
drive the documented Tinker surface:

    service = tinker.ServiceClient()                       # reads TINKER_API_KEY/BASE_URL
    tc = service.create_lora_training_client(base_model, rank=...)
    await tc.forward_backward_async(data, "cross_entropy")
    await tc.optim_step_async(types.AdamParams(learning_rate=...))
    sc = tc.save_weights_and_get_sampling_client(name)     # -> SamplingClient
    await sc.sample_async(prompt, num_samples=..., sampling_params=...)

Requires `pip install tinker==0.6.3` and a MinT key; cannot be exercised offline.
"""

from __future__ import annotations

import asyncio
import random
from collections.abc import Callable

from safetycite.backends.base import (
    AdapterRef,
    Backend,
    Sampler,
    SamplingParams,
)
from safetycite.config import TrainConfig, settings
from safetycite.data.render import render_messages, render_target


def _require_tinker():
    try:
        import tinker  # noqa: F401
        from tinker import types  # noqa: F401
    except ImportError as e:  # pragma: no cover - depends on optional extra
        raise RuntimeError(
            "MinT backend needs the tinker SDK: `pip install tinker==0.6.3` and set "
            "TINKER_API_KEY (sk-mint-*) + TINKER_BASE_URL."
        ) from e
    return tinker, types


def _tokenizer(base_model: str):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(base_model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def _prompt_ids(tok, ex_or_q, domain=None):
    q = ex_or_q if isinstance(ex_or_q, str) else ex_or_q.question
    dom = domain if domain is not None else getattr(ex_or_q, "domain", None)
    text = tok.apply_chat_template(
        render_messages(q, dom), tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    return tok(text, add_special_tokens=False)["input_ids"]


class MintSampler(Sampler):
    def __init__(self, sampling_client, base_model: str):
        self.sc = sampling_client
        self.tok = _tokenizer(base_model)

    def generate(self, messages: list[dict], params: SamplingParams | None = None) -> str:
        _, types = _require_tinker()
        params = params or SamplingParams()
        text = self.tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        prompt = types.ModelInput.from_ints(self.tok(text, add_special_tokens=False)["input_ids"])
        sp = types.SamplingParams(
            max_tokens=params.max_new_tokens,
            temperature=max(params.temperature, 1e-4),
            top_p=params.top_p,
        )
        res = asyncio.run(self.sc.sample_async(prompt=prompt, num_samples=1, sampling_params=sp))
        out_tokens = res.sequences[0].tokens
        return self.tok.decode(out_tokens, skip_special_tokens=True).strip()


class MintBackend(Backend):
    name = "mint"

    def __init__(self, base_model: str | None = None):
        self.base_model = base_model or settings.base_model

    def info(self) -> dict:
        return {
            "backend": "mint",
            "base_model": self.base_model,
            "base_url": settings.__dict__.get("TINKER_BASE_URL", "(env)"),
        }

    def sampler(self, adapter: AdapterRef | None = None) -> Sampler:
        tinker, _ = _require_tinker()
        service = tinker.ServiceClient()
        model = adapter.base_model if adapter else self.base_model
        if adapter and adapter.path:
            # Re-create a sampling client from saved weights uri.
            tc = service.create_lora_training_client(base_model=model)
            tc.load_state(adapter.path)
            sc = tc.save_weights_and_get_sampling_client(name=f"{adapter.domain}-serve")
        else:
            sc = service.create_sampling_client(base_model=model)
        return MintSampler(sc, model)

    # --- SFT ----------------------------------------------------------------
    def _datums(self, tok, examples, types):
        data = []
        for ex in examples:
            p_ids = _prompt_ids(tok, ex)
            t_ids = tok(render_target(ex) + tok.eos_token, add_special_tokens=False)["input_ids"]
            full = p_ids + t_ids
            input_tokens = full[:-1]
            target_tokens = full[1:]
            weights = [0.0 if (i + 1) < len(p_ids) else 1.0 for i in range(len(input_tokens))]
            data.append(
                types.Datum(
                    model_input=types.ModelInput.from_ints(tokens=input_tokens),
                    loss_fn_inputs=dict(weights=weights, target_tokens=target_tokens),
                )
            )
        return data

    def train_sft(self, domain, train, val, cfg: TrainConfig | None = None) -> AdapterRef:
        tinker, types = _require_tinker()
        cfg = cfg or TrainConfig()
        tok = _tokenizer(self.base_model)
        service = tinker.ServiceClient()
        tc = service.create_lora_training_client(base_model=self.base_model, rank=cfg.lora_rank)
        data = self._datums(tok, list(train), types)
        rng = random.Random(0)

        async def run():
            for epoch in range(cfg.sft_epochs):
                rng.shuffle(data)
                for i in range(0, len(data), 8):
                    batch = data[i : i + 8]
                    await tc.forward_backward_async(batch, "cross_entropy")
                    await tc.optim_step_async(types.AdamParams(learning_rate=cfg.sft_lr))
                print(f"[mint-sft:{domain}] epoch {epoch + 1}/{cfg.sft_epochs}")

        asyncio.run(run())
        name = f"{domain}-sft"
        tc.save_weights_and_get_sampling_client(name=name)
        ref = AdapterRef(
            backend="mint",
            base_model=self.base_model,
            domain=domain,
            path=name,
            method="sft",
            notes=f"MinT SFT {cfg.sft_epochs}ep on {len(data)} examples",
        )
        ref.save()
        return ref

    # --- GRPO ---------------------------------------------------------------
    def train_rl(
        self,
        domain,
        train,
        val,
        reward_fn: Callable[[str, list[str]], float],
        cfg: TrainConfig | None = None,
    ) -> AdapterRef:
        tinker, types = _require_tinker()
        cfg = cfg or TrainConfig()
        tok = _tokenizer(self.base_model)
        service = tinker.ServiceClient()
        tc = service.create_lora_training_client(base_model=self.base_model, rank=cfg.lora_rank)
        examples = list(train)
        rng = random.Random(0)
        G = cfg.rl_group_size

        async def run():
            for step in range(cfg.rl_steps):
                ex = rng.choice(examples)
                sc = tc.save_weights_and_get_sampling_client(name=f"{domain}-rl-{step}")
                prompt = types.ModelInput.from_ints(_prompt_ids(tok, ex))
                sp = types.SamplingParams(max_tokens=cfg.max_new_tokens, temperature=1.0, top_p=0.95)
                res = await sc.sample_async(prompt=prompt, num_samples=G, sampling_params=sp)
                texts = [tok.decode(s.tokens, skip_special_tokens=True) for s in res.sequences]
                rewards = [reward_fn(t, ex.gold_citations) for t in texts]
                mean = sum(rewards) / len(rewards)
                # Build weighted SFT-style updates toward above-average samples (REINFORCE).
                p_ids = _prompt_ids(tok, ex)
                data = []
                for s, r in zip(res.sequences, rewards, strict=False):
                    adv = r - mean
                    if adv <= 0:
                        continue
                    full = p_ids + list(s.tokens)
                    input_tokens, target_tokens = full[:-1], full[1:]
                    weights = [
                        (0.0 if (i + 1) < len(p_ids) else adv) for i in range(len(input_tokens))
                    ]
                    data.append(
                        types.Datum(
                            model_input=types.ModelInput.from_ints(tokens=input_tokens),
                            loss_fn_inputs=dict(weights=weights, target_tokens=target_tokens),
                        )
                    )
                if data:
                    await tc.forward_backward_async(data, "cross_entropy")
                    await tc.optim_step_async(types.AdamParams(learning_rate=cfg.rl_lr))
                if (step + 1) % 5 == 0:
                    print(f"[mint-rl:{domain}] step {step + 1}/{cfg.rl_steps} mean_reward={mean:.3f}")

        asyncio.run(run())
        name = f"{domain}-rl"
        tc.save_weights_and_get_sampling_client(name=name)
        ref = AdapterRef(
            backend="mint",
            base_model=self.base_model,
            domain=domain,
            path=name,
            method="rl",
            notes=f"MinT GRPO {cfg.rl_steps} steps, G={G}",
        )
        ref.save()
        return ref
