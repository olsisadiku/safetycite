"""Real LoRA training for the local engine: SFT + GRPO. No trl dependency — small,
explicit loops that run on MPS/CUDA/CPU and stay correct across library versions.

GRPO here is the core of the method: sample a group of completions per prompt, score
each with the *verifiable citation reward*, normalise rewards within the group to get
advantages, and push the policy toward the better-rewarded samples.
"""

from __future__ import annotations

import random
from collections.abc import Callable
from pathlib import Path

from safetycite.backends.base import AdapterRef, adapter_dir
from safetycite.config import TrainConfig
from safetycite.data.render import render_messages, render_target

_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def _device_dtype():
    import torch

    if torch.cuda.is_available():
        return "cuda", torch.bfloat16
    if torch.backends.mps.is_available():
        return "mps", torch.float32
    return "cpu", torch.float32


def _tokenizer(base_model):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(base_model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def _lora_config(cfg: TrainConfig):
    from peft import LoraConfig

    return LoraConfig(
        r=cfg.lora_rank,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=0.05,
        target_modules=_TARGET_MODULES,
        task_type="CAUSAL_LM",
    )


def _fresh_peft(base_model, cfg, device, dtype):
    from peft import get_peft_model
    from transformers import AutoModelForCausalLM

    base = AutoModelForCausalLM.from_pretrained(base_model, dtype=dtype)
    model = get_peft_model(base, _lora_config(cfg))
    model.to(device)
    return model


def _prompt_text(tok, ex):
    return tok.apply_chat_template(
        render_messages(ex.question, ex.domain),
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )


# --- SFT --------------------------------------------------------------------
def train_sft_local(base_model: str, domain: str, train, cfg: TrainConfig) -> AdapterRef:
    import torch

    device, dtype = _device_dtype()
    tok = _tokenizer(base_model)
    model = _fresh_peft(base_model, cfg, device, dtype)
    model.train()
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=cfg.sft_lr)

    examples = list(train)
    rng = random.Random(0)
    step = 0
    for epoch in range(cfg.sft_epochs):
        rng.shuffle(examples)
        running = 0.0
        for ex in examples:
            prompt = _prompt_text(tok, ex)
            target = render_target(ex) + tok.eos_token
            p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
            t_ids = tok(target, add_special_tokens=False)["input_ids"]
            input_ids = torch.tensor([p_ids + t_ids], device=device)
            labels = torch.tensor([[-100] * len(p_ids) + t_ids], device=device)
            loss = model(input_ids=input_ids, labels=labels).loss
            loss.backward()
            opt.step()
            opt.zero_grad()
            running += float(loss.item())
            step += 1
        print(f"[sft:{domain}] epoch {epoch + 1}/{cfg.sft_epochs} avg_loss={running / max(len(examples),1):.4f}")

    path = str(adapter_dir(domain, "local", "sft"))
    model.save_pretrained(path)
    ref = AdapterRef(
        backend="local",
        base_model=base_model,
        domain=domain,
        path=path,
        method="sft",
        notes=f"SFT {cfg.sft_epochs}ep / {step} steps on {len(examples)} examples",
    )
    ref.save()
    print(f"[sft:{domain}] saved -> {path}")
    return ref


# --- GRPO -------------------------------------------------------------------
def _build_for_rl(base_model, domain, cfg, device, dtype):
    """Warm-start RL from the SFT adapter if present, else a fresh LoRA."""
    from peft import PeftModel
    from transformers import AutoModelForCausalLM

    base = AutoModelForCausalLM.from_pretrained(base_model, dtype=dtype)
    sft_path = adapter_dir(domain, "local", "sft")
    if (Path(sft_path) / "adapter_config.json").exists():
        model = PeftModel.from_pretrained(base, str(sft_path), is_trainable=True)
        warm = True
    else:
        from peft import get_peft_model

        model = get_peft_model(base, _lora_config(cfg))
        warm = False
    model.to(device)
    return model, warm


def train_rl_local(
    base_model: str,
    domain: str,
    train,
    reward_fn: Callable[[str, list[str]], float],
    cfg: TrainConfig,
) -> AdapterRef:
    import torch

    device, dtype = _device_dtype()
    tok = _tokenizer(base_model)
    model, warm = _build_for_rl(base_model, domain, cfg, device, dtype)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=cfg.rl_lr)
    eos = tok.eos_token_id
    examples = list(train)
    rng = random.Random(0)
    G = cfg.rl_group_size
    reward_hist: list[float] = []

    for step in range(cfg.rl_steps):
        ex = rng.choice(examples)
        prompt = _prompt_text(tok, ex)
        enc = tok(prompt, return_tensors="pt").to(device)
        p_len = enc["input_ids"].shape[1]

        # Sample G completions ONE AT A TIME -> bounds generation memory so this fits
        # bigger models (e.g. Qwen3-4B) on a 16 GB T4. Sequences have varying lengths.
        model.eval()
        seqs = []
        with torch.no_grad():
            for _ in range(G):
                out = model.generate(
                    **enc,
                    max_new_tokens=min(cfg.max_new_tokens, 200),
                    do_sample=True,
                    temperature=1.0,
                    top_p=0.95,
                    num_return_sequences=1,
                    pad_token_id=eos,
                )
                seqs.append(out[0].detach())
            if device == "mps":
                torch.mps.empty_cache()
        model.train()

        texts = [tok.decode(s[p_len:], skip_special_tokens=True) for s in seqs]
        rewards = torch.tensor(
            [reward_fn(t, ex.gold_citations) for t in texts], device=device, dtype=torch.float32
        )
        adv = (rewards - rewards.mean()) / (rewards.std() + 1e-6)
        reward_hist.append(float(rewards.mean().item()))

        # Policy gradient with group-normalised advantage. Back-prop PER SAMPLE so we
        # never hold G full-vocab autograd graphs at once (the earlier MPS OOM); fused
        # cross-entropy avoids materialising a full log_softmax over the ~150k vocab.
        opt.zero_grad()
        ce_loss = torch.nn.functional.cross_entropy
        for g in range(G):
            a = float(adv[g])
            if abs(a) < 1e-8:
                continue
            seq = seqs[g].unsqueeze(0)
            logits = model(input_ids=seq).logits[:, :-1, :]
            targets = seq[:, 1:]
            nll = ce_loss(
                logits.reshape(-1, logits.size(-1)), targets.reshape(-1), reduction="none"
            ).view(targets.shape)  # -logprob per token
            mask = torch.zeros_like(nll)
            mask[:, p_len - 1 :] = 1.0  # only completion tokens
            seq_lp = -(nll * mask).sum() / mask.sum().clamp(min=1)
            ((-a * seq_lp) / G).backward()
            del logits, nll, seq_lp
            if device == "mps":
                torch.mps.empty_cache()
        opt.step()

        if (step + 1) % 5 == 0 or step == 0:
            recent = sum(reward_hist[-10:]) / len(reward_hist[-10:])
            print(f"[rl:{domain}] step {step + 1}/{cfg.rl_steps} mean_reward={recent:.3f}")

    path = str(adapter_dir(domain, "local", "rl"))
    model.save_pretrained(path)
    ref = AdapterRef(
        backend="local",
        base_model=base_model,
        domain=domain,
        path=path,
        method="rl",
        notes=f"GRPO {cfg.rl_steps} steps, G={G}, warm_start={warm}",
        metrics={"final_mean_reward": reward_hist[-1] if reward_hist else 0.0},
    )
    ref.save()
    print(f"[rl:{domain}] saved -> {path}")
    return ref
