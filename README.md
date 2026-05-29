<div align="center">

# 🦺 SafetyCite

**An OSHA Compliance Copilot trained with [MinT](https://mint-doc.macaron.im)**

Small LLMs that answer workplace-safety questions **and cite the correct OSHA standard** —
trained with reinforcement learning on a *verifiable* citation reward.

</div>

---

## Why this exists

Most "ask the docs" copilots can't tell you *whether they're right*. OSHA is different:
when someone asks about fall protection in construction, the correct answer cites a specific
regulation — **29 CFR §1926.501** — and that's **checkable**. Citation correctness is a hard,
ground-truth signal.

That makes OSHA the rare domain where **RL with verifiable rewards (GRPO)** — exactly what
**MinT** (Macaron's Tinker-compatible LoRA SFT/RL platform) is built for — is the natural fit,
and where "before vs after training" produces a real, demoable number: citation accuracy ↑,
hallucinated citations ↓.

## What it does

- **Per-domain LoRA adapters** — one each for **Construction (1926)**, **General Industry (1910)**,
  and **Recordkeeping (1904)**: a hot-swappable *mixture of compliance experts*. A router picks the
  right one per question.
- **Verifiable reward** — `reward = citation_F1 + format − hallucination_penalty`, checked against
  the **real eCFR corpus**. No judge model — the same function is the RL reward *and* the eval metric.
- **Fully live** — every answer is a real model generation. No mocks. Run it on this Mac (Apple MPS),
  on Colab's free GPU, or on remote MinT GPUs — same code, swappable engine.
- **A UI you can actually test** — ask a scenario, see answers with **clickable, verified** CFR text
  (✓ real / ✗ hallucinated), toggle **base vs fine-tuned** side-by-side, and watch the **eval dashboard**.

```
question ─▶ Router ─▶ Sampler(adapter) ─┬─▶ answer + parsed citations
                                         └─▶ Corpus index ─▶ verify each cite exists + show text
Training:  eCFR corpus ─▶ datasets ─▶ SFT ─▶ GRPO(citation reward) ─▶ adapter ─▶ eval
Engines (swappable, all live):   Local/HF (MPS·CUDA·CPU)   |   MinT (remote)
```

## Quickstart

### ▶ Colab (primary — free GPU, public URL)
Open **`notebooks/colab_app.ipynb`** in Colab, set a T4 GPU, set `REPO_URL`, and run all cells.
It clones, installs, fetches the corpus, trains the adapters, builds the UI, serves the real model,
and prints a public **cloudflared** `https://…trycloudflare.com` link.

### ▶ This Mac (Apple Silicon / MPS)
```bash
uv venv && uv pip install -e ".[local]"      # core + torch/transformers/peft
uv run safetycite fetch                       # real OSHA text from eCFR  -> data/corpus/
uv run safetycite build                       # Q&A datasets             -> data/datasets/
uv run safetycite sft construction            # train a LoRA adapter (real, on MPS)
uv run safetycite rl  construction            # refine with GRPO (verifiable reward)
uv run safetycite eval construction           # base vs adapter report
uv run safetycite serve                       # API + UI on :8000
# in another shell, for hot-reload dev UI:
cd web && bun install && bun run dev          # :5173, proxies /api
```
> `uv run` locks all extras; if the `mint` extra conflicts on your Python, use the venv directly
> (`.venv/bin/safetycite …`) — see notes below.

### ▶ MinT (remote GPUs)
```bash
uv pip install -e ".[mint]"                   # tinker==0.6.3 (sk-mint-* keys)
export SAFETYCITE_BACKEND=mint
export TINKER_API_KEY=sk-mint-...             # macaron.im/mindlab/mint
export TINKER_BASE_URL=https://mint.macaron.xin/
safetycite sft construction --backend mint
safetycite serve
```

## CLI

| command | what |
|---|---|
| `safetycite fetch [--parts 1926,1910,1904] [--full]` | pull OSHA text from eCFR |
| `safetycite build [--domain all]` | build train/val/test Q&A |
| `safetycite sft <domain\|all>` | LoRA supervised fine-tuning |
| `safetycite rl <domain\|all>` | GRPO with the citation reward |
| `safetycite eval <domain\|all>` | base vs adapter metrics → JSON |
| `safetycite serve` | FastAPI API + built UI |

Pick the engine with `--backend local|mint` or `SAFETYCITE_BACKEND`.

## How it works

- **Corpus** (`safetycite/corpus`) — fetches real 29 CFR sections from the public eCFR API and indexes
  them by canonical section id. This index is the **single source of truth for citation verification**,
  used by both the reward and the UI.
- **Reward** (`safetycite/rewards`) — parses every `29 CFR §xxxx.xx` out of an answer, scores citation
  **F1** vs gold (F1, not recall, so over-citing is penalised), checks **format**, and penalises
  **hallucinated** sections that don't exist in the corpus.
- **Data** (`safetycite/data`) — builds Q&A from corpus headings + hand-curated scenarios; every gold
  citation is validated against the index. One renderer defines the answer contract used by training,
  the reward, and the UI.
- **Backends** (`safetycite/backends`) — `local` (HF, shares one base model and hot-swaps adapters /
  disables them for the true base) and `mint` (Tinker SDK). Both implement real SFT and GRPO.

## Project layout

```
safetycite/   corpus · rewards · data · backends · train · eval · serve · cli
web/          Bun + React + Vite + Tailwind UI
notebooks/    colab_app.ipynb  (primary runtime)
data/         corpus · datasets · adapters   (generated)
tests/        pure-logic tests for rewards + corpus
```

## Notes & caveats
- **Not legal advice.** A research/demo project; verify against the actual regulation (links provided).
- **Small model.** `Qwen3-0.6B` is the cheap default — great for showing the *delta*; bump to
  `Qwen/Qwen3-4B-Instruct-2507` (esp. on MinT) for stronger absolute accuracy.
- **Colab is ephemeral** — perfect for a live demo; use a real host or MinT for persistence.
