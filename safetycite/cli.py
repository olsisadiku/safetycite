"""SafetyCite CLI: fetch → build → sft → rl → eval → serve."""

from __future__ import annotations

from dataclasses import replace

import typer
from rich.console import Console
from rich.table import Table

from safetycite.config import DATASETS_DIR, DOMAINS, get_domain, settings

app = typer.Typer(add_completion=False, help="SafetyCite — OSHA compliance copilot on MinT.")
console = Console()


def _domains_arg(domain: str) -> list[str]:
    if domain == "all":
        return list(DOMAINS)
    if domain not in DOMAINS:
        raise typer.BadParameter(f"Unknown domain '{domain}'. Use one of: {', '.join(DOMAINS)}, all")
    return [domain]


@app.command()
def fetch(
    parts: str = typer.Option("1926,1910,1904", help="CFR parts to fetch (comma-separated)."),
    full: bool = typer.Option(False, help="Fetch the entire part instead of the curated set."),
    date: str = typer.Option(None, help="eCFR snapshot date (default from settings)."),
):
    """Pull real OSHA regulation text from the eCFR API into data/corpus/."""
    from safetycite.corpus.fetch_ecfr import fetch_domain
    from safetycite.corpus.index import domain_for_part, save_domain_corpus

    for part in [p.strip() for p in parts.split(",") if p.strip()]:
        dom = domain_for_part(part)
        console.print(f"Fetching {dom} (29 CFR {part}){' [full]' if full else ''}…")
        recs = fetch_domain(dom, date=date, full=full)
        n = save_domain_corpus(dom, recs)
        console.print(f"  [green]✓[/] {n} sections -> data/corpus/{dom}.json")


@app.command()
def build(
    domain: str = typer.Option("all", help="Domain key or 'all'."),
    seed: int = typer.Option(0, help="Shuffle/split seed."),
):
    """Build per-domain Q&A datasets (train/val/test) from corpus + seed scenarios."""
    from safetycite.data.build_dataset import build_domain

    table = Table("domain", "headings", "scenarios", "train", "val", "test")
    for d in _domains_arg(domain):
        r = build_domain(d, seed=seed)
        for w in r["warnings"]:
            console.print(f"  [yellow]warn[/] {w}")
        table.add_row(d, *(str(r[k]) for k in ("headings", "scenarios", "train", "val", "test")))
    console.print(table)


@app.command()
def sft(
    domain: str = typer.Argument(..., help="Domain key or 'all'."),
    backend: str = typer.Option(None, help="local | mint (default from env)."),
    epochs: int = typer.Option(None, help="Override SFT epochs."),
):
    """Train a LoRA adapter with supervised fine-tuning."""
    from safetycite.backends import get_backend
    from safetycite.data.schema import read_jsonl

    be = get_backend(backend)
    for d in _domains_arg(domain):
        cfg = get_domain(d).train
        if epochs:
            cfg = replace(cfg, sft_epochs=epochs)
        train = read_jsonl(DATASETS_DIR / d / "train.jsonl")
        val = read_jsonl(DATASETS_DIR / d / "val.jsonl")
        console.print(f"[bold]SFT[/] {d} ({len(train)} ex) on backend={be.name}…")
        ref = be.train_sft(d, train, val, cfg)
        console.print(f"  [green]✓[/] {ref.notes} -> {ref.path}")


@app.command()
def rl(
    domain: str = typer.Argument(..., help="Domain key or 'all'."),
    backend: str = typer.Option(None, help="local | mint (default from env)."),
    steps: int = typer.Option(None, help="Override GRPO steps."),
):
    """Refine an adapter with GRPO using the verifiable citation reward."""
    from safetycite.backends import get_backend
    from safetycite.corpus.index import CorpusIndex
    from safetycite.data.schema import read_jsonl
    from safetycite.rewards.composite import make_reward_fn

    be = get_backend(backend)
    reward_fn = make_reward_fn(CorpusIndex.load())
    for d in _domains_arg(domain):
        cfg = get_domain(d).train
        if steps:
            cfg = replace(cfg, rl_steps=steps)
        train = read_jsonl(DATASETS_DIR / d / "train.jsonl")
        val = read_jsonl(DATASETS_DIR / d / "val.jsonl")
        console.print(f"[bold]GRPO[/] {d} ({cfg.rl_steps} steps) on backend={be.name}…")
        ref = be.train_rl(d, train, val, reward_fn, cfg)
        console.print(f"  [green]✓[/] {ref.notes} -> {ref.path}")


@app.command(name="eval")
def eval_cmd(
    domain: str = typer.Argument(..., help="Domain key or 'all'."),
    backend: str = typer.Option(None, help="local | mint (default from env)."),
    no_base: bool = typer.Option(False, "--no-base", help="Skip base-model comparison."),
    limit: int = typer.Option(None, help="Limit test examples (faster)."),
):
    """Evaluate base vs adapter on the held-out test set and save the report."""
    from safetycite.eval.evaluate import evaluate_domain

    table = Table("domain", "split", "citation_f1", "exact", "format", "halluc", "reward")
    for d in _domains_arg(domain):
        r = evaluate_domain(d, backend, compare_base=not no_base, limit=limit)
        if "base" in r:
            b = r["base"]
            table.add_row(d, "base", f"{b['citation_f1']:.2f}", f"{b['exact_match']:.2f}",
                          f"{b['format']:.2f}", f"{b['hallucination_rate']:.2f}", f"{b['reward']:.2f}")
        a = r["adapter"]
        tag = f"adapter:{r['adapter_method']}" if r["adapter_present"] else "adapter:none"
        table.add_row(d, tag, f"{a['citation_f1']:.2f}", f"{a['exact_match']:.2f}",
                      f"{a['format']:.2f}", f"{a['hallucination_rate']:.2f}", f"{a['reward']:.2f}")
    console.print(table)


@app.command()
def serve(
    host: str = typer.Option(settings.host),
    port: int = typer.Option(settings.port),
    reload: bool = typer.Option(False),
):
    """Run the FastAPI server (serves the API and, if built, the web UI)."""
    import uvicorn

    console.print(f"[bold]SafetyCite[/] serving on http://{host}:{port}  (backend={settings.backend})")
    uvicorn.run("safetycite.serve.app:app", host=host, port=port, reload=reload)


if __name__ == "__main__":
    app()
