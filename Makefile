# SafetyCite — convenience targets (local/Mac path). Uses the project venv directly
# to avoid `uv run` locking the heavy mint/local extras together.
PY := .venv/bin/python
CLI := .venv/bin/safetycite

.PHONY: setup fetch build data sft rl eval serve ui ui-build test lint clean all

setup:                       ## create venv + install core+local deps
	uv venv
	uv pip install --python .venv/bin/python -e ".[local,dev]"

fetch:                       ## pull real OSHA corpus from eCFR
	$(CLI) fetch

build:                       ## build per-domain Q&A datasets
	$(CLI) build

data: fetch build            ## corpus + datasets

sft:                         ## train all SFT adapters (DOMAIN=construction for one)
	$(CLI) sft $(or $(DOMAIN),all)

rl:                          ## GRPO refine (DOMAIN=construction for one)
	$(CLI) rl $(or $(DOMAIN),all)

eval:                        ## evaluate base vs adapter
	$(CLI) eval $(or $(DOMAIN),all)

serve:                       ## run API + built UI on :8000
	$(CLI) serve

ui:                          ## run the Vite dev server on :5173 (proxies /api)
	cd web && bun install && bun run dev

ui-build:                    ## build the UI for FastAPI to serve
	cd web && bun install && bun run build

test:                        ## run unit tests
	$(PY) -m pytest -q

lint:                        ## ruff check
	$(PY) -m ruff check safetycite

clean:                       ## remove generated artifacts
	rm -rf data/corpus/*.json data/datasets/* data/adapters/* web/dist

all: data sft rl eval        ## full local pipeline
