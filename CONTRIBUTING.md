# Contributing to podfit

podfit is a small, single-maintainer project — contributions are welcome, but
keep changes focused and in line with what's already here.

## Before you open a PR

- **Bug reports**: open an issue with the model repo, GPU, engine, and
  runtime settings that produced the wrong result, plus what you expected.
  For a sizing/compatibility bug, include the exact `config.json` fields
  that matter (`num_hidden_layers`, `hidden_size`, attention/KV heads,
  quantization config) if the repo is private or gated.
- **New hardware**: GPU catalog entries (`app/catalog.py`) need a real
  source — a vendor spec sheet, an official announcement, or a
  well-corroborated third-party benchmark. Don't guess at bandwidth or
  TFLOPS; if a number isn't officially published (true for every Apple
  Silicon GPU), say so in the entry's `note` field the way existing
  entries do.
- **New model architectures**: additions to `app/archsupport.py` need the
  engine's actual kernel requirements (minimum compute capability, which
  of vLLM/SGLang/llama.cpp support it) — not assumptions from the model
  card.
- **Features**: open an issue first for anything beyond a bug fix, so we
  don't duplicate effort or build something that doesn't fit the project's
  scope (a feasibility calculator, not a benchmark suite or a model zoo).

## Dev setup

```bash
python -m venv .venv && source .venv/bin/activate   # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
PODFIT_DB=./data/podfit.db PODFIT_SEED=./data/detected_hardware.json \
  uvicorn app.main:app --reload --port 8510
```

No build step for the frontend (`app/static/`) — it's plain HTML/CSS/JS,
edit and reload. Both `HF_TOKEN` and `AA_API_KEY` are optional; without
them you still get anonymous Hub access and the archived HF Open LLM
Leaderboard for scores. See `README.md` for the full rundown, and the
in-app **Docs** tab's Hardware reference for interconnects, Apple Silicon,
and NVIDIA/L20 background.

## Code conventions

- No comments unless they explain a non-obvious *why* (a hidden constraint,
  a workaround for a specific upstream bug, an invariant that isn't clear
  from the code itself). Don't restate what the code already says.
- Match the existing style: dense, technical prose in docstrings and
  findings text, no filler. The app's `_f(level, title, detail, tag)`
  findings are user-facing copy — they should read like the rest of the
  app, not like generic error messages.
- "Nothing is invented" is the project's core promise (see README). Any
  number the app shows — a parameter count, a benchmark score, a memory
  figure, a GPU spec — should trace back to a real source (the Hub's
  `config.json`/safetensors metadata, a benchmark provider, a vendor spec
  sheet) or be clearly labeled as an estimate and why.
- Keep PRs scoped to one change. A bug fix doesn't need a refactor riding
  along with it.

## License

By contributing, you agree your changes are licensed under the project's
GPL-3.0 license (see `LICENSE`).
