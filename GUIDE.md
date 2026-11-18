# Running podfit locally

Two ways to run it: Docker (matches how it's actually deployed) or a plain
Python venv (faster iteration — edit `app/static/` and reload, no rebuild).

## Option A: Docker

```bash
./detect-hardware.sh        # optional: writes data/detected_hardware.json from nvidia-smi
cp .env.example .env        # optional: HF_TOKEN, AA_API_KEY
docker compose up -d --build
curl -s localhost:8510/api/health
```

`data/` holds the SQLite database (saved rigs + the Hub/benchmark cache) and
survives rebuilds — re-running `detect-hardware.sh` only seeds a rig the app
has never seen, so your edits are never clobbered.

Needs Docker Desktop (or another Docker daemon) actually running — on
Windows, start Docker Desktop first, or `docker compose` fails with something
like `failed to connect to the docker API at npipe:////./pipe/...`.

## Option B: Python venv

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

mkdir -p data
PODFIT_DB=./data/podfit.db PODFIT_SEED=./data/detected_hardware.json \
  uvicorn app.main:app --reload --host 127.0.0.1 --port 8510
```

Open `http://127.0.0.1:8510`. `--reload` picks up backend (`app/*.py`)
changes automatically; `app/static/` has no build step, so a browser refresh
is all you need there.

### Windows: Python version matters

`pydantic-core` is a compiled dependency with prebuilt wheels for specific
Python versions. If your default `python` is newer than what's been built
yet (this bit me with Python 3.14 — the Rust build fails with
`the configured Python interpreter version (3.14) is newer than PyO3's
maximum supported version`), create the venv with an older interpreter
instead:

```bash
py -0                        # list installed Python versions
py -3.13 -m venv .venv        # pick one with prebuilt wheels available
```

## Keys: HF_TOKEN / AA_API_KEY

Both are optional. Without them you get anonymous Hub access (gated repos
like Llama/Gemma report "gated" instead of a size) and only the archived HF
Open LLM Leaderboard for benchmark scores (no Artificial Analysis).

Two ways to set them:

1. **Server-side** (applies to every visitor): put them in `.env` — see
   `.env.example` — and restart. For the Python venv path, export them
   before launching uvicorn instead: `HF_TOKEN=... AA_API_KEY=... uvicorn ...`
   (module-level `os.environ.get()` reads happen once at import, so a
   running process won't pick up a `.env` change without a restart).
2. **Bring your own key** (per-browser, no server change): open the
   **API keys** button in the top bar and paste a personal HF token / AA
   key. It's stored in that browser's `localStorage` and sent as headers,
   overriding the server's key for that request only — see the in-app
   **Docs** tab for how this works and why gated-repo results fetched this
   way are excluded from the shared cache.

## Verifying it's up

```bash
curl -s localhost:8510/api/health
# {"ok":true,"hf_token":true,"cache":{"entries":N,"newest":...}}
curl -s localhost:8510/api/catalog | head -c 200
```

If `/` loads but every search/sizing call fails, check the server log for
the actual HuggingFace/network error — `api()` in `app/static/app.js`
surfaces the backend's error `detail` directly in the UI, so the message
you see client-side is usually the real cause.

## Common issues

- **`ModuleNotFoundError: No module named 'uvicorn'`** — you're running the
  system Python, not the venv's. Activate the venv, or call
  `./.venv/Scripts/python.exe -m uvicorn ...` directly on Windows.
- **`/data/...` path errors on Windows** — `PODFIT_DB`/`PODFIT_SEED` default
  to absolute Unix paths (`/data/podfit.db`) meant for the Linux container.
  Always set both to a local path (as above) when running outside Docker.
- **Docker Desktop not running** — see Option A above; start it first,
  there's no workaround.
