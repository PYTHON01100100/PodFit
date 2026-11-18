"""podfit - can this model run on this hardware?"""
from __future__ import annotations

import asyncio
import json
import os
import pathlib

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import aa_source, archsupport, attention, bench, db, engine, gpuarch, hf
from .catalog import ENGINES, GPUS, KV_DTYPES, WEIGHT_FORMATS

STATIC = pathlib.Path(__file__).parent / "static"
SEED_FILE = os.environ.get("PODFIT_SEED", "/data/detected_hardware.json")

app = FastAPI(title="podfit", version="1.0",
              description="Model-to-GPU feasibility calculator")


@app.on_event("startup")
def _startup() -> None:
    db.init()
    _seed_detected()


def _seed_detected() -> None:
    """Seed a rig from hardware detected on the host at install time."""
    try:
        data = json.loads(pathlib.Path(SEED_FILE).read_text())
    except (OSError, json.JSONDecodeError):
        return
    for rig in data.get("rigs", []):
        db.seed_rig_if_absent(rig["id"], rig["name"],
                              {k: v for k, v in rig.items() if k not in ("id", "name")})


# ---------------------------------------------------------------------------
# Reference data
# ---------------------------------------------------------------------------
@app.get("/api/catalog")
def catalog():
    return {
        "gpus": GPUS,
        "weight_formats": [{"key": k, **v} for k, v in WEIGHT_FORMATS.items()],
        "kv_dtypes": [{"key": k, **v} for k, v in KV_DTYPES.items()],
        "engines": [{"key": k, **v} for k, v in ENGINES.items()],
        "pipelines": [{"key": k, "label": lbl} for k, lbl in hf.PIPELINES],
        "arch_count": len(archsupport.ARCH),
        "hf_token": bool(hf.TOKEN),
        "bench": bench.status(),
    }


class ArchSupportRequest(BaseModel):
    repo: str | None = None
    model: dict | None = None
    runtime: Runtime


@app.post("/api/arch-support")
async def arch_support(req: ArchSupportRequest):
    """Which GPU architecture families can run this model, and what each costs."""
    model = req.model
    if not model:
        if not req.repo:
            raise HTTPException(400, "provide either repo or model")
        try:
            model = await hf.model_detail(req.repo)
        except hf.HFError as e:
            raise HTTPException(status_code=e.status, detail=e.message)
    spec = model.get("spec") or {}
    if not (spec.get("num_layers") and spec.get("hidden_size")):
        raise HTTPException(422, "no usable config.json for this repo")
    model.setdefault("params_total", spec.get("params_total"))
    return {"families": gpuarch.support_matrix(model, req.runtime.model_dump(), engine.evaluate)}


@app.get("/api/aa/search")
async def aa_search(q: str = "", limit: int = Query(25, ge=1, le=60)):
    if not bench.AA_KEY:
        raise HTTPException(400, "Artificial Analysis is not configured (no AA_API_KEY).")
    try:
        return await aa_source.search(q, limit)
    except Exception as e:
        raise HTTPException(502, f"Artificial Analysis search failed: {e}")


@app.get("/api/gpu-families")
def gpu_families():
    return {"families": [gpuarch._row(f) for f in gpuarch.FAMILIES]}


@app.get("/api/arch")
def arch_table():
    rows = []
    for key, e in sorted(archsupport.ARCH.items()):
        rows.append({"model_type": key, "family": e["family"], "vllm": e["vllm"],
                     "sglang": e["sglang"], "llamacpp": e["llamacpp"],
                     "traits": e["traits"], "min_cc": e["min_cc"], "note": e["note"]})
    return {"rows": rows, "labels": archsupport.STATUS_LABEL}


@app.get("/api/health")
def health():
    return {"ok": True, "hf_token": bool(hf.TOKEN), "cache": db.cache_stats()}


# ---------------------------------------------------------------------------
# HuggingFace
# ---------------------------------------------------------------------------
@app.get("/api/models/search")
async def models_search(q: str = "", pipeline: str = "", sort: str = "downloads",
                        limit: int = Query(30, ge=1, le=100), author: str = "",
                        library: str = ""):
    try:
        return {"results": await hf.search(q, pipeline, sort, limit, author, library)}
    except hf.HFError as e:
        raise HTTPException(status_code=e.status, detail=e.message)
    except Exception as e:  # network / upstream shape changes
        raise HTTPException(status_code=502, detail=f"HuggingFace request failed: {e}")


@app.get("/api/models/detail")
async def models_detail(repo: str, revision: str = "main", refresh: bool = False):
    try:
        d = await hf.model_detail(repo, revision, refresh)
    except hf.HFError as e:
        raise HTTPException(status_code=e.status, detail=e.message)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"HuggingFace request failed: {e}")
    spec = d["spec"]
    d["arch_support"] = archsupport.lookup(spec.get("model_type"), spec.get("architectures"))
    d["active_params"] = hf.active_params(spec, d["spec"]["params_total"])
    # The benchmark providers can be slow (upstream throttling). The UI fetches
    # /api/bench separately so sizing renders immediately.
    return d


@app.get("/api/bench")
async def bench_lookup(repo: str, base_model: str = ""):
    try:
        return {"repo": repo, "bench": await bench.lookup(repo, base_model or None)}
    except Exception as e:
        raise HTTPException(502, f"benchmark lookup failed: {e}")


@app.get("/api/avatars")
async def avatars(orgs: str):
    names = [o.strip() for o in orgs.split(",") if o.strip()][:60]
    return {"avatars": await hf.avatars(names)}


@app.get("/api/attention")
def attention_backends():
    return {"backends": [{"key": k, **{kk: vv for kk, vv in v.items()
                                       if kk in ("label", "family", "vendors", "note")}}
                         for k, v in attention.BACKENDS.items()]}


@app.post("/api/cache/clear")
def cache_clear():
    return {"cleared": db.cache_clear()}


# ---------------------------------------------------------------------------
# Rigs
# ---------------------------------------------------------------------------
class Rig(BaseModel):
    name: str
    gpu_id: str | None = None
    custom_gpu: dict | None = None
    gpus_per_node: int = 1
    nodes: int = 1
    host_ram_gb: float = 0
    os: str = "linux"
    interconnect: str = "pcie"
    driver_version: str = ""
    vram_override_gb: float | None = None
    notes: str = ""


@app.get("/api/rigs")
def rigs_list():
    return {"rigs": db.list_rigs()}


@app.post("/api/rigs")
def rig_create(rig: Rig, rig_id: str | None = None):
    rid = db.save_rig(rig.name, rig.model_dump(exclude={"name"}), rig_id=rig_id)
    return {"id": rid, "rig": db.get_rig(rid)}


@app.put("/api/rigs/{rig_id}")
def rig_update(rig_id: str, rig: Rig):
    if not db.get_rig(rig_id):
        raise HTTPException(404, "rig not found")
    db.save_rig(rig.name, rig.model_dump(exclude={"name"}), rig_id=rig_id)
    return {"id": rig_id, "rig": db.get_rig(rig_id)}


@app.delete("/api/rigs/{rig_id}")
def rig_delete(rig_id: str):
    if not db.delete_rig(rig_id):
        raise HTTPException(404, "rig not found")
    return {"deleted": rig_id}


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
class Hardware(BaseModel):
    gpu_id: str | None = None
    custom_gpu: dict | None = None
    gpus_per_node: int = 1
    nodes: int = 1
    host_ram_gb: float = 0
    os: str = "linux"
    interconnect: str = "pcie"
    driver_version: str = ""
    vram_override_gb: float | None = None


class Runtime(BaseModel):
    engine: str = "vllm"
    weight_format: str = "bf16"
    kv_dtype: str = "auto"
    context_len: int = Field(8192, ge=64, le=4_194_304)
    concurrency: int = Field(1, ge=1, le=100_000)
    tp: int = 0
    pp: int = 0
    gpu_mem_util: float = Field(0.90, gt=0.05, le=1.0)
    enforce_eager: bool = False
    max_batched_tokens: int = 8192
    gguf_size: int = 0
    gguf_path: str = ""
    attention: str = "auto"
    gpu_layers: int | str | None = None
    images_per_request: int = 0


class EvalRequest(BaseModel):
    repo: str | None = None
    revision: str = "main"
    model: dict | None = None      # pre-fetched detail, avoids a second Hub call
    hardware: Hardware
    runtime: Runtime


@app.post("/api/evaluate")
async def evaluate(req: EvalRequest):
    model = req.model
    if not model:
        if not req.repo:
            raise HTTPException(400, "provide either repo or model")
        try:
            model = await hf.model_detail(req.repo, req.revision)
        except hf.HFError as e:
            raise HTTPException(status_code=e.status, detail=e.message)
    spec = model.get("spec") or {}
    if not (spec.get("num_layers") and spec.get("hidden_size")):
        raise HTTPException(422, "This repo has no usable config.json (no layer/hidden "
                                 "size), so its memory footprint cannot be derived. "
                                 "GGUF-only or adapter-only repos need their base model.")
    model.setdefault("params_total", spec.get("params_total"))
    try:
        return engine.evaluate(model, req.hardware.model_dump(), req.runtime.model_dump())
    except ValueError as e:
        raise HTTPException(400, str(e))


@app.post("/api/compare")
async def compare(req: EvalRequest, gpu_ids: str = Query(...)):
    """Same model+runtime across several GPUs."""
    model = req.model
    if not model:
        model = await hf.model_detail(req.repo, req.revision)
    out = []
    for gid in [g for g in gpu_ids.split(",") if g]:
        hwd = req.hardware.model_dump()
        hwd["gpu_id"], hwd["custom_gpu"] = gid, None
        try:
            r = engine.evaluate(model, hwd, req.runtime.model_dump())
            out.append({"gpu_id": gid, "name": r["gpu"]["name"], "verdict": r["verdict"],
                        "headline": r["headline"], "parallel": r["parallel"],
                        "memory": r["memory"], "perf": r["perf"],
                        "blockers": [f for f in r["findings"] if f["level"] == "blocker"]})
        except ValueError:
            continue
    return {"results": out}


class RecommendRequest(BaseModel):
    hardware: Hardware
    runtime: Runtime
    pipeline: str = "text-generation"
    sort: str = "downloads"
    query: str = ""
    source: str = "hf"                 # "hf" = Hub lists, "aa" = ranked by intelligence
    pool: int = Field(40, ge=5, le=80)
    min_params: float = Field(0.4, ge=0, le=2000)   # billions


@app.post("/api/recommend")
async def recommend(req: RecommendRequest):
    """Rank real Hub models by benchmark score, keeping only those that fit."""
    aa_meta = None
    if req.source == "aa":
        # Rank by capability first, then ask what the hardware can hold.
        try:
            ranked = await aa_source.ranked_open_models(want=req.pool)
        except Exception as e:
            raise HTTPException(502, f"Artificial Analysis lookup failed: {e}")
        aa_meta = {k: ranked[k] for k in ("considered", "no_open_weights", "total_ranked")}
        pre = {m["repo"]: m for m in ranked["models"]}
        pool = [{"id": m["repo"], "params": None, "gated": False, "downloads": 0,
                 "pipeline_tag": "text-generation", "params_by_dtype": {}}
                for m in ranked["models"]]
    else:
        pre = {}
        try:
            pool = await hf.search(req.query, req.pipeline, req.sort, req.pool)
        except hf.HFError as e:
            raise HTTPException(status_code=e.status, detail=e.message)

    gpu = engine.resolve_gpu(req.hardware.model_dump())
    hw = req.hardware.model_dump()
    total_vram = gpu["vram_gb"] * max(hw["nodes"], 1) * max(hw["gpus_per_node"], 1)
    bpw = engine.WEIGHT_FORMATS.get(req.runtime.weight_format, {}).get("bpw", 16.0)

    # Drop test stubs and toy checkpoints - they always "fit" and would crowd
    # out real answers - then skip anything whose weights cannot physically
    # land, before paying for a config.json fetch.
    # Culls are reported, not silent: a model missing from the results should
    # never be indistinguishable from one that was never looked at.
    skipped = {"gated": [], "no_config": [], "gguf_only": [], "fetch_failed": [],
               "not_a_model": [], "too_large": [], "too_small": []}

    def worth_scanning(m):
        low = m["id"].lower()
        if any(w in low for w in ("tiny-", "-tiny", "test", "random", "debug", "dummy")):
            skipped["not_a_model"].append(m["id"])
            return False
        if m["params"] and m["params"] < req.min_params * 1e9:
            skipped["too_small"].append(m["id"])
            return False
        if m["params"] and m["params"] * bpw / 8 / 1e9 > total_vram * 1.6:
            skipped["too_large"].append(m["id"])
            return False
        return True

    shortlist = [m for m in pool if worth_scanning(m)][:req.pool]

    sem = asyncio.Semaphore(16)

    async def one(m):
        async with sem:
            try:
                return await hf.model_detail(m["id"], light=True)
            except hf.HFError as e:
                if e.kind == "gated":
                    skipped["gated"].append(m["id"])
                else:
                    skipped["fetch_failed"].append(f"{m['id']} ({e.kind})")
                return None
            except Exception as e:
                # Never drop a model without saying so: an unexplained gap in
                # the list is indistinguishable from a ranking bug.
                skipped["fetch_failed"].append(f"{m['id']} ({type(e).__name__})")
                return None

    fetched = [d for d in await asyncio.gather(*(one(m) for m in shortlist)) if d]
    details = []
    for d in fetched:
        sp = d.get("spec") or {}
        if not (sp.get("num_layers") and sp.get("hidden_size")):
            # A GGUF-only mirror has no config.json to size from. That is not a
            # broken repo - it is simply unusable by the engine selected here.
            if d.get("has_gguf") or d["id"].lower().endswith("gguf"):
                skipped["gguf_only"].append(d["id"])
            else:
                skipped["no_config"].append(d["id"])
            continue
        # Re-apply the size floor: search metadata often omits the parameter
        # count, so tiny models only become visible after the config is read.
        if (sp.get("params_total") or 0) < req.min_params * 1e9:
            skipped["too_small"].append(d["id"])
            continue
        details.append(d)

    if req.source == "aa":
        scores = {d["id"]: {
            "source": "artificial-analysis",
            "source_label": "Artificial Analysis Intelligence Index",
            "score": pre[d["id"]]["score"], "name": pre[d["id"]]["aa_name"],
            "benchmarks": pre[d["id"]]["benchmarks"], "scale": "0-100",
        } for d in details if d["id"] in pre}
        bench_available = True
    else:
        try:
            scores = await bench.lookup_many([(d["id"], d.get("base_model")) for d in details])
        except Exception:
            scores = {}
        bench_available = bool(scores.pop("__available__", False))

    out = []
    for d in details:
        d.setdefault("params_total", d["spec"].get("params_total"))
        try:
            r = engine.evaluate(d, hw, req.runtime.model_dump())
        except ValueError as e:
            skipped["fetch_failed"].append(f"{d['id']} (sizing: {e})")
            continue
        b = scores.get(d["id"])
        out.append({
            "id": d["id"], "params": d["spec"].get("params_total"),
            "active_params": d.get("active_params"),
            "pipeline_tag": d.get("pipeline_tag"), "downloads": d.get("downloads"),
            "gated": d.get("gated"), "family": (d.get("arch_support") or {}).get("family"),
            "verdict": r["verdict"], "headline": r["headline"],
            "parallel": r["parallel"], "weights_gib": r["memory"]["weights_gib"],
            "max_context": r["memory"]["max_context_single"],
            "max_concurrent": r["memory"]["max_concurrent_at_ctx"],
            "decode_tps": r["perf"].get("decode_tps_single"),
            "blockers": [f["title"] for f in r["findings"] if f["level"] == "blocker"],
            "bench": b,
        })

    def key(x):
        fits = 0 if x["verdict"] in ("yes", "tight") else 1
        score = x["bench"]["score"] if x.get("bench") and x["bench"].get("score") is not None else -1
        return (fits, -score, -(x["params"] or 0))

    out.sort(key=key)
    return {"results": out, "scanned": len(details), "pool": len(pool),
            "shortlisted": len(shortlist), "skipped": skipped,
            "format_label": engine.WEIGHT_FORMATS.get(req.runtime.weight_format, {}).get("label", ""),
            "bench_available": bench_available, "sources": bench.status(),
            "source": req.source, "aa": aa_meta}


# ---------------------------------------------------------------------------
# Static UI
# ---------------------------------------------------------------------------
app.mount("/static", StaticFiles(directory=STATIC), name="static")


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


@app.get("/favicon.ico")
def favicon():
    return JSONResponse({}, status_code=204)
