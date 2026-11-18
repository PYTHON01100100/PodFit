"""Artificial Analysis as a model *source*, not just a score lookup.

The Hub's trending and download lists rank by attention, which is a poor proxy
for capability. This module inverts the flow: take Artificial Analysis' models
ordered by intelligence, keep the ones with real open weights, and hand those
to the sizing engine.

AA publishes no HuggingFace id and no open-weights flag, so a model is resolved
by searching the Hub for its name and accepting a repo only when BOTH hold:

  * the repo's org is one the model's creator actually publishes under, and
  * the repo ships weights (a safetensors parameter count).

Without the org check, proprietary models resolve to squatter repos - "GPT-6
Astra" happily matches `gptforfree/GPT-6-Astra`, which contains no weights at
all. Anything that fails these tests is reported as having no open weights
rather than guessed at.
"""
from __future__ import annotations

import asyncio

import httpx

from . import bench, db, hf

RESOLVE_TTL = 14 * 24 * 3600
RESOLVE_VERSION = 2

# HuggingFace orgs each AA creator actually publishes weights under. An empty
# list means "this creator publishes no open weights" (API-only labs).
CREATOR_ORGS: dict[str, list[str]] = {
    "openai": ["openai"], "meta": ["meta-llama", "facebook"], "google": ["google"],
    "alibaba": ["Qwen"], "qwen": ["Qwen"], "deepseek": ["deepseek-ai"],
    "zai": ["zai-org", "THUDM"], "moonshot": ["moonshotai"], "kimi": ["moonshotai"],
    "mistral": ["mistralai"], "microsoft": ["microsoft"], "nvidia": ["nvidia"],
    "ibm": ["ibm-granite"], "allenai": ["allenai"], "ai2": ["allenai"],
    "tencent": ["tencent"], "baidu": ["baidu"], "xai": ["xai-org"],
    "cohere": ["CohereLabs", "CohereForAI"], "01ai": ["01-ai"], "upstage": ["upstage"],
    "liquid": ["LiquidAI"], "nous": ["NousResearch"], "inclusionai": ["inclusionAI"],
    "minimax": ["MiniMaxAI"], "stepfun": ["stepfun-ai"], "tii": ["tiiuae"],
    "lg": ["LGAI-EXAONE"], "lgai": ["LGAI-EXAONE"], "huggingface": ["HuggingFaceTB"],
    "bytedance": ["ByteDance-Seed"], "reka": ["RekaAI"], "ai21": ["ai21labs"],
    "databricks": ["databricks"], "snowflake": ["Snowflake"], "apple": ["apple"],
    "arcee": ["arcee-ai"], "opengvlab": ["OpenGVLab"], "internlm": ["internlm"],
    "openbmb": ["openbmb"], "rednote": ["rednote-hilab"], "swiss": ["swiss-ai"],
    # API-only: no open weights exist, so never resolve them.
    "anthropic": [], "amazon": [], "perplexity": [], "cursor": [], "inception": [],
}

# Quantized / community re-uploads: never the canonical repo for a model.
MIRROR_MARK = ("gguf", "awq", "gptq", "exl2", "exl3", "mlx", "nvfp4", "mxfp4",
               "fp8", "bnb", "4bit", "8bit", "int4", "int8", "w4a16", "w8a8",
               "abliterated", "uncensored", "distill")


def _is_mirror(seg: str) -> bool:
    low = seg.lower()
    return any(low.endswith("-" + m) or low.endswith("." + m) or ("-" + m + "-") in low
               for m in MIRROR_MARK)


async def _resolve_one(client: httpx.AsyncClient, rec: dict) -> str | None:
    name = rec.get("name") or ""
    creator = rec.get("creator_slug") or ""
    key = f"aa2hf/v{RESOLVE_VERSION}:{name}"
    cached = db.cache_get(key, RESOLVE_TTL)
    if cached is not None:
        return cached or None

    allowed = CREATOR_ORGS.get(creator)
    if allowed is not None and not allowed:          # known API-only lab
        db.cache_put(key, "")
        return None

    want = bench._tokens(name)
    if not want[1]:
        db.cache_put(key, "")
        return None

    query = name.split("(")[0].strip()
    try:
        r = await client.get(f"{hf.HF_API}/models", headers=hf._headers(),
                             timeout=httpx.Timeout(25.0),
                             params={"search": query, "limit": 20, "sort": "downloads",
                                     "direction": -1,
                                     "expand[]": ["downloads", "safetensors"]})
        if r.status_code != 200:
            return None                              # transient: do not cache a miss
        results = r.json()
    except (httpx.HTTPError, ValueError):
        return None

    best = None
    for m in results:
        rid = m.get("id") or ""
        if "/" not in rid:
            continue
        org, seg = rid.split("/", 1)
        if _is_mirror(seg):
            continue
        if allowed is None:
            if org.lower() != creator.lower():
                continue
        elif org.lower() not in [a.lower() for a in allowed]:
            continue
        if not bench._match(want, bench._tokens(seg)):
            continue
        if not m.get("safetensors"):                 # must actually ship weights
            continue
        dl = m.get("downloads") or 0
        if best is None or dl > best[0]:
            best = (dl, rid)

    db.cache_put(key, best[1] if best else "")
    return best[1] if best else None


async def ranked_open_models(want: int = 30, scan_cap: int = 240) -> dict:
    """AA models by intelligence, filtered to ones with open weights on the Hub.

    Walks the ranked list until `want` distinct repos resolve. AA lists a model
    once per reasoning-effort setting, so entries are collapsed onto their repo
    and keep the best score.
    """
    rows = [r for r in await bench.aa_all() if r.get("score") is not None]
    rows.sort(key=lambda x: -x["score"])

    by_repo: dict[str, dict] = {}
    considered = 0
    no_weights = 0
    sem = asyncio.Semaphore(8)

    async with httpx.AsyncClient() as client:
        async def one(rec):
            async with sem:
                return rec, await _resolve_one(client, rec)

        for i in range(0, min(len(rows), scan_cap), 24):
            batch = rows[i:i + 24]
            considered += len(batch)
            for rec, repo in await asyncio.gather(*(one(r) for r in batch)):
                if not repo:
                    no_weights += 1
                    continue
                prev = by_repo.get(repo)
                if prev is None or rec["score"] > prev["score"]:
                    by_repo[repo] = {
                        "repo": repo, "score": rec["score"], "aa_name": rec["name"],
                        "creator": rec.get("creator") or "",
                        "benchmarks": rec.get("benchmarks") or {},
                        "release_date": rec.get("release_date"),
                    }
            if len(by_repo) >= want:
                break

    out = sorted(by_repo.values(), key=lambda x: -x["score"])[:want]
    return {"models": out, "considered": considered, "no_open_weights": no_weights,
            "total_ranked": len(rows)}


async def search(q: str, limit: int = 25) -> dict:
    """Search AA's ranked models by name, resolving each to a Hub repo."""
    rows = [r for r in await bench.aa_all() if r.get("score") is not None]
    ql = (q or "").strip().lower()
    if ql:
        rows = [r for r in rows if ql in (r["name"] or "").lower()
                or ql in (r.get("creator") or "").lower()]
    rows.sort(key=lambda x: -x["score"])

    seen, picked = set(), []
    for r in rows:
        base = (r["name"] or "").split("(")[0].strip().lower()
        if base in seen:
            continue
        seen.add(base)
        picked.append(r)
        if len(picked) >= limit:
            break

    sem = asyncio.Semaphore(8)
    async with httpx.AsyncClient() as client:
        async def one(rec):
            async with sem:
                return rec, await _resolve_one(client, rec)
        pairs = await asyncio.gather(*(one(r) for r in picked))

    return {"results": [{
        "aa_name": rec["name"], "score": rec["score"],
        "creator": rec.get("creator") or "", "repo": repo,
        "open_weights": bool(repo),
        "benchmarks": rec.get("benchmarks") or {},
        "release_date": rec.get("release_date"),
    } for rec, repo in pairs]}
