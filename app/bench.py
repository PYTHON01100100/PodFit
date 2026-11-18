"""Model quality ("how smart is it") data from public benchmark sources.

Two providers, in preference order:

1. Artificial Analysis - their Intelligence Index, the aggregator most people
   quote. Requires AA_API_KEY in the environment (free tier keys are issued at
   artificialanalysis.ai); without it this provider is simply skipped.
2. HuggingFace Open LLM Leaderboard v2 - free and keyless through the HF
   datasets-server. Real, reproducible numbers (IFEval, BBH, MATH L5, GPQA,
   MUSR, MMLU-PRO), but the leaderboard was archived in 2025, so models
   released after that are absent.

Nothing here is invented: a model with no entry in either source reports no
score rather than a guess.
"""
from __future__ import annotations

import asyncio
import contextvars
import os
import re

import httpx

from . import db

AA_KEY = os.environ.get("AA_API_KEY") or os.environ.get("ARTIFICIAL_ANALYSIS_API_KEY") or ""

# A visitor's own key (BYOK), set per-request from the X-AA-Key header.
_key_override: contextvars.ContextVar[str] = contextvars.ContextVar("aa_key_override", default="")


def set_key(key: str) -> None:
    _key_override.set((key or "").strip())


def current_key() -> str:
    return _key_override.get() or AA_KEY
AA_URL = "https://artificialanalysis.ai/api/v2/data/llms/models"
DS = "https://datasets-server.huggingface.co/filter"
OLLB_DATASET = "open-llm-leaderboard/contents"
OLLB_NOTE = ("HuggingFace Open LLM Leaderboard v2 average of IFEval, BBH, MATH Lvl 5, "
             "GPQA, MUSR and MMLU-PRO. The leaderboard was archived in 2025, so models "
             "released after that have no entry.")
AA_NOTE = ("Artificial Analysis Intelligence Index - their aggregate across MMLU-Pro, "
           "GPQA Diamond, LiveCodeBench, AIME and others, on a 0-100 scale.")

TTL = 7 * 24 * 3600
# Bump when the name-matching rules change: cached verdicts from an older
# matcher would otherwise survive for a week and keep serving wrong scores.
MATCH_VERSION = 4
BATCH = 20
_aa_lock = asyncio.Lock()

# Suffixes community re-uploads bolt onto a base repo name. Stripping these
# lets a GGUF/AWQ mirror inherit the base model's benchmark score.
QUANT_SUFFIX = re.compile(
    r"[-_.]?(gguf|awq|gptq|exl2|exl3|mlx|bnb|nf4|int4|int8|w4a16|w8a8|w4a8|fp8|fp4|"
    r"mxfp4|nvfp4|4bit|8bit|4-bit|8-bit|quantized|compressed[-_]?tensors|"
    r"unquantized|bf16|fp16|hf|abliterated|uncensored)"
    r"([-_.]?(v?\d+(\.\d+)?))?$", re.I)
MIRROR_ORGS = {"unsloth", "bartowski", "thebloke", "lmstudio-community", "mradermacher",
               "redhatai", "neuralmagic", "casperhansen", "modelscope", "second-state",
               "nvidia", "mlx-community", "ggml-org", "tensorblock", "quantfactory"}

# Which upstream org publishes a given model family, so a mirror's copy can be
# resolved back to the original repo.
FAMILY_ORG = [
    ("qwen", "Qwen"), ("llama", "meta-llama"), ("phi", "microsoft"),
    ("gemma", "google"), ("mistral", "mistralai"), ("mixtral", "mistralai"),
    ("magistral", "mistralai"), ("devstral", "mistralai"),
    ("deepseek", "deepseek-ai"), ("gpt-oss", "openai"), ("granite", "ibm-granite"),
    ("olmo", "allenai"), ("yi-", "01-ai"), ("glm", "zai-org"), ("nemotron", "nvidia"),
    ("smol", "HuggingFaceTB"), ("falcon", "tiiuae"), ("exaone", "LGAI-EXAONE"),
    ("command", "CohereLabs"), ("aya", "CohereLabs"), ("internlm", "internlm"),
    ("minicpm", "openbmb"), ("kimi", "moonshotai"), ("seed", "ByteDance-Seed"),
    ("hunyuan", "tencent"), ("ernie", "baidu"), ("apertus", "swiss-ai"),
]


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", (s or "").lower())


def candidate_repos(repo: str, base_model: str | None = None) -> list[str]:
    """Repo ids to try, best first: the repo itself, then its declared base,
    then the repo with quantization suffixes peeled off."""
    out, seen = [], set()

    def add(r):
        if r and r not in seen:
            seen.add(r); out.append(r)

    add(repo)
    if base_model:
        add(base_model if isinstance(base_model, str) else
            (base_model[0] if isinstance(base_model, list) and base_model else None))
    if "/" in repo:
        org, name = repo.split("/", 1)
        stripped = name
        for _ in range(3):
            nxt = QUANT_SUFFIX.sub("", stripped).rstrip("-_.")
            if nxt == stripped:
                break
            stripped = nxt
        if stripped != name:
            add(f"{org}/{stripped}")
        if org.lower() in MIRROR_ORGS or stripped != name:
            # A mirror keeps the original repo name; map it back to the org
            # that actually publishes that family.
            low = stripped.lower()
            for token, real in FAMILY_ORG:
                if low.startswith(token) or token in low:
                    add(f"{real}/{stripped}")
                    break
    return out[:6]


# ---------------------------------------------------------------------------
# Open LLM Leaderboard v2
# ---------------------------------------------------------------------------
def _ollb_record(row: dict) -> dict:
    g = lambda k: row.get(k) if isinstance(row.get(k), (int, float)) else None
    return {
        "source": "open-llm-leaderboard",
        "source_label": "Open LLM Leaderboard v2",
        "repo": row.get("fullname"),
        "score": g("Average ⬆️"),
        "scale": "0-100 (typical range 5-45)",
        "precision": row.get("Precision"),
        "params_b": g("#Params (B)"),
        "benchmarks": {k: v for k, v in {
            "MMLU-Pro": g("MMLU-PRO"), "GPQA": g("GPQA"), "IFEval": g("IFEval"),
            "MATH Lvl 5": g("MATH Lvl 5"), "BBH": g("BBH"), "MUSR": g("MUSR"),
        }.items() if v is not None},
        "note": OLLB_NOTE,
    }


class BenchUnavailable(Exception):
    """The upstream source failed - distinct from "this model has no score"."""


async def _ollb_fetch(names: list[str]) -> dict:
    """One filtered query for up to BATCH exact repo ids."""
    if not names:
        return {}
    where = " OR ".join('"fullname"=\'{}\''.format(n.replace("'", "''")) for n in names)
    params = {"dataset": OLLB_DATASET, "config": "default", "split": "train",
              "where": where, "limit": str(min(len(names) * 3, 100))}
    # datasets-server answers 500 while it warms an index and throttles bursts,
    # so a couple of backed-off retries turn most misses into hits.
    last = "unknown"
    rows = None
    async with httpx.AsyncClient() as client:
        for attempt in range(3):
            try:
                r = await client.get(DS, params=params, timeout=httpx.Timeout(30.0))
                if r.status_code == 200:
                    rows = (r.json() or {}).get("rows") or []
                    break
                last = f"HTTP {r.status_code}"
                if r.status_code in (400, 404, 422):
                    raise BenchUnavailable(last)
            except (httpx.HTTPError, ValueError) as e:
                last = str(e)
            await asyncio.sleep(1.5 * (2 ** attempt))
    if rows is None:
        raise BenchUnavailable(f"datasets-server unavailable ({last})")
    best: dict[str, dict] = {}
    for item in rows:
        rec = _ollb_record(item.get("row") or {})
        key = (rec["repo"] or "").lower()
        # A repo can appear once per precision; keep the best-scoring run.
        if rec["score"] is not None and (key not in best or rec["score"] > best[key]["score"]):
            best[key] = rec
    return best


# ---------------------------------------------------------------------------
# Artificial Analysis
# ---------------------------------------------------------------------------
def _aa_number(d: dict, *names):
    for n in names:
        v = d.get(n)
        if isinstance(v, (int, float)):
            return float(v)
    return None


def _aa_record(m: dict) -> dict | None:
    """AA's schema has moved between versions, so pull fields defensively."""
    evals = m.get("evaluations") or m.get("evaluation") or {}
    if not isinstance(evals, dict):
        evals = {}
    score = (_aa_number(m, "intelligence_index", "artificial_analysis_intelligence_index")
             or _aa_number(evals, "artificial_analysis_intelligence_index",
                           "intelligence_index", "artificial_analysis_index"))
    if score is None:
        return None
    marks = {}
    for key, label in (("mmlu_pro", "MMLU-Pro"), ("gpqa", "GPQA Diamond"),
                       ("gpqa_diamond", "GPQA Diamond"), ("aime", "AIME"),
                       ("livecodebench", "LiveCodeBench"), ("hle", "Humanity's Last Exam"),
                       ("math_500", "MATH-500"), ("scicode", "SciCode")):
        v = _aa_number(evals, key, "artificial_analysis_" + key)
        if v is not None:
            marks[label] = v * 100 if v <= 1 else v
    creator = m.get("model_creator") or {}
    return {
        "source": "artificial-analysis",
        "source_label": "Artificial Analysis Intelligence Index",
        "repo": None,
        "name": m.get("name") or m.get("slug"),
        "slug": m.get("slug"),
        "creator": creator.get("name") or "",
        "creator_slug": (creator.get("slug") or "").lower(),
        "release_date": m.get("release_date"),
        "score": score,
        "scale": "0-100",
        "benchmarks": marks,
        "note": AA_NOTE,
    }


async def aa_all() -> list[dict]:
    """All AA models, cached. Empty list when no key is configured.

    The listing itself is the same public dataset for any valid key, so the
    cache is shared across visitors regardless of whose key populated it."""
    key = current_key()
    if not key:
        return []
    ckey = f"aa:all/v{MATCH_VERSION}"
    hit = db.cache_get(ckey, TTL)
    if hit is not None:
        return hit
    async with _aa_lock:
        hit = db.cache_get(ckey, TTL)
        if hit is not None:
            return hit
        try:
            async with httpx.AsyncClient() as client:
                r = await client.get(AA_URL, headers={"x-api-key": key},
                                     timeout=httpx.Timeout(30.0))
                if r.status_code != 200:
                    return []
                payload = r.json()
        except httpx.HTTPError:
            return []
        rows = payload.get("data") if isinstance(payload, dict) else payload
        out = []
        for m in (rows or []):
            rec = _aa_record(m) if isinstance(m, dict) else None
            if rec:
                out.append(rec)
        db.cache_put(ckey, out)
        return out


# Artificial Analysis names models for humans ("Llama 3.3 Instruct 70B"), not by
# repo id ("meta-llama/Llama-3.3-70B-Instruct"), and appends a reasoning-effort
# qualifier in parentheses. Matching therefore compares token sets: the size
# token must agree exactly, the family words must overlap, and any version
# number must be consistent.
SIZE_TOKEN = re.compile(r"^a?\d+(?:\.\d+)?[bm]$")
VERSION_TOKEN = re.compile(r"^\d+(?:\.\d+)*$")
MODIFIERS = {
    "instruct", "it", "chat", "base", "hf", "preview", "experimental", "exp",
    "reasoning", "nonreasoning", "non", "thinking", "high", "low", "medium",
    "max", "min", "xhigh", "effort", "default", "fallback", "adaptive",
    "vision", "omni", "coder", "code", "math", "distill", "turbo", "mini",
    "latest", "release", "v", "model", "ai", "the",
}


def _tokens(name: str) -> tuple[set, set, set]:
    """(size tokens, family words, version numbers) for a model name."""
    name = re.sub(r"\([^)]*\)", " ", name or "")        # drop "(high)", "(Reasoning)"
    parts = [p for p in re.split(r"[^A-Za-z0-9.]+", name.lower()) if p]
    sizes, family, versions = set(), set(), set()
    for p in parts:
        if SIZE_TOKEN.match(p):
            sizes.add(p)
        elif VERSION_TOKEN.match(p):
            versions.add(p.rstrip("."))
        elif p in MODIFIERS:
            continue
        else:
            # Split glued forms like "gpt-oss" -> already split, but also
            # "qwen3" -> ("qwen", "3") so a name written "Qwen 3" still matches.
            m = re.match(r"^([a-z]+)(\d+(?:\.\d+)?)$", p)
            if m:
                if m.group(1) not in MODIFIERS:
                    family.add(m.group(1))
                versions.add(m.group(2))
            else:
                family.add(p)
    return sizes, family, versions


def _aa_index(rows: list) -> list:
    idx = []
    for rec in rows:
        name = rec.get("name") or rec.get("repo") or ""
        if not name:
            continue
        idx.append((_tokens(name), rec))
    return idx


def _match(want: tuple, have: tuple) -> bool:
    w_size, w_fam, w_ver = want
    h_size, h_fam, h_ver = have
    # Family words must match exactly, not merely overlap: "Qwen3 8B" and
    # "Qwen3 VL 8B" are different models, and crediting one with the other's
    # score is worse than reporting no score at all.
    if w_fam != h_fam:
        return False
    if w_size or h_size:
        if w_size != h_size:
            return False
    if w_ver and h_ver and not (w_ver & h_ver):
        return False
    return True


async def aa_lookup(repo: str, name_hint: str = "") -> dict | None:
    rows = await aa_all()
    if not rows:
        return None
    want = _tokens(name_hint or repo.split("/")[-1])
    if not want[1]:
        return None
    best = None
    for have, rec in _aa_index(rows):
        if _match(want, have) and (best is None or (rec.get("score") or 0) > (best.get("score") or 0)):
            best = rec
    # Several AA entries can describe one model at different reasoning efforts;
    # the highest is the model's best measured capability.
    return best


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
async def lookup(repo: str, base_model: str | None = None) -> dict | None:
    """Best available score for one repo, trying its base model as a fallback."""
    key = f"bench/v{MATCH_VERSION}:{repo}|{base_model or ''}"
    hit = db.cache_get(key, TTL)
    if hit is not None:
        return hit or None

    cands = candidate_repos(repo, base_model)
    rec, via = None, None
    for c in cands[:3]:
        rec = await aa_lookup(c)
        if rec:
            # A score found under a stripped/base name belongs to that model,
            # not to this quantized or fine-tuned derivative - say so.
            via = c if c.lower() != repo.lower() else None
            break
    answered = True
    if rec is None:
        try:
            found = await _ollb_fetch(cands)
        except BenchUnavailable:
            found, answered = {}, False
        for c in cands:
            if c.lower() in found:
                rec = found[c.lower()]
                break
    if rec and via:
        rec = dict(rec, inherited_from=via)
    elif rec and rec.get("repo") and rec["repo"].lower() != repo.lower():
        rec = dict(rec, inherited_from=rec["repo"])
    if rec or answered:
        db.cache_put(key, rec or {})
    return rec


async def lookup_many(pairs: list[tuple[str, str | None]]) -> dict:
    """Scores for many repos at once. pairs = [(repo, base_model), ...]."""
    out: dict[str, dict | None] = {}
    todo: list[tuple[str, list[str]]] = []
    for repo, base in pairs:
        cached = db.cache_get(f"bench/v{MATCH_VERSION}:{repo}|{base or ''}", TTL)
        if cached is not None:
            out[repo] = cached or None
        else:
            todo.append((repo, candidate_repos(repo, base)))

    aa_rows = await aa_all()
    if aa_rows:
        still = []
        for repo, cands in todo:
            rec, via = None, None
            for c in cands[:3]:
                rec = await aa_lookup(c)
                if rec:
                    via = c if c.lower() != repo.lower() else None
                    break
            if rec:
                if via:
                    rec = dict(rec, inherited_from=via)
                out[repo] = rec
                db.cache_put(f"bench/v{MATCH_VERSION}:{repo}|", rec)
            else:
                still.append((repo, cands))
        todo = still

    # One datasets-server query per batch of candidate names.
    names, index = [], []
    for repo, cands in todo:
        for c in cands:
            names.append(c); index.append(repo)
    found: dict[str, dict] = {}
    answered = True
    for i in range(0, len(names), BATCH):
        if i:
            await asyncio.sleep(0.6)      # datasets-server throttles bursts
        try:
            found.update(await _ollb_fetch(names[i:i + BATCH]))
        except BenchUnavailable:
            answered = False
    out["__available__"] = answered or bool(found)   # consumed by /api/recommend
    for repo, cands in todo:
        rec = next((found[c.lower()] for c in cands if c.lower() in found), None)
        if rec and rec.get("repo", "").lower() != repo.lower():
            rec = dict(rec, inherited_from=rec["repo"])
        out[repo] = rec
        if rec or answered:
            db.cache_put(f"bench/v{MATCH_VERSION}:{repo}|", rec or {})
    return out


def status() -> dict:
    return {
        "artificial_analysis": {"configured": bool(current_key()), "note": AA_NOTE},
        "open_llm_leaderboard": {"configured": True, "note": OLLB_NOTE},
    }
