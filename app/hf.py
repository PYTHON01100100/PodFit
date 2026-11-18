"""HuggingFace Hub client: search, metadata, config.json parsing, GGUF files.

Everything is read-only and anonymous unless HF_TOKEN is set in the
environment. Responses are cached in SQLite with a TTL so repeated sizing runs
against the same model do not re-hit the Hub.
"""
from __future__ import annotations

import asyncio
import contextvars
import json
import os
import time

import httpx

from . import db

HF_API = "https://huggingface.co/api"
HF_RESOLVE = "https://huggingface.co/{repo}/resolve/{rev}/{path}"
TOKEN = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN") or ""
TIMEOUT = httpx.Timeout(20.0, connect=10.0)

# A visitor's own token (BYOK), set per-request from the X-HF-Token header.
# contextvars are task-local, so concurrent requests never see each other's key.
_token_override: contextvars.ContextVar[str] = contextvars.ContextVar("hf_token_override", default="")


def set_token(token: str) -> None:
    _token_override.set((token or "").strip())


def current_token() -> str:
    return _token_override.get() or TOKEN


def using_personal_token() -> bool:
    """True when this request supplied its own key rather than the server's."""
    return bool(_token_override.get())

SEARCH_TTL = 300          # 5 min - trending/download counts move
DETAIL_TTL = 6 * 3600     # 6 h  - configs are effectively immutable per revision

# Pipelines worth sizing on a GPU, grouped for the UI filter.
PIPELINES = [
    ("text-generation", "Text generation (LLM)"),
    ("image-text-to-text", "Vision-language (VLM)"),
    ("text2text-generation", "Seq2seq"),
    ("feature-extraction", "Embeddings"),
    ("sentence-similarity", "Sentence similarity"),
    ("text-classification", "Classification / reranking"),
    ("automatic-speech-recognition", "Speech recognition"),
    ("audio-text-to-text", "Audio-language"),
    ("any-to-any", "Omni / any-to-any"),
    ("fill-mask", "Masked LM"),
    ("token-classification", "Token classification"),
    ("question-answering", "Question answering"),
    ("summarization", "Summarization"),
    ("translation", "Translation"),
    ("visual-question-answering", "Visual QA"),
    ("image-to-text", "Image captioning"),
    ("text-to-image", "Text to image"),
    ("text-to-video", "Text to video"),
    ("text-to-speech", "Text to speech"),
    ("video-text-to-text", "Video-language"),
]


def _headers() -> dict:
    h = {"User-Agent": "podfit/1.0"}
    tok = current_token()
    if tok:
        h["Authorization"] = f"Bearer {tok}"
    return h


class HFError(Exception):
    def __init__(self, message: str, status: int = 502, kind: str = "error"):
        super().__init__(message)
        self.message, self.status, self.kind = message, status, kind


async def _get_json(client: httpx.AsyncClient, url: str, params=None):
    r = await client.get(url, params=params, headers=_headers(), timeout=TIMEOUT,
                         follow_redirects=True)
    if r.status_code in (401, 403):
        raise HFError(
            "This repository is gated or private. Accept its licence on the Hub "
            "and set HF_TOKEN in the container environment to size it.",
            status=r.status_code, kind="gated")
    if r.status_code == 404:
        raise HFError("Not found on the Hub.", status=404, kind="not_found")
    if r.status_code == 429:
        raise HFError("HuggingFace rate limit hit. Set HF_TOKEN to raise the limit.",
                      status=429, kind="rate_limit")
    r.raise_for_status()
    return r.json()


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------
async def search(q: str = "", pipeline: str = "", sort: str = "downloads",
                 limit: int = 30, author: str = "", library: str = "") -> list:
    key = f"search:{q}|{pipeline}|{sort}|{limit}|{author}|{library}"
    hit = db.cache_get(key, SEARCH_TTL)
    if hit is not None:
        return hit

    params = {
        "limit": min(limit, 100),
        "sort": sort if sort in ("downloads", "likes", "lastModified", "trendingScore", "createdAt") else "downloads",
        "direction": -1,
        "expand[]": ["downloads", "likes", "pipeline_tag", "library_name", "tags",
                     "lastModified", "gated", "safetensors", "createdAt", "private"],
    }
    if q:
        params["search"] = q
    if pipeline:
        params["pipeline_tag"] = pipeline
    if author:
        params["author"] = author
    if library:
        params["library"] = library

    async with httpx.AsyncClient() as client:
        rows = await _get_json(client, f"{HF_API}/models", params)

    out = []
    for m in rows:
        st = m.get("safetensors") or {}
        params_total = st.get("total")
        tags = m.get("tags") or []
        out.append({
            "id": m.get("id") or m.get("modelId"),
            "author": m.get("author") or (m.get("id", "").split("/")[0] if "/" in m.get("id", "") else ""),
            "pipeline_tag": m.get("pipeline_tag"),
            "library": m.get("library_name"),
            "downloads": m.get("downloads") or 0,
            "likes": m.get("likes") or 0,
            "gated": bool(m.get("gated")),
            "private": bool(m.get("private")),
            "lastModified": m.get("lastModified"),
            "createdAt": m.get("createdAt"),
            "params": params_total,
            "params_by_dtype": st.get("parameters") or {},
            "gguf": any(t == "gguf" for t in tags),
            "quant_tags": [t for t in tags if t in
                           ("4-bit", "8-bit", "awq", "gptq", "fp8", "mxfp4", "nvfp4",
                            "bitsandbytes", "compressed-tensors", "exl2", "gguf")],
            "tags": [t for t in tags if not t.startswith(("arxiv:", "base_model:", "dataset:", "doi:", "region:"))][:12],
        })
    # Skipped for a personal token: results could include repos that token
    # alone can see (private/gated), which must not leak to other visitors.
    if not using_personal_token():
        db.cache_put(key, out)
    return out


# ---------------------------------------------------------------------------
# Model detail
# ---------------------------------------------------------------------------
async def model_detail(repo: str, revision: str = "main", refresh: bool = False,
                       light: bool = False) -> dict:
    """light=True skips the recursive file-tree call (no GGUF sizes), halving
    the Hub round-trips - used when scanning many models at once."""
    key = f"detail:{repo}@{revision}"
    if not refresh:
        # A full record satisfies a light request, but not the other way round.
        hit = db.cache_get(key, DETAIL_TTL)
        if hit is None and light:
            hit = db.cache_get(key + ":light", DETAIL_TTL)
        if hit is not None:
            return hit

    async with httpx.AsyncClient() as client:
        meta = await _get_json(client, f"{HF_API}/models/{repo}", {
            "expand[]": ["downloads", "likes", "pipeline_tag", "library_name", "tags",
                         "lastModified", "gated", "safetensors", "cardData",
                         "usedStorage", "createdAt", "siblings", "config"],
        })

        files = []
        if light:
            files = [{"path": sib.get("rfilename"), "size": 0}
                     for sib in (meta.get("siblings") or []) if sib.get("rfilename")]
        else:
            try:
                tree = await _get_json(client, f"{HF_API}/models/{repo}/tree/{revision}",
                                       {"recursive": "true"})
                files = [{"path": f["path"], "size": f.get("size") or 0}
                         for f in tree if f.get("type") == "file"]
            except (HFError, httpx.HTTPError):
                files = [{"path": sib.get("rfilename"), "size": 0}
                         for sib in (meta.get("siblings") or []) if sib.get("rfilename")]

        paths = {f["path"] for f in files}
        cfg, gen_cfg = {}, {}
        if "config.json" in paths:
            try:
                cfg = await _get_json(
                    client, HF_RESOLVE.format(repo=repo, rev=revision, path="config.json"))
            except HFError as e:
                cfg = meta.get("config") or {}
                if not cfg.get("num_hidden_layers") and e.kind == "gated":
                    raise HFError(
                        f"{repo} is gated: its config.json cannot be read anonymously. "
                        "Accept the licence on the Hub, create a read token, and add it "
                        "under API keys in this app (or set HF_TOKEN in the podfit "
                        "container environment).",
                        status=403, kind="gated")
            except (httpx.HTTPError, json.JSONDecodeError):
                cfg = meta.get("config") or {}
        else:
            cfg = meta.get("config") or {}
        if "generation_config.json" in paths:
            try:
                gen_cfg = await _get_json(
                    client, HF_RESOLVE.format(repo=repo, rev=revision, path="generation_config.json"))
            except Exception:
                gen_cfg = {}

    st = meta.get("safetensors") or {}
    tags = meta.get("tags") or []
    ggufs = sorted([f for f in files if f["path"].lower().endswith(".gguf")],
                   key=lambda f: f["size"])
    weight_bytes = sum(f["size"] for f in files
                       if f["path"].endswith((".safetensors", ".bin", ".pt", ".pth")))

    spec = parse_config(cfg)
    if not spec["params_total"]:
        spec["params_total"] = st.get("total") or estimate_params(spec)
        spec["params_source"] = "safetensors index" if st.get("total") else "estimated from config.json"
    if st.get("total"):
        spec["params_total"] = st["total"]
        spec["params_source"] = "safetensors metadata"

    detail = {
        "id": meta.get("id") or repo,
        "revision": revision,
        "pipeline_tag": meta.get("pipeline_tag"),
        "library": meta.get("library_name"),
        "downloads": meta.get("downloads") or 0,
        "likes": meta.get("likes") or 0,
        "gated": bool(meta.get("gated")),
        "lastModified": meta.get("lastModified"),
        "createdAt": meta.get("createdAt"),
        "base_model": _base_model(meta, tags),
        "license": ((meta.get("cardData") or {}).get("license")
                    or next((t.split(":", 1)[1] for t in tags if t.startswith("license:")), None)),
        "tags": [t for t in tags if not t.startswith(("arxiv:", "doi:", "region:"))][:20],
        "params_by_dtype": st.get("parameters") or {},
        "repo_bytes": meta.get("usedStorage") or 0,
        "weight_file_bytes": weight_bytes,
        "gguf_files": [{"path": g["path"], "size": g["size"]} for g in ggufs][:60],
        "has_gguf": bool(ggufs),
        "num_files": len(files),
        "config": cfg,
        "generation_config": gen_cfg,
        "spec": spec,
        "light": light,
        "fetched_at": time.time(),
    }
    # Content fetched with a visitor's own token (gated repos, private search
    # hits) must not leak into the shared cache for the next visitor to read
    # back out. Only cache what the server's own (or no) token produced.
    if not using_personal_token():
        db.cache_put(key + (":light" if light else ""), detail)
    return detail


# ---------------------------------------------------------------------------
# config.json -> normalized model spec
# ---------------------------------------------------------------------------
# Families whose config declares `sliding_window` but omits the interleave
# pattern. Transformers supplies the default, so we must too - otherwise every
# layer looks windowed and the KV estimate collapses to a fraction of reality.
# Value is the repeat period: one full-attention layer per N layers.
SWA_PATTERN_DEFAULT = {
    "gemma2": 2,        # alternating local/global
    "gemma3": 6,        # 5 sliding : 1 full
    "gemma3_text": 6,
    "cohere2": 4,       # 3 sliding : 1 full
    "ministral": 1,     # every layer windowed - no full layers
}


def _base_model(meta: dict, tags: list) -> str | None:
    bm = (meta.get("cardData") or {}).get("base_model")
    if isinstance(bm, list):
        bm = bm[0] if bm else None
    if isinstance(bm, str) and "/" in bm:
        return bm
    for t in tags:
        if t.startswith("base_model:") and "/" in t:
            part = t.split(":")[-1]
            if "/" in part:
                return part
    return None


def _first(d: dict, *names, default=None):
    for n in names:
        if d.get(n) is not None:
            return d[n]
    return default


def parse_config(cfg: dict) -> dict:
    """Flatten a HF config.json into the fields the memory model needs.

    Multimodal configs nest the language model under text_config / llm_config /
    language_config; the KV-cache math must use that inner block, while the
    vision tower is accounted for separately as extra weights.
    """
    cfg = cfg or {}
    text = cfg
    nested_key = None
    for k in ("text_config", "llm_config", "language_config", "language_model_config", "decoder"):
        if isinstance(cfg.get(k), dict) and cfg[k].get("num_hidden_layers"):
            text, nested_key = cfg[k], k
            break

    hidden = _first(text, "hidden_size", "n_embd", "d_model", "model_dim", default=0) or 0
    layers = _first(text, "num_hidden_layers", "n_layer", "num_layers", "n_layers",
                    "decoder_layers", "num_decoder_layers", default=0) or 0
    heads = _first(text, "num_attention_heads", "n_head", "num_heads",
                   "decoder_attention_heads", "num_heads_decoder", default=0) or 0
    kvh_explicit = _first(text, "num_key_value_heads", "num_kv_heads", "n_head_kv", default=None)
    kv_heads = kvh_explicit or heads
    explicit_head_dim = _first(text, "head_dim", "attention_head_dim", "d_kv", default=None)
    head_dim = explicit_head_dim or (hidden // heads if heads else 0) or 0
    inter = _first(text, "intermediate_size", "ffn_dim", "n_inner", "d_ff",
                   default=(4 * hidden if hidden else 0)) or 0
    vocab = _first(text, "vocab_size", default=0) or 0
    max_ctx = _first(text, "max_position_embeddings", "n_positions", "max_seq_len",
                     "seq_length", "max_sequence_length", "max_target_positions",
                     default=0) or 0

    # RoPE scaling can legitimately extend usable context past max_position_embeddings.
    rope = text.get("rope_scaling") or cfg.get("rope_scaling") or {}
    rope_factor = rope.get("factor") if isinstance(rope, dict) else None
    rope_type = (rope.get("rope_type") or rope.get("type")) if isinstance(rope, dict) else None

    # Sliding window / interleaved attention.
    sw = _first(text, "sliding_window", "attention_window_size", default=None)
    if text.get("use_sliding_window") is False:
        sw = None
    layer_types = text.get("layer_types") or cfg.get("layer_types") or []
    sw_pattern = _first(text, "sliding_window_pattern", "layer_types_pattern", default=None)
    mtype = (cfg.get("model_type") or text.get("model_type") or "").lower()
    full_layers, swa_layers = layers, 0
    swa_assumed = False
    if layer_types and isinstance(layer_types, list):
        swa_layers = sum(1 for t in layer_types if "slid" in str(t).lower())
        full_layers = len(layer_types) - swa_layers
    elif sw and isinstance(sw_pattern, int) and sw_pattern > 1:
        swa_layers = layers - (layers // sw_pattern)
        full_layers = layers - swa_layers
    elif sw:
        default_pattern = SWA_PATTERN_DEFAULT.get(mtype)
        if default_pattern and default_pattern > 1:
            swa_layers = layers - (layers // default_pattern)
            full_layers = layers - swa_layers
            swa_assumed = True
        else:
            swa_layers, full_layers = layers, 0

    # Mixture of experts.
    n_exp = _first(text, "num_experts", "num_local_experts", "n_routed_experts",
                   "moe_num_experts", "num_experts_per_layer", default=0) or 0
    n_exp_tok = _first(text, "num_experts_per_tok", "moe_topk", "num_selected_experts",
                       "top_k", default=0) or 0
    moe_inter = _first(text, "moe_intermediate_size", "expert_intermediate_size",
                       "ffn_hidden_size", default=inter) or inter
    n_shared = _first(text, "n_shared_experts", "num_shared_experts",
                      "shared_expert_intermediate_size", default=0) or 0
    if isinstance(n_shared, int) and n_shared > 64:      # it was an intermediate size
        n_shared = 1
    moe_layer_freq = _first(text, "moe_layer_freq", "decoder_sparse_step", default=1) or 1
    first_dense = _first(text, "first_k_dense_replace", "num_dense_layers", default=0) or 0

    # Sparse attention (DeepSeek Sparse Attention / the V4 indexer). A model
    # that selects a subset of keys per query needs indexer kernels, which are
    # a much narrower set than ordinary attention.
    index_topk = _first(text, "index_topk", "indexer_topk", "sparse_topk", default=0) or 0
    index_heads = _first(text, "index_n_heads", "indexer_n_heads", default=0) or 0
    index_head_dim = _first(text, "index_head_dim", "indexer_head_dim", default=0) or 0
    is_sparse_attn = bool(index_topk or index_heads)

    # Multi-head Latent Attention (DeepSeek V2/V3, Kimi).
    kv_lora = _first(text, "kv_lora_rank", default=0) or 0
    qk_rope = _first(text, "qk_rope_head_dim", default=0) or 0
    qk_nope = _first(text, "qk_nope_head_dim", default=0) or 0
    v_head_dim = _first(text, "v_head_dim", default=0) or 0
    q_lora = _first(text, "q_lora_rank", default=0) or 0
    is_mla = bool(kv_lora and qk_rope)
    # Sparse/compressed caches (a per-query key subset, compress_ratios, an
    # indexer) are not derivable from these fields, so the KV figure for them
    # is an upper-bound approximation and must be labelled as one.
    kv_layout_unknown = bool(is_sparse_attn or (q_lora and qk_rope and not kv_lora))
    if is_mla and not explicit_head_dim and (qk_nope + qk_rope):
        # For MLA the query/key head is qk_nope + qk_rope wide; hidden/heads is
        # meaningless here and would misreport the kernel's head size.
        head_dim = qk_nope + qk_rope

    # Vision / audio towers.
    vis = cfg.get("vision_config") or cfg.get("vision_tower_config") or {}
    aud = cfg.get("audio_config") or cfg.get("audio_tower_config") or {}

    quant = cfg.get("quantization_config") or (text.get("quantization_config") or {})
    quant_block = quant.get("weight_block_size") or quant.get("block_size") or None
    quant_scale_fmt = (quant.get("scale_fmt") or "").lower()
    quant_method = (quant.get("quant_method") or quant.get("method") or "").lower()
    quant_bits = quant.get("bits") or quant.get("weight_bits") or quant.get("w_bit")
    if quant_method == "compressed-tensors":
        fmt = json.dumps(quant.get("config_groups") or {}).lower()
        if "float8" in fmt or "fp8" in fmt:
            quant_method, quant_bits = "fp8", 8
        elif "int4" in fmt or '"num_bits": 4' in fmt:
            quant_method, quant_bits = "int4", 4

    dtype = (_first(text, "torch_dtype", "dtype", default=None)
             or _first(cfg, "torch_dtype", "dtype", default="bfloat16"))

    enc_layers = _first(text, "encoder_layers", "num_encoder_layers", default=0) or 0
    enc_heads = _first(text, "encoder_attention_heads", default=0) or 0

    return {
        "encoder_layers": enc_layers, "encoder_heads": enc_heads,
        "is_encoder_decoder": bool(text.get("is_encoder_decoder") or cfg.get("is_encoder_decoder")
                                   or (enc_layers and layers)),
        "model_type": cfg.get("model_type") or text.get("model_type") or "",
        "architectures": cfg.get("architectures") or text.get("architectures") or [],
        "nested_text_config": nested_key,
        "hidden_size": hidden, "num_layers": layers,
        "num_attention_heads": heads, "num_kv_heads": kv_heads,
        "head_dim": head_dim, "intermediate_size": inter,
        "vocab_size": vocab, "max_position_embeddings": max_ctx,
        "rope_factor": rope_factor, "rope_type": rope_type,
        "sliding_window": sw, "swa_layers": swa_layers, "full_attn_layers": full_layers,
        "swa_pattern_assumed": swa_assumed,
        "swa_pattern": sw_pattern if isinstance(sw_pattern, int) else SWA_PATTERN_DEFAULT.get(mtype),
        "tie_word_embeddings": bool(_first(text, "tie_word_embeddings", default=False)),
        "num_experts": n_exp, "experts_per_token": n_exp_tok,
        "moe_intermediate_size": moe_inter, "shared_experts": n_shared,
        "moe_layer_freq": moe_layer_freq if isinstance(moe_layer_freq, int) else 1,
        "first_dense_layers": first_dense if isinstance(first_dense, int) else 0,
        "is_moe": bool(n_exp and n_exp > 1),
        "is_mla": is_mla, "kv_lora_rank": kv_lora, "q_lora_rank": q_lora,
        "kv_layout_unknown": kv_layout_unknown,
        "is_sparse_attn": is_sparse_attn, "index_topk": index_topk,
        "index_n_heads": index_heads, "index_head_dim": index_head_dim,
        "quant_block_size": quant_block, "quant_scale_fmt": quant_scale_fmt,
        "qk_rope_head_dim": qk_rope,
        "qk_nope_head_dim": qk_nope, "v_head_dim": v_head_dim,
        "has_vision": bool(vis), "has_audio": bool(aud),
        "vision_hidden": vis.get("hidden_size") or 0,
        "vision_layers": vis.get("depth") or vis.get("num_hidden_layers") or 0,
        "torch_dtype": dtype,
        "quant_method": quant_method, "quant_bits": quant_bits,
        "params_total": 0, "params_source": "",
    }


def estimate_params(s: dict) -> int:
    """Analytic parameter count from config, for repos with no safetensors index."""
    h, L, vocab = s["hidden_size"], s["num_layers"], s["vocab_size"]
    if not (h and L):
        return 0
    heads, kvh, hd = s["num_attention_heads"], s["num_kv_heads"], s["head_dim"]
    emb = vocab * h * (1 if s["tie_word_embeddings"] else 2)

    if s["is_mla"]:
        q_lora = s.get("q_lora_rank") or 0
        attn = (h * (s["qk_nope_head_dim"] + s["qk_rope_head_dim"]) * heads
                + h * (s["kv_lora_rank"] + s["qk_rope_head_dim"])
                + s["kv_lora_rank"] * heads * (s["qk_nope_head_dim"] + s["v_head_dim"])
                + heads * s["v_head_dim"] * h)
    else:
        attn = h * heads * hd + 2 * h * kvh * hd + heads * hd * h

    if s["is_moe"]:
        dense_l = s["first_dense_layers"]
        moe_l = max(L - dense_l, 0)
        per_expert = 3 * h * s["moe_intermediate_size"]
        mlp_total = (dense_l * 3 * h * s["intermediate_size"]
                     + moe_l * (s["num_experts"] * per_expert
                                + s["shared_experts"] * per_expert))
    else:
        mlp_total = L * 3 * h * s["intermediate_size"]

    vision = 0
    if s["has_vision"] and s["vision_hidden"] and s["vision_layers"]:
        vh, vl = s["vision_hidden"], s["vision_layers"]
        vision = vl * (4 * vh * vh + 8 * vh * vh)
    return int(L * attn + mlp_total + emb + vision + L * 4 * h)


def active_params(s: dict, total: int) -> int:
    """Params touched per token - drives compute, not memory."""
    if not s["is_moe"] or not s["num_experts"]:
        return total
    h = s["hidden_size"]
    per_expert = 3 * h * s["moe_intermediate_size"]
    L = s["num_layers"]
    dense_l = s["first_dense_layers"]
    moe_l = max(L - dense_l, 0)
    routed_all = moe_l * s["num_experts"] * per_expert
    routed_active = moe_l * max(s["experts_per_token"], 1) * per_expert
    return max(int(total - routed_all + routed_active), int(total * 0.03))


# ---------------------------------------------------------------------------
# Org / user avatars, used as model logos in the UI
# ---------------------------------------------------------------------------
AVATAR_TTL = 30 * 24 * 3600


async def avatars(names: list[str]) -> dict:
    """Map a HuggingFace org or user name to its avatar URL.

    Orgs and individual accounts live at different endpoints and the Hub
    returns 404 (or an error body) for the wrong one, so try both. Misses are
    cached too - most mirror accounts simply have no avatar.
    """
    out, todo = {}, []
    for n in names:
        hit = db.cache_get(f"avatar:{n}", AVATAR_TTL)
        if hit is not None:
            out[n] = hit or None
        else:
            todo.append(n)
    if not todo:
        return out

    async def one(client, name):
        for path in (f"{HF_API}/organizations/{name}/avatar",
                     f"{HF_API}/users/{name}/avatar"):
            try:
                r = await client.get(path, headers=_headers(),
                                     timeout=httpx.Timeout(10.0))
                if r.status_code == 200:
                    url = (r.json() or {}).get("avatarUrl")
                    if url:
                        return name, url
            except (httpx.HTTPError, ValueError):
                continue
        return name, None

    async with httpx.AsyncClient() as client:
        results = await asyncio.gather(*(one(client, n) for n in todo))
    for name, url in results:
        out[name] = url
        db.cache_put(f"avatar:{name}", url or "")
    return out
