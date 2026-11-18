"""Feasibility engine: memory model, compatibility rules, throughput estimates.

All memory is computed in bytes and reported in GiB. The vLLM/SGLang path
models a per-GPU budget of `gpu_memory_utilization * total VRAM`, subtracts
weights, framework overhead, activation peak and CUDA graphs, and gives what is
left to the KV cache - which is what those engines actually do at startup.
"""
from __future__ import annotations

import math

from . import archsupport, attention, gpuarch
from .catalog import GPU_BY_ID, KV_DTYPES, WEIGHT_FORMATS, ENGINES

GIB = 1024 ** 3


def _f(level, title, detail, tag=""):
    return {"level": level, "title": title, "detail": detail, "tag": tag}


# ---------------------------------------------------------------------------
# Hardware resolution
# ---------------------------------------------------------------------------
def resolve_gpu(hw: dict) -> dict:
    if hw.get("custom_gpu"):
        c = hw["custom_gpu"]
        return {
            "id": "custom", "name": c.get("name") or "Custom GPU",
            "vendor": c.get("vendor", "nvidia"), "arch": c.get("arch", "custom"),
            "vram_gb": float(c.get("vram_gb") or 0),
            "bandwidth": float(c.get("bandwidth") or 0),
            "fp16": float(c.get("fp16") or 0), "fp8": float(c.get("fp8") or 0),
            "fp4": float(c.get("fp4") or 0), "cc": float(c.get("cc") or 8.0),
            "nvlink": float(c.get("nvlink") or 0), "tdp": float(c.get("tdp") or 0),
            "form": "Custom", "note": "",
        }
    g = GPU_BY_ID.get(hw.get("gpu_id"))
    if not g:
        raise ValueError(f"unknown gpu_id {hw.get('gpu_id')!r}")
    g = dict(g)
    if hw.get("vram_override_gb"):
        g["vram_gb"] = float(hw["vram_override_gb"])
    return g


# ---------------------------------------------------------------------------
# Weight bytes
# ---------------------------------------------------------------------------
def weight_bytes(spec: dict, params: int, fmt_key: str, gguf_size: int = 0):
    """Returns (bytes, bits_per_weight, explanation)."""
    fmt = WEIGHT_FORMATS[fmt_key]
    if gguf_size:
        bpw = gguf_size * 8 / params if params else fmt["bpw"]
        return gguf_size, bpw, "measured GGUF file size from the Hub"
    if fmt["kind"] == "gguf":
        return int(params * fmt["bpw"] / 8), fmt["bpw"], "GGUF bits-per-weight table"
    # Embeddings and the LM head are usually left in higher precision by
    # AWQ/GPTQ/FP8 quantizers, so account for them separately.
    if fmt["kind"] in ("int4", "int8", "fp8", "fp4") and spec.get("vocab_size") and spec.get("hidden_size"):
        emb = spec["vocab_size"] * spec["hidden_size"] * (1 if spec.get("tie_word_embeddings") else 2)
        emb = min(emb, int(params * 0.5))
        body = max(params - emb, 0)
        b = body * fmt["bpw"] / 8 + emb * 2
        return int(b), b * 8 / params if params else fmt["bpw"], \
            "quantized body, embeddings/LM head kept at 16-bit"
    return int(params * fmt["bpw"] / 8), fmt["bpw"], "uniform bits-per-weight"


# ---------------------------------------------------------------------------
# KV cache
# ---------------------------------------------------------------------------
def kv_per_token_per_layer(spec: dict, kv_bytes_per_elem: float, tp: int) -> float:
    """Bytes of KV cache one token consumes in one layer, on one GPU."""
    if spec.get("no_kv_cache"):
        return 0.0
    if spec.get("is_mla"):
        # MLA stores one compressed latent (+ rope part) per token per layer, and
        # vLLM replicates that latent across tensor-parallel ranks rather than
        # sharding it - so TP does not reduce it.
        return (spec["kv_lora_rank"] + spec["qk_rope_head_dim"]) * kv_bytes_per_elem
    kvh = spec.get("num_kv_heads") or spec.get("num_attention_heads") or 1
    kvh_per_gpu = max(1, math.ceil(kvh / tp))
    return 2 * kvh_per_gpu * (spec.get("head_dim") or 0) * kv_bytes_per_elem


def kv_bytes_for(spec: dict, ctx: int, seqs: int, kv_elem: float, tp: int, pp: int) -> float:
    per_layer_token = kv_per_token_per_layer(spec, kv_elem, tp)
    layers = spec.get("num_layers") or 0
    full = spec.get("full_attn_layers") if spec.get("full_attn_layers") is not None else layers
    swa = spec.get("swa_layers") or 0
    if full + swa != layers:
        full, swa = layers, 0
    sw = spec.get("sliding_window") or 0
    swa_ctx = min(ctx, sw) if sw else ctx
    layers_per_stage_full = full / max(pp, 1)
    layers_per_stage_swa = swa / max(pp, 1)
    return per_layer_token * (layers_per_stage_full * ctx + layers_per_stage_swa * swa_ctx) * seqs


def max_ctx_for_budget(spec, budget, seqs, kv_elem, tp, pp, cap):
    if budget <= 0:
        return 0
    lo, hi = 0, cap
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if kv_bytes_for(spec, mid, seqs, kv_elem, tp, pp) <= budget:
            lo = mid
        else:
            hi = mid - 1
    return lo


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def evaluate(model: dict, hw: dict, rt: dict, _suggest: bool = True) -> dict:
    spec = dict(model.get("spec") or {})
    params = int(model.get("params_total") or spec.get("params_total") or 0)
    findings: list[dict] = []

    gpu = resolve_gpu(hw)
    fam = gpuarch.family_for(gpu)
    gpu = dict(gpu, family=fam["name"] if fam else gpu.get("arch"),
               family_key=fam["key"] if fam else None,
               sm=fam["sm"] if fam else (f"sm_{int(gpu['cc']*10)}" if gpu.get("cc", 0) > 0 else ""))
    nodes = max(1, int(hw.get("nodes") or 1))
    gpus_per_node = max(1, int(hw.get("gpus_per_node") or 1))
    total_gpus = nodes * gpus_per_node
    host_ram = float(hw.get("host_ram_gb") or 0)
    os_name = (hw.get("os") or "linux").lower()
    interconnect = hw.get("interconnect") or ("nvlink" if gpu.get("nvlink") else "pcie")
    driver = str(hw.get("driver_version") or "")

    eng = rt.get("engine") or "vllm"
    eng_meta = ENGINES.get(eng, ENGINES["vllm"])
    fmt_key = rt.get("weight_format") or "bf16"
    if fmt_key not in WEIGHT_FORMATS:
        fmt_key = "bf16"
    kv_key = rt.get("kv_dtype") or "auto"
    ctx = max(1, int(rt.get("context_len") or 4096))
    seqs = max(1, int(rt.get("concurrency") or 1))
    util = float(rt.get("gpu_mem_util") or 0.90)
    eager = bool(rt.get("enforce_eager"))
    batched_tokens = int(rt.get("max_batched_tokens") or 8192)
    gguf_size = int(rt.get("gguf_size") or 0)
    images = int(rt.get("images_per_request") or 0)

    is_local = eng in ("llamacpp", "ollama")
    arch = archsupport.lookup(spec.get("model_type"), spec.get("architectures"))

    # ---- parallelism ------------------------------------------------------
    tp = int(rt.get("tp") or 0)
    pp = int(rt.get("pp") or 0)
    auto_par = not (tp and pp)
    if is_local:
        tp, pp = 1, 1

    # ---- capability + support gates --------------------------------------
    cc = gpu["cc"]
    fmt = WEIGHT_FORMATS[fmt_key]

    if os_name not in eng_meta["os"]:
        findings.append(_f("blocker", f"{eng_meta['label']} does not run on {os_name}",
                           f"{eng_meta['label']} supports {', '.join(eng_meta['os'])}. "
                           "On Windows, run it under WSL2, or switch to llama.cpp/Ollama "
                           "which have native builds.", "os"))
    if fmt_key not in eng_meta["weights"]:
        findings.append(_f("blocker", f"{fmt['label']} is not a {eng_meta['label']} weight format",
                           f"{eng_meta['label']} loads: "
                           f"{', '.join(WEIGHT_FORMATS[w]['label'] for w in eng_meta['weights'][:8])}...",
                           "format"))
    if gpu["vendor"] == "apple" and not is_local:
        findings.append(_f("blocker", "Apple silicon cannot run vLLM/SGLang",
                           "These engines are CUDA/ROCm only. Use llama.cpp (Metal) or MLX.", "vendor"))
    # FP8 and MXFP4 have a dequantize-on-load fallback in vLLM/SGLang, so an
    # under-spec GPU degrades (more memory, no speedup) rather than refusing.
    DEQUANT_FALLBACK = {"fp8", "mxfp4"}
    if gpu["vendor"] == "nvidia" and cc > 0:
        if fmt["min_cc"] > cc and fmt_key not in DEQUANT_FALLBACK:
            findings.append(_f("blocker", f"{fmt['label']} needs compute capability "
                               f"{fmt['min_cc']}+, this GPU is sm_{int(cc*10)}",
                               f"{gpu['name']} is {gpu['family']} ({gpu['sm']}), which has no "
                               f"hardware path for {fmt['label']}. Pick a format its tensor "
                               "cores support.", "cc"))
        if fmt_key == "bf16" and cc < 8.0:
            findings.append(_f("blocker", "No bfloat16 on this GPU",
                               f"{gpu['name']} is {gpu['family']}, which predates Ampere and has "
                               "no bfloat16 tensor cores. Use FP16 instead.", "cc"))
        if arch["min_cc"] > cc:
            findings.append(_f("blocker", f"{arch['family']} kernels need sm_{int(arch['min_cc']*10)}+",
                               f"The attention/MoE kernels this architecture relies on are not "
                               f"compiled for sm_{int(cc*10)}.", "arch"))

    eng_support = arch.get(eng if eng != "ollama" else "llamacpp", "?")
    if eng_support == "n":
        findings.append(_f("blocker", f"{arch['family']} is not supported by {eng_meta['label']}",
                           arch.get("note") or "This architecture has no implementation in "
                           f"{eng_meta['label']}. Try a different engine.", "arch"))
    elif eng_support == "p":
        findings.append(_f("warn", f"{arch['family']} support in {eng_meta['label']} is partial",
                           arch.get("note") or "Support exists but is newer or carries caveats "
                           "(feature gaps, no chunked prefill, or version-gated). Verify against "
                           "your engine version.", "arch"))
    elif eng_support == "?":
        findings.append(_f("warn", f"{arch['family']} is not in the support table",
                           arch.get("note") or "Check the engine's supported-models list.", "arch"))
    if not arch.get("known"):
        findings.append(_f("info", "Architecture matched by inference",
                           f"config.json model_type is '{spec.get('model_type') or 'missing'}'; "
                           f"treated as {arch['family']}.", "arch"))

    # ---- native quantization interactions --------------------------------
    native_q = (spec.get("quant_method") or "").lower()
    upcast_factor = 1.0
    q_kind = fmt["kind"]
    if (native_q in ("mxfp4",) or fmt_key == "mxfp4") and gpu["vendor"] == "nvidia" \
            and 0 < cc < 9.0 and not is_local:
        upcast_factor = 4.0
        findings.append(_f("warn", "MXFP4 weights get upcast to BF16 on this GPU",
                           f"The repo ships MXFP4 MoE weights, but native MXFP4 kernels need "
                           f"Hopper/Blackwell (sm90+). {gpu['name']} is {gpu['family']} "
                           f"({gpu['sm']}), so vLLM "
                           "dequantizes the experts to BF16 at load, so plan for roughly 4x the "
                           "advertised weight size. The estimate below already reflects that.",
                           "quant"))
    if (native_q in ("fp8", "fbgemm_fp8") or fmt_key == "fp8") and gpu["vendor"] == "nvidia" \
            and 0 < cc < 8.9 and not is_local:
        upcast_factor = 2.0
        findings.append(_f("warn", "FP8 weights get upcast to BF16 on this GPU",
                           f"The repo ships FP8 weights but {gpu['family']} ({gpu['sm']}) "
                           "has no FP8 tensor "
                           "cores; the loader converts to BF16, doubling weight memory. The "
                           "estimate below reflects that.", "quant"))
    if native_q and fmt["kind"] not in ("gguf",) and native_q not in fmt_key and upcast_factor == 1.0:
        findings.append(_f("info", f"Repo ships {native_q.upper()} weights",
                           f"config.json declares quantization_config.quant_method="
                           f"'{native_q}'. Selecting a different format means re-quantizing "
                           "offline, not a runtime flag.", "quant"))

    # ---- weights ----------------------------------------------------------
    if upcast_factor > 1.0:
        # Runs as BF16 after the loader dequantizes.
        w_bytes, bpw, w_src = weight_bytes(spec, params, "bf16", 0)
        w_src = f"dequantized to BF16 on sm_{int(cc*10)}"
    else:
        w_bytes, bpw, w_src = weight_bytes(spec, params, fmt_key, gguf_size)

    # ---- auto parallelism -------------------------------------------------
    vram_b = gpu["vram_gb"] * GIB
    kv_elem_pre = KV_DTYPES.get(kv_key, KV_DTYPES["auto"])["bytes"]
    if auto_par and not is_local:
        tp, pp = _auto_parallel(spec, w_bytes, vram_b, util, gpus_per_node, nodes,
                                interconnect, ctx, seqs, kv_elem_pre)
    tp = max(1, tp); pp = max(1, pp)
    used_gpus = tp * pp

    if not is_local and used_gpus > total_gpus:
        findings.append(_f("blocker", f"TP{tp} x PP{pp} needs {used_gpus} GPUs, you have {total_gpus}",
                           "Reduce parallelism, add GPUs, or use a smaller weight format.", "parallel"))
    heads = spec.get("num_attention_heads") or 0
    kvh = spec.get("num_kv_heads") or heads
    if not is_local and tp > 1:
        if heads and heads % tp:
            findings.append(_f("blocker", f"num_attention_heads={heads} is not divisible by TP={tp}",
                               "Tensor parallelism splits attention heads evenly; pick a TP that "
                               f"divides {heads}.", "parallel"))
        if kvh and tp > kvh and tp % kvh:
            findings.append(_f("warn", f"TP={tp} exceeds num_key_value_heads={kvh}",
                               "KV heads get replicated across ranks, so the KV cache stops "
                               "shrinking with TP and total KV memory grows.", "parallel"))
        if spec.get("intermediate_size") and spec["intermediate_size"] % tp:
            findings.append(_f("warn", f"intermediate_size={spec['intermediate_size']} is not "
                               f"divisible by TP={tp}", "Some engines pad the MLP shards; others "
                               "refuse to start.", "parallel"))
    if not is_local and pp > 1 and spec.get("num_layers") and spec["num_layers"] % pp:
        findings.append(_f("info", f"{spec['num_layers']} layers over PP={pp} splits unevenly",
                           "Stages get different layer counts, so the pipeline bubbles are "
                           "slightly larger.", "parallel"))
    if not is_local and tp > gpus_per_node and nodes > 1:
        if interconnect in ("eth",):
            findings.append(_f("blocker", f"TP={tp} spans nodes over plain Ethernet",
                               "Tensor parallelism all-reduces activations at every layer. Over "
                               "TCP Ethernet this is not viable - it will either fail NCCL "
                               "initialisation or run an order of magnitude slower. Use "
                               "pipeline parallelism across nodes instead.", "parallel"))
        else:
            findings.append(_f("warn", f"TP={tp} spans {nodes} nodes",
                               "Cross-node tensor parallelism needs InfiniBand or RoCE at 200Gb+ "
                               "with GPUDirect RDMA. Prefer TP inside a node and PP across nodes.",
                               "parallel"))
    if not is_local and pp > 1 and nodes > 1 and interconnect == "eth":
        findings.append(_f("info", "Pipeline parallelism is the right choice here",
                           "PP only ships one activation tensor per stage boundary, so it "
                           "tolerates Ethernet far better than TP does.", "parallel"))
    if not is_local and tp > 2 and not gpu.get("nvlink") and gpu["vendor"] == "nvidia":
        findings.append(_f("warn", f"{gpu['name']} has no NVLink",
                           f"TP={tp} all-reduces cross PCIe. Expect noticeably sub-linear scaling "
                           "past TP=4 on this card.", "parallel"))

    # ---- per-GPU memory budget -------------------------------------------
    kv_elem = KV_DTYPES.get(kv_key, KV_DTYPES["auto"])["bytes"]
    if kv_key not in ("auto",) and gpu["vendor"] == "nvidia" and cc > 0 \
            and KV_DTYPES[kv_key]["min_cc"] > cc:
        findings.append(_f("blocker", f"{KV_DTYPES[kv_key]['label']} KV cache needs "
                           f"sm_{int(KV_DTYPES[kv_key]['min_cc']*10)}+",
                           f"{gpu['name']} cannot store the KV cache in that dtype.", "kv"))
    if kv_key == "fp8":
        findings.append(_f("info", "FP8 KV cache halves cache memory",
                           "Quality impact is small for most models but measurable on long "
                           "reasoning traces. Worth an eval before production.", "kv"))

    traits = set(arch.get("traits") or [])
    if "embed" in traits or (spec.get("num_layers") and not spec.get("head_dim")):
        spec["no_kv_cache"] = True
    if "embed" in traits:
        findings.append(_f("info", "Encoder model - no KV cache",
                           "Encoder-only models run a single forward pass per input, so there is "
                           "no autoregressive cache. Memory is weights plus the activation "
                           "working set, which scales with batch size x sequence length.", "kv"))
    elif spec.get("is_encoder_decoder"):
        findings.append(_f("info", "Encoder-decoder model",
                           f"Only the {spec.get('num_layers')} decoder layers hold a self-attention "
                           "cache, and generated sequences are short. The cross-attention cache "
                           "over encoder states is approximated here.", "kv"))

    if is_local:
        result = _size_local(spec, gpu, w_bytes, ctx, seqs, kv_elem, host_ram,
                             total_gpus, rt, findings)
    else:
        result = _size_server(spec, gpu, w_bytes, params, ctx, seqs, kv_elem, tp, pp,
                              util, eager, batched_tokens, images, findings)

    # ---- context / model limits ------------------------------------------
    max_pos = spec.get("max_position_embeddings") or 0
    if max_pos and ctx > max_pos:
        rf = spec.get("rope_factor")
        detail = (f"config.json declares max_position_embeddings={max_pos:,}. "
                  f"Serving {ctx:,} needs RoPE scaling (--rope-scaling / --rope-scale) and "
                  "quality degrades past the trained length.")
        if rf:
            detail += f" The repo already declares rope_scaling factor {rf}."
        findings.append(_f("warn", f"Requested context exceeds the model's native {max_pos:,}",
                           detail, "context"))
    if spec.get("sliding_window") and spec.get("swa_layers"):
        detail = (f"{spec['swa_layers']} of {spec.get('num_layers')} layers only keep "
                  f"{spec['sliding_window']:,} tokens of cache, so KV memory grows much "
                  "more slowly than context length. The remaining "
                  f"{spec.get('full_attn_layers')} full-attention layer(s) are what actually "
                  "drive the number above.")
        if spec.get("swa_pattern_assumed"):
            detail += (f" config.json does not declare the interleave, so the transformers "
                       f"default for this family (1 full layer per {spec.get('swa_pattern')}) "
                       "was assumed.")
        findings.append(_f("info", "Sliding-window attention caps KV growth", detail, "kv"))
    if spec.get("is_sparse_attn"):
        findings.append(_f("info", "Sparse attention",
                           f"This model selects the top {spec.get('index_topk') or 'k'} keys per "
                           f"query through a {spec.get('index_n_heads') or ''}-head indexer. The "
                           "kernels for that are far narrower than ordinary attention — see the "
                           "attention backend section.", "attention"))
    if spec.get("kv_layout_unknown"):
        findings.append(_f("warn", "KV cache size is an upper bound for this architecture",
                           "The config declares a compressed or sparse cache whose per-token "
                           "layout cannot be derived from it. The KV figure here is computed from "
                           "the declared head count and width, which over-states what the engine "
                           "actually allocates. Treat the memory total as a ceiling.", "kv"))
    # Block-scaled FP8 needs DeepGEMM/CUTLASS paths that only exist on newer parts.
    if (spec.get("quant_block_size") and (spec.get("quant_method") or "").startswith("fp8")
            and gpu["vendor"] == "nvidia" and 0 < cc < 9.0 and not is_local):
        bs = spec["quant_block_size"]
        findings.append(_f("warn", "Block-scaled FP8 has no fast kernel on this GPU",
                           f"The checkpoint is FP8 with a {bs} weight-block scale. The DeepGEMM "
                           "and CUTLASS block-FP8 paths are Hopper-and-newer; below that the "
                           "engine falls back to a much slower generic path, where it works at "
                           "all.", "quant"))
    if spec.get("quant_scale_fmt") in ("ue8m0",) and gpu["vendor"] == "nvidia" \
            and 0 < cc < 9.0 and not is_local:
        findings.append(_f("warn", "UE8M0 scale format is a Hopper-era assumption",
                           "The checkpoint stores its FP8 scales as UE8M0, which the older "
                           "quantization paths do not read. Verify the engine accepts it on this "
                           "architecture before planning around it.", "quant"))
    if spec.get("is_mla"):
        findings.append(_f("info", "MLA compresses the KV cache",
                           f"Latent rank {spec['kv_lora_rank']} + {spec['qk_rope_head_dim']} rope "
                           "dims per token per layer, replicated (not sharded) across TP ranks.", "kv"))
    if spec.get("is_moe"):
        findings.append(_f("info", f"MoE: {spec['num_experts']} experts, "
                           f"{spec['experts_per_token']} active per token",
                           "Every expert must be resident in memory even though only a few run "
                           "per token - size VRAM on total params, speed on active params.", "moe"))
    if spec.get("has_vision") and not is_local:
        findings.append(_f("info", "Vision tower adds encoder memory",
                           f"Image tokens are appended to the sequence; {images or 1} image(s) per "
                           "request can add hundreds to thousands of tokens each. The estimate "
                           "reserves a multimodal encoder buffer.", "vision"))
    if model.get("gated"):
        findings.append(_f("warn", "Gated repository",
                           "You must accept the licence on the Hub and pass HF_TOKEN to the "
                           "serving container to download the weights.", "access"))

    # ---- driver / toolkit -------------------------------------------------
    drv_major = 0
    try:
        drv_major = int(float(driver.split(".")[0])) if driver else 0
    except ValueError:
        drv_major = 0
    if drv_major:
        need = 570 if cc >= 10.0 else (525 if cc >= 7.0 else 470)
        if drv_major < need:
            findings.append(_f("blocker", f"Driver {driver} is too old",
                               f"{gpu['arch']} with a current CUDA 12.x wheel needs driver "
                               f"{need}+.", "driver"))
    if gpu["vendor"] == "amd" and eng in ("vllm", "sglang"):
        findings.append(_f("warn", "ROCm build required",
                           "You need the ROCm image of the engine; a handful of kernels "
                           "(some quant schemes, FlashInfer) are CUDA-only.", "vendor"))
    if gpu["vendor"] == "nvidia" and 0 < cc < 8.0 and not is_local:
        findings.append(_f("warn", "No FlashAttention-2 below sm_80",
                           "The engine falls back to xformers/Triton paths - slower, and some "
                           "features (sliding window, chunked prefill) may be unavailable.", "kernel"))

    # ---- attention backend ------------------------------------------------
    attn = attention.analyze(spec, gpu, eng, kv_key, rt.get("attention") or "auto")
    for f in attn["findings"]:
        findings.append(_f(f["level"], f["title"], f["detail"], f["tag"]))

    # ---- throughput -------------------------------------------------------
    compute_kind = "native" if upcast_factor > 1.0 else fmt["kind"]
    perf = _throughput(spec, gpu, params, w_bytes, ctx, seqs, kv_elem, tp, pp,
                       is_local, result, compute_kind)

    # ---- verdict ----------------------------------------------------------
    blockers = [f for f in findings if f["level"] == "blocker"]
    if blockers:
        verdict, headline = "no", "Will not run as configured"
    elif not result["fits"]:
        verdict, headline = "no", "Does not fit in memory"
    elif result["headroom_ratio"] < 0.08:
        verdict, headline = "tight", "Fits, but with almost no headroom"
    elif result.get("kv_ok_ratio", 1) < 1.2:
        verdict, headline = "tight", "Fits at this context, little room to grow"
    else:
        verdict, headline = "yes", "Fits comfortably"

    out = {
        "verdict": verdict, "headline": headline,
        "gpu": gpu, "arch": {**arch, "engine_status": eng_support,
                             "engine_status_label": archsupport.STATUS_LABEL.get(eng_support, "Unknown")},
        "parallel": {"tp": tp, "pp": pp, "gpus_used": used_gpus,
                     "total_gpus": total_gpus, "auto": auto_par, "nodes": nodes,
                     "gpus_per_node": gpus_per_node},
        "weights": {"bytes": w_bytes, "gib": w_bytes / GIB, "bpw": round(bpw, 3),
                    "source": w_src, "params": params,
                    "active_params": _active(spec, params),
                    "upcast": upcast_factor},
        "memory": result, "perf": perf, "attention": attn, "findings": findings,
        "command": _command(model, gpu, spec, rt, tp, pp, util, ctx, seqs, eng, fmt_key,
                            kv_key, eager, result),
    }
    if _suggest and verdict == "no":
        out["suggestions"] = _suggestions(model, hw, rt, gpu)
    return out


# ---------------------------------------------------------------------------
# "What would make it fit" - every option is a real re-evaluation, not a guess
# ---------------------------------------------------------------------------
# Weight formats ordered by descending size, per engine, so we can walk down to
# the next thing that would actually fit.
_FORMAT_LADDER = {
    "vllm":   ["bf16", "fp16", "fp8", "int8", "awq", "gptq4", "bnb4"],
    "sglang": ["bf16", "fp16", "fp8", "int8", "awq", "gptq4"],
    "llamacpp": ["gguf_q8_0", "gguf_q6_k", "gguf_q5_k_m", "gguf_q4_k_m",
                 "gguf_iq4_xs", "gguf_q3_k_m", "gguf_q2_k"],
    "ollama": ["gguf_q8_0", "gguf_q6_k", "gguf_q5_k_m", "gguf_q4_k_m",
               "gguf_iq4_xs", "gguf_q3_k_m", "gguf_q2_k"],
}


def _ok(r) -> bool:
    return r["verdict"] in ("yes", "tight") and not any(
        f["level"] == "blocker" for f in r["findings"])


def _suggestions(model: dict, hw: dict, rt: dict, gpu: dict) -> list:
    out, seen = [], set()

    def attempt(kind, title, hw2, rt2, patch, note=""):
        key = (kind, title)
        if key in seen or len(out) >= 4:
            return
        try:
            r = evaluate(model, hw2, rt2, _suggest=False)
        except (ValueError, KeyError, ZeroDivisionError):
            return
        if not _ok(r):
            return
        seen.add(key)
        m = r["memory"]
        out.append({
            "kind": kind, "title": title, "note": note,
            "verdict": r["verdict"],
            "patch": patch,
            "weights_gib": m["weights_gib"],
            "kv_available_gib": max(m["kv_available_gib"], 0),
            "max_context": m["max_context_single"],
            "max_concurrent": m["max_concurrent_at_ctx"],
            "tp": r["parallel"]["tp"], "pp": r["parallel"]["pp"],
            "gpus": r["parallel"]["gpus_used"],
            "decode_tps": (r["perf"] or {}).get("decode_tps_single"),
        })

    eng = rt.get("engine") or "vllm"
    is_local = eng in ("llamacpp", "ollama")
    have = max(1, int(hw.get("nodes") or 1)) * max(1, int(hw.get("gpus_per_node") or 1))

    # 1. More of the same card. Walk up node count to the smallest that works.
    if not is_local:
        per_node = max(1, int(hw.get("gpus_per_node") or 1))
        for nodes in (2, 3, 4, 6, 8):
            total = per_node * nodes
            if total <= have or total > 128:
                continue
            hw2 = dict(hw, nodes=nodes)
            rt2 = dict(rt, tp=0, pp=0)
            before = len(out)
            attempt("hardware",
                    f"Use {total} GPUs — {nodes} nodes of {per_node} × {gpu['name']}",
                    hw2, rt2, {"hardware": {"nodes": nodes}, "runtime": {"tp": 0, "pp": 0}},
                    note="Needs the extra nodes reachable over InfiniBand or RoCE."
                         if nodes > 1 else "")
            if len(out) > before:
                break

    # 2. Quantize further on the hardware already in hand.
    ladder = _FORMAT_LADDER.get(eng, [])
    cur = rt.get("weight_format") or "bf16"
    if cur in ladder:
        for nxt in ladder[ladder.index(cur) + 1:]:
            rt2 = dict(rt, weight_format=nxt, tp=0, pp=0)
            before = len(out)
            attempt("weights",
                    f"Quantize to {WEIGHT_FORMATS[nxt]['label']}",
                    dict(hw), rt2, {"runtime": {"weight_format": nxt, "tp": 0, "pp": 0}},
                    note="Only works if a checkpoint in this format exists, or you "
                         "quantize it yourself offline.")
            if len(out) > before:
                break

    # 3. Halve the KV cache with an FP8 cache.
    if rt.get("kv_dtype", "auto") == "auto" and gpu.get("cc", 0) >= 8.9 and not is_local:
        attempt("kv", "Store the KV cache in FP8", dict(hw),
                dict(rt, kv_dtype="fp8", tp=0, pp=0),
                {"runtime": {"kv_dtype": "fp8", "tp": 0, "pp": 0}},
                note="Halves cache memory. Quality impact is small for most models but "
                     "worth an eval on long reasoning traces.")

    # 4. Ask for less: shorter context, or fewer concurrent streams.
    ctx = int(rt.get("context_len") or 8192)
    for frac in (0.5, 0.25, 0.125):
        c = max(1024, int(ctx * frac) // 1024 * 1024)
        if c >= ctx:
            continue
        before = len(out)
        attempt("context", f"Serve {c:,} tokens of context instead of {ctx:,}",
                dict(hw), dict(rt, context_len=c, tp=0, pp=0),
                {"runtime": {"context_len": c, "tp": 0, "pp": 0}})
        if len(out) > before:
            break

    seqs = int(rt.get("concurrency") or 1)
    if seqs > 1:
        for frac in (0.5, 0.25, 0.125):
            n = max(1, int(seqs * frac))
            if n >= seqs:
                continue
            before = len(out)
            attempt("concurrency", f"Run {n} concurrent stream{'s' if n > 1 else ''} instead of {seqs}",
                    dict(hw), dict(rt, concurrency=n, tp=0, pp=0),
                    {"runtime": {"concurrency": n, "tp": 0, "pp": 0}})
            if len(out) > before:
                break

    return out


def _active(spec, params):
    from .hf import active_params
    return active_params(spec, params)


def _auto_parallel(spec, w_bytes, vram_b, util, gpus_per_node, nodes, interconnect,
                   ctx=4096, seqs=1, kv_elem=2.0):
    """Smallest TP x PP that holds the weights *and* the requested KV working set.

    Tensor parallelism is kept inside a node and pipeline parallelism spans
    nodes, which is the arrangement that survives a slow interconnect.
    """
    budget = vram_b * util
    heads = spec.get("num_attention_heads") or 0
    max_gpus = gpus_per_node * nodes

    def legal_tp(t):
        t = min(t, gpus_per_node)
        while t > 1 and heads and heads % t:
            t //= 2
        return max(t, 1)

    best = None
    for total in (1, 2, 4, 8, 16, 32, 64, 128, 256):
        if total > max_gpus:
            break
        tp = legal_tp(total)
        pp = max(1, total // tp)
        if tp * pp > max_gpus:
            continue
        need = (w_bytes / (tp * pp) + 2.5 * GIB
                + kv_bytes_for(spec, ctx, seqs, kv_elem, tp, pp))
        if best is None:
            best = (tp, pp)
        if need <= budget:
            return tp, pp
    # Nothing satisfies the request: fall back to the widest legal shape so the
    # caller reports the real shortfall rather than an artificially small one.
    tp = legal_tp(min(gpus_per_node, max_gpus))
    return tp, max(1, min(max_gpus // max(tp, 1), 16))


# ---------------------------------------------------------------------------
# vLLM / SGLang sizing
# ---------------------------------------------------------------------------
def _size_server(spec, gpu, w_bytes, params, ctx, seqs, kv_elem, tp, pp, util,
                 eager, batched_tokens, images, findings):
    vram_b = gpu["vram_gb"] * GIB
    budget = vram_b * util
    w_pg = w_bytes / (tp * pp)

    # Framework overhead: CUDA context, cuBLAS/cuDNN handles, NCCL buffers.
    overhead = 0.85 * GIB + (0.25 * GIB * math.log2(tp) if tp > 1 else 0)

    # Peak activation for one scheduler step of `batched_tokens` tokens.
    h = spec.get("hidden_size") or 0
    inter = spec.get("intermediate_size") or 0
    heads = spec.get("num_attention_heads") or 1
    kvh = spec.get("num_kv_heads") or heads
    hd = spec.get("head_dim") or 0
    tokens = batched_tokens
    if spec.get("no_kv_cache"):
        tokens = max(batched_tokens, ctx * seqs)
    act = (tokens * (2 * inter + (heads + 2 * kvh) * hd) / tp
           + tokens * h * 4) * 2
    act += seqs * (spec.get("vocab_size") or 0) * 4 / tp     # logits buffer
    if spec.get("has_vision"):
        act += max(images, 1) * 0.35 * GIB                   # encoder + pixel buffers
    act = max(act, 0.3 * GIB)

    graphs = 0 if eager else min(2.0 * GIB, 0.4 * GIB + 0.03 * w_pg)

    kv_avail = budget - w_pg - overhead - act - graphs
    kv_need = kv_bytes_for(spec, ctx, seqs, kv_elem, tp, pp)

    fits = kv_avail > 0 and kv_need <= kv_avail
    if kv_avail <= 0 and w_pg + overhead + act + graphs > budget:
        findings.append(_f("blocker", "Weights alone exceed the per-GPU budget",
                           f"{w_pg/GIB:.1f} GiB of weights + {(overhead+act+graphs)/GIB:.1f} GiB "
                           f"of runtime overhead per GPU vs a {budget/GIB:.1f} GiB budget "
                           f"({util:.0%} of {gpu['vram_gb']} GB). Increase TP/PP, quantize "
                           "further, or use bigger GPUs.", "memory"))
    elif not fits:
        findings.append(_f("blocker", "KV cache does not fit at this context and concurrency",
                           f"Need {kv_need/GIB:.1f} GiB per GPU, only {max(kv_avail,0)/GIB:.1f} GiB "
                           "is left after weights and overhead.", "memory"))

    cap = 4_194_304
    if spec.get("no_kv_cache"):
        # Activation working set per sequence, not cache.
        per_seq = max(act / max(seqs, 1), 1)
        max_ctx1 = spec.get("max_position_embeddings") or ctx
        max_seqs = int((kv_avail + act) // per_seq) if kv_avail + act > 0 else 0
        total_tokens = 0
    else:
        max_ctx1 = max_ctx_for_budget(spec, kv_avail, 1, kv_elem, tp, pp, cap)
        one_seq = kv_bytes_for(spec, ctx, 1, kv_elem, tp, pp)
        max_seqs = int(kv_avail // one_seq) if one_seq > 0 and kv_avail > 0 else 0
        total_tokens = max_ctx1

    used = w_pg + overhead + act + graphs + max(kv_need, 0)
    return {
        "mode": "server",
        "budget_gib": budget / GIB, "vram_gib": vram_b / GIB, "util": util,
        "weights_gib": w_pg / GIB, "overhead_gib": overhead / GIB,
        "activations_gib": act / GIB, "cudagraph_gib": graphs / GIB,
        "kv_available_gib": max(kv_avail, 0) / GIB, "kv_needed_gib": kv_need / GIB,
        "used_gib": used / GIB,
        "free_after_gib": max(budget - used, 0) / GIB,
        "unused_reserve_gib": (vram_b - budget) / GIB,
        "fits": fits,
        "headroom_ratio": max(budget - used, 0) / budget if budget else 0,
        "kv_ok_ratio": (kv_avail / kv_need) if kv_need > 0 else 999,
        "max_context_single": max_ctx1,
        "max_concurrent_at_ctx": max_seqs,
        "kv_total_token_capacity": total_tokens,
        "no_kv_cache": bool(spec.get("no_kv_cache")),
        "kv_per_token_bytes": kv_bytes_for(spec, 1, 1, kv_elem, tp, pp),
        "per_gpu": True,
    }


# ---------------------------------------------------------------------------
# llama.cpp / Ollama sizing
# ---------------------------------------------------------------------------
def _size_local(spec, gpu, w_bytes, ctx, seqs, kv_elem, host_ram, total_gpus, rt, findings):
    layers = spec.get("num_layers") or 1
    vram_total = gpu["vram_gb"] * GIB * total_gpus
    # llama.cpp needs room for the compute buffer and the context, not a fixed
    # utilization fraction; reserve conservatively.
    reserve = 0.6 * GIB * total_gpus + 0.25 * GIB
    kv_need = kv_bytes_for(spec, ctx, seqs, kv_elem, 1, 1)
    compute_buf = max(0.25 * GIB, (spec.get("hidden_size") or 4096) * 2048 * 4)

    auto_layers = rt.get("gpu_layers")
    if auto_layers in (None, "", -1, "auto"):
        avail = vram_total - reserve - kv_need - compute_buf
        n_off = layers if avail >= w_bytes else int(max(0, avail) / (w_bytes / layers))
        n_off = max(0, min(layers, n_off))
    else:
        n_off = max(0, min(layers, int(auto_layers)))

    w_gpu = w_bytes * n_off / layers
    w_cpu = w_bytes - w_gpu
    # KV lives on the GPU only when all layers are offloaded (llama.cpp default).
    kv_gpu = kv_need if n_off >= layers else kv_need * n_off / layers
    kv_cpu = kv_need - kv_gpu

    vram_used = w_gpu + kv_gpu + compute_buf + reserve
    ram_used = w_cpu + kv_cpu + 0.5 * GIB
    fits_vram = vram_used <= vram_total
    fits_ram = (host_ram <= 0) or (ram_used <= host_ram * GIB * 0.92)

    if not fits_vram:
        findings.append(_f("blocker", "Does not fit in VRAM even with offload accounting",
                           f"{vram_used/GIB:.1f} GiB needed vs {vram_total/GIB:.1f} GiB available.",
                           "memory"))
    if n_off < layers:
        findings.append(_f("warn", f"Only {n_off} of {layers} layers fit on the GPU",
                           f"{w_cpu/GIB:.1f} GiB of weights stay in system RAM. CPU layers are "
                           "5-20x slower per token; expect the offloaded fraction to dominate "
                           "generation speed.", "offload"))
    if host_ram > 0 and not fits_ram:
        findings.append(_f("blocker", f"Not enough system RAM for the CPU-resident part",
                           f"Needs about {ram_used/GIB:.1f} GiB of {host_ram:.0f} GB host RAM.",
                           "memory"))
    if total_gpus > 1:
        findings.append(_f("info", f"{total_gpus} GPUs are pooled by layer split",
                           "llama.cpp splits by layer rather than tensor-parallel, so VRAM adds "
                           "up but only one GPU is busy at a time during decode.", "parallel"))

    avail_for_kv = vram_total - reserve - w_gpu - compute_buf
    max_ctx1 = (spec.get("max_position_embeddings") or ctx) if spec.get("no_kv_cache") \
        else max_ctx_for_budget(spec, avail_for_kv, 1, kv_elem, 1, 1, 4_194_304)
    one_seq = kv_bytes_for(spec, ctx, 1, kv_elem, 1, 1)
    return {
        "mode": "local",
        "budget_gib": vram_total / GIB, "vram_gib": vram_total / GIB, "util": 1.0,
        "weights_gib": w_gpu / GIB, "weights_cpu_gib": w_cpu / GIB,
        "overhead_gib": reserve / GIB, "activations_gib": compute_buf / GIB,
        "cudagraph_gib": 0.0,
        "kv_available_gib": max(avail_for_kv, 0) / GIB, "kv_needed_gib": kv_gpu / GIB,
        "kv_cpu_gib": kv_cpu / GIB,
        "used_gib": vram_used / GIB,
        "free_after_gib": max(vram_total - vram_used, 0) / GIB,
        "unused_reserve_gib": 0.0,
        "ram_used_gib": ram_used / GIB, "host_ram_gib": host_ram,
        "gpu_layers": n_off, "total_layers": layers,
        "fits": fits_vram and fits_ram,
        "headroom_ratio": max(vram_total - vram_used, 0) / vram_total if vram_total else 0,
        "kv_ok_ratio": (avail_for_kv / kv_gpu) if kv_gpu > 0 else 999,
        "max_context_single": max_ctx1,
        "max_concurrent_at_ctx": int(avail_for_kv // one_seq) if one_seq > 0 and avail_for_kv > 0 else 0,
        "kv_per_token_bytes": kv_bytes_for(spec, 1, 1, kv_elem, 1, 1),
        "no_kv_cache": bool(spec.get("no_kv_cache")),
        "per_gpu": False,
    }


# ---------------------------------------------------------------------------
# Throughput (order-of-magnitude, memory-bandwidth and FLOP roofline)
# ---------------------------------------------------------------------------
def _throughput(spec, gpu, params, w_bytes, ctx, seqs, kv_elem, tp, pp, is_local, mem,
                compute_kind="native"):
    bw = gpu["bandwidth"] * 1e9
    if bw <= 0 or not params:
        return {"available": False}
    active = _active(spec, params)
    active_bytes = w_bytes * (active / params) if params else w_bytes
    eff = 0.80 if not is_local else 0.65          # achievable fraction of peak BW

    read_per_step = active_bytes / (tp if not is_local else 1)
    kv_read = kv_bytes_for(spec, ctx // 2, 1, kv_elem, tp, pp) if not is_local \
        else kv_bytes_for(spec, ctx // 2, 1, kv_elem, 1, 1)

    single_tps = (bw * eff) / max(read_per_step + kv_read, 1)
    if pp > 1:
        single_tps *= 0.92 ** (pp - 1)             # pipeline stage handoff

    # Batched: weights are read once per step and amortised over the batch.
    denom = read_per_step + seqs * kv_read
    batch_tps = seqs * (bw * eff) / max(denom, 1)
    if compute_kind == "fp4" and gpu.get("fp4"):
        flops = gpu["fp4"] * 1e12
    elif compute_kind in ("fp8", "int8") and gpu.get("fp8"):
        flops = gpu["fp8"] * 1e12
    else:
        # Weight-only quantization (AWQ/GPTQ/GGUF) still does 16-bit math.
        flops = gpu["fp16"] * 1e12
    mfu = 0.45
    compute_ceiling = (flops * mfu * tp * pp) / max(2 * active, 1) if flops else float("inf")
    batch_tps = min(batch_tps, compute_ceiling)

    prefill_tps = (flops * mfu * tp * pp) / max(2 * active, 1) if flops else 0
    ttft = (ctx / prefill_tps) if prefill_tps else 0

    return {
        "available": True,
        "decode_tps_single": single_tps,
        "decode_tps_total": batch_tps,
        "prefill_tps": prefill_tps,
        "ttft_s": ttft,
        "compute_ceiling_tps": compute_ceiling if compute_ceiling != float("inf") else None,
        "bound": "memory bandwidth" if batch_tps < compute_ceiling * 0.95 else "compute",
        "active_params": active,
        "bandwidth_gbs": gpu["bandwidth"],
    }


# ---------------------------------------------------------------------------
# Launch command
# ---------------------------------------------------------------------------
def _command(model, gpu, spec, rt, tp, pp, util, ctx, seqs, eng, fmt_key, kv_key,
             eager, mem):
    repo = model.get("id", "MODEL")
    if eng in ("vllm", "sglang"):
        q = {"awq": "awq_marlin", "gptq4": "gptq_marlin", "gptq8": "gptq_marlin",
             "fp8": "fp8", "int8": "compressed-tensors", "bnb4": "bitsandbytes",
             "mxfp4": "mxfp4", "nvfp4": "modelopt_fp4"}.get(fmt_key)
        dtype = {"fp16": "float16", "bf16": "bfloat16", "fp32": "float32"}.get(fmt_key, "auto")
        if eng == "vllm":
            parts = [f"vllm serve {repo}", f"  --tensor-parallel-size {tp}"]
            if pp > 1:
                parts.append(f"  --pipeline-parallel-size {pp}")
            parts += [f"  --max-model-len {ctx}",
                      f"  --max-num-seqs {seqs}",
                      f"  --gpu-memory-utilization {util:g}",
                      f"  --dtype {dtype}"]
            if q:
                parts.append(f"  --quantization {q}")
            if kv_key in ("fp8", "int8"):
                parts.append(f"  --kv-cache-dtype fp8")
            if eager:
                parts.append("  --enforce-eager")
            if spec.get("max_position_embeddings") and ctx > spec["max_position_embeddings"]:
                f = round(ctx / spec["max_position_embeddings"] + 0.05, 2)
                parts.append(f'  --rope-scaling \'{{"rope_type":"yarn","factor":{f},'
                             f'"original_max_position_embeddings":{spec["max_position_embeddings"]}}}\'')
            if spec.get("has_vision"):
                parts.append(f'  --limit-mm-per-prompt \'{{"image":{max(rt.get("images_per_request") or 1,1)}}}\'')
            return " \\\n".join(parts)
        parts = [f"python -m sglang.launch_server --model-path {repo}",
                 f"  --tp {tp}"]
        if pp > 1:
            parts.append(f"  --pp {pp}")
        parts += [f"  --context-length {ctx}", f"  --mem-fraction-static {util:g}"]
        if q:
            parts.append(f"  --quantization {q}")
        if kv_key in ("fp8", "int8"):
            parts.append("  --kv-cache-dtype fp8_e4m3")
        return " \\\n".join(parts)

    gguf = rt.get("gguf_path") or "model.gguf"
    if eng == "llamacpp":
        parts = [f"llama-server -m {gguf}",
                 f"  -ngl {mem.get('gpu_layers', 99)}",
                 f"  -c {ctx * seqs}",
                 f"  -np {seqs}"]
        if kv_key in ("q4",):
            parts += ["  --cache-type-k q4_0", "  --cache-type-v q4_0"]
        elif kv_key in ("fp8", "int8"):
            parts += ["  --cache-type-k q8_0", "  --cache-type-v q8_0"]
        if (rt.get("nodes_gpus") or 1) > 1:
            parts.append("  --split-mode layer")
        parts.append("  --host 0.0.0.0 --port 8080")
        return " \\\n".join(parts)
    tag = repo.split("/")[-1].lower()
    return (f"ollama run {tag}\n"
            f"# context and offload are set per-model:\n"
            f"#   /set parameter num_ctx {ctx}\n"
            f"#   OLLAMA_NUM_GPU={mem.get('gpu_layers', 99)} ollama serve")
