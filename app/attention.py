"""Attention backend feasibility.

A model can pass every memory check and still refuse to start because no
attention kernel on that GPU supports what it needs: sliding windows, an FP8 KV
cache, a 256-wide head, or MLA. This module picks the backend the engine would
actually choose and explains what that costs.

Capabilities are the well-established ones (compute-capability floors, MLA
kernel availability, sliding-window and FP8-KV support). Exact coverage moves
between engine releases, so findings say "typical" where a version could differ.
"""
from __future__ import annotations

# Sentinels shared with catalog.py
CC_CDNA2, CC_CDNA3, CC_RDNA3, CC_METAL = -2.0, -3.0, -4.0, -9.0
AMD_CCS = {CC_CDNA2, CC_CDNA3, CC_RDNA3}

BACKENDS = {
    # --- standard (GQA/MHA) attention -----------------------------------
    "FLASH_ATTN": dict(
        label="FlashAttention 2/3", family="std", vendors=["nvidia"],
        min_cc=8.0, max_cc=None, swa=True, fp8_kv_min_cc=8.9,
        head_dims=(32, 64, 96, 128, 160, 192, 224, 256),
        note="vLLM's default on Ampere through Blackwell. FA3 kernels are used "
             "on Hopper and newer, FA2 below that."),
    "FLASHINFER": dict(
        label="FlashInfer", family="std", vendors=["nvidia"],
        min_cc=7.5, max_cc=None, swa=True, fp8_kv_min_cc=8.9,
        head_dims=(64, 128, 256),
        note="Strong on Hopper/Blackwell and the usual choice when FlashAttention "
             "wheels do not cover the architecture. Head sizes outside 64/128 have "
             "thinner kernel coverage."),
    "TRITON_ATTN": dict(
        label="Triton attention", family="std", vendors=["nvidia", "amd"],
        min_cc=7.0, max_cc=None, swa=True, fp8_kv_min_cc=8.0,
        head_dims=None, portable=True,
        note="Portable JIT fallback. Runs almost anywhere, but is materially "
             "slower than a hand-written FlashAttention kernel."),
    "XFORMERS": dict(
        label="xFormers memory-efficient attention", family="std",
        vendors=["nvidia"], min_cc=7.0, max_cc=8.9, swa=False, fp8_kv_min_cc=None,
        head_dims=None,
        note="The last resort on Volta and Turing, which have no FlashAttention-2. "
             "No sliding-window support in the V1 engine path."),
    "ROCM_AITER_FA": dict(
        label="ROCm AITER FlashAttention", family="std", vendors=["amd"],
        min_cc=CC_CDNA3, max_cc=CC_CDNA3, swa=True, fp8_kv_min_cc=CC_CDNA3,
        head_dims=(64, 128),
        note="AMD's tuned kernels, CDNA3 (MI300/MI325) only."),
    # --- Multi-head Latent Attention (DeepSeek-style) -------------------
    "FLASHMLA": dict(
        label="FlashMLA", family="mla", vendors=["nvidia"],
        min_cc=9.0, max_cc=9.9, swa=True, fp8_kv_min_cc=9.0, head_dims=None,
        note="Hopper-only MLA kernels, and by far the fastest MLA path."),
    "CUTLASS_MLA": dict(
        label="CUTLASS MLA", family="mla", vendors=["nvidia"],
        min_cc=10.0, max_cc=None, swa=True, fp8_kv_min_cc=10.0, head_dims=None,
        note="Blackwell MLA kernels."),
    "FLASHINFER_MLA": dict(
        label="FlashInfer MLA", family="mla", vendors=["nvidia"],
        min_cc=9.0, max_cc=None, swa=True, fp8_kv_min_cc=9.0, head_dims=None,
        note="MLA through FlashInfer on Hopper and newer."),
    "TRITON_MLA": dict(
        label="Triton MLA", family="mla", vendors=["nvidia", "amd"],
        min_cc=8.0, max_cc=None, swa=True, fp8_kv_min_cc=8.0, head_dims=None, portable=True,
        note="The only MLA path below Hopper. It works, but gives up most of the "
             "speed advantage MLA is chosen for."),
    # --- sparse MLA (DeepSeek Sparse Attention / V4 indexer) -------------
    # Verified against vLLM 0.27.1 on an sm_89 card. Every sparse-MLA backend
    # in that build gates on compute-capability MAJOR:
    #   FlashMLASparseBackend      capability.major in [9, 10]
    #   FlashAttnMLASparseBackend  capability.major == 9
    #   FlashInfer DSv4 / SM120    capability.major == 12
    # Nothing accepts major == 8, and there is no Triton sparse path - so
    # unlike dense MLA this family has no portable fallback on Ampere or Ada.
    # Kernel coverage moves between releases; re-check on a version bump.
    "FLASHMLA_SPARSE": dict(
        label="FlashMLA sparse", family="mla_sparse", vendors=["nvidia"],
        min_cc=9.0, max_cc=10.9, swa=True, fp8_kv_min_cc=9.0, head_dims=None,
        note="Sparse MLA kernels for Hopper and data-center Blackwell."),
    "FLASHATTN_MLA_SPARSE": dict(
        label="FlashAttention sparse MLA", family="mla_sparse", vendors=["nvidia"],
        min_cc=9.0, max_cc=9.9, swa=True, fp8_kv_min_cc=9.0, head_dims=None,
        note="In-tree sparse MLA on FlashAttention. Its own gate is "
             "capability.major == 9, so Hopper only."),
    "FLASHINFER_MLA_SPARSE_DSV4": dict(
        label="FlashInfer sparse MLA (DSv4)", family="mla_sparse", vendors=["nvidia"],
        min_cc=12.0, max_cc=None, swa=True, fp8_kv_min_cc=12.0, head_dims=None,
        note="The RTX/workstation Blackwell (sm_120) sparse path."),
    "ROCM_AITER_SPARSE_MLA": dict(
        label="ROCm AITER sparse MLA", family="mla_sparse", vendors=["amd"],
        min_cc=CC_CDNA3, max_cc=CC_CDNA3, swa=True, fp8_kv_min_cc=CC_CDNA3,
        head_dims=None, note="AMD's sparse MLA path, CDNA3 only."),

    # --- non-CUDA --------------------------------------------------------
    "METAL": dict(
        label="Metal (llama.cpp)", family="std", vendors=["apple"],
        min_cc=CC_METAL, max_cc=CC_METAL, swa=True, fp8_kv_min_cc=CC_METAL,
        head_dims=None, note="llama.cpp's Metal backend on Apple silicon."),
    "LLAMACPP_CUDA": dict(
        label="llama.cpp CUDA kernels", family="std", vendors=["nvidia", "amd"],
        min_cc=6.0, max_cc=None, swa=True, fp8_kv_min_cc=6.0, head_dims=None, portable=True,
        note="llama.cpp ships its own kernels and supports far older GPUs than "
             "vLLM does; -fa enables its flash-attention path."),
}

# Preference order the engines themselves use, best first.
ORDER_STD = ["FLASH_ATTN", "ROCM_AITER_FA", "FLASHINFER", "TRITON_ATTN", "XFORMERS"]
ORDER_MLA = ["FLASHMLA", "CUTLASS_MLA", "FLASHINFER_MLA", "TRITON_MLA"]
ORDER_MLA_SPARSE = ["FLASHMLA_SPARSE", "FLASHATTN_MLA_SPARSE",
                    "FLASHINFER_MLA_SPARSE_DSV4", "ROCM_AITER_SPARSE_MLA"]
ORDER_LOCAL = ["METAL", "LLAMACPP_CUDA"]


def _vendor_ok(b, gpu):
    return gpu["vendor"] in b["vendors"]


def _cc_ok(b, cc):
    """Compute capability is a NVIDIA number; AMD and Apple use sentinels.

    A numeric floor is meaningless against a sentinel, so the two worlds are
    kept apart: sentinel GPUs match a backend either by exact sentinel (the
    vendor's own kernels) or by the backend being portable (the Triton and
    llama.cpp paths, which compile for ROCm too).
    """
    if cc > 0:                                   # a real NVIDIA compute capability
        if b["min_cc"] <= 0:                     # vendor-specific, non-CUDA backend
            return False
        return b["min_cc"] <= cc and (b["max_cc"] is None or cc <= b["max_cc"])
    if b["min_cc"] <= 0:                         # sentinel-to-sentinel
        return b["min_cc"] == cc
    return bool(b.get("portable"))


def analyze(spec: dict, gpu: dict, engine: str, kv_key: str, requested: str = "auto") -> dict:
    """Returns the viable backends, the one that would be picked, and findings."""
    cc = gpu["cc"]
    is_local = engine in ("llamacpp", "ollama")
    is_mla = bool(spec.get("is_mla"))
    is_sparse = bool(spec.get("is_sparse_attn"))
    head_dim = spec.get("head_dim") or 0
    swa_needed = bool(spec.get("sliding_window") and spec.get("swa_layers"))
    fp8_kv = kv_key in ("fp8", "int8")

    if is_local:
        order = ORDER_LOCAL
    elif is_sparse:
        order = ORDER_MLA_SPARSE
    else:
        order = ORDER_MLA if is_mla else ORDER_STD

    candidates, findings = [], []
    for key in order:
        b = BACKENDS[key]
        reasons, status = [], "ok"
        if not _vendor_ok(b, gpu):
            continue
        if not _cc_ok(b, cc):
            if cc > 0:
                reasons.append(f"needs sm_{int(b['min_cc']*10)}"
                               + (f"-sm_{int(b['max_cc']*10)}" if b["max_cc"] else "+")
                               + f", this GPU is sm_{int(cc*10)}")
            else:
                reasons.append("not available for this accelerator")
            status = "no"
        if status == "ok" and swa_needed and not b["swa"]:
            reasons.append("no sliding-window support")
            status = "no"
        if status == "ok" and fp8_kv:
            f = b["fp8_kv_min_cc"]
            if f is None:
                reasons.append("cannot store an FP8/INT8 KV cache")
                status = "no"
            elif cc > 0 and f > 0 and cc < f:
                reasons.append(f"FP8 KV cache needs sm_{int(f*10)}+")
                status = "no"
        if status == "ok" and b["head_dims"] and head_dim and head_dim not in b["head_dims"]:
            reasons.append(f"head_dim {head_dim} is outside its supported set "
                           f"({', '.join(str(h) for h in b['head_dims'])})")
            status = "warn" if key != "FLASH_ATTN" else "no"
        candidates.append({"key": key, "label": b["label"], "status": status,
                           "reason": "; ".join(reasons), "note": b["note"]})

    usable = [c for c in candidates if c["status"] in ("ok", "warn")]
    if requested and requested != "auto":
        pick = next((c for c in candidates if c["key"] == requested), None)
        if pick is None:
            findings.append(dict(level="blocker", tag="attention",
                                 title=f"{requested} is not applicable here",
                                 detail=("That backend does not exist for this "
                                         "vendor/model combination. Leave the backend on "
                                         "auto unless you are reproducing a specific setup.")))
            chosen = usable[0] if usable else None
        elif pick["status"] == "no":
            findings.append(dict(level="blocker", tag="attention",
                                 title=f"{pick['label']} cannot run this model here",
                                 detail=f"{pick['reason'].capitalize()}. "
                                        + (f"The engine would otherwise select "
                                           f"{usable[0]['label']}." if usable else
                                           "No other backend is viable either.")))
            chosen = pick
        else:
            chosen = pick
    else:
        chosen = usable[0] if usable else None

    if chosen is None:
        findings.append(dict(level="blocker", tag="attention",
                             title="No attention backend supports this combination",
                             detail=_no_backend_detail(spec, gpu, engine, swa_needed,
                                                       fp8_kv, head_dim, is_mla)))
    else:
        if chosen["status"] == "warn":
            findings.append(dict(level="warn", tag="attention",
                                 title=f"{chosen['label']} is a partial fit",
                                 detail=chosen["reason"].capitalize()
                                        + ". It may fall back to a slower path or fail on "
                                          "some engine versions; verify before relying on it."))
        # Notable degradations worth calling out even when everything "works".
        if is_mla and chosen["key"] == "TRITON_MLA":
            findings.append(dict(level="warn", tag="attention",
                                 title="MLA falls back to Triton on this GPU",
                                 detail="FlashMLA needs Hopper (sm90) and CUTLASS MLA needs "
                                        "Blackwell (sm100). On this card the Triton MLA path "
                                        "runs, but gives up much of the throughput that makes "
                                        "MLA models attractive."))
        if chosen["key"] == "XFORMERS":
            findings.append(dict(level="warn", tag="attention",
                                 title="Falling back to xFormers",
                                 detail="This GPU predates FlashAttention-2 (sm80). Expect "
                                        "lower throughput, and note that sliding-window "
                                        "models and FP8 KV caches are not supported on this path."))
        if chosen["key"] == "TRITON_ATTN" and not is_local:
            findings.append(dict(level="info", tag="attention",
                                 title="Triton attention selected",
                                 detail="No hand-tuned kernel covers this combination, so the "
                                        "portable Triton path is used. It is correct but slower, "
                                        "and it JIT-compiles on first request."))
        if not is_local and cc >= 10.0 and chosen["key"] == "FLASH_ATTN":
            findings.append(dict(level="info", tag="attention",
                                 title="Blackwell kernel coverage is version-sensitive",
                                 detail="Not every published wheel ships FlashAttention kernels "
                                        "for sm100. If startup complains, FlashInfer is the "
                                        "usual working alternative on this card."))
    return {"candidates": candidates, "chosen": chosen,
            "needs": {"mla": is_mla, "sparse": is_sparse, "swa": swa_needed,
                      "fp8_kv": fp8_kv, "head_dim": head_dim},
            "findings": findings}


def _no_backend_detail(spec, gpu, engine, swa, fp8_kv, head_dim, is_mla):
    wants = []
    if spec.get("is_sparse_attn"):
        wants.append(f"sparse attention (top-{spec.get('index_topk') or 'k'} indexer)")
    if is_mla:
        wants.append("multi-head latent attention")
    if swa:
        wants.append(f"a {spec.get('sliding_window'):,}-token sliding window")
    if fp8_kv:
        wants.append("an FP8/INT8 KV cache")
    if head_dim:
        wants.append(f"head_dim {head_dim}")
    need = ", ".join(wants) or "this configuration"
    fixes = []
    if spec.get("is_sparse_attn"):
        fixes.append("sparse-attention models need Hopper, data-center Blackwell, or "
                     "sm_120 — there is no portable kernel for them")
    if fp8_kv:
        fixes.append("set the KV cache dtype back to auto")
    if swa and gpu["cc"] > 0 and gpu["cc"] < 8.0:
        fixes.append("use a model without sliding-window attention on this GPU")
    if is_mla and gpu["cc"] > 0 and gpu["cc"] < 8.0:
        fixes.append("MLA models need Ampere or newer")
    fix = ("Try: " + "; ".join(fixes) + ".") if fixes else \
        "This model needs a newer GPU architecture than the one selected."
    return (f"{gpu['name']} has no attention kernel that provides {need}. {fix}")
