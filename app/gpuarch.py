"""GPU architecture families, and whether a model's architecture runs on each.

Compute capability is what the kernels actually gate on, but "sm_89" is not how
anyone thinks about their fleet — they think "my L20s are Ada". This module
names the families, records what each one can and cannot do, and answers the
question that follows: *will this model run on Ada at all, and what do I lose?*

The per-family verdict is not a second rule set. It runs the real evaluator
against a representative card of each family with memory taken out of the
equation, so the matrix can never disagree with the sizing on the main screen.
"""
from __future__ import annotations

from .catalog import CC_CDNA2, CC_CDNA3, CC_METAL, CC_RDNA3, GPU_BY_ID

# Ordered oldest to newest; `cc` is the representative compute capability and
# `rep` a real card from the catalog used to evaluate the family.
FAMILIES = [
    dict(key="volta", name="Volta", vendor="nvidia", cc=7.0, sm="sm_70", rep="v100-32",
         cards="V100",
         yes=["FP16 tensor cores"],
         no=["bfloat16", "FP8", "FP4", "FlashAttention-2"],
         note="The last generation without bfloat16. Everything must run in FP16, and "
              "attention falls back to xFormers."),
    dict(key="turing", name="Turing", vendor="nvidia", cc=7.5, sm="sm_75", rep="t4",
         cards="T4, RTX 20-series",
         yes=["FP16", "INT8", "4-bit weight-only (AWQ/GPTQ)"],
         no=["bfloat16", "FP8", "FP4", "FlashAttention-2"],
         note="No bfloat16 and no FlashAttention-2; FlashInfer is the usable attention path."),
    dict(key="ampere", name="Ampere", vendor="nvidia", cc=8.0, sm="sm_80 / sm_86", rep="a100-80",
         cards="A100, A10, A30, A6000, RTX 30-series",
         yes=["bfloat16", "FlashAttention-2", "INT8", "4-bit weight-only", "Marlin kernels"],
         no=["FP8 tensor cores", "FP4", "MXFP4", "FlashAttention-3", "FlashMLA"],
         note="The first bfloat16 generation. FP8 checkpoints load but are dequantized to "
              "BF16, so they take twice the memory they advertise."),
    dict(key="ada", name="Ada Lovelace", vendor="nvidia", cc=8.9, sm="sm_89", rep="l40s",
         cards="L20, L40S, L40, L4, RTX 6000 Ada, RTX 4090",
         yes=["bfloat16", "FP8 (E4M3/E5M2)", "FP8 KV cache", "FlashAttention-2",
              "INT8", "4-bit weight-only"],
         no=["MXFP4 kernels", "NVFP4", "FlashAttention-3", "FlashMLA", "NVLink on most cards"],
         note="Full FP8 tensor cores, but none of the Hopper-and-newer kernels: MXFP4 "
              "checkpoints get dequantized to BF16, MLA models fall back to a much slower "
              "Triton path, and most Ada cards have no NVLink, so tensor parallelism "
              "crosses PCIe."),
    dict(key="hopper", name="Hopper", vendor="nvidia", cc=9.0, sm="sm_90", rep="h100-sxm",
         cards="H100, H200, H20, H800",
         yes=["bfloat16", "FP8", "MXFP4", "FlashAttention-3", "FlashMLA", "NVLink/NVSwitch"],
         no=["NVFP4 tensor cores"],
         note="The widest kernel coverage in practice — nearly everything ships an sm_90 path."),
    dict(key="blackwell", name="Blackwell (data center)", vendor="nvidia", cc=10.0,
         sm="sm_100", rep="b200", cards="B200, B100, GB200",
         yes=["bfloat16", "FP8", "MXFP4", "NVFP4", "CUTLASS MLA", "NVLink 5"],
         no=[],
         note="Newest and fastest, but kernel coverage is version-sensitive — not every "
              "published wheel ships sm_100 builds yet."),
    dict(key="blackwell-rtx", name="Blackwell (RTX / workstation)", vendor="nvidia", cc=12.0,
         sm="sm_120", rep="rtx-pro-6000", cards="RTX PRO 6000 Blackwell, RTX 5090",
         yes=["bfloat16", "FP8", "NVFP4"],
         no=["NVLink", "data-center MLA kernels"],
         note="Shares Blackwell's number formats but is a different compile target from "
              "sm_100, and has no NVLink — multi-GPU goes over PCIe."),
    dict(key="cdna3", name="AMD CDNA 3", vendor="amd", cc=CC_CDNA3, sm="gfx942", rep="mi300x",
         cards="MI300X, MI325X",
         yes=["bfloat16", "FP8", "huge HBM per GPU", "AITER FlashAttention"],
         no=["CUDA-only kernels (FlashInfer, some quant schemes)"],
         note="Needs the ROCm build of the engine; a handful of kernels remain CUDA-only."),
    dict(key="cdna2", name="AMD CDNA 2", vendor="amd", cc=CC_CDNA2, sm="gfx90a", rep="mi250x",
         cards="MI250X",
         yes=["bfloat16"], no=["FP8", "FP4", "AITER kernels"],
         note="No FP8; expect the Triton attention path."),
    dict(key="rdna3", name="AMD RDNA 3", vendor="amd", cc=CC_RDNA3, sm="gfx1100", rep="rx-7900xtx",
         cards="RX 7900 XTX",
         yes=["bfloat16"], no=["FP8", "FP4", "most server kernels"],
         note="Consumer AMD. llama.cpp is the dependable path here."),
    dict(key="metal", name="Apple silicon", vendor="apple", cc=CC_METAL, sm="Metal",
         rep="m4-max-128", cards="M3/M4 Max, M3 Ultra",
         yes=["unified memory", "GGUF via llama.cpp / MLX"],
         no=["vLLM", "SGLang", "CUDA or ROCm kernels"],
         note="Unified memory makes big models loadable, but only llama.cpp-class runtimes exist."),
]

FAMILY_BY_KEY = {f["key"]: f for f in FAMILIES}

# Findings that describe the architecture rather than the size of the box.
ARCH_TAGS = {"cc", "arch", "quant", "kernel", "attention", "vendor", "format", "kv", "os"}
# Of those, the ones that mean "runs, but you give something up".
DEGRADE_TAGS = {"quant", "attention", "kernel"}


def family_for(gpu: dict) -> dict | None:
    """Which family a catalog GPU belongs to."""
    cc, vendor = gpu.get("cc"), gpu.get("vendor")
    best = None
    for f in FAMILIES:
        if f["vendor"] != vendor:
            continue
        if vendor == "nvidia":
            if cc >= f["cc"] and (best is None or f["cc"] > best["cc"]):
                best = f
        elif f["cc"] == cc:
            best = f
    return best


def support_matrix(model: dict, runtime: dict, evaluate) -> list:
    """Per-family verdict for this model, with memory removed from the question.

    Each family is evaluated on a representative card given far more VRAM and
    GPUs than the model needs, so anything left is architectural.
    """
    out = []
    for f in FAMILIES:
        rep = GPU_BY_ID.get(f["rep"])
        if not rep:
            continue
        hw = {
            "gpu_id": f["rep"], "custom_gpu": None,
            "gpus_per_node": 8, "nodes": 4, "host_ram_gb": 4096,
            "os": "macos" if f["vendor"] == "apple" else "linux",
            "interconnect": "nvlink" if rep.get("nvlink") else "pcie",
            "driver_version": "", "vram_override_gb": 640,
        }
        rt = dict(runtime, tp=0, pp=0)
        try:
            r = evaluate(model, hw, rt, _suggest=False)
        except (ValueError, KeyError, ZeroDivisionError) as e:
            out.append({**_row(f), "status": "unknown", "reasons": [str(e)], "caveats": []})
            continue

        arch_f = [x for x in r["findings"] if x.get("tag") in ARCH_TAGS]
        blockers = [x for x in arch_f if x["level"] == "blocker"]
        degrades = [x for x in arch_f if x["level"] == "warn" and x["tag"] in DEGRADE_TAGS]

        # Warnings that do not cost capability (a different wheel, say) are
        # worth saying but should not downgrade the verdict.
        caveats = [x["title"] for x in arch_f
                   if x["level"] == "warn" and x["tag"] not in DEGRADE_TAGS]
        if blockers:
            status, reasons = "no", _dedupe([x["title"] for x in blockers])
        elif degrades:
            status, reasons = "degraded", _dedupe([x["title"] for x in degrades])
        else:
            status, reasons = "yes", []

        out.append({
            **_row(f), "status": status, "reasons": reasons, "caveats": _dedupe(caveats),
            "attention": (r.get("attention") or {}).get("chosen", {}) and
                         (r["attention"]["chosen"] or {}).get("label"),
            "bpw": r["weights"]["bpw"],
            "weights_gib_total": r["weights"]["gib"],
        })
    return out


def _dedupe(xs: list) -> list:
    seen, out = set(), []
    for x in xs:
        if x not in seen:
            seen.add(x); out.append(x)
    return out


def _row(f: dict) -> dict:
    return {"key": f["key"], "name": f["name"], "vendor": f["vendor"], "sm": f["sm"],
            "cc": f["cc"], "cards": f["cards"], "note": f["note"],
            "yes": f["yes"], "no": f["no"]}
