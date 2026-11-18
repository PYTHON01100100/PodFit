"""GPU catalog + dtype/quantization tables.

Bandwidth is theoretical peak (GB/s). TFLOPS figures are DENSE tensor-core
numbers (no 2:1 sparsity marketing multiplier). cc = CUDA compute capability;
for AMD/Apple it is a synthetic sentinel used only for gating.
"""

# cc sentinels for non-CUDA vendors
CC_CDNA2 = -2.0   # MI250X
CC_CDNA3 = -3.0   # MI300X / MI325X
CC_RDNA3 = -4.0   # Radeon 7900
CC_METAL = -9.0   # Apple silicon unified memory

GPUS = [
    # --- NVIDIA Hopper -----------------------------------------------------
    dict(id="h100-sxm", name="H100 SXM5 80GB", vendor="nvidia", arch="Hopper",
         vram_gb=80, bandwidth=3350, fp16=989, fp8=1979, fp4=0, cc=9.0,
         nvlink=900, tdp=700, form="SXM"),
    dict(id="h100-pcie", name="H100 PCIe 80GB", vendor="nvidia", arch="Hopper",
         vram_gb=80, bandwidth=2000, fp16=756, fp8=1513, fp4=0, cc=9.0,
         nvlink=600, tdp=350, form="PCIe"),
    dict(id="h200-sxm", name="H200 SXM5 141GB", vendor="nvidia", arch="Hopper",
         vram_gb=141, bandwidth=4800, fp16=989, fp8=1979, fp4=0, cc=9.0,
         nvlink=900, tdp=700, form="SXM"),
    dict(id="h200-nvl", name="H200 NVL 141GB", vendor="nvidia", arch="Hopper",
         vram_gb=141, bandwidth=4800, fp16=835, fp8=1671, fp4=0, cc=9.0,
         nvlink=900, tdp=600, form="PCIe/NVL"),
    dict(id="h800-sxm", name="H800 SXM5 80GB", vendor="nvidia", arch="Hopper",
         vram_gb=80, bandwidth=3350, fp16=989, fp8=1979, fp4=0, cc=9.0,
         nvlink=400, tdp=700, form="SXM",
         note="China-market H100 with NVLink cut to 400 GB/s."),
    dict(id="h20", name="H20 96GB (HBM3)", vendor="nvidia", arch="Hopper",
         vram_gb=96, bandwidth=4000, fp16=148, fp8=296, fp4=0, cc=9.0,
         nvlink=900, tdp=400, form="SXM",
         note="China-market Hopper. Huge HBM + bandwidth but only ~15% of "
              "H100 tensor FLOPS - memory-bound decode is strong, prefill "
              "and large-batch throughput are compute-limited."),
    dict(id="h20-141", name="H20 141GB (HBM3e)", vendor="nvidia", arch="Hopper",
         vram_gb=141, bandwidth=4800, fp16=148, fp8=296, fp4=0, cc=9.0,
         nvlink=900, tdp=500, form="SXM",
         note="HBM3e refresh of H20. Same weak tensor FLOPS, more capacity."),
    # --- NVIDIA Blackwell --------------------------------------------------
    dict(id="b200", name="B200 SXM 180GB", vendor="nvidia", arch="Blackwell",
         vram_gb=180, bandwidth=8000, fp16=2250, fp8=4500, fp4=9000, cc=10.0,
         nvlink=1800, tdp=1000, form="SXM"),
    dict(id="b100", name="B100 SXM 180GB", vendor="nvidia", arch="Blackwell",
         vram_gb=180, bandwidth=8000, fp16=1800, fp8=3500, fp4=7000, cc=10.0,
         nvlink=1800, tdp=700, form="SXM"),
    dict(id="gb200", name="GB200 (per-GPU 186GB)", vendor="nvidia", arch="Blackwell",
         vram_gb=186, bandwidth=8000, fp16=2500, fp8=5000, fp4=10000, cc=10.0,
         nvlink=1800, tdp=1200, form="Superchip"),
    dict(id="rtx-pro-6000", name="RTX PRO 6000 Blackwell 96GB", vendor="nvidia",
         arch="Blackwell", vram_gb=96, bandwidth=1792, fp16=503, fp8=1007,
         fp4=2014, cc=12.0, nvlink=0, tdp=600, form="PCIe"),
    dict(id="rtx-5090", name="GeForce RTX 5090 32GB", vendor="nvidia",
         arch="Blackwell", vram_gb=32, bandwidth=1792, fp16=210, fp8=419,
         fp4=838, cc=12.0, nvlink=0, tdp=575, form="Consumer"),
    # --- NVIDIA Ada --------------------------------------------------------
    dict(id="l40s", name="L40S 48GB", vendor="nvidia", arch="Ada",
         vram_gb=48, bandwidth=864, fp16=362, fp8=733, fp4=0, cc=8.9,
         nvlink=0, tdp=350, form="PCIe"),
    dict(id="l40", name="L40 48GB", vendor="nvidia", arch="Ada",
         vram_gb=48, bandwidth=864, fp16=181, fp8=362, fp4=0, cc=8.9,
         nvlink=0, tdp=300, form="PCIe"),
    dict(id="l20", name="L20 48GB", vendor="nvidia", arch="Ada",
         vram_gb=48, bandwidth=864, fp16=119.5, fp8=239, fp4=0, cc=8.9,
         nvlink=0, tdp=275, form="PCIe",
         note="China-market Ada (AD102). 48GB GDDR6 at 864 GB/s, no NVLink - "
              "tensor-parallel traffic crosses PCIe Gen4 x16, so TP>4 scales "
              "poorly. FP8 is supported natively (sm89)."),
    dict(id="l4", name="L4 24GB", vendor="nvidia", arch="Ada",
         vram_gb=24, bandwidth=300, fp16=121, fp8=242, fp4=0, cc=8.9,
         nvlink=0, tdp=72, form="PCIe"),
    dict(id="rtx-6000-ada", name="RTX 6000 Ada 48GB", vendor="nvidia", arch="Ada",
         vram_gb=48, bandwidth=960, fp16=364, fp8=728, fp4=0, cc=8.9,
         nvlink=0, tdp=300, form="PCIe"),
    dict(id="rtx-4090", name="GeForce RTX 4090 24GB", vendor="nvidia", arch="Ada",
         vram_gb=24, bandwidth=1008, fp16=165, fp8=330, fp4=0, cc=8.9,
         nvlink=0, tdp=450, form="Consumer"),
    # --- NVIDIA Ampere -----------------------------------------------------
    dict(id="a100-80", name="A100 SXM 80GB", vendor="nvidia", arch="Ampere",
         vram_gb=80, bandwidth=2039, fp16=312, fp8=0, fp4=0, cc=8.0,
         nvlink=600, tdp=400, form="SXM"),
    dict(id="a100-40", name="A100 SXM 40GB", vendor="nvidia", arch="Ampere",
         vram_gb=40, bandwidth=1555, fp16=312, fp8=0, fp4=0, cc=8.0,
         nvlink=600, tdp=400, form="SXM"),
    dict(id="a800-80", name="A800 80GB", vendor="nvidia", arch="Ampere",
         vram_gb=80, bandwidth=2039, fp16=312, fp8=0, fp4=0, cc=8.0,
         nvlink=400, tdp=400, form="SXM"),
    dict(id="a6000", name="RTX A6000 48GB", vendor="nvidia", arch="Ampere",
         vram_gb=48, bandwidth=768, fp16=155, fp8=0, fp4=0, cc=8.6,
         nvlink=112, tdp=300, form="PCIe"),
    dict(id="a10", name="A10 24GB", vendor="nvidia", arch="Ampere",
         vram_gb=24, bandwidth=600, fp16=125, fp8=0, fp4=0, cc=8.6,
         nvlink=0, tdp=150, form="PCIe"),
    dict(id="a30", name="A30 24GB", vendor="nvidia", arch="Ampere",
         vram_gb=24, bandwidth=933, fp16=165, fp8=0, fp4=0, cc=8.0,
         nvlink=200, tdp=165, form="PCIe"),
    dict(id="rtx-3090", name="GeForce RTX 3090 24GB", vendor="nvidia", arch="Ampere",
         vram_gb=24, bandwidth=936, fp16=71, fp8=0, fp4=0, cc=8.6,
         nvlink=112, tdp=350, form="Consumer"),
    # --- NVIDIA older ------------------------------------------------------
    dict(id="v100-32", name="V100 SXM2 32GB", vendor="nvidia", arch="Volta",
         vram_gb=32, bandwidth=900, fp16=125, fp8=0, fp4=0, cc=7.0,
         nvlink=300, tdp=300, form="SXM",
         note="No bfloat16 support - everything must run in fp16."),
    dict(id="t4", name="Tesla T4 16GB", vendor="nvidia", arch="Turing",
         vram_gb=16, bandwidth=320, fp16=65, fp8=0, fp4=0, cc=7.5,
         nvlink=0, tdp=70, form="PCIe",
         note="No bfloat16, no FlashAttention-2."),
    # --- AMD ---------------------------------------------------------------
    dict(id="mi300x", name="AMD MI300X 192GB", vendor="amd", arch="CDNA3",
         vram_gb=192, bandwidth=5300, fp16=1307, fp8=2614, fp4=0, cc=CC_CDNA3,
         nvlink=896, tdp=750, form="OAM"),
    dict(id="mi325x", name="AMD MI325X 256GB", vendor="amd", arch="CDNA3",
         vram_gb=256, bandwidth=6000, fp16=1307, fp8=2614, fp4=0, cc=CC_CDNA3,
         nvlink=896, tdp=1000, form="OAM"),
    dict(id="mi250x", name="AMD MI250X 128GB", vendor="amd", arch="CDNA2",
         vram_gb=128, bandwidth=3277, fp16=383, fp8=0, fp4=0, cc=CC_CDNA2,
         nvlink=800, tdp=560, form="OAM"),
    dict(id="rx-7900xtx", name="Radeon RX 7900 XTX 24GB", vendor="amd", arch="RDNA3",
         vram_gb=24, bandwidth=960, fp16=123, fp8=0, fp4=0, cc=CC_RDNA3,
         nvlink=0, tdp=355, form="Consumer"),
    # --- Apple (llama.cpp / MLX only) --------------------------------------
    dict(id="m4-max-128", name="Apple M4 Max 128GB unified", vendor="apple",
         arch="Metal", vram_gb=96, bandwidth=546, fp16=34, fp8=0, fp4=0,
         cc=CC_METAL, nvlink=0, tdp=90, form="Unified",
         note="Unified memory; ~75% is addressable by the GPU by default."),
    dict(id="m3-ultra-512", name="Apple M3 Ultra 512GB unified", vendor="apple",
         arch="Metal", vram_gb=384, bandwidth=819, fp16=57, fp8=0, fp4=0,
         cc=CC_METAL, nvlink=0, tdp=140, form="Unified",
         note="Unified memory; ~75% is addressable by the GPU by default."),
]

GPU_BY_ID = {g["id"]: g for g in GPUS}

# ---------------------------------------------------------------------------
# Weight formats. bpw = effective bits per weight including quant metadata
# (group scales / zero points), which is why 4-bit schemes are not exactly 4.
# ---------------------------------------------------------------------------
WEIGHT_FORMATS = {
    "fp32":     dict(label="FP32",              bpw=32.0, min_cc=7.0,  kind="native"),
    "fp16":     dict(label="FP16",              bpw=16.0, min_cc=7.0,  kind="native"),
    "bf16":     dict(label="BF16",              bpw=16.0, min_cc=8.0,  kind="native"),
    "fp8":      dict(label="FP8 (W8A8, E4M3)",  bpw=8.0,  min_cc=8.9,  kind="fp8"),
    "int8":     dict(label="INT8 (W8A8)",       bpw=8.0,  min_cc=7.5,  kind="int8"),
    "awq":      dict(label="AWQ 4-bit",         bpw=4.25, min_cc=7.5,  kind="int4"),
    "gptq4":    dict(label="GPTQ 4-bit",        bpw=4.25, min_cc=7.5,  kind="int4"),
    "gptq8":    dict(label="GPTQ 8-bit",        bpw=8.25, min_cc=7.5,  kind="int8"),
    "bnb4":     dict(label="bitsandbytes NF4",  bpw=4.5,  min_cc=7.5,  kind="int4"),
    "mxfp4":    dict(label="MXFP4",             bpw=4.25, min_cc=9.0,  kind="fp4"),
    "nvfp4":    dict(label="NVFP4",             bpw=4.25, min_cc=10.0, kind="fp4"),
    # GGUF ladder - bpw values are llama.cpp measured averages over Llama-class
    # models; mixed-precision K-quants keep some tensors at higher precision.
    "gguf_q8_0":   dict(label="GGUF Q8_0",   bpw=8.50, min_cc=0.0, kind="gguf"),
    "gguf_q6_k":   dict(label="GGUF Q6_K",   bpw=6.56, min_cc=0.0, kind="gguf"),
    "gguf_q5_k_m": dict(label="GGUF Q5_K_M", bpw=5.68, min_cc=0.0, kind="gguf"),
    "gguf_q4_k_m": dict(label="GGUF Q4_K_M", bpw=4.85, min_cc=0.0, kind="gguf"),
    "gguf_q4_0":   dict(label="GGUF Q4_0",   bpw=4.55, min_cc=0.0, kind="gguf"),
    "gguf_iq4_xs": dict(label="GGUF IQ4_XS", bpw=4.25, min_cc=0.0, kind="gguf"),
    "gguf_q3_k_m": dict(label="GGUF Q3_K_M", bpw=3.91, min_cc=0.0, kind="gguf"),
    "gguf_iq3_xs": dict(label="GGUF IQ3_XS", bpw=3.30, min_cc=0.0, kind="gguf"),
    "gguf_q2_k":   dict(label="GGUF Q2_K",   bpw=3.35, min_cc=0.0, kind="gguf"),
}

KV_DTYPES = {
    "auto": dict(label="auto (model dtype)", bytes=2.0, min_cc=7.0),
    "fp16": dict(label="FP16",               bytes=2.0, min_cc=7.0),
    "bf16": dict(label="BF16",               bytes=2.0, min_cc=8.0),
    "fp8":  dict(label="FP8 E4M3",           bytes=1.0, min_cc=8.9),
    "int8": dict(label="INT8",               bytes=1.0, min_cc=7.5),
    "q4":   dict(label="4-bit (llama.cpp q4_0)", bytes=0.5, min_cc=0.0),
}

ENGINES = {
    "vllm":     dict(label="vLLM", os=["linux"], weights=["fp16","bf16","fp32","fp8","int8","awq","gptq4","gptq8","bnb4","mxfp4","nvfp4"]),
    "sglang":   dict(label="SGLang", os=["linux"], weights=["fp16","bf16","fp8","int8","awq","gptq4","gptq8","mxfp4","nvfp4"]),
    "llamacpp": dict(label="llama.cpp", os=["linux","windows","macos"], weights=[k for k in WEIGHT_FORMATS if k.startswith("gguf")] + ["fp16"]),
    "ollama":   dict(label="Ollama", os=["linux","windows","macos"], weights=[k for k in WEIGHT_FORMATS if k.startswith("gguf")]),
}
