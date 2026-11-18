"""Model-architecture support matrix.

Keyed by config.json `model_type`. For each family we record which engines can
serve it, the traits that drive memory/kernel behaviour, and the minimum CUDA
compute capability the required kernels need. `y` = supported, `p` = partial /
caveats apply, `n` = not supported, `?` = unknown to this table.
"""

# traits: moe, mla, swa (sliding window), ssm (mamba/hybrid), vision, audio,
#         encdec (encoder-decoder), embed (embedding/rerank only)
A = lambda family, vllm="y", sglang="y", llamacpp="n", traits=(), min_cc=8.0, note="": dict(
    family=family, vllm=vllm, sglang=sglang, llamacpp=llamacpp,
    traits=list(traits), min_cc=min_cc, note=note)

ARCH = {
    # ---- dense text ------------------------------------------------------
    "llama":        A("Llama / Llama 2-3.x", llamacpp="y", min_cc=7.0),
    "mistral":      A("Mistral", llamacpp="y", traits=["swa"], min_cc=7.0),
    "qwen2":        A("Qwen2 / Qwen2.5", llamacpp="y", min_cc=7.0),
    "qwen3":        A("Qwen3 dense", llamacpp="y", min_cc=7.0),
    "gemma":        A("Gemma", llamacpp="y", min_cc=7.0),
    "gemma2":       A("Gemma 2", llamacpp="y", traits=["swa"], min_cc=8.0,
                      note="Interleaved sliding-window attention; needs a backend with SWA support (FlashAttention or FlashInfer)."),
    "gemma3_text":  A("Gemma 3 (text)", llamacpp="y", traits=["swa"], min_cc=8.0),
    "phi3":         A("Phi-3 / Phi-3.5", llamacpp="y", traits=["swa"], min_cc=7.0),
    "phi4":         A("Phi-4", llamacpp="y", min_cc=7.0),
    "cohere":       A("Command-R", llamacpp="y", min_cc=7.0),
    "cohere2":      A("Command-R7B / Command-A", llamacpp="y", traits=["swa"], min_cc=8.0),
    "glm4":         A("GLM-4", llamacpp="y", min_cc=7.0),
    "internlm2":    A("InternLM2", llamacpp="y", min_cc=7.0),
    "baichuan":     A("Baichuan", llamacpp="p", min_cc=7.0),
    "chatglm":      A("ChatGLM3", llamacpp="y", min_cc=7.0),
    "yi":           A("Yi", llamacpp="y", min_cc=7.0),
    "stablelm":     A("StableLM", llamacpp="y", min_cc=7.0),
    "starcoder2":   A("StarCoder2", llamacpp="y", min_cc=7.0),
    "gpt_bigcode":  A("StarCoder / SantaCoder", llamacpp="y", min_cc=7.0),
    "gpt_neox":     A("GPT-NeoX / Pythia", llamacpp="y", min_cc=7.0),
    "gptj":         A("GPT-J", llamacpp="y", min_cc=7.0),
    "opt":          A("OPT", llamacpp="n", min_cc=7.0),
    "mpt":          A("MPT", llamacpp="y", min_cc=7.0),
    "falcon":       A("Falcon", llamacpp="y", min_cc=7.0),
    "olmo":         A("OLMo", llamacpp="y", min_cc=7.0),
    "olmo2":        A("OLMo 2", llamacpp="y", min_cc=7.0),
    "granite":      A("IBM Granite", llamacpp="y", min_cc=7.0),
    "exaone":       A("EXAONE", llamacpp="y", min_cc=7.0),
    "minicpm":      A("MiniCPM", llamacpp="y", min_cc=7.0),
    "minicpm3":     A("MiniCPM3", llamacpp="p", min_cc=7.0),
    "nemotron":     A("Nemotron", llamacpp="p", min_cc=8.0),
    "smollm3":      A("SmolLM3", llamacpp="y", min_cc=7.0),
    "seed_oss":     A("Seed-OSS", llamacpp="p", min_cc=8.0),
    "apertus":      A("Apertus", llamacpp="n", min_cc=8.0),
    "ernie4_5":     A("ERNIE 4.5 dense", llamacpp="p", min_cc=8.0),
    "hunyuan_v1_dense": A("Hunyuan dense", llamacpp="p", min_cc=8.0),
    "lfm2":         A("LFM2", sglang="p", llamacpp="y", min_cc=8.0),

    # ---- MoE -------------------------------------------------------------
    "mixtral":      A("Mixtral MoE", llamacpp="y", traits=["moe","swa"], min_cc=8.0),
    "qwen2_moe":    A("Qwen2 MoE", llamacpp="y", traits=["moe"], min_cc=8.0),
    "qwen3_moe":    A("Qwen3 MoE", llamacpp="y", traits=["moe"], min_cc=8.0),
    "qwen3_next":   A("Qwen3-Next (hybrid MoE)", sglang="y", llamacpp="p",
                      traits=["moe","ssm"], min_cc=8.0,
                      note="Gated DeltaNet linear attention + MoE. Needs a recent vLLM/SGLang with the hybrid mamba cache; the state cache is allocated per sequence and does not shrink with context."),
    "phimoe":       A("Phi-3.5-MoE", llamacpp="p", traits=["moe"], min_cc=8.0),
    "dbrx":         A("DBRX", llamacpp="p", traits=["moe"], min_cc=8.0),
    "arctic":       A("Snowflake Arctic", llamacpp="n", traits=["moe"], min_cc=8.0),
    "olmoe":        A("OLMoE", llamacpp="y", traits=["moe"], min_cc=8.0),
    "granitemoe":   A("Granite MoE", llamacpp="y", traits=["moe"], min_cc=8.0),
    "glm4_moe":     A("GLM-4.5 / GLM-4.6 MoE", llamacpp="y", traits=["moe"], min_cc=8.0),
    "minimax_text_01": A("MiniMax-Text-01", sglang="p", llamacpp="n",
                      traits=["moe","ssm"], min_cc=8.0),
    "hunyuan_v1_moe": A("Hunyuan MoE", llamacpp="p", traits=["moe"], min_cc=8.0),
    "ernie4_5_moe": A("ERNIE 4.5 MoE", llamacpp="p", traits=["moe"], min_cc=8.0),
    "gpt_oss":      A("GPT-OSS (20B/120B)", llamacpp="y", traits=["moe","swa"], min_cc=8.0,
                      note="Ships pre-quantized MXFP4 MoE weights. Native MXFP4 kernels need Hopper/Blackwell (sm90+); on Ada and older the experts are upcast to bf16 at load time, roughly quadrupling weight memory."),
    "deepseek_v2":  A("DeepSeek-V2", llamacpp="y", traits=["moe","mla"], min_cc=8.0),
    "deepseek_v3":  A("DeepSeek-V3 / R1 / V3.x", llamacpp="y", traits=["moe","mla"], min_cc=8.0,
                      note="Multi-head Latent Attention: the KV cache is a single compressed latent per token, so it is roughly an order of magnitude smaller than an equivalent GQA model. Official weights are FP8 block-scaled."),
    "kimi_k2":      A("Kimi K2", llamacpp="p", traits=["moe","mla"], min_cc=8.0),
    "deepseek_v4":  A("DeepSeek-V4 (Flash / Pro)", vllm="p", sglang="p", llamacpp="p",
                      traits=["moe","mla","sparse"], min_cc=9.0,
                      note="Sparse MLA with a top-k indexer. The CUDA implementation routes "
                           "everything that is not sm_120 to FlashMLA, which is Hopper and "
                           "data-center Blackwell only, and ships no Triton sparse fallback — so "
                           "Ada, Ampere and older cannot serve it at any size. Weights are "
                           "block-scaled FP8 with UE8M0 scales, which carry the same assumption."),
    "deepseek_v32": A("DeepSeek-V3.2 (sparse)", vllm="p", sglang="p", llamacpp="p",
                      traits=["moe","mla","sparse"], min_cc=9.0,
                      note="Adds a sparse indexer on top of V3's MLA; the sparse kernels are "
                           "much less portable than V3's dense MLA path."),
    "longcat_flash":A("LongCat-Flash", sglang="p", llamacpp="n", traits=["moe"], min_cc=8.0),

    # ---- hybrid / state-space -------------------------------------------
    "mamba":        A("Mamba", sglang="p", llamacpp="p", traits=["ssm"], min_cc=8.0),
    "mamba2":       A("Mamba-2", sglang="p", llamacpp="p", traits=["ssm"], min_cc=8.0),
    "jamba":        A("Jamba", sglang="p", llamacpp="n", traits=["moe","ssm"], min_cc=8.0),
    "bamba":        A("Bamba", sglang="p", llamacpp="n", traits=["ssm"], min_cc=8.0),
    "zamba2":       A("Zamba2", sglang="p", llamacpp="n", traits=["ssm"], min_cc=8.0),
    "nemotron_h":   A("Nemotron-H hybrid", sglang="p", llamacpp="p", traits=["ssm"], min_cc=8.0),
    "falcon_h1":    A("Falcon-H1", sglang="p", llamacpp="y", traits=["ssm"], min_cc=8.0),
    "granitemoehybrid": A("Granite 4 hybrid", sglang="p", llamacpp="y", traits=["moe","ssm"], min_cc=8.0),

    # ---- vision-language -------------------------------------------------
    "llava":            A("LLaVA", llamacpp="y", traits=["vision"], min_cc=7.0),
    "llava_next":       A("LLaVA-NeXT", llamacpp="y", traits=["vision"], min_cc=7.0),
    "llava_onevision":  A("LLaVA-OneVision", llamacpp="p", traits=["vision"], min_cc=8.0),
    "qwen2_vl":         A("Qwen2-VL", llamacpp="y", traits=["vision","swa"], min_cc=8.0),
    "qwen2_5_vl":       A("Qwen2.5-VL", llamacpp="y", traits=["vision","swa"], min_cc=8.0),
    "qwen3_vl":         A("Qwen3-VL", llamacpp="y", traits=["vision"], min_cc=8.0),
    "qwen3_vl_moe":     A("Qwen3-VL MoE", llamacpp="p", traits=["vision","moe"], min_cc=8.0),
    "gemma3":           A("Gemma 3 (multimodal)", llamacpp="y", traits=["vision","swa"], min_cc=8.0),
    "mllama":           A("Llama 3.2 Vision", sglang="p", llamacpp="n", traits=["vision"], min_cc=8.0,
                          note="Cross-attention vision architecture; vLLM allocates a separate encoder cache and does not support chunked prefill for it."),
    "llama4":           A("Llama 4 (Scout/Maverick)", llamacpp="y", traits=["vision","moe"], min_cc=8.0),
    "internvl_chat":    A("InternVL", llamacpp="p", traits=["vision"], min_cc=8.0),
    "minicpmv":         A("MiniCPM-V", llamacpp="y", traits=["vision"], min_cc=8.0),
    "phi3_v":           A("Phi-3-Vision", llamacpp="p", traits=["vision"], min_cc=8.0),
    "phi4_multimodal":  A("Phi-4-multimodal", sglang="p", llamacpp="p", traits=["vision","audio"], min_cc=8.0),
    "pixtral":          A("Pixtral", llamacpp="y", traits=["vision"], min_cc=8.0),
    "idefics3":         A("Idefics3", llamacpp="p", traits=["vision"], min_cc=8.0),
    "smolvlm":          A("SmolVLM", llamacpp="y", traits=["vision"], min_cc=8.0),
    "molmo":            A("Molmo", sglang="p", llamacpp="n", traits=["vision"], min_cc=8.0),
    "aria":             A("Aria", sglang="p", llamacpp="n", traits=["vision","moe"], min_cc=8.0),
    "deepseek_vl_v2":   A("DeepSeek-VL2", llamacpp="p", traits=["vision","moe"], min_cc=8.0),
    "glm4v":            A("GLM-4V", llamacpp="p", traits=["vision"], min_cc=8.0),
    "kimi_vl":          A("Kimi-VL", sglang="p", llamacpp="n", traits=["vision","moe"], min_cc=8.0),
    "ovis":             A("Ovis", sglang="p", llamacpp="p", traits=["vision"], min_cc=8.0),
    "step3_vl":         A("Step3-VL", sglang="p", llamacpp="n", traits=["vision"], min_cc=8.0),
    "chameleon":        A("Chameleon", sglang="p", llamacpp="n", traits=["vision"], min_cc=8.0),
    "fuyu":             A("Fuyu", sglang="n", llamacpp="n", traits=["vision"], min_cc=8.0),
    "mistral3":         A("Mistral Small 3.x multimodal", llamacpp="y", traits=["vision"], min_cc=8.0),

    # ---- audio -----------------------------------------------------------
    "whisper":      A("Whisper", vllm="p", sglang="p", llamacpp="y",
                      traits=["audio","encdec"], min_cc=7.0,
                      note="Encoder-decoder ASR. vLLM serves it through the transcription API, not chat completions; whisper.cpp is the usual local path."),
    "ultravox":     A("Ultravox", sglang="p", llamacpp="n", traits=["audio"], min_cc=8.0),
    "voxtral":      A("Voxtral", sglang="p", llamacpp="y", traits=["audio"], min_cc=8.0),
    "qwen2_audio":  A("Qwen2-Audio", sglang="p", llamacpp="p", traits=["audio"], min_cc=8.0),
    "granite_speech": A("Granite Speech", sglang="p", llamacpp="n", traits=["audio"], min_cc=8.0),

    # ---- embedding / reranking ------------------------------------------
    "bert":         A("BERT encoder", llamacpp="y", traits=["embed"], min_cc=7.0),
    "roberta":      A("RoBERTa encoder", llamacpp="y", traits=["embed"], min_cc=7.0),
    "xlm-roberta":  A("XLM-RoBERTa", llamacpp="y", traits=["embed"], min_cc=7.0),
    "modernbert":   A("ModernBERT", llamacpp="p", traits=["embed"], min_cc=8.0),
    "nomic_bert":   A("Nomic BERT", llamacpp="y", traits=["embed"], min_cc=7.0),
    "gte":          A("GTE encoder", llamacpp="y", traits=["embed"], min_cc=7.0),

    # ---- encoder-decoder -------------------------------------------------
    "t5":           A("T5 / FLAN-T5", vllm="p", sglang="n", llamacpp="p", traits=["encdec"], min_cc=7.0),
    "bart":         A("BART", vllm="p", sglang="n", llamacpp="n", traits=["encdec"], min_cc=7.0),
}

# Architectures that are not autoregressive LM serving at all.
NON_LM = {
    "stable_diffusion", "sd3", "flux", "unet", "vae", "clip", "clip_vision_model",
    "siglip", "siglip_vision_model", "dinov2", "vit", "wav2vec2", "hubert",
    "sam", "detr", "yolos", "swin", "resnet", "convnext", "esm", "musicgen",
    "speecht5", "vits", "parler_tts",
}

STATUS_LABEL = {"y": "Supported", "p": "Partial", "n": "Not supported", "?": "Unknown"}


def lookup(model_type: str, architectures=None) -> dict:
    """Resolve a config.json model_type to a support entry."""
    mt = (model_type or "").lower()
    if mt in ARCH:
        return dict(ARCH[mt], key=mt, known=True)
    if mt in NON_LM:
        return dict(A(f"{model_type} (not a text-generation model)",
                      vllm="n", sglang="n", llamacpp="n", min_cc=0.0,
                      note="This is not an autoregressive language model, so LLM "
                           "serving engines do not apply. Size it as a plain "
                           "PyTorch workload instead."),
                    key=mt, known=True, non_lm=True)
    # Fall back to substring matching on architectures / model_type, so
    # community fine-tunes with novel model_type strings still resolve.
    hay = " ".join([mt] + [a.lower() for a in (architectures or [])])
    for key, entry in ARCH.items():
        if key in hay:
            return dict(entry, key=key, known=False, inferred_from=key)
    if "forcausallm" in hay.replace("_", ""):
        return dict(A("Unrecognised causal LM", vllm="?", sglang="?", llamacpp="?",
                      min_cc=8.0,
                      note="Not in the built-in support table. It exposes a CausalLM "
                           "head, so a Llama-style engine path is likely, but confirm "
                           "against the engine's supported-models list before relying on it."),
                    key=mt or "unknown", known=False)
    return dict(A("Unknown architecture", vllm="?", sglang="?", llamacpp="?", min_cc=8.0,
                  note="No config.json model_type matched the support table."),
                key=mt or "unknown", known=False)
