# podfit

A web app that answers one question: **can this model run on my hardware, and how well?**

Point it at any model on HuggingFace, describe the GPUs you have (or pick your
saved rig), choose an engine and quantization, and it computes the actual memory
breakdown — weights, KV cache, activations, CUDA graphs, framework overhead —
tells you what fits, how much context and concurrency you get, roughly how fast
it will run, what will block you, and the exact command to launch it.

The interface is answer-first. The whole configuration collapses to a single
line (`model › GPUs › engine · weights › context`) that you click to open;
the verdict, the headline numbers and the memory breakdown are the page.
Findings are triaged — blockers and warnings in full, informational notes folded
into one counted row — and the attention backend, launch command and GPU
comparison sit in disclosure rows you open when you want them. When something
does not fit, **What would make it fit** offers concrete alternatives, each one
a real re-evaluation against your hardware rather than a rule of thumb: more
nodes, a smaller weight format, an FP8 KV cache, less context, fewer streams.

## What it covers

**Model data** comes from the Hub at request time, never from a bundled list:
search across every model type (LLM, VLM, audio, embedding, reranker, seq2seq),
real parameter counts from safetensors metadata, and the actual `config.json` —
layers, hidden size, attention/KV heads, head dim, vocab, native context, RoPE
scaling, sliding-window layout, MoE expert counts, MLA latent ranks, vision and
audio towers, and any `quantization_config` the repo ships with. GGUF repos are
sized from real file sizes, not estimates.

**Hardware**: a catalog of 33 accelerators (Hopper incl. H20 and H800, Blackwell,
Ada incl. L20 and L40S, Ampere, Volta/Turing, AMD CDNA, Apple unified memory),
plus a form for anything not listed, plus saved rigs. This host's GPUs are
detected and seeded automatically as a rig.

**Engines**: vLLM and SGLang for serving, llama.cpp and Ollama for local GGUF
(including partial GPU offload with CPU spillover).

**GPU architecture families**: Volta through Blackwell, plus CDNA and Apple
silicon — each with what it can and cannot do, and a per-model matrix showing
which families can serve the model at all, with memory taken out of the
question. Your card's family is named everywhere the GPU is (`8 × L20 48GB
(Ada Lovelace)`), because "sm_89" is not how anyone thinks about their fleet.

**Attention backends**: a model can pass every memory check and still fail to
start because no attention kernel on that GPU provides what it needs. podfit
picks the backend the engine would actually choose and shows why the others were
rejected — FlashAttention-2 needs sm80, xFormers has no sliding-window path, an
FP8 KV cache needs sm89, and MLA models need FlashMLA (Hopper) or CUTLASS MLA
(Blackwell), falling back to a much slower Triton MLA below that. You can also
pin a specific backend to reproduce a known setup.

**Architecture support** (`Architecture support` button): a table of ~100 model
families — which of the three engines serves each one, the minimum CUDA compute
capability its kernels need, and the traits that drive memory (`moe`, `mla`,
`swa`, `ssm`, `vision`). This is what catches "the model loads on an H100 but
not on this card" before you spend an hour finding out.

**Intelligence scores** (`Best models for my hardware`): benchmark scores so you
can pick the strongest model that actually fits. Two sources, both real — nothing
is estimated or invented, and a model with no published score says so:

| Source | Key needed | Coverage |
|---|---|---|
| Artificial Analysis Intelligence Index | `AA_API_KEY` | current models; used first when configured |
| HF Open LLM Leaderboard v2 | none | broad, but archived in 2025 — newer models have no entry |

Scores are inherited onto quantized mirrors (`unsloth/…-GGUF`, `RedHatAI/…-w4a16`)
from the base model they declare, and labelled as inherited.

**Artificial Analysis as a source, not just a score.** The Hub's trending and
download lists rank by attention, which is a poor proxy for capability — a model
can be excellent and simply not be trending. So both the model picker and *Best
for my hardware* can take their candidates from AA's intelligence ranking
instead, strongest first, and only then ask what the hardware can hold.

AA publishes no HuggingFace id and no open-weights flag, so a model is resolved
by searching the Hub for its name and accepting a repo only when **both** hold:
the org is one that creator actually publishes under, and the repo ships
safetensors weights. Without the org check, proprietary models resolve to
squatters — "GPT-6 Astra" happily matches `gptforfree/GPT-6-Astra`, which
contains no weights at all. Anything failing those tests is reported as API-only
rather than guessed at; of the top 168 ranked entries, 125 are currently
API-only.

## Run it

```bash
./detect-hardware.sh        # writes data/detected_hardware.json from nvidia-smi
cp .env.example .env        # optional: HF_TOKEN, AA_API_KEY
docker compose up -d --build
curl -s localhost:8510/api/health
```

`data/` holds the SQLite database (saved rigs + the Hub/benchmark cache) and
survives rebuilds. Re-run `detect-hardware.sh` after a hardware change; it only
seeds a rig the app has never seen, so your edits are never clobbered.

### Optional environment

- `HF_TOKEN` — required for gated repos (Llama, Gemma, some Mistral). Without it
  those models report "gated" instead of a size.
- `AA_API_KEY` — Artificial Analysis key, for intelligence scores on models newer
  than the archived HF leaderboard.

Set both with:

```bash
./set-keys.sh
```

It prompts without echoing, so neither key lands in your shell history or in a
process listing; blank input keeps the current value. It then recreates the
container (an env change needs a recreate, not a restart) and prints whether each
key is live. Editing `.env` by hand works too — follow it with
`docker compose up -d --force-recreate`.

### nginx

`nginx/podfit.vhost.conf` is a template reverse-proxy block for `127.0.0.1:8510`.
Fill in your own domain and certificate paths, then append it to your site config
and reload:

```bash
cat nginx/podfit.vhost.conf | sudo tee -a /etc/nginx/sites-available/your-site.conf
sudo nginx -t && sudo systemctl reload nginx
```

It expects a `$connection_upgrade` map to already be defined elsewhere in your
nginx config.

## How the numbers are produced

Per GPU, for vLLM/SGLang:

```
budget      = VRAM × gpu_memory_utilization
weights     = params × bits_per_weight / 8 / (TP × PP)
              (embeddings and LM head stay 16-bit under 4/8-bit schemes)
overhead    = CUDA context + NCCL buffers, scaled by TP
activations = max_num_batched_tokens through the MLP/attention working set + logits
cuda graphs = 0 when enforce_eager, else ~0.4–2 GiB
KV cache    = whatever is left  →  max context and max concurrency
```

KV per token per layer is `2 × kv_heads/TP × head_dim × kv_dtype_bytes`, except
for MLA models (DeepSeek-style), where it is one compressed latent per token that
TP **replicates** rather than shards. Sliding-window layers are capped at the
window instead of the full context. Encoder-only models get no KV cache at all;
encoder-decoder models cache only the decoder side.

Speed figures are a roofline, not a benchmark: decode from memory bandwidth at
80% efficiency, prefill from dense tensor FLOPS at 45% MFU, using the tensor-core
path the chosen format actually runs on. They are an upper bound.

Some deliberate honesty about the model: a pre-quantized repo that lands on a GPU
without the matching tensor cores (MXFP4 on Ada, FP8 on Ampere) is sized as the
BF16 it gets dequantized into, which is usually the surprise that breaks a
deployment.

**Sparse attention is a hard architectural gate, not a slow path.** A config that
declares an indexer (`index_topk`, `index_n_heads` — DeepSeek V3.2 and V4) needs
sparse-MLA kernels, and in vLLM 0.27.1 every one of them gates on compute
capability *major*: `FlashMLASparse` accepts 9 and 10, `FlashAttnMLASparse`
accepts 9, the FlashInfer DSv4 path accepts 12. Nothing accepts 8, and there is
no Triton sparse fallback — so Ampere and Ada cannot serve those models at any
size, which is not something a memory calculation would ever reveal. Dense MLA
(V2, V3, R1) is unaffected: it keeps its Triton fallback. Kernel coverage moves
between releases, so re-check on a version bump.

## Hardware reference

### Interconnects: NVLink, RDMA, PCIe, Ethernet

What links the GPUs together changes how far tensor parallelism (TP) can
scale, and podfit's `interconnect` field (Setup → Hardware → the advanced
disclosure) drives the warnings and blockers it raises about that:

- **NVLink / NVSwitch** — NVIDIA's proprietary GPU-to-GPU fabric, the
  fastest option and the one TP really wants. Bandwidth depends on the SKU:
  900 GB/s on H100/H200 SXM, 1.8 TB/s on Blackwell SXM (B100/B200/GB200).
  Many PCIe-form-factor cards (L40S, L4, RTX 6000 Ada, the RTX 50-series)
  ship with none at all — TP on those falls back to PCIe and degrades hard
  past a few GPUs.
- **RDMA (InfiniBand / RoCE, with GPUDirect RDMA)** — the standard way to
  extend TP *across nodes*: direct GPU-to-GPU memory access over the
  network, bypassing the CPU. Needs 200 Gb/s+ to be worth it; below that,
  prefer pipeline parallelism (PP) across nodes instead, since PP only
  ships one activation tensor per stage boundary rather than an all-reduce
  every layer.
- **RDMA over Thunderbolt 5 (Apple Silicon)** — a newer entry in the same
  dropdown, for clustering multiple Mac Studios rather than NVIDIA GPUs.
  Apple's own MLX research uses exactly this transport for distributed
  inference across machines. See the Apple Silicon section below.
- **PCIe only / Ethernet** — no dedicated GPU fabric. PCIe-only is fine
  inside one box for PP; plain Ethernet is viable for PP across nodes but
  not for TP, which podfit flags as a blocker if you try it.

### Apple Silicon: unified memory, UltraFusion, and clustering

podfit treats Apple Silicon as its own vendor family ("Metal", a synthetic
compute-capability sentinel of `-9.0`), gated to llama.cpp and MLX only —
vLLM and SGLang are CUDA/ROCm-only and will never run on it.

- **Unified memory.** CPU, GPU and Neural Engine share one memory pool;
  by convention (and by default in every Apple catalog entry) ~75% of it
  is treated as GPU-addressable, matching what llama.cpp/MLX actually get
  to use.
- **Max vs. Ultra.** A Max chip is a single die. An Ultra chip fuses two
  Max dies together over Apple's UltraFusion interposer — roughly double
  the CPU/GPU/Neural Engine cores and double the memory bandwidth of the
  Max it's built from, not a multi-GPU setup in the NVLink sense.
  M5 Ultra is the first Ultra built this way with per-core Neural
  Accelerators (matrix-multiply units) in its GPU, which Apple says
  speeds up prefill far more than decode — decode stays bandwidth-bound
  either way.
- **Clustering multiple Macs.** For models too large for even a single
  512 GB M3/M5 Ultra, llama.cpp's RPC backend or MLX's distributed
  runtime can pool several machines over Thunderbolt 5 (RDMA) or plain
  Ethernet. Set `nodes` > 1 in podfit to model this: memory pools across
  machines (layer-split, same as multi-GPU llama.cpp), but the speed
  figures still assume each layer's compute stays within *one* chip's own
  bandwidth — the inter-machine transfer time itself isn't simulated,
  which is exactly what the finding podfit raises in this situation
  (`"N separate machines, not one box"`) says.

### NVIDIA: compute capability, architecture families, and the L20

Every NVIDIA kernel-availability gate in podfit (FlashAttention-2, FP8,
sparse-MLA, and so on) keys off **CUDA compute capability** (`sm_XX`), not
the marketing name. Roughly, oldest to newest: Volta (`sm_70`) → Turing
(`sm_75`) → Ampere (`sm_80`/`sm_86`) → Ada (`sm_89`) → Hopper (`sm_90`) →
Blackwell (`sm_100` datacenter / `sm_120` RTX-class). NVLink availability
is a function of SKU and form factor, not architecture alone — SXM parts
get it, most PCIe and all consumer cards don't.

**The L20** is an Ada-generation (AD102) card built specifically for the
China market after US export controls cut off H100/A100-class chips
there: 48 GB GDDR6 at 864 GB/s, FP8-capable (`sm_89`) but roughly a
quarter of an H100's dense tensor FLOPS, and **no NVLink** — tensor-
parallel traffic crosses PCIe Gen4 x16, so TP > 4 scales poorly on it.
It has compliance-driven siblings in the catalog with the same story:
H20/H20-141 (Hopper-class HBM bandwidth, but tensor FLOPS cut to a
fraction of H100's), H800 (an H100 with NVLink capped at 400 GB/s instead
of 900), A800 (an A100 with NVLink capped at 400 GB/s). podfit sizes all
of these against their real, cut-down numbers — not the uncapped chip's
spec sheet.

### Renting GPUs: the cloud landscape

- **Hyperscalers** (AWS, Azure, GCP, Oracle Cloud) — the broadest catalog
  and the strongest reliability/compliance guarantees, generally the most
  expensive on-demand. Worth it when you need the rest of their ecosystem
  (networking, IAM, managed services) alongside the GPU, not just the GPU.
- **GPU-specialist "neoclouds"** (Lambda, CoreWeave, Nebius, Crusoe,
  Voltage Park, and similar) — a narrower catalog focused on current
  NVIDIA silicon, usually meaningfully cheaper than hyperscalers for the
  same card, and often first to get new SKUs (H200, B200, GB200) in volume.
- **Marketplaces / spot platforms** (RunPod, Vast.ai, Paperspace) — the
  cheapest on-demand pricing and the most variable reliability and
  availability. A good fit for a short sizing or benchmarking run, a
  risky one for production serving.
- **Regional / compliance-driven clouds** (Alibaba Cloud, Tencent Cloud,
  Huawei Cloud) — the primary source for the export-compliant chips in
  podfit's catalog (H20, H800, L20), since those SKUs largely don't exist
  outside China-market deployments.
- **Bare metal / colo** — full control and the best $/GPU-hour at scale,
  but you own the power, cooling, networking, and NVLink/InfiniBand
  fabric design yourself. It's also the only path to an Apple Silicon
  cluster — no cloud provider rents Mac Studios by the hour at any real
  scale, so a 4-node M5 Ultra RDMA cluster is hardware you buy, not rent.

## Layout

```
app/catalog.py     GPU specs, weight formats, KV dtypes, engine capabilities
app/archsupport.py ~100 model families x engine support x kernel requirements
app/attention.py   attention backend selection and its GPU/feature constraints
app/gpuarch.py     GPU architecture families and the per-model support matrix
app/hf.py          Hub client: search, config.json parsing, GGUF files, caching
app/bench.py       Artificial Analysis + Open LLM Leaderboard providers
app/aa_source.py   AA's ranking as a candidate pool, resolved to open-weights repos
app/engine.py      memory model, compatibility rules, throughput, launch command
app/db.py          SQLite: saved rigs + TTL cache
app/main.py        FastAPI routes
app/static/        the UI (no build step)
design/            the .dc.html artboards behind the published design canvas
```
