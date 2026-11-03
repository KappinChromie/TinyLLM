# Phase 1: Architecture Speed Sweep & Knee Analysis

**Hardware:** NVIDIA GeForce RTX 3070 Ti (8 GB VRAM, Driver 595.97, CUDA 12.4)  
**CPU:** AMD Ryzen 9 5900XT (16 Cores / 32 Threads)  
**Power Profile:** High Performance  
**Benchmark Configuration:** `llama-bench -ngl 99 -fa 1 -p 512 -n 256 -r 3` (fp16, tied embeddings)

---

## 1. Executive Summary: The Overhead Floor & "The Knee"

In the micro-model regime on an RTX 3070 Ti, execution speed is dominated by **kernel launch overhead and sequential layer synchronization**, not raw FLOPs or memory bandwidth.

- **Peak Measured Decode Speed:** **2,478.8 tokens/sec** (`1L / 256H / 768FFN / v8192`, 2.88M params).
- **The 10% Floor ("The Knee"):** Any model achieving $\ge 2,231\text{ t/s}$ (within 10% of peak). The largest model still sitting on this overhead floor is **`1L x 512H` (6.82M params, 2,261.8 t/s)** and **`2L x 256H` (3.67M params, 2,292.0 t/s)**.
- **The 25% Tier (1,859 – 2,231 t/s):** Balances extreme speed with 4x–8x more capacity. Best candidate: **`2L x 768H` (18.49M params, 1,982.0 t/s)** or **`3L x 512H` (12.06M params, 1,922.2 t/s)**.
- **The 50% Tier (1,240 – 1,859 t/s):** Delivers full LLM depth (4–6 layers) and 30M–60M parameters while still delivering **1,250 – 1,650 t/s** (over 2.5x to 3x faster than the SmolLM2-135M baseline at 543 t/s).

---

## 2. Speed Tiers & Best Candidate Shapes

### Tier 1: The Knee Floor ($\ge 90\%$ of peak: $\ge 2,231\text{ t/s}$)
*Maximum possible decode speed. Best if the goal is purely a 2,000+ t/s speed demon.*

| Rank | Architecture | Params (Total) | Params (Non-Emb) | Vocab | TG Speed (t/s) | PP Speed (t/s) | Notes |
|:---:|:---|:---:|:---:|:---:|:---:|:---:|:---|
| 1 | **1L / 512H / 1280FFN** | 6.82M | 2.62M | 8,192 | **2,261.8** $\pm$ 4.3 | 242,325 | Largest capacity on 10% floor |
| 2 | **2L / 256H / 768FFN** | 3.67M | 1.57M | 8,192 | **2,292.0** $\pm$ 2.7 | 230,716 | 2 layers for basic compositional routing |
| 3 | **1L / 256H / 768FFN** | 2.88M | 0.79M | 8,192 | **2,478.8** $\pm$ 5.9 | 270,049 | Peak fastest single configuration |
| 4 | **2L / 256H / 512FFN** | 3.28M | 1.18M | 8,192 | **2,250.1** $\pm$ 3.8 | 243,795 | Slim FFN (2.0x) variant |

---

### Tier 2: The Practical Frontier ($75\% - 90\%$ of peak: $1,859 - 2,231\text{ t/s}$)
*Recommended sweet spot. Gives 3x to 8x the non-embedding parameter capacity of Tier 1 while losing less than 20% speed.*

| Rank | Architecture | Params (Total) | Params (Non-Emb) | Vocab | TG Speed (t/s) | PP Speed (t/s) | Notes |
|:---:|:---|:---:|:---:|:---:|:---:|:---:|:---|
| 1 | **2L / 768H / 2048FFN** | 18.49M | 12.19M | 8,192 | **1,982.0** $\pm$ 8.2 | 191,900 | High width, 12M non-emb params |
| 2 | **3L / 512H / 1280FFN** | 12.06M | 7.87M | 8,192 | **1,922.2** $\pm$ 5.7 | 193,844 | 3 layers provides better non-linear depth |
| 3 | **4L / 256H / 768FFN** | 5.24M | 3.15M | 8,192 | **1,979.8** $\pm$ 4.3 | 187,459 | 4 layers, narrow width |
| 4 | **4L / 512H / 1280FFN** | 12.59M | 10.49M | 4,096 | **1,881.7** $\pm$ 4.7 | 176,272 | 4 layers with compact 4k vocab |
| 5 | **1L / 1024H / 2816FFN**| 19.40M | 11.01M | 8,192 | **2,068.9** $\pm$ 6.1 | 214,664 | Extreme width single layer |

---

### Tier 3: Quality-First Fast Tier ($50\% - 75\%$ of peak: $1,240 - 1,859\text{ t/s}$)
*Capable language modeling capacity (30M–60M parameters) running at 1,250–1,650 t/s (still 2x to 3x faster than SmolLM2-135M).*

| Rank | Architecture | Params (Total) | Params (Non-Emb) | Vocab | TG Speed (t/s) | PP Speed (t/s) | Notes |
|:---:|:---|:---:|:---:|:---:|:---:|:---:|:---|
| 1 | **4L / 768H / 2048FFN** | 30.65M | 24.39M | 8,192 | **1,663.9** $\pm$ 5.2 | 148,599 | Strong 4-layer 24M non-emb capacity |
| 2 | **4L / 512H / 1280FFN** | 14.68M | 10.49M | 8,192 | **1,808.5** $\pm$ 4.6 | 169,875 | Standard balanced small architecture |
| 3 | **6L / 512H / 1280FFN** | 19.93M | 15.73M | 8,192 | **1,526.1** $\pm$ 4.1 | 143,782 | 6 layers gives real transformer depth |
| 4 | **4L / 768H / 2048FFN** | 48.96M | 24.39M | 32,000 | **1,412.4** $\pm$ 6.4 | 150,454 | Distillation-ready (32k vocab) |
| 5 | **6L / 768H / 2048FFN** | 61.16M | 36.58M | 32,000 | **1,260.6** $\pm$ 5.9 | 112,330 | Full 36M non-emb params, 32k vocab |

---

## 3. Key Architectural Findings

### A. Depth Costs Far More Than Width
Kernel launch overhead is sequential. Every layer adds a mandatory barrier:
- Moving from 1L to 12L at hidden=256 reduces speed from **2,479 t/s to 1,384 t/s (-44%)**.
- Widening from 256H to 1024H at 1L reduces speed only from **2,479 t/s to 2,069 t/s (-16%)**.
- **Isocost Insight:** `2L x 1024H` (18.5M non-emb params, **1,827 t/s**) is faster than `6L x 256H` (4.7M non-emb params, **1,759 t/s**), despite having **nearly 4x the capacity**.

### B. Hidden Size 128 is an Antipattern
At `hidden=128`, decode speed collapsed to **1,694 t/s (1L)** and **872 t/s (4L)**. The tensor dimensions are too small to saturate the 48 SMs and tensor cores on the RTX 3070 Ti, resulting in severe GPU underutilization. `hidden=256` or `hidden=512` is the true minimum practical width.

### C. The Cost of Vocabulary Size
Evaluating vocabulary sizes from 2,048 to 49,152:
- Moving from 2k to 8k costs almost nothing ($< 1\%$).
- Moving from 8k to 32k costs **11% to 14%** decode speed across all tested shapes.
- Moving from 8k to 49k (SmolLM2 tokenizer) costs **15% to 21.5%** decode speed.
- **Phase 3 Recommendation:** A custom SentencePiece tokenizer with 8,192 tokens preserves peak throughput. If full logit distillation is chosen in Phase 5, a 32,000 tokenizer costs ~12-14% speed, whereas a 49k tokenizer crosses the 20% penalty line.

---

## 4. Deliverable Artifacts
- **CSV Data:** [`results/sweep.csv`](file:///c:/Users/luche/Documents/projects/00000000/tinyllm/results/sweep.csv) (94 configurations tested with 3 repetitions each, complete GPU telemetry)
- **Plots Generated:**
  - [`results/tg_vs_params.png`](file:///c:/Users/luche/Documents/projects/00000000/tinyllm/results/tg_vs_params.png): Decode speed vs total params with tier cutoffs marked
  - [`results/tg_vs_layers.png`](file:///c:/Users/luche/Documents/projects/00000000/tinyllm/results/tg_vs_layers.png): Speed vs layer count by hidden size
  - [`results/tg_vs_vocab.png`](file:///c:/Users/luche/Documents/projects/00000000/tinyllm/results/tg_vs_vocab.png): Speed vs vocabulary size curve
