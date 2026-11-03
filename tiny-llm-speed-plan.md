# Project: Fastest-possible custom LLM on an RTX 3070 Ti, with as much quality as the speed budget allows

## Goal

Design, train, and benchmark a small Llama-architecture language model that:

1. **Maximizes decode tokens/sec** on an RTX 3070 Ti (8 GB), in two modes: (A) single-stream (batch 1) and (B) aggregate throughput under batching.
2. **Packs in as much capability as possible without losing that speed**, using the best available training data and (if it proves worthwhile) distillation.

Speed is the primary metric. Quality is the thing we maximize *subject to* the speed target.

## Core principles (read these first)

1. **Speed depends on architecture, not weights.** A random-init model and a trained model of the same shape run at the same speed (same dtype/quant). So we find fast shapes first, with zero training, then spend all our effort on making the chosen shape smart.
2. **There is an overhead floor.** Below some size, decode speed stops improving because kernel-launch and CPU overhead dominate, not memory bandwidth. Models at or below the floor are all roughly equally fast. **The target design is the largest model that is still on the floor** (the "knee"). That is free capacity.
3. **Depth costs more speed than width.** Layers run sequentially. Widening a layer is cheap until you leave the overhead-bound regime; adding layers is not. But small models often prefer depth for quality, so measure both speed *and* validation loss.
4. **Tokens/sec is gameable.** A tiny vocabulary splits text into more tokens, inflating t/s without increasing words/sec. Always also report **characters/sec** (or words/sec) on real generated text.
5. **Never report a number that isn't in a results CSV.** Every claim comes from a logged, reproducible run.

## Ground rules

- Change one variable at a time. Commit every config, script, and result.
- Benchmark protocol: 1 warmup run, then 5 repetitions, report mean and std. Log GPU clocks, power, and temperature (`nvidia-smi`) on every run.
- Use the **Llama architecture** (RMSNorm, RoPE, SwiGLU, GQA, tied embeddings) so llama.cpp supports it.
- Make hidden size and FFN size multiples of **256** if you want K-quants (Q4_K_M) to apply to every tensor. Multiples of 32 are the minimum for Q4_0/Q8_0.
- Keep vocab size below 65,536 so token IDs fit in `uint16` on disk.
- Dataset and model names below may have changed. Verify on Hugging Face before downloading.

---

## Phase 0: Environment and baselines

1. Install CUDA toolkit, PyTorch (CUDA build), `transformers`, `datasets`, `tokenizers`, `sentencepiece`, `numpy`, `matplotlib`, `pandas`.
2. Build llama.cpp with CUDA (`-DGGML_CUDA=ON`). Confirm `llama-bench`, `llama-batched-bench`, `llama-quantize`, and `convert_hf_to_gguf.py` work.
3. Baselines: benchmark SmolLM2-135M and Qwen3-0.6B at F16, Q8_0, Q4_0:
   ```
   llama-bench -m model.gguf -ngl 99 -fa 1 -p 512 -n 256
   ```
4. Stabilize the machine: high-performance power plan, no other GPU load, note driver version and CPU model.

**Deliverable:** `results/baseline.csv`

---

## Phase 1: Speed sweep with random weights (no training), find the knee

Goal: map speed against shape, locate the overhead floor, and identify the **largest shape that is still on the floor**.

1. Generate random-init `LlamaConfig` models over a grid:
   - layers: 1, 2, 3, 4, 6, 8, 12
   - hidden size: 128, 256, 512, 768, 1024
   - FFN ratio: 2x, 2.67x (SwiGLU standard), 4x
   - KV heads: 1, 2, full
   - vocab size: 2k, 4k, 8k, 16k, 32k, 49k (use a placeholder tokenizer; speed doesn't depend on the text)
   - tied embeddings: on
2. For each config: save in fp16, convert to GGUF, benchmark with `llama-bench -ngl 99 -fa 1 -n 256`. Log layers, hidden, ffn, vocab, kv_heads, total params, non-embedding params, file size, `pp` t/s, `tg` t/s.
3. Plot:
   - `tg` t/s vs. total params
   - `tg` t/s vs. layer count (expect this to matter most)
   - `tg` t/s vs. vocab size (this tells us what a big tokenizer actually costs. It decides the distillation tokenizer question in Phase 3)
4. Define the **knee**: the largest configs whose `tg` t/s is within **10%** of the fastest configuration. Also record a second tier within **25%** and a third within **50%**.

**Deliverable:** `results/sweep.csv`, plots, and a short note listing the best 3-5 candidate shapes per speed tier.

---

## Phase 2: Data & length distribution

Principle: for small models, **data quality and data fit matter more than volume**. Choose data matched to model capacity and context length requirements.

### Sequence of execution
1. **Data length-distribution analysis**: Compute token-length distributions across all candidate sources to verify volume of documents at $\ge 2\text{k}$, $\ge 8\text{k}$, and $\ge 32\text{k}$ tokens.
2. **Pause and report**: Present distribution analysis, license status, proposed mixture, and download sizes for user review.
3. **Train custom tokenizer**: Train the custom 8k SentencePiece tokenizer **only after the final data mixture is approved by the user**.

### Candidate sources (verified on Hugging Face)

| Source | Hugging Face ID | License | Length Profile | Role |
|---|---|---|---|---|
| **TinyStories** | `roneneldan/TinyStories` | CDLA-Permissive-1.0 | Short (mean 208 tok, max 988) | Coherent simple reasoning for tiny models |
| **Cosmopedia v2** | `HuggingFaceTB/cosmopedia-v2` | Apache 2.0 | Medium (mean 720 tok, max 1.8k) | Dense synthetic textbooks and stories |
| **FineWeb-Edu** | `HuggingFaceFW/fineweb-edu` | ODC-By 1.0 | Varied (mean 1.1k tok, max 26.9k) | Educational filtered web text |
| **Wikipedia (en)** | `wikimedia/wikipedia` | CC BY-SA 3.0 / 4.0 | Medium-Long (mean 852.2 tok, 5.46B tok total, 1.77% docs $\ge 8\text{k}$) | Factual reference; neutral evaluation set (~10M tokens) |
| **PG-19 (Books)** | `emozilla/pg19` | Public Domain / CC0 | Very Long (mean 107.2k tok, 3.08B tok total, 80.3% docs $\ge 32\text{k}$) | Primary source for genuine $\ge 8\text{k}$ and $\ge 32\text{k}$ long context |

*Note: arXiv (`ccdv/arxiv-summarization`) was audited and dropped from training due to placeholder contamination (`@xmath`, `@xcite`) and licensing ambiguity.*

### Processing & Holdouts
1. **Holdouts taken by document before tokenizing**: Hold out a fixed 5% validation set per source by document. Never mix train and val.
2. **Neutral reference set**: 10M tokens held out from Wikipedia shard 40, reserved as a neutral evaluation corpus neither model sees during Phase 5 data ablation, audited for 13-gram non-contamination.
3. **Long-document evaluation set**: Dedicated held-out pool of 170+ books and long articles $\ge 32\text{k}$ tokens from PG-19 and Wikipedia for token-position bucket perplexity and needle retrieval.
4. **Tokenization**: Memmapped `uint16` binary shards (`.bin`) with EOS between documents.

**Deliverable:** `results/length_distribution.csv`, `data/` tokenized shards, `data/manifest.json`.

---

## Phase 3: Tokenizer decision

**Decision (Post-Phase 1): Custom 8k SentencePiece tokenizer.**

- Phase 1 speed sweep demonstrated that moving from 8k to 32k incurs an 11% to 14% decode speed penalty, and 49k incurs a 15% to 21.5% penalty.
- To maximize decode tokens/sec within the speed budget, the project adopts **Option A: custom 8,192 (8k) SentencePiece BPE** trained directly on the approved training data mix.
- **Training Mixture for Tokenizer**: Train primarily on the Stage A foundation mix (~90% share across Cosmopedia v2, FineWeb-Edu, and TinyStories), with a small ~10% share from long-form sources (Wikipedia, PG-19) to ensure common book and factual vocabulary are represented. Trained with `byte_fallback=True` and `remove_extra_whitespaces=False` to prevent indentation collapsing.
- **Compression Efficiency Verification**: Immediately after tokenizer training, compute and report **compression efficiency** (characters per token and bytes per token) across all candidate sources (TinyStories, Cosmopedia, FineWeb-Edu, Wikipedia, and PG-19) to verify that long-form text is not under-served. Also verify 100% round-trip tokenization consistency in `llama.cpp` using `llama-tokenize.exe`.
- Logit-level distillation is dropped for now (revisit logit KD only if the 8k model plateaus). Sequence-level distillation (training on teacher-generated synthetic text) remains fully supported.

**Deliverable:** `tokenizer/` with trained SentencePiece model, vocab files, and compression ratio report across sources.

---

## Phase 4: Training recipe, 1-hour finalist comparison trials, and token budget

### Architecture Finalists & Control Trial
All models use the custom 8k SentencePiece tokenizer, RMSNorm, RoPE, SwiGLU, and tied embeddings:
- **Finalist 1**: `4L / 768H / 2048FFN` (1 KV head)
- **Finalist 2**: `6L / 512H / 1280FFN` (1 KV head)
- **Finalist 3**: `6L / 768H / 2048FFN` (1 KV head)
- **Control Trial**: `4L / 768H / 2048FFN` with **2 KV heads** (benchmarked against Finalist 1 to quantify the exact quality vs. speed tradeoff of 1 KV head).

### Implementation & Training Recipe
- **Attention & Kernels**: bf16 autocast, fused AdamW, native PyTorch FlashAttention-2 / SDPA under WSL2 (Ubuntu 24.04 LTS, CUDA 12.4), gradient checkpointing. Peak VRAM is only 2.18–2.48 GB even at 32k context.
- **Chunked Loss**: Cross-entropy computed in chunks over the sequence dimension to eliminate peak VRAM spikes from large logit matrices at extended sequence lengths.
- **Global Batch Sizing per Stage**:
  - **Stage A (2k context)**: Global batch = 128 sequences (262,144 tokens ~0.26M tokens). Micro-batch = 8 sequences, 16 grad accum steps.
  - **Stage B (8k context)**: Global batch = 32 sequences (262,144 tokens ~0.26M tokens). Micro-batch = 2 sequences, 16 grad accum steps.
  - **Stage C (32k context)**: Global batch = 4 sequences (131,072 tokens ~0.13M tokens). Micro-batch = 1 sequence, 4 grad accum steps (yields 875 optimizer steps over 3,500 sequences).
- **Learning Rate Tuning**: Quick 3-point LR sweep ($\eta_0 \in \{1\times 10^{-3}, 2\times 10^{-3}, 3\times 10^{-3}\}$) on a short 200-step run per finalist on real tokenized data in WSL2 to select optimal peak learning rate. Weight decay 0.1, grad clip 1.0, tied embeddings.

### Measured Training Throughput (RTX 3070 Ti, WSL2, Flash SDPA)
Measured empirically on real hardware across all trial shapes (from `results/train_throughput.csv`):

| Architecture | Total Params | 2k Tok/s (M tok/hr) | 8k Tok/s (M tok/hr) | 32k Tok/s (M tok/hr) | Peak VRAM |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **`4L_768H_1KV`** | 30.28M | 87,375 (314.5M/hr) | 58,146 (209.3M/hr) | 29,627 (106.7M/hr) | 2.32 GB |
| **`6L_512H_1KV`** | 19.54M | 101,053 (363.8M/hr) | 63,744 (229.5M/hr) | 31,080 (111.9M/hr) | 2.19 GB |
| **`6L_768H_1KV`** | 42.28M | 62,210 (224.0M/hr) | 40,146 (144.5M/hr) | 19,973 (71.9M/hr) | 2.48 GB |
| **`4L_768H_2KV` (Control)** | 30.68M | 85,553 (308.0M/hr) | 57,381 (206.6M/hr) | 29,279 (105.4M/hr) | 2.32 GB |

```python
# budget.py
import sys

def count_params(vocab, d, layers, ffn, heads, kv_heads, tied=True):
    head_dim = d // heads
    attn = d * d + 2 * d * (kv_heads * head_dim) + d * d  # q, k, v, o
    mlp = 3 * d * ffn                                     # gate, up, down
    norms = 2 * d
    per_layer = attn + mlp + norms
    emb = vocab * d
    total = layers * per_layer + emb * (1 if tied else 2) + d
    return total, total - emb

def plan(tok_per_sec_measured, hours, params):
    tokens = tok_per_sec_measured * hours * 3600
    return tokens, tokens / params     # total tokens, tokens-per-parameter
```

### 1-Hour Finalist Comparison Trials (Short Context, 2k)
Before committing to the full multi-stage training run:
1. Train all 4 trial models (Finalist 1, Finalist 2, Finalist 3, and Control) on identical approved dataset mix, identical 8k tokenizer, and equal wall-clock time (~1 hour each at 2,048 context).
2. **Terminal Decay in Trials**: To ensure fair evaluation, each 1-hour trial spends the first 90% of its budget in the stable phase and executes a short 10% terminal decay down to $0.1\times \eta_0$ so final validation losses reflect converged models.
3. Evaluate held-out validation loss across sources and compare 1-KV vs. 2-KV quality gap.
4. Plot held-out validation loss vs. decode speed (at 0 and 32k depth).
5. Review results with the user to select the single winning architecture for the full staged run.

**Deliverable:** `results/train_throughput.csv`, 1-hour trial comparison curves, speed vs. val loss Pareto plot, finalist decision note.

---

## Phase 5: Data mixture ablation (Plain CE on Web Text vs. Teacher-Generated Synthetic)

**Question to answer:** Does training on teacher-generated synthetic data (Cosmopedia v2) beat plain CE on human web text (FineWeb-Edu) under equal wall-clock time, and what is the optimal blend?

Train the chosen winning finalist shape under equal wall-clock budget (1–2 hours):
- **Run A: Human web text** (`FineWeb-Edu`).
- **Run B: Synthetic educational text** (`Cosmopedia v2`).

### Apples-to-Apples Evaluation Protocol
To avoid in-distribution evaluation bias (where each model predictably wins on its own training distribution):
1. **Neutral Common Reference Set**: Evaluate both runs on a 10M-token held-out dataset that **neither model trained on** (English Wikipedia held-out split, verified for zero contamination).
2. **Cross-Distribution Evaluation**: Evaluate Run A on held-out Cosmopedia, and Run B on held-out FineWeb-Edu.
3. **Qualitative Evaluation**: Compare greedy and sampled generations on 20 fixed prompts assessing coherence, factual consistency, and reasoning.

**Decision Rule:** Determine whether pure synthetic, pure web, or an interpolated blend (e.g. 60/40 or 70/30 synthetic/web) achieves the lowest perplexity on the neutral reference set. The winning distribution ratio sets the data blend for Stage A pretraining.

**Deliverable:** `results/distill_ablation.csv`, loss curves on neutral reference set, cross-distribution loss table, generation samples, mixture ratio verdict.

---

## Phase 6: Full staged training run with long-context extension, quality anneal, and fallback decay branches

Execute the full training curriculum on the winning architecture using a multi-stage Warmup-Stable-Decay (WSD) schedule, progressing from short context up to 32k tokens, incorporating short-context replay, verification gates, and fallback decay branches.

### Staged Curriculum & Learning Rate Schedule

```
LR
^
|      [Stage A: 2k Stable]            [Stage B: 8k Stable]         [Stage C1: 32k]
|      +------------------------+      +-----------------------+    +-----+
|     /                          \    /                         \  /       \ [Stage C2: Anneal]
|    /     [Decay Branch A] \     \  /    [Decay Branch B] \     \/         \
|   /      \-----------------\     \/     \-----------------\                \---> (0.1x peak)
|  /       (100M-tok fallback)\           (100M-tok fallback)\
+--+---------------------------+----------+-------------------+-------------+---------> Tokens
   0                         Stage A                          Stage B       Stage C
```

#### Total Training Token Allocation & Reserve Budget
The training capacity on the RTX 3070 Ti is planned around 3.2B tokens:
- **Stage A (2k Short Foundation)**: 2,500M tokens (78.1%)
- **Stage B (8k Long Extension)**: 250M tokens (7.8%)
- **Stage C (32k Long Extension & Anneal)**: 115M tokens (3.6%)
- **Quality Anneal (Stage C2)**: 35M tokens (1.1%)
- **Reserve Budget (Safety & Contingency)**: 300M tokens (9.4%)
- **Total Planned Capacity**: 3,200M tokens (100.0%)

*The 300M-token reserve is explicitly set aside for dynamic stage extensions (e.g. +2,000 sequences in Stage C) or on-demand fallback decay branches without exceeding the overall compute budget.*

#### Stage Details & Data Mixtures

1. **Stage A: Foundation Pretraining (Context 1,024–2,048 tokens, RoPE base $\theta = 10{,}000$)**
   - **Data Mixture**: Winning data blend from Phase 5 (Cosmopedia v2 + FineWeb-Edu + TinyStories).
   - **LR Schedule**: Linear warmup over first 1% of Stage A steps to peak LR $\eta_0$, remaining flat in the **stable phase** at $\eta_0$.
   - **On-Demand Fallback Decay Branch A**: If needed, copy the checkpoint at the end of Stage A and run a 100M-token cosine decay down to $0.1\times \eta_0$ to produce a fully converged short-context release checkpoint. The undecayed checkpoint continues into Stage B.

2. **Stage B: Long-Context Extension to 8k (Context 8,192 tokens, RoPE base $\theta = 500{,}000$–$1{,}000{,}000$)**
   - **Token Budget**: 250M tokens (~30,500 sequences of 8k).
   - **Global Batch**: 32 sequences (262,144 tokens ~0.26M tokens, ~950 optimizer steps).
   - **Internally Consistent Data Mixture (100% Total)**:
     - **Wikipedia (Long articles $\ge 8\text{k}$)**: 35% (87.5M tokens)
     - **FineWeb-Edu (Long articles $\ge 8\text{k}$)**: 25% (62.5M tokens)
     - **PG-19 (Books $\ge 8\text{k}$)**: 15% (37.5M tokens)
     - **Short Replay (Cosmopedia v2 + FineWeb-Edu short)**: 25% (62.5M tokens)
     - **Sum**: $35\% + 25\% + 15\% + 25\% = 100.0\%$ (250.0M tokens).
   - **Replay Implementation**: Short replay sequences are batched exclusively in separate 2,048-token batches (alternating with long batches) to prevent cross-document attention leakage across packed short documents.
   - **LR Schedule**: Continues directly at **$0.7\times \eta_0$** (with a brief 0.1% micro-ramp). Does NOT re-warm to full peak $\eta_0$ and does NOT start from a decayed LR.
   - **On-Demand Fallback Decay Branch B**: If Stage C is not pursued or fails its entrance checks, branch off a 100M-token cosine decay to $0.1\times \eta_0$ from the Stage B checkpoint to produce a verified, stable 8k-context release model.
   - **Verification Gate**: Evaluate both (1) position-bucket loss and passkey retrieval at 1k, 2k, 4k, 8k, AND (2) short-context validation loss on the 2k benchmark to confirm zero capability degradation. Both must pass before launching Stage C.

3. **Stage C: Long-Context Extension to 32k & Quality Anneal (Context 32,768 tokens, RoPE base $\theta = 1{,}000{,}000$–$4{,}000{,}000$)**
   - **Token Budget**: Exactly **3,500 sequences of 32,768 tokens** (~114.7M tokens).
   - **Global Batch**: 4 sequences of 32k (~131k tokens, exactly 875 optimizer steps). Micro-batch = 1 seq, 4 grad accum steps.
   - **Data Mixture**:
     - **PG-19 Books (Continuous sequences $\ge 32\text{k}$)**: 75% (86.0M tokens, ~2,625 sequences).
     - **Short Replay (Cosmopedia v2 + FineWeb-Edu, in separate 2k batches)**: 25% (28.7M tokens, ~875 sequence equivalents).
     - **Sum**: 100.0% (114.7M tokens).
   - **Sub-stage C1 (Sequences 1–1,750)**: Stable 32k adaptation at flat **$0.5\times \eta_0$** (with 0.1% micro-ramp).
   - **Dynamic Extension Rule**: If 32k passkey retrieval is below 80% or loss slope is still decreasing steeply at sequence 1,750, draw up to 2,000 additional sequences (+500 optimizer steps, +65.5M tokens) from the 300M reserve budget before initiating the terminal anneal.
   - **Sub-stage C2 (Final 1,750 Sequences) - Quality Anneal**: LR decays smoothly via cosine schedule down to **$0.1\times \eta_0$**, while shifting the mixture toward the highest-quality educational and long-form data.

### Long-Document Data Strategy
- FineWeb-Edu and Wikipedia have **0 documents ≥ 32k tokens**. PG-19 is the verified large public-domain corpus with continuous documents ≥ 32k tokens (**2.95B tokens ≥ 32k** across 28,752 books).
- **No document packing without disclosure**: Do not pack unrelated short documents together as a substitute for long context without explicitly reporting it. For Stage C, 3,500 continuous sequences of 32k tokens require ~114.7M tokens, which is comfortably supplied by PG-19 books without packing.

### Continuous Monitoring & Evaluation Protocol
1. **Short-Context Preservation**: Evaluate and log **short-context validation loss (2k)** after **EVERY stage** (Stage A, Stage B, and Stage C) to monitor and guarantee short-range capabilities do not degrade during long-context extension.
2. **Position-Bucket Validation Loss**: Measure held-out perplexity across buckets: [0-1k], [1-2k], [2-4k], [4-8k], [8-16k], and [16-32k] on held-out long documents.
3. **Passkey / Needle Retrieval Accuracy**: Needle-in-a-haystack retrieval at 1k, 2k, 4k, 8k, 16k, and 32k context depths (at least 20 trials per depth with needle inserted at varied depth percentages).
4. **Success Criteria**: Flat loss curve across position buckets and non-trivial passkey accuracy count as success for that length. Report the longest context depth where it genuinely succeeds.

### Export & Long-Context Speed Verification
1. Export final PyTorch checkpoint to GGUF; verify metadata encodes context length 32,768 and extended RoPE base.
2. Verify llama.cpp perplexity and passkey retrieval match PyTorch results at 32k context.
3. Benchmark final model in llama.cpp at depths 0, 8k, 16k, and 32k with 5 repetitions and error bars.

**Deliverables:** Stage checkpoints, fallback decay branch checkpoints, position-bucket loss curves, 
esults/long_context_eval.csv, passkey accuracy table, short-context regression logs, 
esults/quality.csv.

---

## Phase 7: Export, quantization, and final speed benchmark

1. Convert the final model to GGUF at F16, then quantize to Q8_0, Q4_0, Q4_K_M (if dimensions allow).
2. Benchmark every variant with `llama-bench -ngl 99 -fa 1 -p 512 -n 256` and the same protocol as before. On tiny models, simple quants often tie or beat K-quants, and quantizing may not speed anything up at all.
3. **Verify the key claim:** the trained model's speed matches its random-init twin from Phase 1. If not, investigate.
4. Check that quantization doesn't badly hurt quality: compute perplexity on the held-out set for each quant with `llama-perplexity`.
5. Measure **characters/sec and words/sec** on real generations, not just tokens/sec.

**Deliverable:** `results/final_quant.csv`

---

## Phase 8: Aggregate throughput

1. `llama-batched-bench` (and optionally `llama-server` with parallel slots) at concurrency 1, 8, 32, 64, 128, 256.
2. Find where total t/s plateaus or VRAM runs out. Keep context small to save KV cache memory.
3. Optional: repeat in vLLM (easiest on Linux/WSL2) and compare.

**Deliverable:** `results/batched.csv`, throughput-vs-concurrency plot.

---

## Phase 9: Stretch goals (beat llama.cpp)

At the overhead floor, launch overhead is the bottleneck, not bandwidth. Try:

1. PyTorch with **CUDA graphs** capturing the entire decode step (`torch.compile(mode="reduce-overhead")` or manual capture), with a static KV cache.
2. Fused ops, then a hand-written fused decode kernel (llama2.c-style, in CUDA).
3. Compare each against the Phase 7 llama.cpp number and report the delta.
4. Vocab trimming (see Phase 3) if a larger teacher tokenizer was used.

---

## Final report

- **Headline numbers:** max single-stream t/s and max aggregate t/s, each with the exact reproduction command.
- **Main table:** shape, params, file size, quant, single-stream t/s, chars/sec, peak aggregate t/s, held-out val loss.
- **Plots:** speed vs. params/layers (with the knee marked), speed vs. val loss frontier, throughput vs. concurrency.
- **Distillation verdict:** what the ablation showed at equal wall-clock.
- **Data notes:** final mixture, token counts per source, epochs of repetition.
- **Surprises:** depth vs. width, vocab cost, quantization effects.

## Pitfalls checklist

- Tokenizer/vocab mismatch is the most common GGUF conversion failure. Check it first.
- Thermal or power throttling produces noisy numbers. Log clocks every run.
- Windows makes vLLM hard; use Linux/WSL2 for it.
- Compare at identical prompt and generation lengths every time.
- Don't compare val loss across different tokenizers directly. Use bits-per-byte or bits-per-character if tokenizers differ.
- Keep validation data strictly separate from training data, including after mixing sources.
