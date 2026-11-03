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

## Phase 2: Data

Principle: for small models, **data quality and data fit matter more than volume**. Choose data matched to model capacity.

### Candidate sources (verify availability)

| Tier | Dataset | Notes |
|---|---|---|
| Simple, coherent | **TinyStories** (`roneneldan/TinyStories`) | ~0.5B tokens. Models under ~30M params can write coherent stories on this. |
| Synthetic educational | **Cosmopedia v2** (in `HuggingFaceTB/smollm-corpus`) | LLM-generated textbooks, stories, and how-tos. High quality and dense. |
| Filtered web | **FineWeb-Edu** (`HuggingFaceFW/fineweb-edu`, also the deduped copy in `smollm-corpus`) | Educational-quality-filtered web text. Use `sample-10BT` or stream subsets. |
| Code (optional) | **Python-Edu** (in `smollm-corpus`) | Only if you want code ability. |
| Math (optional) | **FineMath** | Only if you want math ability. |
| Reference | Wikipedia / Simple Wikipedia | Small, clean, factual. |

### Mixture guidance by model size

- **Under ~30M params:** mostly TinyStories + Cosmopedia. Broad web text will make a very small model's output worse and less coherent, because it lacks the capacity to model it.
- **30M-150M params:** Cosmopedia + FineWeb-Edu as the bulk, with some TinyStories-style simple text and optionally small code/math slices.
- **150M+ params:** FineWeb-Edu as the bulk, Cosmopedia as a large share, then code/math slices.

### Processing

1. Take validation splits **by document** *before* tokenizing, and keep them fixed for the whole project. Hold out a separate val set per source.
2. Prefer already-deduplicated sources. If mixing multiple sources, run a quick exact-hash dedup across them.
3. Tokenize once, write to `uint16` memmapped `.bin` files (nanoGPT style), with an EOS between documents.
4. Record the exact token count per source.

**Deliverable:** `data/` with tokenized shards, `data/manifest.json` (source, token count, hash).

---

## Phase 3: Tokenizer decision

**Decision (Post-Phase 1): Custom 8k SentencePiece tokenizer.**

- Phase 1 speed sweep demonstrated that moving from 8k to 32k incurs an 11% to 14% decode speed penalty, and 49k incurs a 15% to 21.5% penalty.
- To maximize decode tokens/sec within the speed budget, the project adopts **Option A: custom 8,192 (8k) SentencePiece BPE** trained directly on the training data mix.
- Logit-level distillation is dropped for now (revisit logit KD only if the 8k model plateaus). Sequence-level distillation (training on teacher-generated synthetic text) remains fully supported.

**Deliverable:** `tokenizer/` with trained SentencePiece model and vocabulary.

---

## Phase 4: Training recipe, staged context schedule, and token budget

### Staged Context Schedule
Rather than training at a fixed context length, training follows a staged curriculum up to 32k tokens:
- **Stage A: Pretraining (about 85-90% of token budget)**: 1024-2048 context, RoPE base 10,000.
- **Stage B: Extend to 8k (about 7-10% of token budget)**: Raise RoPE base (test 500k-1M), continue training at 8k context on genuinely long documents. Pass checks before proceeding.
- **Stage C: Extend to 32k (about 3-5% of token budget)**: Raise RoPE base (test 1M-4M), continue training at 32k context on documents that are actually >= 32k tokens.

Finalize exact budget splits after measuring training tokens/sec at 2k, 8k, and 32k context. Propose the split and verify with measured numbers before launching.

### Memory and Implementation Notes
- **Attention & Kernels**: bf16 autocast, fused AdamW, SDPA / FlashAttention, gradient checkpointing, and micro-batch size 1 at long context.
- **Chunked Loss**: Compute loss in chunks over the sequence dimension to avoid instantiating the massive 32k logits tensor in VRAM.
- **Environment**: If Windows blocks `torch.compile` or FlashAttention (Triton), run training in WSL2 (which has CUDA 12.4 and full Triton/compile support), alerting the user before moving files.
- **Hyperparameters**: Global batch around 0.25-0.5M tokens (gradient accumulation), peak LR 1e-3 to 3e-3, linear warmup (~1% of steps), then WSD (warmup-stable-decay) or cosine down to ~10% of peak. Weight decay 0.1, grad clip 1.0, tied embeddings.
- **Logging**: Checkpoint frequently. Log train/val loss every N steps, and sample generations at checkpoints.

### Token budget calculator

Don't guess. **Measure training throughput early**, then plan.

```python
# budget.py
import sys

def count_params(vocab, d, layers, ffn, heads, kv_heads, tied=True):
    head_dim = d // heads
    attn = d*d + 2*d*(kv_heads*head_dim) + d*d       # q, k, v, o
    mlp = 3 * d * ffn                                # gate, up, down
    norms = 2 * d
    per_layer = attn + mlp + norms
    emb = vocab * d
    total = layers * per_layer + emb * (1 if tied else 2) + d
    return total, total - emb

def plan(tok_per_sec_measured, hours, params):
    tokens = tok_per_sec_measured * hours * 3600
    return tokens, tokens / params     # total tokens, tokens-per-parameter

if __name__ == "__main__":
    # Example: fill in measured numbers from a 200-step training benchmark
    p_total, p_nonemb = count_params(32000, 512, 4, 1536, 8, 2)
    toks, tpp = plan(tok_per_sec_measured=100_000, hours=12, params=p_total)
    print(f"params={p_total/1e6:.1f}M  tokens={toks/1e9:.2f}B  tokens/param={tpp:.0f}")
```

Rule of thumb: training compute is roughly `6 x params x tokens` FLOPs, and small models on consumer GPUs reach only a modest fraction of peak throughput. **Measure tokens/sec on a 200-step run for each finalist shape** and plug it into the calculator. Treat any figure not measured on this machine as a guess.

Guidance on how much data to use:
- The compute-optimal rule is about 20 tokens/param, but small models are best **heavily overtrained** (hundreds to thousands of tokens/param). The best small public models do this.
- Repeating data for up to about 4 epochs is generally mildly worse than fresh data, not catastrophic. If the high-quality pool is smaller than the token budget, repeat it rather than dilute with low-quality text.
- Fit as many *fresh, high-quality* tokens as the time budget allows, then repeat.

**Deliverable:** `results/train_throughput.csv`, chosen token budget per finalist.

---

## Phase 5: Data source & distillation ablation (Plain CE vs. Teacher-Generated Synthetic)

**Question to answer:** does training on teacher-generated synthetic data (Cosmopedia) beat plain CE on human web text (FineWeb-Edu) under equal wall-clock time?

With the custom 8k SentencePiece tokenizer chosen to maximize decode speed, logit-level KD is dropped for now (revisit logit KD only if the 8k model plateaus). Phase 5 focuses on sequence-level comparison:

### Ablation design
Train the same finalist shape under equal wall-clock budget (e.g. 1-2 hours):
- **Run A: Plain CE on filtered human web text** (FineWeb-Edu).
- **Run B: Plain CE on teacher-generated synthetic text** (Cosmopedia v2 / synthetic stories).

Compare validation loss on held-out sets (both synthetic and natural text), generation coherence, and training throughput.

**Decision rule:** Choose the winning data distribution / ratio for Stage A pretraining.

**Deliverable:** `results/distill_ablation.csv`, loss curves, sample outputs, verdict note.

---

## Phase 6: Long-Context Staged Training & Extension Phase

Train and progressively extend context window up to 32k tokens:

### Staged Curriculum
1. **Stage A: Pretraining (about 85-90% of token budget)**
   - Context: 1024-2048 tokens, standard RoPE base 10,000.
   - Broad, high-quality mix with LR in the stable phase.
2. **Stage B: Extend to 8k (about 7-10% of token budget)**
   - Raise RoPE base (test 500k-1M), continue training at 8k context on genuinely long documents.
   - Run verification checks below. If they fail, stop, report, and do not start Stage C.
3. **Stage C: Extend to 32k (about 3-5% of token budget)**
   - Raise RoPE base again (test 1M-4M), continue training at 32k context on documents that are actually >= 32k tokens.
   - Run verification checks below.

Finalize exact budget splits after measuring training tokens/sec at 2k, 8k, and 32k. Propose the split and report numbers before launching. If Stage C would get too few tokens to take effect, report and propose alternatives.

### Long-Document Data Strategy
- FineWeb-Edu documents are mostly short (< 2k tokens). First compute the token length distribution across candidate datasets and measure available tokens at $\ge 8\text{k}$ and $\ge 32\text{k}$ lengths.
- Genuine long-form sources to consider:
  - Wikipedia long articles
  - Public-domain books (e.g. PG-19, Project Gutenberg)
  - ArXiv / scientific long-form papers
  - Other long-form open corpora
- Verify availability and licenses for each source before downloading.
- Never pack unrelated short documents together as a substitute for long context without explicitly reporting it (keep any packing to a minimum). If long-form data is insufficient for Stage C, repeat long documents across a few epochs and report the repetition count.

### Evaluation Protocol (run after Stage A, Stage B, and Stage C)
1. **Validation loss by position buckets:** Measure held-out perplexity across token buckets: `[0-1k]`, `[1-2k]`, `[2-4k]`, `[4-8k]`, `[8-16k]`, and `[16-32k]` on long documents.
2. **Passkey / Needle Retrieval Accuracy:** Evaluate needle-in-a-haystack retrieval at 1k, 2k, 4k, 8k, 16k, and 32k context depths (at least 20 trials per depth with needle inserted at varied depth percentages).
3. **Success Criteria:** A flat loss curve across position buckets and non-trivial passkey accuracy count as success for that length. Report the longest context depth where it genuinely works.

### Export and Long-Context Speed Verification
- Confirm GGUF export encodes the extended RoPE base and context length so llama.cpp runs correctly at 32k (`-c 32768`).
- Verify that perplexity / passkey retrieval in llama.cpp matches PyTorch results.
- Benchmark final trained, quantized models at depths 0, 8k, 16k, and 32k.

**Deliverable:** Stage checkpoints, position-bucket loss curves, `results/long_context_eval.csv`, passkey accuracy table.

---

## Phase 6: Final training run (staged curriculum)

1. **Stage 1: Foundation (about 60-70% of budget).** Broad, high-quality mix appropriate to model size (see Phase 2), with the LR in the stable phase.
2. **Stage 2: Anneal (about 20-30% of budget).** Decay the LR while shifting the mixture toward the **highest-quality** data (Cosmopedia, cleanest FineWeb-Edu shards, and any code/math you chose). Late-stage quality upweighting reliably helps small models.
3. Use the winning method from Phase 5 (CE or distillation).
4. Keep a rolling checkpoint and sample generations at each stage. If val loss is still improving at the end, extend the stable phase (this is why WSD) and re-decay.
5. Run the train-time speed sanity check at the end (see Phase 7).

**Evaluation (quality):**
- Primary: held-out **validation loss/perplexity** per source.
- Secondary: 20 fixed prompts with generations saved for eyeballing coherence.
- Optional: `lm-evaluation-harness` on HellaSwag/ARC-Easy/PIQA. Models this small may sit near chance on several tasks, so don't over-read them.

**Deliverable:** final checkpoint, loss curves, `results/quality.csv`, sample outputs.

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
