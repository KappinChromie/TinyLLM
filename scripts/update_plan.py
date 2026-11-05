import sys

with open('tiny-llm-speed-plan.md', 'r', encoding='utf-8') as f:
    text = f.read()

# 1. Update Architecture Finalists section in Phase 4
idx1 = text.find('### Architecture Finalists & Control Trial')
idx2 = text.find('### Implementation & Training Recipe')
if idx1 == -1 or idx2 == -1:
    print(f"Failed to find Architecture Finalists block (idx1={idx1}, idx2={idx2})")
    sys.exit(1)

new_p4_arch = """### Architecture Selection & Winner Decision
All initial candidates used the custom 8k SentencePiece tokenizer, RMSNorm, RoPE, SwiGLU, and tied embeddings:
- `4L / 768H / 2048FFN` (1 KV head) - 30.28M params
- `6L / 512H / 1280FFN` (1 KV head) - 19.54M params
- `6L / 768H / 2048FFN` (1 KV head) - 42.28M params
- `4L / 768H / 2048FFN` with 2 KV heads (control) - 30.68M params

> [!IMPORTANT]
> **DECISION: `6L_768H_1KV` (42.28M params) IS LOCKED AS THE SOLE WINNING ARCHITECTURE.**
> 
> Following the Phase 5A 1-hour trials and an extended 3-hour head-to-head comparison against `4L_768H_1KV`, `6L_768H_1KV` won decisively across all evaluation criteria:
> 1. **Iso-Token Dominance**: At 250M tokens, 6L led by -0.1018 loss (decayed) and -0.1062 (undecayed). At 500M tokens, 6L led by -0.1190 (decayed). At 750M tokens, the lead expanded to -0.1345 loss (decayed) and -0.1337 (undecayed), demonstrating that 6L's quality advantage widens continuously with scale (+32% expansion over training).
> 2. **Iso-Wall-Clock Dominance**: Even when 4L was trained for the exact same wall-clock time (3.0 hours) and consumed 1,054M tokens (+36.7% more tokens than 6L's 770.8M tokens), 6L still defeated 4L by -0.0632 loss on neutral Wikipedia (3.7381 vs 3.8013) and -0.0617 on average validation loss.
> 3. **Statistical Significance**: Both gaps exceed the empirical validation noise floor ($\Delta_{\\text{val}} = 0.0076$) by > 8x, and exceed the training noise floor ($\Delta_{\\text{train}} = 0.0254$) by > 2.4x.
> 4. **1 KV Head Decision**: 1 KV head is locked for all stages based on 2k parity. *Caveat*: Long-context retrieval at 8k–32k with 1 KV head remains untested and will be strictly validated during Stage B/C passkey tests.

"""

text = text[:idx1] + new_p4_arch + text[idx2:]

# 2. Update Throughput table in Phase 4
idx3 = text.find('### Measured Training Throughput (RTX 3070 Ti, WSL2, Flash SDPA)')
idx4 = text.find('```python\n# budget.py')
if idx3 == -1 or idx4 == -1:
    print(f"Failed to find Measured Training Throughput block (idx3={idx3}, idx4={idx4})")
    sys.exit(1)

new_throughput = """### Measured Training Throughput (RTX 3070 Ti, WSL2, Flash SDPA)
Measured empirically on real hardware for `6L_768H_1KV` (PyTorch 2.6, CUDA 12.4, fused AdamW, bf16):

| Context Length | Micro Batch | Grad Ckpt | Chunked Loss | Tok/s | M Tok/hr | Step Time (ms) | Peak VRAM | Status |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **2,048 (Stage A)** | 8 | False | False | **68,453 – 71,373** | **246.4 – 256.9M** | 239.3 ms | 4.90 GB | Validated in 3h run |
| **8,192 (Stage B)** | 1 | False | False | **48,795** | **175.7M** | 167.9 ms | 2.59 GB | Fits in VRAM w/o ckpt |
| **8,192 (Stage B)** | 1 | True | True | **40,128** | **144.5M** | 204.2 ms | 0.96 GB | Conservative planning base |
| **32,768 (Stage C)** | 1 | True | False | **19,647** | **70.7M** | 1,667.8 ms | 3.83 GB | Verified on hardware |

"""

text = text[:idx3] + new_throughput + text[idx4:]

# 3. Update Phase 5 section completely
idx5 = text.find('## Phase 5: 1-Hour Finalist Trials & Data Mixture Ablation')
idx6 = text.find('## Phase 6: Full staged training run with long-context extension')
if idx5 == -1 or idx6 == -1:
    print(f"Failed to find Phase 5 block (idx5={idx5}, idx6={idx6})")
    sys.exit(1)

new_p5 = """## Phase 5: Architecture Decision, Quality Audits, Reference Speeds, & Phase 5B Mixture Ablation

### 1. Architecture Selection Decision (Completed)
- **Outcome**: `6L_768H_1KV` (42.28M parameters) selected as the winning architecture.
- **Head-to-Head Evidence**:
  - *Iso-Token (250M tokens)*: 6L led by -0.1018 (decayed) and -0.1062 (undecayed).
  - *Iso-Token (500M tokens)*: 6L led by -0.1190 (decayed) and -0.1189 (undecayed).
  - *Iso-Token (750M tokens)*: 6L led by -0.1345 (decayed) and -0.1337 (undecayed).
  - *Iso-Wall-Clock (3.0 hours)*: Despite 4L processing +36.7% more tokens (1,054M vs 770.8M), 6L beat 4L on neutral Wikipedia by -0.0632 loss (3.7381 vs 3.8013) and across all validation sets by -0.0617 loss. Both gaps decisively exceed the validation noise floor ($\Delta_{\\text{val}} = 0.0076$) and training noise floor ($\Delta_{\\text{train}} = 0.0254$).
- **1 KV Head Locked**: Locked for all subsequent stages (with long-context retrieval at 8k–32k to be validated in Stage B/C passkey tests).

### 2. Step 1: Data and Text-Quality Diagnostic Results (Completed)
- **Hypothesis Investigated**: Undertrained models dropping function words ("was happy have an", "went to park play his").
- **Audit Findings**:
  1. *Shard Window Decodes*: 20 random windows from `train_tinystories.bin`, `train_cosmopedia.bin`, and `train_fineweb_edu.bin` decoded into 100% natural, grammatical English with all function words present.
  2. *Round-Trip Verification*: 1,000 raw documents per source tested; exact-match rate was **100.00%** on TinyStories, **100.00%** on Cosmopedia, and **100.00%** on FineWeb-Edu. Tokenizer and binary shards are completely bug-free.
  3. **CRITICAL ROOT CAUSE IDENTIFIED (Double Label Shift Bug)**: In `StageAMixDataLoader`, batches were created with `x = shard[idx:idx+seq_len]` and `y = shard[idx+1:idx+1+seq_len]`. However, Hugging Face's `LlamaForCausalLM.forward(x, labels=y)` internally applies `shift_labels = labels[..., 1:]`. This resulted in a **double shift**, training $x_t$ to predict $x_{t+2}$ instead of $x_{t+1}$!
  4. *Empirical Confirmation*: Teacher-forced accuracy on held-out TinyStories documents was only 2.81% for $t+1$, but **~41.2%** for $t+2$.
  5. *Resolution*: Standard explicit cross-entropy `F.cross_entropy(logits.view(-1, vocab_size), y.view(-1))` implemented in all training scripts without passing `labels` into Hugging Face forward pass.

### 3. Step 2: Reference Decode Speed Benchmark (Completed)
Benchmarked on Windows-native llama.cpp (build `51fe42c5d`, CUDA 12.4, RTX 3070 Ti 8GB, `-fa 1 -ngl 99 -n 256 -r 5`):

| Model | Quantization | File Size | 0k Context Depth | 8k Context Depth | 32k Context Depth |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **`6L_768H_1KV` (Winner)** | **F16** | **80.8 MB** | **1,384.9 ± 10.2 t/s** | **1,274.9 ± 8.6 t/s** | **1,101.2 ± 7.9 t/s** |
| **`6L_768H_1KV` (Winner)** | **Q8_0** | **43.0 MB** | **1,481.0 ± 28.1 t/s** | **1,351.8 ± 17.5 t/s** | **1,170.6 ± 11.7 t/s** |
| **`6L_768H_1KV` (Winner)** | **Q4_0** | **24.4 MB** | **1,554.9 ± 21.7 t/s** | **1,411.1 ± 16.3 t/s** | **1,213.1 ± 10.2 t/s** |
| `4L_768H_1KV` | F16 | 57.9 MB | 1,660.7 ± 27.1 t/s | 1,511.6 ± 31.3 t/s | 1,314.8 ± 11.0 t/s |
| `4L_768H_1KV` | Q8_0 | 30.9 MB | 1,781.1 ± 14.2 t/s | 1,645.0 ± 5.5 t/s | 1,404.5 ± 8.7 t/s |
| `4L_768H_1KV` | Q4_0 | 18.0 MB | 1,840.3 ± 39.3 t/s | 1,676.8 ± 4.1 t/s | 1,418.8 ± 5.0 t/s |

### 4. Step 4: Phase 5B Data-Mixture Ablation Protocol
Train `6L_768H_1KV` for four runs of 250M tokens each (~1 hour each) with identical seeds and data ordering:
- **Run A**: 100% FineWeb-Edu (`train_fineweb_edu.bin`)
- **Run B**: 100% Cosmopedia v2 (`train_cosmopedia.bin`)
- **Run C (Default Blend)**: 45% Cosmopedia v2, 45% FineWeb-Edu, 10% TinyStories
- **Run D (Simpler-Text Blend)**: 40% TinyStories, 40% Cosmopedia v2, 20% FineWeb-Edu

#### Training & Evaluation Protocol
- **Setup**: Context 2,048 tokens, LR $\eta_0 = 1\times 10^{-3}$, batch 65,536 tokens, WSD schedule (flat for 225M tokens, 25M token cosine decay to $0.1\times \eta_0$).
- **Primary Decision Metric**: Held-out cross-entropy and bits per byte (BPB) on neutral `val_wikipedia_neutral`.
- **Secondary Metrics**: Cross-entropy and BPB on `val_cosmopedia`, `val_fineweb_edu`, and `val_tinystories`.
- **Qualitative Generation**: 10 standard evaluation prompts plus 5 basic grammar prompts (short sentences, simple stories).
- **Noise Floor Comparison**: Compare differences against both empirical thresholds: $\Delta_{\\text{val}} = 0.0076$ (validation spread) and $\Delta_{\\text{train}} = 0.0254$ (training spread).

---

"""

text = text[:idx5] + new_p5 + text[idx6:]

# 4. Update Phase 6 Budget section
idx7 = text.find('#### Total Training Token Allocation & Reserve Budget')
idx8 = text.find('#### Stage Details & Data Mixtures')
if idx7 == -1 or idx8 == -1:
    print(f"Failed to find Budget block (idx7={idx7}, idx8={idx8})")
    sys.exit(1)

new_budget = """#### Recomputed 12-Hour Staged Training Budget (`6L_768H_1KV`)
Based on measured training throughputs for `6L_768H_1KV` on the RTX 3070 Ti (8GB) with PyTorch 2.6 CUDA 12.4 in WSL2:
- **2k Context Throughput**: ~71,373 tok/s (256.9M tokens/hr)
- **8k Context Throughput**: 40,128 tok/s to 48,795 tok/s (planned at conservative 40,500 tok/s = 145.8M tokens/hr)
- **32k Context Throughput**: 19,647 tok/s (70.73M tokens/hr)

| Training Stage | Context Length | Target Tokens | Sequence Count | Global Batch Size | Optimizer Steps | Wall-Clock Hours | % of 12h Budget | % of Active Time |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Stage A (Short Foundation)** | 2,048 | 1,900.5M | 928,000 | 128 seqs (262k tok) | 7,250 | 7.40 hrs | 61.7% | 68.9% |
| **Stage B (8k Long Extension)** | 8,192 | 250.1M | 30,528 | 32 seqs (262k tok) | 954 | 1.71 hrs | 14.3% | 15.9% |
| **Stage C (32k Extension & Anneal)** | 32,768 | 114.7M | 3,500 | 4 seqs (131k tok) | 875 | 1.62 hrs | 13.5% | 15.1% |
| **10% Compute Reserve** | — | — | — | — | — | 1.20 hrs | 10.0% | — |
| **Total Planned Pipeline** | — | **2,265.3M** | **962,028** | — | **9,079** | **11.93 hrs (~12.0h)** | **100.0%** | **100.0%** |

> [!IMPORTANT]
> **COMPUTE DISTRIBUTION SHIFT**: Under `6L_768H_1KV`, Stage B and Stage C consume **3.33 hours (30.8% of active training time)**, compared to 2.26 hours (22.3%) under earlier 4L estimates. Stage A receives 7.40 hours (1,900.5M tokens). The 10% reserve (1.20 hours) is preserved intact for dynamic Stage C extensions or retry buffers.
>
> **DECAY BRANCHES (ON-DEMAND ONLY)**:
> Decay branches are non-default fallback checkpoints branched on demand; they do NOT consume main-line training budget:
> - **Branch A (2k Release)**: 100M tokens (~384 optimizer steps, ~0.39 hr) branched from Stage A if a dedicated 2k release model is needed.
> - **Branch B (8k Release)**: 100M tokens (~381 optimizer steps, ~0.69 hr) branched from Stage B if Stage C fails entrance criteria.

"""

text = text[:idx7] + new_budget + text[idx8:]

with open('tiny-llm-speed-plan.md', 'w', encoding='utf-8') as f:
    f.write(text)

with open(r'C:\Users\luche\Downloads\tiny-llm-speed-plan.md', 'w', encoding='utf-8') as f:
    f.write(text)

print("Plan successfully updated!")
