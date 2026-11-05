"""
Step 3: Update tiny-llm-speed-plan.md with:
1. Post-Fix Re-Verification Suite Results (Step 2A, 2B, 2C).
2. Recomputed 12-Hour Staged Training Budget & Wall-Clock Estimates.
3. Summary Table of Final Training Specifications.
4. Sync to C:/Users/luche/Downloads/tiny-llm-speed-plan.md.
"""

from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parent.parent
PLAN_PATH = ROOT / "tiny-llm-speed-plan.md"
DOWNLOADS_PATH = Path("C:/Users/luche/Downloads/tiny-llm-speed-plan.md")

with open(PLAN_PATH, "r", encoding="utf-8") as f:
    text = f.read()

# 1. Prepare Re-Verification Suite text to insert before Phase 6
reverification_section = """### 5. Step 5: Post-Fix Re-Verification Suite Results (Completed)

Following the label-shift fix, three mandatory re-verifications were executed on real hardware:

#### 1. Shape Re-Verification (`4L_768H_1KV` vs `6L_768H_1KV`)
- **Setup**: `4L_768H_1KV` trained for 250M tokens under identical conditions to 6L Run C (45% Cosmo / 45% FineWeb / 10% TinyStories, micro-bs 4, grad accum 8, LR 1e-3, seed 42, governor 83/78).
- **Results (`results/shape_comparison_reverification.csv`)**:
  - `val_wikipedia_neutral`: **6L = 2.9364 CE** (1.0338 BPB) vs **4L = 3.0277 CE** (1.0659 BPB) -> **6L Lead = -0.0913 CE** (-0.0321 BPB).
  - `val_cosmopedia`: **6L = 2.1149 CE** vs **4L = 2.2214 CE** -> **6L Lead = -0.1065 CE**.
  - `val_fineweb_edu`: **6L = 2.8695 CE** vs **4L = 2.9667 CE** -> **6L Lead = -0.0972 CE**.
  - `val_tinystories`: **6L = 1.6175 CE** vs **4L = 1.6974 CE** -> **6L Lead = -0.0799 CE**.
- **Significance**: The -0.0913 CE lead on neutral Wikipedia exceeds the $\ge 0.05$ CE confirmation threshold by $+0.0413$ CE ($12.0\times$ the provisional validation noise floor $\Delta_{\text{val}} = 0.0076$).
- **Verdict**: `6L_768H_1KV` is **decisively confirmed as the winning architecture**. Second seed (seed 43) is not needed.

#### 2. Learning Rate Check at Real Stage A Global Batch (128 Sequences = 262,144 Tokens)
- **Setup**: Evaluated `6L_768H_1KV` on approved Stage A mix (60% FineWeb-Edu / 30% Cosmopedia v2 / 10% TinyStories) at batch 128 sequences (micro-bs 4, grad accum 32) across peak LRs `5e-4`, `1e-3`, `2e-3` for 150M tokens each (572 steps: 28 warmup, flat to 515, 57 terminal cosine decay).
- **Results (`results/lr_reverification_summary.csv`)**:

| Peak LR | Active Time | Pauses (Paused Min) | Active Tok/s | Final Train Loss | Neutral Wiki CE (BPB) | Cosmopedia CE (BPB) | FineWeb-Edu CE (BPB) | TinyStories CE (BPB) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **$5.0 \times 10^{-4}$** | 33.63 min | 84 (7.04 min) | 74,306.6 | 3.0516 | 3.6588 (1.2881) | 2.9109 (1.0122) | 3.4712 (1.2050) | 2.1798 (0.9766) |
| **$1.0 \times 10^{-3}$** | 33.87 min | 94 (7.88 min) | 73,783.8 | 2.8092 | 3.2936 (1.1595) | 2.6667 (0.9273) | 3.2112 (1.1147) | 2.0005 (0.8963) |
| **$2.0 \times 10^{-3}$ (Winner)** | **34.60 min** | **89 (7.49 min)** | **72,223.3** | **2.7670** | **3.2631 (1.1488)** | **2.6130 (0.9086)** | **3.1795 (1.1037)** | **1.9434 (0.8707)** |

- **Verdict**: **$\eta_0 = 2.0 \times 10^{-3}$ is the decisive winner across all 4 validation sets**. Due to the 4x larger global batch size (128 seqs vs 32 seqs), gradient variance is reduced, enabling the model to learn substantially faster and reach lower loss. Peak LR for Stage A is locked at **$2.0 \times 10^{-3}$**.

#### 3. Final Training Codebase Throughput Verification
- **Setup**: Benchmarked on RTX 3070 Ti with PyTorch 2.6 CUDA 12.4 in WSL2 using final production training code (`results/final_training_throughput.csv`):
  - **2,048 Context (Stage A)**: micro_bs=4, grad_ckpt=False -> **74,900.0 tok/s** (269.64M tok/hr) | Step time: 109.4 ms | Peak VRAM: 4,308 MB.
  - **8,192 Context (Stage B)**: micro_bs=1, grad_ckpt=False -> **51,324.2 tok/s** (184.77M tok/hr) | Step time: 159.6 ms | Peak VRAM: 4,444 MB.
  - **32,768 Context (Stage C)**: micro_bs=1, grad_ckpt=True -> **21,569.1 tok/s** (77.65M tok/hr) | Step time: 1,519.2 ms | Peak VRAM: 5,715 MB.

---
"""

target_before_phase6 = """---

## Phase 6: Full staged training run with long-context extension, quality anneal, and fallback decay branches"""

if target_before_phase6 in text:
    text = text.replace(target_before_phase6, reverification_section + "\n## Phase 6: Full staged training run with long-context extension, quality anneal, and fallback decay branches")
    print("Inserted Re-Verification Suite section.")
else:
    print("Warning: target_before_phase6 not found!")

# 2. Update Budget Table and add Summary Table of Final Training Specifications
old_budget_block = """#### Recomputed 12-Hour Staged Training Budget (`6L_768H_1KV`)
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
> - **Branch B (8k Release)**: 100M tokens (~381 optimizer steps, ~0.69 hr) branched from Stage B if Stage C fails entrance criteria."""

new_budget_block = """#### Recomputed 12-Hour Staged Training Budget & Wall-Clock Estimates (`6L_768H_1KV`)
Based on measured active training throughputs on the RTX 3070 Ti (8GB) with PyTorch 2.6 CUDA 12.4 in WSL2 (`results/final_training_throughput.csv`):
- **2k Context (Stage A)**: **74,900.0 tok/s** (269.64M tokens/hr) | micro_bs=4, grad_accum=32, grad_ckpt=False, Peak VRAM: 4,308 MB
- **8k Context (Stage B)**: **51,324.2 tok/s** (184.77M tokens/hr) | micro_bs=1, grad_accum=32, grad_ckpt=False, Peak VRAM: 4,444 MB
- **32k Context (Stage C)**: **21,569.1 tok/s** (77.65M tokens/hr) | micro_bs=1, grad_accum=4, grad_ckpt=True, Peak VRAM: 5,715 MB

| Training Stage | Context Length | Target Tokens | Sequence Count | Global Batch Size | Optimizer Steps | Active Hours | Wall-Clock Hours (Typical 18–22% Pauses) | Wall-Clock Range (7.5%–25% Pause Bounds) | % of Active Time |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Stage A (Short Foundation)** | 2,048 | 2,149.6M | 1,049,600 | 128 seqs (262k tok) | 8,200 | **7.97 hrs** | 9.40 – 9.72 hrs | 8.57 – 10.63 hrs | 73.8% |
| **Stage B (8k Long Extension)** | 8,192 | 250.1M | 30,528 | 32 seqs (262k tok) | 954 | **1.35 hrs** | 1.59 – 1.65 hrs | 1.45 – 1.80 hrs | 12.5% |
| **Stage C (32k Extension & Anneal)** | 32,768 | 114.7M | 3,500 | 4 seqs (131k tok) | 875 | **1.48 hrs** | 1.75 – 1.81 hrs | 1.59 – 1.97 hrs | 13.7% |
| **Active Training Subtotal** | — | **2,514.4M** | **1,083,628** | — | **10,029** | **10.80 hrs** | **12.74 – 13.18 hrs** | **11.61 – 14.40 hrs** | **100.0%** |
| **10% Compute Reserve** | — | — | — | — | — | **1.20 hrs** | 1.42 – 1.46 hrs | 1.29 – 1.60 hrs | — |
| **Total Pipeline Allocation** | — | **2,514.4M** | **1,083,628** | — | **10,029** | **12.00 hrs** | **14.16 – 14.64 hrs** | **12.90 – 16.00 hrs** | — |

> [!IMPORTANT]
> **WALL-CLOCK TIME & TEMPERATURE GOVERNOR DYNAMICS**:
> - Training time is strictly budgeted by **active compute hours** (10.80 active hours for Stages A–C + 1.20 active hours reserve = 12.00 active hours total).
> - Thermal pauses under the approved **83°C pause / 78°C resume** threshold were empirically measured across all runs:
>   - Step 2A (4L reverification): 3.27 min paused across 45.77 min wall clock = **7.1% pause share** (7.7% of active time).
>   - Step 2B (6L LR check, 3 trials of 150M tokens each): 22.41 min paused across 124.53 min wall clock = **18.0% pause share** (21.9% of active time).
> - At the measured steady-state overhead of 18–22%, the 10.80 active hours of training will require **12.7 to 13.2 wall-clock hours**.
> - The governor automatically alerts if cumulative pauses exceed 25% of elapsed wall-clock time (which caps maximum wall-clock duration at 14.40 wall hours for active training).
>
> **DECAY BRANCHES (ON-DEMAND ONLY)**:
> Decay branches are non-default fallback checkpoints branched on demand; they do NOT consume main-line training budget:
> - **Branch A (2k Release)**: 100M tokens (~381 optimizer steps, ~0.37 hr) branched from Stage A if a dedicated 2k release model is needed.
> - **Branch B (8k Release)**: 100M tokens (~381 optimizer steps, ~0.54 hr) branched from Stage B if Stage C fails entrance criteria.

#### Summary Table of Final Training Specifications

| Specification Item | Final Chosen Setting | Rationale / Source |
| :--- | :--- | :--- |
| **Model Architecture** | `6L_768H_1KV` (42.28M params) | Won iso-token (-0.1371 CE) and iso-wall (-0.1005 CE) head-to-head; re-verified in Step 2A (+0.0913 CE lead over 4L on Wikipedia). |
| **Stage A Data Mixture** | **60% FineWeb-Edu / 30% Cosmopedia v2 / 10% TinyStories** | Interpolates FineWeb-Edu encyclopedic breadth (2.8349 CE) with Cosmopedia v2 reasoning and TinyStories fluency. |
| **Stage A Peak Learning Rate** | **$2.0 \times 10^{-3}$** | Decisive winner in Step 2B 3-point sweep at batch 128 (3.2631 CE on Wiki vs 3.2936 at 1e-3 and 3.6588 at 5e-4). |
| **Stage A Global Batch Size** | **128 sequences (262,144 tokens)** | Standard foundation batch size; allows efficient gradient accumulation while stabilizing optimization. |
| **Micro-Batch Size (Stage A)** | **4 sequences (8,192 tokens)** | Tested in `results/micro_batch_vram_test.csv`; caps total GPU memory at ~4.3 GB (safely below 5.0 GB limit). |
| **Gradient Accumulation Steps** | **32 steps** | $4 \text{ seqs} \times 32 = 128 \text{ sequences}$ ($262,144\text{ tokens}$ per optimizer step). |
| **Warmup Schedule** | **100 optimizer steps (~1.2% of Stage A)** | Linear ramp from 0 to peak LR $2.0 \times 10^{-3}$ (first 26.2M tokens). |
| **Learning Rate Schedule** | **Warmup-Stable-Decay (WSD)** | Flat at peak LR through step 8,200; undecayed checkpoint advances directly to Stage B ($0.7 \times \eta_0$). |
| **Checkpoint Interval** | **Every 15 minutes of active training** | Resumable state (weights, optimizer, RNG, dataloader) with rolling 2-checkpoint retention. Verified $\Delta = 0.0031$. |
| **Temperature Governor** | **Pause at $\ge 83^\circ$C, resume at $\le 78^\circ$C** | Monitored between optimizer steps; pause log written to `results/pause_log.csv`; alerts if pause share $> 25\%$. |
| **Pre-Flight Regression Test** | **Mandatory dummy sequence test** | Asserts `[1,2,3,4,5] -> x:[1,2,3,4], y:[2,3,4,5]` and verifies loss function does not apply secondary shift. |"""

if old_budget_block in text:
    text = text.replace(old_budget_block, new_budget_block)
    print("Replaced budget table and added summary specification table.")
else:
    print("Warning: old_budget_block not matched!")

with open(PLAN_PATH, "w", encoding="utf-8") as f:
    f.write(text)

# Sync to Downloads copy
if DOWNLOADS_PATH.parent.exists():
    try:
        shutil.copyfile(PLAN_PATH, DOWNLOADS_PATH)
        print(f"Synced update to {DOWNLOADS_PATH}")
    except Exception as e:
        print(f"Warning: could not sync to {DOWNLOADS_PATH}: {e}")

print("Plan update complete.")
