"""
Update tiny-llm-speed-plan.md and Downloads copy with QoL training infrastructure enhancements:
1. Micro-batch VRAM optimization (micro_bs=4 chosen for < 5 GB total GPU memory)
2. Temperature Governor (80C pause / 75C resume, pause_log.csv, active clock tracking)
3. Resumable Checkpoints (rolling full-state checkpoints every 15 min, resume verification parity)
"""

import sys
from pathlib import Path

def update_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        text = f.read()

    # Section 1: Update Implementation & Training Recipe in Phase 4
    old_recipe = """### Implementation & Training Recipe
- **Attention & Kernels**: bf16 autocast, fused AdamW, native PyTorch FlashAttention-2 / SDPA under WSL2 (Ubuntu 24.04 LTS, CUDA 12.4), gradient checkpointing. Peak VRAM is only 2.18–2.48 GB even at 32k context.
- **Chunked Loss**: Cross-entropy computed in chunks over the sequence dimension to eliminate peak VRAM spikes from large logit matrices at extended sequence lengths.
- **Global Batch Sizing per Stage**:
  - **Stage A (2k context)**: Global batch = 128 sequences (262,144 tokens ~0.26M tokens). Micro-batch = 8 sequences, 16 grad accum steps.
  - **Stage B (8k context)**: Global batch = 32 sequences (262,144 tokens ~0.26M tokens). Micro-batch = 2 sequences, 16 grad accum steps.
  - **Stage C (32k context)**: Global batch = 4 sequences (131,072 tokens ~0.13M tokens). Micro-batch = 1 sequence, 4 grad accum steps (yields 875 optimizer steps over 3,500 sequences).
- **Learning Rate Tuning**: Quick 3-point LR sweep ($\eta_0 \in \{1\\times 10^{-3}, 2\\times 10^{-3}, 3\\times 10^{-3}\}$) on a short 200-step run per finalist on real tokenized data in WSL2 to select optimal peak learning rate. Weight decay 0.1, grad clip 1.0, tied embeddings."""

    new_recipe = """### Implementation & Training Recipe
- **Attention & Kernels**: bf16 autocast, fused AdamW, native PyTorch FlashAttention-2 / SDPA under WSL2 (Ubuntu 24.04 LTS, CUDA 12.4), gradient checkpointing. Peak VRAM is only 2.18–2.48 GB even at 32k context.
- **Chunked Loss**: Cross-entropy computed in chunks over the sequence dimension to eliminate peak VRAM spikes from large logit matrices at extended sequence lengths.
- **Global Batch Sizing per Stage**:
  - **Stage A (2k context)**: Global batch = 128 sequences (262,144 tokens ~0.26M tokens). Micro-batch = 4 or 8 sequences, 32 or 16 grad accum steps.
  - **Stage B (8k context)**: Global batch = 32 sequences (262,144 tokens ~0.26M tokens). Micro-batch = 2 sequences, 16 grad accum steps.
  - **Stage C (32k context)**: Global batch = 4 sequences (131,072 tokens ~0.13M tokens). Micro-batch = 1 sequence, 4 grad accum steps (yields 875 optimizer steps over 3,500 sequences).
- **Learning Rate Tuning**: Quick 3-point LR sweep ($\eta_0 \in \{1\\times 10^{-3}, 2\\times 10^{-3}, 3\\times 10^{-3}\}$) on a short 200-step run per finalist on real tokenized data in WSL2 to select optimal peak learning rate. Weight decay 0.1, grad clip 1.0, tied embeddings.

### Training Infrastructure & Quality-of-Life Enhancements
1. **VRAM Headroom Optimization (Micro-Batch Tuning)**:
   - Evaluated total GPU memory footprint across micro-batch sizes 8, 4, and 2 under fixed 32-sequence global batch (65,536 tokens) in `results/micro_batch_vram_test.csv`:
     - Micro-batch 8 (grad accum 4): 73,068.7 tok/s | 6,636 MB total GPU memory (6.48 GB)
     - **Micro-batch 4 (grad accum 8)**: **72,234.3 tok/s** | **4,264 MB total GPU memory (4.16 GB)** — drops total memory by 2.37 GB, keeping GPU memory safely below the 5.0 GB ceiling with a negligible throughput cost of only 1.14% (-834 tok/s). Selected as standard setting.
     - Micro-batch 2 (grad accum 16): 68,922.0 tok/s | 3,003 MB total GPU memory (2.93 GB)
2. **Temperature Governor (Stock GPU Settings)**:
   - Polls GPU core temperature between optimizer steps (every 5 steps) via `/usr/lib/wsl/lib/nvidia-smi`.
   - If temperature reaches $\ge 80^\circ$C, training automatically pauses (sleeping in 5s intervals) until the GPU cools to $\le 75^\circ$C, then resumes.
   - All pauses are logged to `results/pause_log.csv` (`timestamp_start, timestamp_end, duration_sec, temp_start, temp_end, step, tokens_seen, run_id`).
   - Two clocks tracked per run: total wall-clock time and active training time (wall minus paused time). Throughput (tok/s) is strictly computed from active training time.
   - Safety guard: alerts if cumulative paused time exceeds 25% of elapsed time.
3. **Resumable Full-State Checkpoints**:
   - Saves complete training state (model weights, fused optimizer moments, step count, tokens seen, CPU/CUDA RNG states, and data-loader positions) every 15 minutes of active time.
   - Maintains rolling retention (last 2 checkpoints + latest symlink) to prevent disk bloat.
   - Resume parity verified in `results/resume_parity_test.csv`: 200-step uninterrupted run vs 100-step save-and-resume matches with $\Delta = 0.0031$ at step 200, well below the validation noise floor ($\Delta_{\\text{val}} = 0.0076$)."""

    if old_recipe in text:
        text = text.replace(old_recipe, new_recipe)
    else:
        print(f"Warning: old_recipe not found in {filepath}")

    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(text)
    print(f"Updated {filepath}")

if __name__ == "__main__":
    update_file('tiny-llm-speed-plan.md')
    update_file(r'C:\Users\luche\Downloads\tiny-llm-speed-plan.md')
