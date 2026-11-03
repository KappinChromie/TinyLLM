"""
Extended Learning Rate Sweep on 4L_768H_1KV
- Tests peak LRs in {5e-4, 1e-3, 2e-3}
- Runs for exactly 20 minutes (1,200 seconds) per LR run on the REAL Stage A mix:
    45% Cosmopedia v2, 45% FineWeb-Edu, 10% TinyStories
- Context length: 2,048 tokens
- Global batch size: 32 sequences (65,536 tokens)
  (Micro-batch = 8, Gradient Accumulation = 4 steps)
- Schedule: WSD with 100-step linear warmup, then flat at peak LR.
- Identical random seed (42) across runs for strict fair comparison.
- Outputs:
    results/lr_sweep_extended_curves.csv
    results/lr_sweep_extended_summary.csv
    results/lr_sweep_extended.png
"""

import os
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from transformers import LlamaConfig, LlamaForCausalLM

sys.stdout.reconfigure(line_buffering=True)

ROOT = Path(__file__).resolve().parent.parent
SHARDS_DIR = ROOT / "data" / "shards"
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

CURVES_CSV = RESULTS_DIR / "lr_sweep_extended_curves.csv"
SUMMARY_CSV = RESULTS_DIR / "lr_sweep_extended_summary.csv"
PLOT_PNG = RESULTS_DIR / "lr_sweep_extended.png"

# Hyperparameters
SHAPE_CFG = {
    "num_hidden_layers": 4,
    "hidden_size": 768,
    "intermediate_size": 2048,
    "num_attention_heads": 12,
    "num_key_value_heads": 1,
    "vocab_size": 8192,
}

LRS = [5e-4, 1e-3, 2e-3]
TARGET_DURATION_SEC = 1200  # 20 minutes
SEQ_LEN = 2048
MICRO_BS = 8
GRAD_ACCUM_STEPS = 4
GLOBAL_BS_TOKENS = MICRO_BS * GRAD_ACCUM_STEPS * SEQ_LEN  # 65,536 tokens
WARMUP_STEPS = 100  # ~76s warmup

class StageAMixDataLoader:
    def __init__(self, shards_dir, seq_len):
        self.seq_len = seq_len
        cosmo_path = shards_dir / "train_cosmopedia.bin"
        fineweb_path = shards_dir / "train_fineweb_edu.bin"
        tinystories_path = shards_dir / "train_tinystories.bin"
        
        print("Opening Stage A shards (memmapped):", flush=True)
        print(f"  Cosmopedia: {cosmo_path.name}", flush=True)
        print(f"  FineWeb-Edu: {fineweb_path.name}", flush=True)
        print(f"  TinyStories: {tinystories_path.name}", flush=True)
        
        self.shards = [
            np.memmap(cosmo_path, dtype=np.uint16, mode="r"),
            np.memmap(fineweb_path, dtype=np.uint16, mode="r"),
            np.memmap(tinystories_path, dtype=np.uint16, mode="r"),
        ]
        self.weights = np.array([0.45, 0.45, 0.10], dtype=np.float64)
        self.shard_lengths = [len(s) for s in self.shards]
        
    def get_batch(self, batch_size, rng, device):
        # Choose shards per sequence according to weights
        shard_indices = rng.choice(3, size=batch_size, p=self.weights)
        
        x_list, y_list = [], []
        for s_idx in shard_indices:
            shard = self.shards[s_idx]
            max_idx = self.shard_lengths[s_idx] - self.seq_len - 1
            idx = rng.randint(0, max_idx)
            x_list.append(shard[idx : idx + self.seq_len])
            y_list.append(shard[idx + 1 : idx + 1 + self.seq_len])
            
        x = np.stack(x_list)
        y = np.stack(y_list)
        
        tx = torch.from_numpy(x.astype(np.int64)).to(device=device, non_blocking=True)
        ty = torch.from_numpy(y.astype(np.int64)).to(device=device, non_blocking=True)
        return tx, ty

def run_single_lr(lr, loader, device):
    print(f"\n=======================================================", flush=True)
    print(f"  STARTING RUN: LR = {lr:.1e} | Target Duration = {TARGET_DURATION_SEC}s (20.0 min)", flush=True)
    print(f"=======================================================", flush=True)
    
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    # Clean reproducible seed across all runs
    torch.manual_seed(42)
    np.random.seed(42)
    rng = np.random.RandomState(42)
    
    cfg = LlamaConfig(
        vocab_size=SHAPE_CFG["vocab_size"],
        hidden_size=SHAPE_CFG["hidden_size"],
        intermediate_size=SHAPE_CFG["intermediate_size"],
        num_hidden_layers=SHAPE_CFG["num_hidden_layers"],
        num_attention_heads=SHAPE_CFG["num_attention_heads"],
        num_key_value_heads=SHAPE_CFG["num_key_value_heads"],
        tie_word_embeddings=True,
        torch_dtype="bfloat16",
        _attn_implementation="sdpa",
        max_position_embeddings=SEQ_LEN,
    )
    
    model = LlamaForCausalLM(cfg).to(device=device, dtype=torch.bfloat16)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.1, fused=True)
    
    curve_records = []
    step = 0
    total_tokens_seen = 0
    step_losses = []
    
    t_start = time.perf_counter()
    last_print = t_start
    
    while True:
        t_elapsed = time.perf_counter() - t_start
        if t_elapsed >= TARGET_DURATION_SEC:
            break
            
        # Determine LR via WSD schedule (warmup then flat)
        if step < WARMUP_STEPS:
            curr_lr = lr * (step + 1) / WARMUP_STEPS
        else:
            curr_lr = lr
            
        for pg in optimizer.param_groups:
            pg["lr"] = curr_lr
            
        optimizer.zero_grad(set_to_none=True)
        accum_loss = 0.0
        
        for micro_step in range(GRAD_ACCUM_STEPS):
            x, y = loader.get_batch(MICRO_BS, rng, device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                outputs = model(x, labels=y)
                loss = outputs.loss / GRAD_ACCUM_STEPS
                
            loss.backward()
            accum_loss += loss.item()
            total_tokens_seen += MICRO_BS * SEQ_LEN
            
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        step += 1
        step_losses.append(accum_loss)
        
        # Log every 10 steps
        if step % 10 == 0 or step == 1:
            t_now = time.perf_counter() - t_start
            window = step_losses[-50:] if len(step_losses) >= 50 else step_losses
            smoothed_loss = float(np.mean(window))
            tok_per_sec = total_tokens_seen / max(1e-3, t_now)
            
            curve_records.append({
                "lr": lr,
                "step": step,
                "wall_clock_sec": round(t_now, 2),
                "loss": round(accum_loss, 4),
                "smoothed_loss": round(smoothed_loss, 4),
                "current_lr": curr_lr,
                "tokens_seen": total_tokens_seen,
                "tok_per_sec": round(tok_per_sec, 1)
            })
            
            if (time.perf_counter() - last_print) >= 30.0 or step == 10:
                last_print = time.perf_counter()
                print(f"  Step {step:4d} | Elapsed: {t_now:6.1f}s / {TARGET_DURATION_SEC}s | Loss: {accum_loss:.4f} (smooth: {smoothed_loss:.4f}) | LR: {curr_lr:.2e} | Speed: {tok_per_sec:,.0f} tok/s", flush=True)
                
    torch.cuda.synchronize()
    total_time = time.perf_counter() - t_start
    final_tokens_per_sec = total_tokens_seen / total_time
    peak_vram = torch.cuda.max_memory_allocated(device) / (1024 * 1024)
    
    # Compute metrics over last 100 steps
    tail = step_losses[-100:] if len(step_losses) >= 100 else step_losses
    final_loss = float(np.mean(tail))
    min_loss = float(min(step_losses))
    
    print(f"\n--- Run LR {lr:.1e} Complete ---", flush=True)
    print(f"  Total Steps: {step:,} | Elapsed: {total_time:.1f}s | Tokens: {total_tokens_seen:,}", flush=True)
    print(f"  Final Loss (last 100 avg): {final_loss:.4f} | Min Loss: {min_loss:.4f}", flush=True)
    print(f"  Effective Throughput: {final_tokens_per_sec:,.1f} tok/s | Peak VRAM: {peak_vram:.1f} MiB", flush=True)
    
    del model, optimizer
    torch.cuda.empty_cache()
    
    summary = {
        "lr": lr,
        "total_steps": step,
        "elapsed_sec": round(total_time, 2),
        "tokens_seen": total_tokens_seen,
        "final_loss": round(final_loss, 4),
        "min_loss": round(min_loss, 4),
        "tok_per_sec": round(final_tokens_per_sec, 1),
        "peak_vram_mib": round(peak_vram, 1),
    }
    return curve_records, summary

def main():
    if not torch.cuda.is_available():
        print("CUDA not available.")
        sys.exit(1)
        
    device = torch.device("cuda:0")
    print("=" * 70, flush=True)
    print("  EXTENDED LR SWEEP: 4L_768H_1KV (5e-4, 1e-3, 2e-3 | 20 MIN EACH)", flush=True)
    print("  Global Batch: 32 seqs (65,536 tok) | WSD Schedule (100-step warmup)", flush=True)
    print(f"  Device: {torch.cuda.get_device_name(device)}", flush=True)
    print("=" * 70, flush=True)
    
    loader = StageAMixDataLoader(SHARDS_DIR, SEQ_LEN)
    
    all_curves = []
    all_summaries = []
    
    for lr in LRS:
        curves, summary = run_single_lr(lr, loader, device)
        all_curves.extend(curves)
        all_summaries.append(summary)
        
        # Incrementally save curves and summary after each run
        df_curves = pd.DataFrame(all_curves)
        df_curves.to_csv(CURVES_CSV, index=False)
        df_sum = pd.DataFrame(all_summaries)
        df_sum.to_csv(SUMMARY_CSV, index=False)
        print(f"Saved interim progress to {CURVES_CSV} and {SUMMARY_CSV}", flush=True)
        
    print("\n" + "=" * 70, flush=True)
    print("  ALL 3 LR RUNS FINISHED — GENERATING PLOTS & FINAL REPORT", flush=True)
    print("=" * 70, flush=True)
    
    # Plot curves
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), dpi=150)
    
    df_curves = pd.DataFrame(all_curves)
    for lr in LRS:
        sub = df_curves[df_curves["lr"] == lr]
        label = f"LR {lr:.0e}"
        ax1.plot(sub["step"], sub["smoothed_loss"], label=label, linewidth=1.8)
        ax2.plot(sub["wall_clock_sec"] / 60.0, sub["smoothed_loss"], label=label, linewidth=1.8)
        
    ax1.set_xlabel("Optimizer Step (Global Batch = 65,536 tok)", fontsize=11)
    ax1.set_ylabel("Loss (50-Step Smoothed)", fontsize=11)
    ax1.set_title("Training Loss vs. Optimizer Step", fontsize=12, fontweight="bold")
    ax1.grid(True, linestyle="--", alpha=0.6)
    ax1.legend(fontsize=10)
    
    ax2.set_xlabel("Wall-Clock Time (Minutes)", fontsize=11)
    ax2.set_ylabel("Loss (50-Step Smoothed)", fontsize=11)
    ax2.set_title("Training Loss vs. Wall-Clock Time", fontsize=12, fontweight="bold")
    ax2.grid(True, linestyle="--", alpha=0.6)
    ax2.legend(fontsize=10)
    
    plt.tight_layout()
    plt.savefig(PLOT_PNG)
    plt.close()
    print(f"Plot saved to {PLOT_PNG}", flush=True)
    
    df_sum = pd.DataFrame(all_summaries)
    print("\n=== FINAL SUMMARY ===")
    print(df_sum.to_string(), flush=True)

if __name__ == "__main__":
    main()
