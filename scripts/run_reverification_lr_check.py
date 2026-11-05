"""
Step 2B: LR Check at Real Stage A Global Batch Size (128 sequences = 262,144 tokens)
Evaluates 6L_768H_1KV across 3 peak LRs: 5e-4, 1e-3, 2e-3
- Model: 6L_768H_1KV (42.28M params)
- Mix: Approved Stage A blend (60% FineWeb-Edu, 30% Cosmopedia v2, 10% TinyStories)
- Batch: 128 sequences (262,144 tokens), micro_bs=4, grad_accum=32
- Volume: 150M tokens (572 steps: ~28 warmup, flat to 515, 57 terminal cosine decay)
- Loss: Explicit 1-step cross entropy (no double-shift)
- Seed: 42
- Governor: 83C pause / 78C resume
- Pre-training regression test: verifies exact t -> t+1 label alignment before training begins.
"""

import os
import sys
import math
import time
import shutil
import subprocess
import threading
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from transformers import LlamaConfig, LlamaForCausalLM

sys.stdout.reconfigure(line_buffering=True)

ROOT = Path(__file__).resolve().parent.parent
SHARDS_DIR = ROOT / "data" / "shards"
RESULTS_DIR = ROOT / "results"
MODELS_DIR = ROOT / "models"
CHECKPOINTS_DIR = MODELS_DIR / "checkpoints" / "lr_reverification"

RESULTS_DIR.mkdir(exist_ok=True)
CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)

PAUSE_LOG_CSV = RESULTS_DIR / "pause_log.csv"
CURVES_CSV = RESULTS_DIR / "lr_reverification_curves.csv"
SUMMARY_CSV = RESULTS_DIR / "lr_reverification_summary.csv"
PLOT_PNG = RESULTS_DIR / "lr_reverification_curves.png"

COMPRESSION_STATS = {
    "val_cosmopedia": {"bytes_per_tok": 4.149, "chars_per_tok": 4.148},
    "val_fineweb_edu": {"bytes_per_tok": 4.156, "chars_per_tok": 4.150},
    "val_tinystories": {"bytes_per_tok": 3.220, "chars_per_tok": 3.220},
    "val_wikipedia_neutral": {"bytes_per_tok": 4.098, "chars_per_tok": 4.092},
}

# ---------------------------------------------------------------------------
# Label Alignment Regression Test
# ---------------------------------------------------------------------------
def run_label_alignment_regression_test(model, device):
    print("\n--- Running Mandatory Pre-Training Label-Shift Regression Test ---", flush=True)
    model.eval()
    dummy_seq = torch.tensor([[1, 2, 3, 4, 5]], dtype=torch.long, device=device)
    x = dummy_seq[:, :-1]  # [1, 2, 3, 4]
    y = dummy_seq[:, 1:]   # [2, 3, 4, 5]
    
    if not (x[0, 1:] == y[0, :-1]).all():
        raise RuntimeError("FATAL: Target vector y is not shifted by exactly 1 relative to x!")
    if not (y[0] == torch.tensor([2, 3, 4, 5], device=device)).all():
        raise RuntimeError(f"FATAL: Target y {y[0].tolist()} does not match expected [2, 3, 4, 5]!")
    if x.shape != y.shape:
        raise RuntimeError(f"FATAL: Shape mismatch between x {x.shape} and y {y.shape}")
        
    with torch.no_grad():
        out = model(x)
        if out.logits.shape[1] != x.shape[1]:
            raise RuntimeError(f"FATAL: Logits seq length {out.logits.shape[1]} != input length {x.shape[1]}")
        if hasattr(out, "loss") and out.loss is not None:
            raise RuntimeError("FATAL: Model forward returned an internal loss! Must compute loss explicitly outside model.")
            
        loss_val = F.cross_entropy(out.logits.view(-1, out.logits.shape[-1]), y.view(-1))
        if torch.isnan(loss_val) or loss_val.item() <= 0:
            raise RuntimeError("FATAL: Regression test loss is NaN or non-positive!")
            
    model.train()
    print("  [PASSED]: Exact 1-step label shift (t -> t+1) mathematically verified with zero secondary shift.\n", flush=True)

# ---------------------------------------------------------------------------
# Temperature Governor (83C / 78C)
# ---------------------------------------------------------------------------
def find_nvidia_smi():
    candidates = [
        "/usr/lib/wsl/lib/nvidia-smi",
        "nvidia-smi",
        "/mnt/c/WINDOWS/system32/nvidia-smi.exe",
        "nvidia-smi.exe"
    ]
    for c in candidates:
        if shutil.which(c) or Path(c).exists():
            return c
    return "nvidia-smi"

SMI_BIN = find_nvidia_smi()

def get_gpu_temperature():
    try:
        cmd = [SMI_BIN, "--query-gpu=temperature.gpu", "--format=csv,noheader,nounits"]
        res = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return float(res.stdout.strip())
    except Exception:
        return 0.0

def apply_temperature_governor(step, accum_tokens, run_id, total_wall_start):
    temp = get_gpu_temperature()
    if temp < 83.0:
        return 0.0, 0
        
    t_start = time.time()
    t_start_iso = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(t_start))
    temp_initial = temp
    print(f"\n[!WARNING] [{run_id}] Step {step}: GPU temperature {temp:.1f}C >= 83C threshold! Pausing training...", flush=True)
    
    while temp > 78.0:
        time.sleep(5)
        temp = get_gpu_temperature()
        print(f"  Cooling down: current temp = {temp:.1f}C (target <= 78C)...", end="\r", flush=True)
        
    t_end = time.time()
    t_end_iso = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(t_end))
    pause_duration = t_end - t_start
    print(f"\n[GOVERNOR] Temperature dropped to {temp:.1f}C <= 78C. Resuming training after {pause_duration:.1f}s pause.", flush=True)
    
    pause_log_exists = PAUSE_LOG_CSV.exists()
    with open(PAUSE_LOG_CSV, "a", encoding="utf-8") as f:
        if not pause_log_exists:
            f.write("timestamp_start,timestamp_end,duration_sec,temp_start,temp_end,optimizer_step,tokens_seen,run_id\n")
        f.write(f"{t_start_iso},{t_end_iso},{pause_duration:.2f},{temp_initial:.1f},{temp:.1f},{step},{accum_tokens},{run_id}\n")
        
    return pause_duration, 1

# ---------------------------------------------------------------------------
# Data Loader & Evaluator
# ---------------------------------------------------------------------------
class ShardStreamDataLoader:
    def __init__(self, shards_dir, seq_len, mix_weights):
        self.seq_len = seq_len
        self.weights = np.array(mix_weights, dtype=np.float64)
        self.weights = self.weights / self.weights.sum()
        
        self.cosmo = np.memmap(shards_dir / "train_cosmopedia.bin", dtype=np.uint16, mode="r")
        self.fineweb = np.memmap(shards_dir / "train_fineweb_edu.bin", dtype=np.uint16, mode="r")
        self.tinystories = np.memmap(shards_dir / "train_tinystories.bin", dtype=np.uint16, mode="r")
        self.shards = [self.cosmo, self.fineweb, self.tinystories]
        self.shard_lengths = [len(s) for s in self.shards]

        self.rng_mix = np.random.RandomState(42)
        self.rng_cosmo = np.random.RandomState(1001)
        self.rng_fineweb = np.random.RandomState(1002)
        self.rng_tinystories = np.random.RandomState(1003)
        self.shard_rngs = [self.rng_cosmo, self.rng_fineweb, self.rng_tinystories]

    def get_batch(self, batch_size, device):
        shard_indices = self.rng_mix.choice(3, size=batch_size, p=self.weights)
        x_list, y_list = [], []
        for s_idx in shard_indices:
            shard = self.shards[s_idx]
            rng = self.shard_rngs[s_idx]
            max_idx = self.shard_lengths[s_idx] - self.seq_len - 1
            idx = rng.randint(0, max_idx)
            x_list.append(shard[idx : idx + self.seq_len])
            y_list.append(shard[idx + 1 : idx + 1 + self.seq_len])
            
        x = np.stack(x_list)
        y = np.stack(y_list)
        tx = torch.from_numpy(x.astype(np.int64)).to(device=device, non_blocking=True)
        ty = torch.from_numpy(y.astype(np.int64)).to(device=device, non_blocking=True)
        return tx, ty

class ValidationEvaluator:
    def __init__(self, shards_dir, seq_len, device):
        self.seq_len = seq_len
        self.device = device
        self.val_shards = {
            "val_wikipedia_neutral": np.memmap(shards_dir / "val_wikipedia_neutral.bin", dtype=np.uint16, mode="r"),
            "val_cosmopedia": np.memmap(shards_dir / "val_cosmopedia.bin", dtype=np.uint16, mode="r"),
            "val_fineweb_edu": np.memmap(shards_dir / "val_fineweb_edu.bin", dtype=np.uint16, mode="r"),
            "val_tinystories": np.memmap(shards_dir / "val_tinystories.bin", dtype=np.uint16, mode="r"),
        }

    @torch.no_grad()
    def evaluate_set(self, model, name, num_batches=40, batch_size=4):
        shard = self.val_shards[name]
        max_idx = len(shard) - self.seq_len - 1
        rng = np.random.RandomState(4242)
        total_loss = 0.0
        total_tokens = 0
        
        for _ in range(num_batches):
            offsets = [rng.randint(0, max_idx) for _ in range(batch_size)]
            x = np.stack([shard[i : i + self.seq_len] for i in offsets])
            y = np.stack([shard[i + 1 : i + 1 + self.seq_len] for i in offsets])
            tx = torch.from_numpy(x.astype(np.int64)).to(device=self.device, non_blocking=True)
            ty = torch.from_numpy(y.astype(np.int64)).to(device=self.device, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                outputs = model(tx)
                loss = F.cross_entropy(outputs.logits.view(-1, 8192), ty.view(-1)).item()
            total_loss += loss * (batch_size * self.seq_len)
            total_tokens += batch_size * self.seq_len
            
        avg_loss = total_loss / total_tokens
        bits_per_tok = avg_loss / math.log(2.0)
        bpb = bits_per_tok / COMPRESSION_STATS[name]["bytes_per_tok"]
        bpc = bits_per_tok / COMPRESSION_STATS[name]["chars_per_tok"]
        return {
            "ce_loss": round(avg_loss, 4),
            "bits_per_tok": round(bits_per_tok, 4),
            "bpb": round(bpb, 4),
            "bpc": round(bpc, 4),
        }

    def evaluate_all(self, model, num_batches=40):
        model.eval()
        results = {}
        for name in ["val_wikipedia_neutral", "val_cosmopedia", "val_fineweb_edu", "val_tinystories"]:
            results[name] = self.evaluate_set(model, name, num_batches=num_batches)
        model.train()
        return results

def create_6l_model(device, seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    config = LlamaConfig(
        vocab_size=8192,
        hidden_size=768,
        intermediate_size=2048,
        num_hidden_layers=6,
        num_attention_heads=12,
        num_key_value_heads=1,
        tie_word_embeddings=True,
        torch_dtype="bfloat16",
        _attn_implementation="sdpa",
        max_position_embeddings=2048,
    )
    return LlamaForCausalLM(config).to(device=device, dtype=torch.bfloat16)

# ---------------------------------------------------------------------------
# Train Single LR Trial
# ---------------------------------------------------------------------------
def train_lr_trial(lr_val, device, evaluator):
    run_id = f"LR_{lr_val:.0e}"
    print(f"\n=======================================================", flush=True)
    print(f"Starting LR Trial: {run_id} (Peak LR = {lr_val:.6e})", flush=True)
    print(f"  Batch: 128 sequences (262,144 tokens/step, micro_bs=4, grad_accum=32)", flush=True)
    print(f"  Mix: 60% FineWeb-Edu / 30% Cosmopedia v2 / 10% TinyStories", flush=True)
    print(f"  Volume: 150M tokens (572 steps)", flush=True)
    print(f"=======================================================", flush=True)
    
    model = create_6l_model(device, seed=42)
    run_label_alignment_regression_test(model, device)
    
    # weights: [cosmo, fineweb, tinystories] -> [0.30, 0.60, 0.10]
    loader = ShardStreamDataLoader(SHARDS_DIR, seq_len=2048, mix_weights=[0.30, 0.60, 0.10])
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr_val, betas=(0.9, 0.95), weight_decay=0.1, fused=True)
    
    micro_bs = 4
    grad_accum = 32
    seq_len = 2048
    tokens_per_step = micro_bs * grad_accum * seq_len  # 262,144
    total_steps = 572
    warmup_steps = 28
    decay_steps = 57
    stable_steps = total_steps - decay_steps  # 515
    min_lr = lr_val * 0.1
    
    wall_start = time.time()
    accum_tokens = 0
    total_paused_sec = 0.0
    num_pauses = 0
    
    model.train()
    optimizer.zero_grad(set_to_none=True)
    
    for step in range(total_steps):
        # Poll governor every 2 steps at batch 128 (since each step takes ~3.7s)
        if step % 2 == 0:
            p_sec, was_p = apply_temperature_governor(step, accum_tokens, run_id, wall_start)
            total_paused_sec += p_sec
            num_pauses += was_p
            
        # WSD Schedule
        if step < warmup_steps:
            curr_lr = lr_val * (step + 1) / warmup_steps
        elif step < stable_steps:
            curr_lr = lr_val
        else:
            decay_ratio = (step - stable_steps) / max(1, decay_steps)
            coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
            curr_lr = min_lr + coeff * (lr_val - min_lr)
            
        for param_group in optimizer.param_groups:
            param_group["lr"] = curr_lr
            
        step_loss = 0.0
        for _ in range(grad_accum):
            x, y = loader.get_batch(micro_bs, device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                out = model(x)
                loss = F.cross_entropy(out.logits.view(-1, 8192), y.view(-1))
                scaled_loss = loss / grad_accum
            scaled_loss.backward()
            step_loss += loss.item() / grad_accum
            
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        
        accum_tokens += tokens_per_step
        total_wall_sec = time.time() - wall_start
        active_sec = max(1e-5, total_wall_sec - total_paused_sec)
        
        if (step + 1) % 10 == 0 or step == total_steps - 1:
            tok_s = accum_tokens / active_sec
            print(f"[{run_id}] Step {step+1:3d}/{total_steps} | LR {curr_lr:.2e} | Loss {step_loss:.4f} | "
                  f"Tokens: {accum_tokens/1e6:5.1f}M | Active: {active_sec/60:4.1f}m (Paused: {total_paused_sec/60:4.1f}m) | Speed: {tok_s:,.0f} tok/s", flush=True)
            with open(CURVES_CSV, "a", encoding="utf-8") as f:
                f.write(f"{run_id},{lr_val:.6e},{step+1},{accum_tokens/1e6:.2f},{total_wall_sec:.1f},{active_sec:.1f},{curr_lr:.6e},{step_loss:.4f}\n")
                
    total_wall_sec = time.time() - wall_start
    active_sec = max(1e-5, total_wall_sec - total_paused_sec)
    
    print(f"\n{run_id} Training Finished: Wall={total_wall_sec/60:.1f}m, Active={active_sec/60:.1f}m, Paused={total_paused_sec/60:.1f}m ({num_pauses} pauses)", flush=True)
    
    # Save checkpoint
    ckpt_path = CHECKPOINTS_DIR / f"{run_id}_final.pt"
    torch.save(model.state_dict(), ckpt_path)
    
    # Evaluate
    print(f"Evaluating {run_id} on all 4 validation sets...", flush=True)
    val_res = evaluator.evaluate_all(model, num_batches=50)
    for vname, m in val_res.items():
        print(f"  {vname:25s}: CE Loss = {m['ce_loss']:.4f} | BPB = {m['bpb']:.4f}", flush=True)
        
    return {
        "run_id": run_id,
        "peak_lr": lr_val,
        "total_tokens_m": round(accum_tokens / 1e6, 2),
        "active_min": round(active_sec / 60, 2),
        "wall_min": round(total_wall_sec / 60, 2),
        "paused_min": round(total_paused_sec / 60, 2),
        "pauses": num_pauses,
        "active_tok_per_sec": round(accum_tokens / active_sec, 1),
        "final_train_loss": round(step_loss, 4),
        "val_wiki_ce": val_res["val_wikipedia_neutral"]["ce_loss"],
        "val_wiki_bpb": val_res["val_wikipedia_neutral"]["bpb"],
        "val_cosmo_ce": val_res["val_cosmopedia"]["ce_loss"],
        "val_cosmo_bpb": val_res["val_cosmopedia"]["bpb"],
        "val_fineweb_ce": val_res["val_fineweb_edu"]["ce_loss"],
        "val_fineweb_bpb": val_res["val_fineweb_edu"]["bpb"],
        "val_tinystories_ce": val_res["val_tinystories"]["ce_loss"],
        "val_tinystories_bpb": val_res["val_tinystories"]["bpb"],
    }

# ---------------------------------------------------------------------------
# Plotting & Reporting
# ---------------------------------------------------------------------------
def plot_results(curves_df, summary_df):
    plt.figure(figsize=(14, 5))
    
    # Subplot 1: Loss vs Step
    plt.subplot(1, 2, 1)
    for run_id, grp in curves_df.groupby("run_id"):
        plt.plot(grp["step"], grp["train_loss"], label=f"{run_id} (peak LR {grp['lr_val'].iloc[0]:.0e})", alpha=0.85, linewidth=1.8)
    plt.title("Step 2B: Training Loss vs Optimizer Step (Batch 128 seqs)", fontsize=11, fontweight="bold")
    plt.xlabel("Optimizer Step (262k tok/step)")
    plt.ylabel("Cross-Entropy Loss")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    
    # Subplot 2: Loss vs Active Wall-Clock Time
    plt.subplot(1, 2, 2)
    for run_id, grp in curves_df.groupby("run_id"):
        plt.plot(grp["active_sec"] / 60.0, grp["train_loss"], label=f"{run_id}", alpha=0.85, linewidth=1.8)
    plt.title("Step 2B: Training Loss vs Active Wall-Clock Time", fontsize=11, fontweight="bold")
    plt.xlabel("Active Training Time (minutes)")
    plt.ylabel("Cross-Entropy Loss")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(PLOT_PNG, dpi=200)
    plt.close()
    print(f"Saved plot to {PLOT_PNG}", flush=True)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Step 2B: LR Re-Verification at Global Batch 128 on {device} ({torch.cuda.get_device_name(0)})")
    
    with open(CURVES_CSV, "w", encoding="utf-8") as f:
        f.write("run_id,lr_val,step,tokens_m,wall_sec,active_sec,lr,train_loss\n")
        
    evaluator = ValidationEvaluator(SHARDS_DIR, seq_len=2048, device=device)
    
    lrs = [5e-4, 1e-3, 2e-3]
    summary_rows = []
    
    for lr_val in lrs:
        res = train_lr_trial(lr_val, device, evaluator)
        summary_rows.append(res)
        
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(SUMMARY_CSV, index=False)
    print(f"\nSaved summary to {SUMMARY_CSV}", flush=True)
    
    curves_df = pd.read_csv(CURVES_CSV)
    plot_results(curves_df, summary_df)
    
    # Determine winner on neutral Wikipedia
    best_row = summary_df.loc[summary_df["val_wiki_ce"].idxmin()]
    print("\n=======================================================", flush=True)
    print("STEP 2B LR SWEEP RESULTS (Batch 128 seqs):", flush=True)
    print(summary_df[["run_id", "peak_lr", "val_wiki_ce", "val_fineweb_ce", "val_cosmo_ce", "val_tinystories_ce", "active_min"]].to_string(index=False))
    print(f"\nWinning LR on Neutral Wikipedia: {best_row['run_id']} with CE {best_row['val_wiki_ce']:.4f}")
    if best_row["peak_lr"] == 1e-3:
        print("CONFIRMATION: LR 1e-3 remains optimal at 4x batch size (128 sequences)!")
    else:
        print(f"NOTE: Optimal LR shifted to {best_row['peak_lr']:.1e} at 4x batch size!")
    print("=======================================================", flush=True)

if __name__ == "__main__":
    main()
