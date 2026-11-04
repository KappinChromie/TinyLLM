"""
Noise Floor Estimation on 4L_768H_1KV
Runs 4L_768H_1KV for 20 minutes (1,200s) on Stage A mix with seed 43 and seed 44
(seed 42 already run, final_loss = 4.4780).
Evaluates held-out validation loss across all 4 clean validation sets:
- val_cosmopedia
- val_fineweb_edu
- val_tinystories
- val_wikipedia_neutral
Computes Cross-Entropy loss, Bits/token, BPB (Bits Per Byte), and BPC (Bits Per Character).
Computes the multi-seed empirical spread and defines the noise floor.
Saves to results/noise_estimate.csv and results/noise_report.txt.
"""

import os
import sys
import math
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from transformers import LlamaConfig, LlamaForCausalLM

sys.stdout.reconfigure(line_buffering=True)

ROOT = Path(__file__).resolve().parent.parent
SHARDS_DIR = ROOT / "data" / "shards"
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

OUT_CSV = RESULTS_DIR / "noise_estimate.csv"
OUT_REPORT = RESULTS_DIR / "noise_report.txt"

# Architecture config: 4L_768H_1KV
SHAPE_CFG = {
    "num_hidden_layers": 4,
    "hidden_size": 768,
    "intermediate_size": 2048,
    "num_attention_heads": 12,
    "num_key_value_heads": 1,
    "vocab_size": 8192,
}

PEAK_LR = 1e-3
TARGET_DURATION_SEC = 1200  # 20 minutes
SEQ_LEN = 2048
MICRO_BS = 8
GRAD_ACCUM_STEPS = 4
GLOBAL_BS_TOKENS = MICRO_BS * GRAD_ACCUM_STEPS * SEQ_LEN  # 65,536 tokens
WARMUP_STEPS = 100

# Bytes per token and chars per token measured in Phase 2
COMPRESSION_STATS = {
    "val_cosmopedia": {"bytes_per_tok": 3.478, "chars_per_tok": 3.474},
    "val_fineweb_edu": {"bytes_per_tok": 3.160, "chars_per_tok": 3.148},
    "val_tinystories": {"bytes_per_tok": 3.269, "chars_per_tok": 3.266},
    "val_wikipedia_neutral": {"bytes_per_tok": 2.813, "chars_per_tok": 2.797},
}

class StageAMixDataLoader:
    def __init__(self, shards_dir, seq_len):
        self.seq_len = seq_len
        cosmo_path = shards_dir / "train_cosmopedia.bin"
        fineweb_path = shards_dir / "train_fineweb_edu.bin"
        tinystories_path = shards_dir / "train_tinystories.bin"
        
        self.shards = [
            np.memmap(cosmo_path, dtype=np.uint16, mode="r"),
            np.memmap(fineweb_path, dtype=np.uint16, mode="r"),
            np.memmap(tinystories_path, dtype=np.uint16, mode="r"),
        ]
        self.weights = np.array([0.45, 0.45, 0.10], dtype=np.float64)
        self.shard_lengths = [len(s) for s in self.shards]
        
    def get_batch(self, batch_size, rng, device):
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

class ValidationEvaluator:
    def __init__(self, shards_dir, seq_len, device):
        self.seq_len = seq_len
        self.device = device
        self.val_shards = {
            "val_cosmopedia": np.memmap(shards_dir / "val_cosmopedia.bin", dtype=np.uint16, mode="r"),
            "val_fineweb_edu": np.memmap(shards_dir / "val_fineweb_edu.bin", dtype=np.uint16, mode="r"),
            "val_tinystories": np.memmap(shards_dir / "val_tinystories.bin", dtype=np.uint16, mode="r"),
            "val_wikipedia_neutral": np.memmap(shards_dir / "val_wikipedia_neutral.bin", dtype=np.uint16, mode="r"),
        }
        
    @torch.no_grad()
    def evaluate_set(self, model, name, num_batches=40):
        # 40 batches x 8 sequences = 320 sequences = 655,360 tokens
        shard = self.val_shards[name]
        max_idx = len(shard) - self.seq_len - 1
        rng = np.random.RandomState(12345)  # deterministic eval
        
        total_loss = 0.0
        total_tokens = 0
        
        batch_size = 8
        for _ in range(num_batches):
            offsets = [rng.randint(0, max_idx) for _ in range(batch_size)]
            x = np.stack([shard[i : i + self.seq_len] for i in offsets])
            y = np.stack([shard[i + 1 : i + 1 + self.seq_len] for i in offsets])
            
            tx = torch.from_numpy(x.astype(np.int64)).to(device=self.device, non_blocking=True)
            ty = torch.from_numpy(y.astype(np.int64)).to(device=self.device, non_blocking=True)
            
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                outputs = model(tx, labels=ty)
                loss = outputs.loss.item()
                
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
        
    def evaluate_all(self, model):
        model.eval()
        results = {}
        for name in ["val_cosmopedia", "val_fineweb_edu", "val_tinystories", "val_wikipedia_neutral"]:
            metrics = self.evaluate_set(model, name)
            results[name] = metrics
            
        # Weighted Stage A validation loss (45% cosmo, 45% fineweb, 10% tinystories)
        stage_a_val_loss = (
            0.45 * results["val_cosmopedia"]["ce_loss"] +
            0.45 * results["val_fineweb_edu"]["ce_loss"] +
            0.10 * results["val_tinystories"]["ce_loss"]
        )
        results["stage_a_val_loss"] = round(stage_a_val_loss, 4)
        model.train()
        return results

def train_seed(seed, loader, evaluator, device):
    print(f"\n=======================================================", flush=True)
    print(f"  STARTING RUN: SEED = {seed} | LR = {PEAK_LR:.1e} | Duration = {TARGET_DURATION_SEC}s", flush=True)
    print(f"=======================================================", flush=True)
    
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    torch.manual_seed(seed)
    np.random.seed(seed)
    rng = np.random.RandomState(seed)
    
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
    optimizer = torch.optim.AdamW(model.parameters(), lr=PEAK_LR, weight_decay=0.1, fused=True)
    
    step = 0
    total_tokens_seen = 0
    step_losses = []
    
    t_start = time.perf_counter()
    last_print = t_start
    
    while True:
        t_elapsed = time.perf_counter() - t_start
        if t_elapsed >= TARGET_DURATION_SEC:
            break
            
        if step < WARMUP_STEPS:
            curr_lr = PEAK_LR * (step + 1) / WARMUP_STEPS
        else:
            curr_lr = PEAK_LR
            
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
        
        if (time.perf_counter() - last_print) >= 30.0 or step == 10:
            last_print = time.perf_counter()
            t_now = time.perf_counter() - t_start
            window = step_losses[-50:] if len(step_losses) >= 50 else step_losses
            smooth = float(np.mean(window))
            spd = total_tokens_seen / max(1e-3, t_now)
            print(f"  Step {step:4d} | Elapsed: {t_now:6.1f}s / {TARGET_DURATION_SEC}s | Loss: {accum_loss:.4f} (smooth: {smooth:.4f}) | Speed: {spd:,.0f} tok/s", flush=True)
            
    torch.cuda.synchronize()
    total_time = time.perf_counter() - t_start
    final_tokens_per_sec = total_tokens_seen / total_time
    peak_vram = torch.cuda.max_memory_allocated(device) / (1024 * 1024)
    
    tail = step_losses[-100:] if len(step_losses) >= 100 else step_losses
    final_train_loss = float(np.mean(tail))
    min_train_loss = float(min(step_losses))
    
    print(f"\n--- Training Finished for Seed {seed} ---", flush=True)
    print(f"  Total Steps: {step:,} | Elapsed: {total_time:.1f}s | Tokens Seen: {total_tokens_seen:,}", flush=True)
    print(f"  Final Train Loss (last 100 avg): {final_train_loss:.4f} | Min Loss: {min_train_loss:.4f}", flush=True)
    
    print("\nRunning full validation evaluation suite (all 4 clean held-out sets)...", flush=True)
    t_val_start = time.perf_counter()
    val_metrics = evaluator.evaluate_all(model)
    print(f"Validation complete in {time.perf_counter()-t_val_start:.1f}s:", flush=True)
    print(f"  Stage A Mix Val Loss: {val_metrics['stage_a_val_loss']:.4f}", flush=True)
    print(f"  Cosmopedia: {val_metrics['val_cosmopedia']['ce_loss']:.4f} (BPB: {val_metrics['val_cosmopedia']['bpb']:.4f})", flush=True)
    print(f"  FineWeb-Edu: {val_metrics['val_fineweb_edu']['ce_loss']:.4f} (BPB: {val_metrics['val_fineweb_edu']['bpb']:.4f})", flush=True)
    print(f"  TinyStories: {val_metrics['val_tinystories']['ce_loss']:.4f} (BPB: {val_metrics['val_tinystories']['bpb']:.4f})", flush=True)
    print(f"  Wikipedia Neutral: {val_metrics['val_wikipedia_neutral']['ce_loss']:.4f} (BPB: {val_metrics['val_wikipedia_neutral']['bpb']:.4f})", flush=True)
    
    del model, optimizer
    torch.cuda.empty_cache()
    
    summary = {
        "seed": seed,
        "lr": PEAK_LR,
        "steps": step,
        "elapsed_sec": round(total_time, 1),
        "tokens_seen": total_tokens_seen,
        "train_loss_final": round(final_train_loss, 4),
        "train_loss_min": round(min_train_loss, 4),
        "val_stage_a_loss": val_metrics["stage_a_val_loss"],
        "val_cosmopedia_loss": val_metrics["val_cosmopedia"]["ce_loss"],
        "val_cosmopedia_bpb": val_metrics["val_cosmopedia"]["bpb"],
        "val_fineweb_loss": val_metrics["val_fineweb_edu"]["ce_loss"],
        "val_fineweb_bpb": val_metrics["val_fineweb_edu"]["bpb"],
        "val_tinystories_loss": val_metrics["val_tinystories"]["ce_loss"],
        "val_tinystories_bpb": val_metrics["val_tinystories"]["bpb"],
        "val_wiki_neutral_loss": val_metrics["val_wikipedia_neutral"]["ce_loss"],
        "val_wiki_neutral_bpb": val_metrics["val_wikipedia_neutral"]["bpb"],
        "tok_per_sec": round(final_tokens_per_sec, 1),
        "peak_vram_mib": round(peak_vram, 1),
    }
    return summary

def main():
    if not torch.cuda.is_available():
        print("CUDA not available.")
        sys.exit(1)
        
    device = torch.device("cuda:0")
    print("=" * 70, flush=True)
    print("  NOISE FLOOR ESTIMATION ON 4L_768H_1KV (SEEDS 43 & 44)", flush=True)
    print("  Seed 42 Reference: 120.8M tokens, final train loss = 4.4780", flush=True)
    print(f"  Device: {torch.cuda.get_device_name(device)}", flush=True)
    print("=" * 70, flush=True)
    
    loader = StageAMixDataLoader(SHARDS_DIR, SEQ_LEN)
    evaluator = ValidationEvaluator(SHARDS_DIR, SEQ_LEN, device)
    
    # Prior result for seed 42 from extended LR sweep
    seed42_summary = {
        "seed": 42,
        "lr": 1e-3,
        "steps": 1844,
        "elapsed_sec": 1200.1,
        "tokens_seen": 120848384,
        "train_loss_final": 4.4780,
        "train_loss_min": 4.1156,
        "val_stage_a_loss": None,
        "val_cosmopedia_loss": None,
        "val_cosmopedia_bpb": None,
        "val_fineweb_loss": None,
        "val_fineweb_bpb": None,
        "val_tinystories_loss": None,
        "val_tinystories_bpb": None,
        "val_wiki_neutral_loss": None,
        "val_wiki_neutral_bpb": None,
        "tok_per_sec": 100698.4,
        "peak_vram_mib": 4155.4,
    }
    
    summaries = [seed42_summary]
    
    for s in [43, 44]:
        res = train_seed(s, loader, evaluator, device)
        summaries.append(res)
        df_temp = pd.DataFrame(summaries)
        df_temp.to_csv(OUT_CSV, index=False)
        print(f"Progress checkpoint saved to {OUT_CSV}", flush=True)
        
    df_all = pd.DataFrame(summaries)
    
    # Compute noise statistics
    train_losses = df_all["train_loss_final"].tolist()
    train_spread = max(train_losses) - min(train_losses)
    train_std = float(np.std(train_losses, ddof=1))
    
    val_43 = df_all.loc[df_all["seed"] == 43, "val_stage_a_loss"].values[0]
    val_44 = df_all.loc[df_all["seed"] == 44, "val_stage_a_loss"].values[0]
    val_spread = abs(val_43 - val_44)
    
    wiki_43 = df_all.loc[df_all["seed"] == 43, "val_wiki_neutral_loss"].values[0]
    wiki_44 = df_all.loc[df_all["seed"] == 44, "val_wiki_neutral_loss"].values[0]
    wiki_spread = abs(wiki_43 - wiki_44)
    
    report = "=" * 80 + "\n"
    report += " NOISE FLOOR ESTIMATION REPORT (4L_768H_1KV, 20-MIN TRIALS, STAGE A MIX)\n"
    report += "=" * 80 + "\n\n"
    report += df_all.to_string(index=False) + "\n\n"
    report += f"Train Loss Across 3 Seeds (42, 43, 44):\n"
    report += f"  Seed 42: {train_losses[0]:.4f} | Seed 43: {train_losses[1]:.4f} | Seed 44: {train_losses[2]:.4f}\n"
    report += f"  Mean: {np.mean(train_losses):.4f} | Std Dev: {train_std:.4f} | Max Spread: {train_spread:.4f}\n\n"
    report += f"Validation Loss (Stage A Mix) Between Seeds 43 and 44:\n"
    report += f"  Seed 43: {val_43:.4f} | Seed 44: {val_44:.4f} | Spread (|43 - 44|): {val_spread:.4f}\n\n"
    report += f"Neutral Reference (Wikipedia Shard 40) Validation Loss:\n"
    report += f"  Seed 43: {wiki_43:.4f} | Seed 44: {wiki_44:.4f} | Spread (|43 - 44|): {wiki_spread:.4f}\n\n"
    report += f"EMPIRICAL NOISE FLOOR (Delta_noise): {max(train_spread, val_spread):.4f} loss points\n"
    report += f"Decision Rule: Any architectural or KV-head quality difference <= {max(train_spread, val_spread):.4f} is within empirical noise and cannot be claimed as real.\n"
    report += "=" * 80 + "\n"
    
    with open(OUT_REPORT, "w") as f:
        f.write(report)
        
    print("\n" + report, flush=True)

if __name__ == "__main__":
    main()
