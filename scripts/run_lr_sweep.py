"""
Real-Data 3-Point Learning Rate Sweep
Tests LR in {1e-3, 2e-3, 3e-3} over 200 steps on real tokenized data for all 4 shapes:
1. 4L_768H_1KV
2. 6L_512H_1KV
3. 6L_768H_1KV
4. 4L_768H_2KV (Control)

Measures:
- Loss after 200 steps
- Real-data training tokens/sec
- Peak VRAM
- Gradient stability
Saves to results/lr_sweep.csv.
"""

import os
import sys
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
OUT_CSV = RESULTS_DIR / "lr_sweep.csv"

SHAPES = {
    "4L_768H_1KV": {"num_hidden_layers": 4, "hidden_size": 768, "intermediate_size": 2048, "num_attention_heads": 12, "num_key_value_heads": 1},
    "6L_512H_1KV": {"num_hidden_layers": 6, "hidden_size": 512, "intermediate_size": 1280, "num_attention_heads": 8, "num_key_value_heads": 1},
    "6L_768H_1KV": {"num_hidden_layers": 6, "hidden_size": 768, "intermediate_size": 2048, "num_attention_heads": 12, "num_key_value_heads": 1},
    "4L_768H_2KV": {"num_hidden_layers": 4, "hidden_size": 768, "intermediate_size": 2048, "num_attention_heads": 12, "num_key_value_heads": 2},
}

LRS = [1e-3, 2e-3, 3e-3]
STEPS = 200
SEQ_LEN = 2048
MICRO_BS = 4  # 8,192 tokens per step
VOCAB_SIZE = 8192

class ShardDataLoader:
    def __init__(self, bin_path, seq_len, batch_size):
        self.data = np.memmap(bin_path, dtype=np.uint16, mode="r")
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.total_tokens = len(self.data)
        
    def get_batch(self, device):
        # Sample random offsets
        max_idx = self.total_tokens - self.seq_len - 1
        ix = np.random.randint(0, max_idx, size=self.batch_size)
        x = np.stack([self.data[i : i + self.seq_len] for i in ix])
        y = np.stack([self.data[i + 1 : i + 1 + self.seq_len] for i in ix])
        
        tx = torch.from_numpy(x.astype(np.int64)).to(device=device)
        ty = torch.from_numpy(y.astype(np.int64)).to(device=device)
        return tx, ty

def run_single_sweep(model_name, shape_cfg, lr, loader, device):
    print(f"\n--- Model: {model_name} | LR: {lr:.0e} ---", flush=True)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    cfg = LlamaConfig(
        vocab_size=VOCAB_SIZE,
        hidden_size=shape_cfg["hidden_size"],
        intermediate_size=shape_cfg["intermediate_size"],
        num_hidden_layers=shape_cfg["num_hidden_layers"],
        num_attention_heads=shape_cfg["num_attention_heads"],
        num_key_value_heads=shape_cfg["num_key_value_heads"],
        tie_word_embeddings=True,
        torch_dtype="bfloat16",
        _attn_implementation="sdpa",
        max_position_embeddings=SEQ_LEN
    )
    
    torch.manual_seed(42)
    model = LlamaForCausalLM(cfg).to(device=device, dtype=torch.bfloat16)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.1, fused=True)
    
    # Linear warmup over first 20 steps (10%)
    warmup_steps = 20
    
    tokens_per_step = MICRO_BS * SEQ_LEN
    loss_history = []
    
    t0 = time.perf_counter()
    for step in range(STEPS):
        # Warmup schedule
        if step < warmup_steps:
            curr_lr = lr * (step + 1) / warmup_steps
            for pg in optimizer.param_groups:
                pg["lr"] = curr_lr
                
        optimizer.zero_grad(set_to_none=True)
        x, y = loader.get_batch(device)
        
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            outputs = model(x, labels=y)
            loss = outputs.loss
            
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        loss_val = loss.item()
        loss_history.append(loss_val)
        
        if (step + 1) % 50 == 0 or step == STEPS - 1:
            print(f"  Step {step+1:3d}/{STEPS}: Loss = {loss_val:.4f} (lr = {optimizer.param_groups[0]['lr']:.2e})", flush=True)
            
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    
    real_tokens_per_sec = (STEPS * tokens_per_step) / elapsed
    peak_vram = torch.cuda.max_memory_allocated(device) / (1024 * 1024)
    final_loss = float(np.mean(loss_history[-10:])) # Average last 10 steps
    
    del model, optimizer
    torch.cuda.empty_cache()
    
    return {
        "model": model_name,
        "lr": lr,
        "final_loss": round(final_loss, 4),
        "min_loss": round(min(loss_history), 4),
        "real_tok_per_sec": round(real_tokens_per_sec, 1),
        "elapsed_sec": round(elapsed, 2),
        "peak_vram_mib": round(peak_vram, 1)
    }

def main():
    if not torch.cuda.is_available():
        print("CUDA not available.")
        sys.exit(1)
    device = torch.device("cuda:0")
    
    train_bin = SHARDS_DIR / "train_cosmopedia.bin"
    if not train_bin.exists():
        # fallback to any available bin
        bins = list(SHARDS_DIR.glob("train_*.bin"))
        if not bins:
            print("Error: No training shards found in data/shards/. Please run tokenize_shards.py first.")
            sys.exit(1)
        train_bin = bins[0]
        
    print(f"Loading real data from {train_bin.name}...")
    loader = ShardDataLoader(str(train_bin), SEQ_LEN, MICRO_BS)
    print(f"Data tokens available: {loader.total_tokens:,}")
    
    results = []
    for model_name, shape_cfg in SHAPES.items():
        for lr in LRS:
            res = run_single_sweep(model_name, shape_cfg, lr, loader, device)
            results.append(res)
            
    df = pd.DataFrame(results)
    df.to_csv(OUT_CSV, index=False)
    print(f"\nSaved LR sweep results to {OUT_CSV}")
    print("\n" + df.to_string(), flush=True)

if __name__ == "__main__":
    main()
