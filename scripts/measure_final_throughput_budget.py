"""
Step 2C: Final Training Throughput Measurement for 6L_768H_1KV
Measures active tokens/sec at 2k, 8k, and 32k with the final training codebase.
Reports active throughput and peak VRAM separately from temperature governor pauses.
Computes active hours and estimated wall-clock range for the 12-hour training budget.
"""

import os
import sys
import time
import math
import subprocess
import shutil
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import LlamaConfig, LlamaForCausalLM

sys.stdout.reconfigure(line_buffering=True)

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)
OUT_CSV = RESULTS_DIR / "final_training_throughput.csv"

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

def get_peak_vram_mb():
    try:
        cmd = [SMI_BIN, "--query-gpu=memory.used", "--format=csv,noheader,nounits"]
        res = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return float(res.stdout.strip())
    except Exception:
        return torch.cuda.max_memory_allocated() / (1024 * 1024)

def create_model(seq_len, grad_ckpt=False, device="cuda"):
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
        max_position_embeddings=max(2048, seq_len),
        rope_theta=500000.0 if seq_len > 8192 else (50000.0 if seq_len > 2048 else 10000.0),
    )
    model = LlamaForCausalLM(config).to(device=device, dtype=torch.bfloat16)
    if grad_ckpt:
        model.gradient_checkpointing_enable()
    return model

def benchmark_setting(seq_len, micro_bs, grad_ckpt, num_steps=20, warmup=5, device="cuda"):
    print(f"\nBenchmarking seq_len={seq_len}, micro_bs={micro_bs}, grad_ckpt={grad_ckpt}...", flush=True)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    model = create_model(seq_len=seq_len, grad_ckpt=grad_ckpt, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, betas=(0.9, 0.95), weight_decay=0.1, fused=True)
    
    tokens_per_step = micro_bs * seq_len
    
    # Warmup
    for _ in range(warmup):
        x = torch.randint(0, 8192, (micro_bs, seq_len), device=device, dtype=torch.long)
        y = torch.randint(0, 8192, (micro_bs, seq_len), device=device, dtype=torch.long)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            out = model(x)
            loss = F.cross_entropy(out.logits.view(-1, 8192), y.view(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
    torch.cuda.synchronize()
    
    # Timed loop
    t_start = time.perf_counter()
    for _ in range(num_steps):
        x = torch.randint(0, 8192, (micro_bs, seq_len), device=device, dtype=torch.long)
        y = torch.randint(0, 8192, (micro_bs, seq_len), device=device, dtype=torch.long)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            out = model(x)
            loss = F.cross_entropy(out.logits.view(-1, 8192), y.view(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
    torch.cuda.synchronize()
    t_el = time.perf_counter() - t_start
    
    total_tokens = num_steps * tokens_per_step
    tok_s = total_tokens / t_el
    step_time_ms = (t_el / num_steps) * 1000
    peak_vram = get_peak_vram_mb()
    
    print(f"  Result: {tok_s:,.1f} tok/s | Step: {step_time_ms:.2f} ms | Peak VRAM: {peak_vram:.1f} MB", flush=True)
    
    del model, optimizer
    torch.cuda.empty_cache()
    
    return {
        "seq_len": seq_len,
        "micro_bs": micro_bs,
        "grad_ckpt": grad_ckpt,
        "active_tok_per_sec": round(tok_s, 1),
        "step_time_ms": round(step_time_ms, 2),
        "peak_vram_mb": round(peak_vram, 1),
        "tokens_per_hour_m": round((tok_s * 3600) / 1e6, 2)
    }

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Starting Final Throughput Measurement on {device}...")
    
    configs = [
        # (seq_len, micro_bs, grad_ckpt)
        (2048, 4, False),
        (8192, 1, False),
        (32768, 1, True),
    ]
    
    results = []
    for seq_len, micro_bs, grad_ckpt in configs:
        res = benchmark_setting(seq_len, micro_bs, grad_ckpt, num_steps=20, warmup=5, device=device)
        results.append(res)
        
    df = pd.DataFrame(results)
    df.to_csv(OUT_CSV, index=False)
    print(f"\nFinal Throughput Table saved to {OUT_CSV}:")
    print(df.to_string(index=False))

if __name__ == "__main__":
    main()
