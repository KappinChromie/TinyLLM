"""
Test Micro-Batch Sizes (8, 4, 2) on RTX 3070 Ti (8 GB)
Measures:
- Tokens/sec
- Step time (ms)
- PyTorch Peak Allocated Memory (MiB)
- Peak Total GPU Memory (as reported by nvidia-smi, including desktop/OS usage)
Keeps global batch at 32 sequences (65,536 tokens), grad_ckpt=False, 2k context.
"""

import os
import sys
import time
import subprocess
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
OUT_CSV = RESULTS_DIR / "micro_batch_vram_test.csv"

def get_nvidia_smi_memory_mb():
    try:
        cmd = ["/usr/lib/wsl/lib/nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"]
        res = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return float(res.stdout.strip())
    except Exception:
        try:
            cmd = ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"]
            res = subprocess.run(cmd, capture_output=True, text=True, check=True)
            return float(res.stdout.strip())
        except Exception:
            return 0.0

def benchmark_micro_batch(micro_bs, grad_accum, seq_len=2048, warmup_steps=5, measure_steps=15):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n--- Testing Micro-Batch {micro_bs} (Grad Accum {grad_accum}) ---", flush=True)
    
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
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
        max_position_embeddings=seq_len,
    )
    model = LlamaForCausalLM(config).to(device=device, dtype=torch.bfloat16)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, betas=(0.9, 0.95), weight_decay=0.1, fused=True)
    
    # Generate fixed dummy inputs
    inputs_x = [torch.randint(0, 8192, (micro_bs, seq_len), device=device, dtype=torch.long) for _ in range(grad_accum)]
    inputs_y = [torch.randint(0, 8192, (micro_bs, seq_len), device=device, dtype=torch.long) for _ in range(grad_accum)]
    
    # Warmup
    for _ in range(warmup_steps):
        optimizer.zero_grad(set_to_none=True)
        for i in range(grad_accum):
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                out = model(inputs_x[i])
                loss = F.cross_entropy(out.logits.view(-1, 8192), inputs_y[i].view(-1)) / grad_accum
            loss.backward()
        optimizer.step()
        
    torch.cuda.synchronize()
    start_time = time.perf_counter()
    peak_smi_mb = 0.0
    
    for step in range(measure_steps):
        optimizer.zero_grad(set_to_none=True)
        for i in range(grad_accum):
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                out = model(inputs_x[i])
                loss = F.cross_entropy(out.logits.view(-1, 8192), inputs_y[i].view(-1)) / grad_accum
            loss.backward()
            
            smi_mem = get_nvidia_smi_memory_mb()
            if smi_mem > peak_smi_mb:
                peak_smi_mb = smi_mem
                
        optimizer.step()
        
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start_time
    
    total_tokens = measure_steps * micro_bs * grad_accum * seq_len
    tok_s = total_tokens / elapsed
    step_time_ms = (elapsed / measure_steps) * 1000
    pytorch_peak_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
    
    print(f"Results for Micro-Batch {micro_bs}:")
    print(f"  Throughput: {tok_s:,.1f} tok/s")
    print(f"  Step Time:  {step_time_ms:.2f} ms")
    print(f"  PyTorch Peak VRAM: {pytorch_peak_mb:.1f} MB ({pytorch_peak_mb/1024:.2f} GB)")
    print(f"  nvidia-smi Total GPU Memory: {peak_smi_mb:.1f} MB ({peak_smi_mb/1024:.2f} GB)")
    
    del model, optimizer
    torch.cuda.empty_cache()
    
    return {
        "micro_batch": micro_bs,
        "grad_accum": grad_accum,
        "global_batch_seqs": micro_bs * grad_accum,
        "global_batch_tokens": micro_bs * grad_accum * seq_len,
        "tok_per_sec": tok_s,
        "step_time_ms": step_time_ms,
        "pytorch_peak_mb": pytorch_peak_mb,
        "smi_total_peak_mb": peak_smi_mb,
    }

def main():
    configs = [
        (8, 4),
        (4, 8),
        (2, 16),
    ]
    results = []
    for mb, ga in configs:
        res = benchmark_micro_batch(mb, ga)
        results.append(res)
        
    df = pd.DataFrame(results)
    df.to_csv(OUT_CSV, index=False)
    print(f"\nSaved results to {OUT_CSV}")
    print("\nSummary Table:")
    print(df.to_string(index=False))

if __name__ == "__main__":
    main()
