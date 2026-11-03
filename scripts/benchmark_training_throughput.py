"""
Training Throughput Benchmark
Measures actual training throughput (tokens/sec, step time, peak VRAM) on RTX 3070 Ti (8GB)
for the 4 finalist model shapes across context lengths: 2k, 8k, and 32k.
Uses:
- bf16 autocast
- fused AdamW
- PyTorch SDPA (Scaled Dot-Product Attention with FlashAttention backend in WSL2)
- Gradient checkpointing for 8k and 32k
- Chunked cross-entropy loss over sequence dimension for 8k and 32k
Tests torch.compile and records system capabilities.
"""

import os
import sys
import time
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import LlamaConfig, LlamaForCausalLM
import pandas as pd

sys.stdout.reconfigure(line_buffering=True)

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)
OUT_CSV = RESULTS_DIR / "train_throughput.csv"

# Model configurations
SHAPES = {
    "4L_768H_1KV": {
        "num_hidden_layers": 4,
        "hidden_size": 768,
        "intermediate_size": 2048,
        "num_attention_heads": 12,
        "num_key_value_heads": 1,
        "vocab_size": 8192,
    },
    "6L_512H_1KV": {
        "num_hidden_layers": 6,
        "hidden_size": 512,
        "intermediate_size": 1280,
        "num_attention_heads": 8,
        "num_key_value_heads": 1,
        "vocab_size": 8192,
    },
    "6L_768H_1KV": {
        "num_hidden_layers": 6,
        "hidden_size": 768,
        "intermediate_size": 2048,
        "num_attention_heads": 12,
        "num_key_value_heads": 1,
        "vocab_size": 8192,
    },
    "4L_768H_2KV": {
        "num_hidden_layers": 4,
        "hidden_size": 768,
        "intermediate_size": 2048,
        "num_attention_heads": 12,
        "num_key_value_heads": 2,
        "vocab_size": 8192,
    },
}

CONTEXTS = [2048, 8192, 32768]

def test_torch_compile(device):
    print("\n--- Testing torch.compile support ---", flush=True)
    try:
        model = nn.Linear(64, 64).to(device)
        x = torch.randn(8, 64, device=device)
        compiled_model = torch.compile(model)
        out = compiled_model(x)
        out.sum().backward()
        print("  torch.compile: SUCCESS", flush=True)
        return True, "Supported"
    except Exception as e:
        err_msg = str(e).split("\n")[0]
        print(f"  torch.compile: FAILED ({err_msg})", flush=True)
        return False, err_msg

def run_step(model, optimizer, input_ids, labels, use_chunked=False, chunk_size=4096):
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        if use_chunked:
            out = model.model(input_ids)
            h_shift = out.last_hidden_state[:, :-1, :]
            l_shift = labels[:, 1:]
            total_tokens = l_shift.numel()
            total_loss = 0.0
            for i in range(0, total_tokens, chunk_size):
                h_c = h_shift[:, i:i+chunk_size, :]
                l_c = l_shift[:, i:i+chunk_size]
                logits_c = model.lm_head(h_c)
                loss_c = F.cross_entropy(logits_c.reshape(-1, logits_c.shape[-1]), l_c.reshape(-1), reduction="sum") / total_tokens
                total_loss = total_loss + loss_c
            total_loss.backward()
        else:
            outputs = model(input_ids, labels=labels)
            loss = outputs.loss
            loss.backward()
    optimizer.step()

def benchmark_shape_context(name, shape_cfg, context_len, device):
    print(f"\nBenchmarking {name} at context {context_len}...", flush=True)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    if context_len == 2048:
        micro_bs = 2
        grad_ckpt = False
        use_chunked = False
        warmup_steps = 5
        measure_steps = 20
    elif context_len == 8192:
        micro_bs = 1
        grad_ckpt = True
        use_chunked = True
        warmup_steps = 3
        measure_steps = 10
    else:  # 32768
        micro_bs = 1
        grad_ckpt = True
        use_chunked = True
        warmup_steps = 2
        measure_steps = 6
        
    config = LlamaConfig(
        vocab_size=shape_cfg["vocab_size"],
        hidden_size=shape_cfg["hidden_size"],
        intermediate_size=shape_cfg["intermediate_size"],
        num_hidden_layers=shape_cfg["num_hidden_layers"],
        num_attention_heads=shape_cfg["num_attention_heads"],
        num_key_value_heads=shape_cfg["num_key_value_heads"],
        tie_word_embeddings=True,
        torch_dtype="bfloat16",
        _attn_implementation="sdpa",
        max_position_embeddings=context_len
    )
    
    model = LlamaForCausalLM(config).to(device=device, dtype=torch.bfloat16)
    if grad_ckpt:
        model.gradient_checkpointing_enable()
        
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, fused=True)
    
    input_ids = torch.randint(0, shape_cfg["vocab_size"], (micro_bs, context_len), device=device, dtype=torch.long)
    labels = input_ids.clone()
    total_tokens_per_step = micro_bs * context_len
    
    # Warmup
    for _ in range(warmup_steps):
        run_step(model, optimizer, input_ids, labels, use_chunked=use_chunked)
    torch.cuda.synchronize()
    
    # Measurement
    t0 = time.perf_counter()
    for _ in range(measure_steps):
        run_step(model, optimizer, input_ids, labels, use_chunked=use_chunked)
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    
    elapsed_sec = t1 - t0
    step_time_ms = (elapsed_sec / measure_steps) * 1000.0
    tokens_per_sec = (total_tokens_per_step * measure_steps) / elapsed_sec
    tokens_per_hour = tokens_per_sec * 3600
    peak_vram_mib = torch.cuda.max_memory_allocated(device) / (1024 * 1024)
    
    total_params = sum(p.numel() for p in model.parameters())
    non_emb_params = total_params - model.get_input_embeddings().weight.numel()
    
    print(f"  Result: {tokens_per_sec:,.1f} tok/s | {step_time_ms:.1f} ms/step | Peak VRAM: {peak_vram_mib:.1f} MiB | Tokens/hr: {tokens_per_hour/1e6:.1f}M", flush=True)
    
    del model, optimizer, input_ids, labels
    torch.cuda.empty_cache()
    
    return {
        "model": name,
        "layers": shape_cfg["num_hidden_layers"],
        "hidden": shape_cfg["hidden_size"],
        "ffn": shape_cfg["intermediate_size"],
        "kv_heads": shape_cfg["num_key_value_heads"],
        "total_params_m": round(total_params / 1e6, 2),
        "non_emb_params_m": round(non_emb_params / 1e6, 2),
        "context_len": context_len,
        "micro_bs": micro_bs,
        "grad_ckpt": grad_ckpt,
        "tokens_per_sec": round(tokens_per_sec, 1),
        "step_time_ms": round(step_time_ms, 2),
        "peak_vram_mib": round(peak_vram_mib, 1),
        "tokens_per_hour_m": round(tokens_per_hour / 1e6, 2)
    }

def main():
    if not torch.cuda.is_available():
        print("CUDA not available. Exiting.", flush=True)
        sys.exit(1)
        
    device = torch.device("cuda:0")
    print(f"Benchmarking on {torch.cuda.get_device_name(0)} (WSL2)", flush=True)
    
    compile_ok, compile_status = test_torch_compile(device)
    
    results = []
    for name, cfg in SHAPES.items():
        for ctx in CONTEXTS:
            try:
                res = benchmark_shape_context(name, cfg, ctx, device)
                res["torch_compile_status"] = compile_status
                results.append(res)
            except Exception as e:
                print(f"  {name} at context {ctx}: Failed with {e}", flush=True)
                
    df = pd.DataFrame(results)
    df.to_csv(OUT_CSV, index=False)
    print(f"\nSaved throughput results to {OUT_CSV}", flush=True)
    print("\n" + df.to_string(), flush=True)

if __name__ == "__main__":
    main()
