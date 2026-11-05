"""
Measure 6L_768H_1KV Training Throughput at 2k, 8k, and 32k context
Tests optimal micro-batch size and chunking settings on RTX 3070 Ti (8GB)
WSL2 PyTorch 2.6 CUDA 12.4
"""

import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import LlamaConfig, LlamaForCausalLM

def get_model(context_len, grad_ckpt=False):
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
        max_position_embeddings=context_len,
    )
    model = LlamaForCausalLM(config).to("cuda").to(torch.bfloat16)
    if grad_ckpt:
        model.gradient_checkpointing_enable()
    return model

def benchmark_config(context_len, micro_bs, grad_ckpt, use_chunked, chunk_size=4096, warmup_steps=5, measure_steps=20):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    try:
        model = get_model(context_len, grad_ckpt=grad_ckpt)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, fused=True)
        
        # Fixed random inputs
        x = torch.randint(0, 8192, (micro_bs, context_len), device="cuda", dtype=torch.long)
        y = torch.randint(0, 8192, (micro_bs, context_len), device="cuda", dtype=torch.long)
        
        # Warmup
        for _ in range(warmup_steps):
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                if use_chunked:
                    out = model.model(x)
                    h = out.last_hidden_state
                    total_tokens = x.numel()
                    total_loss = 0.0
                    for i in range(0, context_len, chunk_size):
                        h_c = h[:, i:i+chunk_size, :]
                        y_c = y[:, i:i+chunk_size]
                        logits_c = model.lm_head(h_c)
                        loss_c = F.cross_entropy(logits_c.reshape(-1, 8192), y_c.reshape(-1), reduction="sum") / total_tokens
                        total_loss = total_loss + loss_c
                    total_loss.backward()
                else:
                    out = model(x)
                    loss = F.cross_entropy(out.logits.view(-1, 8192), y.view(-1))
                    loss.backward()
            optimizer.step()
            
        torch.cuda.synchronize()
        start_time = time.perf_counter()
        
        for _ in range(measure_steps):
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                if use_chunked:
                    out = model.model(x)
                    h = out.last_hidden_state
                    total_tokens = x.numel()
                    total_loss = 0.0
                    for i in range(0, context_len, chunk_size):
                        h_c = h[:, i:i+chunk_size, :]
                        y_c = y[:, i:i+chunk_size]
                        logits_c = model.lm_head(h_c)
                        loss_c = F.cross_entropy(logits_c.reshape(-1, 8192), y_c.reshape(-1), reduction="sum") / total_tokens
                        total_loss = total_loss + loss_c
                    total_loss.backward()
                else:
                    out = model(x)
                    loss = F.cross_entropy(out.logits.view(-1, 8192), y.view(-1))
                    loss.backward()
            optimizer.step()
            
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start_time
        
        total_tokens = measure_steps * micro_bs * context_len
        tok_s = total_tokens / elapsed
        step_time_ms = (elapsed / measure_steps) * 1000
        peak_vram_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
        
        print(f"Context {context_len} | micro_bs {micro_bs} | grad_ckpt {grad_ckpt} | chunked {use_chunked}: "
              f"{tok_s:,.1f} tok/s | {step_time_ms:.2f} ms/step | Peak VRAM: {peak_vram_mb:.1f} MB")
        return {
            "context_len": context_len,
            "micro_bs": micro_bs,
            "grad_ckpt": grad_ckpt,
            "chunked": use_chunked,
            "tok_s": tok_s,
            "step_time_ms": step_time_ms,
            "peak_vram_mb": peak_vram_mb,
            "status": "OK"
        }
    except Exception as e:
        print(f"Context {context_len} | micro_bs {micro_bs} | grad_ckpt {grad_ckpt} | chunked {use_chunked}: FAILED ({e})")
        return {
            "context_len": context_len,
            "micro_bs": micro_bs,
            "grad_ckpt": grad_ckpt,
            "chunked": use_chunked,
            "tok_s": 0,
            "step_time_ms": 0,
            "peak_vram_mb": 0,
            "status": f"FAILED: {e}"
        }

if __name__ == "__main__":
    results = []
    print("=== Testing 2k Configurations ===")
    results.append(benchmark_config(2048, micro_bs=2, grad_ckpt=False, use_chunked=False, warmup_steps=5, measure_steps=30))
    results.append(benchmark_config(2048, micro_bs=4, grad_ckpt=False, use_chunked=False, warmup_steps=5, measure_steps=30))
    results.append(benchmark_config(2048, micro_bs=8, grad_ckpt=False, use_chunked=False, warmup_steps=5, measure_steps=30))

    print("\n=== Testing 8k Configurations ===")
    results.append(benchmark_config(8192, micro_bs=1, grad_ckpt=False, use_chunked=False, warmup_steps=3, measure_steps=15))
    results.append(benchmark_config(8192, micro_bs=1, grad_ckpt=True, use_chunked=False, warmup_steps=3, measure_steps=15))
    results.append(benchmark_config(8192, micro_bs=1, grad_ckpt=True, use_chunked=True, chunk_size=4096, warmup_steps=3, measure_steps=15))
    results.append(benchmark_config(8192, micro_bs=2, grad_ckpt=True, use_chunked=False, warmup_steps=3, measure_steps=15))
    results.append(benchmark_config(8192, micro_bs=2, grad_ckpt=True, use_chunked=True, chunk_size=4096, warmup_steps=3, measure_steps=15))

    print("\n=== Testing 32k Configurations ===")
    results.append(benchmark_config(32768, micro_bs=1, grad_ckpt=True, use_chunked=False, warmup_steps=2, measure_steps=8))
    results.append(benchmark_config(32768, micro_bs=1, grad_ckpt=True, use_chunked=True, chunk_size=4096, warmup_steps=2, measure_steps=8))
    results.append(benchmark_config(32768, micro_bs=1, grad_ckpt=True, use_chunked=True, chunk_size=8192, warmup_steps=2, measure_steps=8))
