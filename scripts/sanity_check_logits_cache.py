"""
Sanity Check 1.a: Verify that cached/incremental generation produces
the exact same logits as a full forward pass over the same token sequence.
Tests both bfloat16 and float32.
"""

import sys
from pathlib import Path
import torch
from transformers import LlamaConfig, LlamaForCausalLM

ROOT = Path(__file__).resolve().parent.parent
CHECKPOINT_PATH = ROOT / "models" / "checkpoints" / "4L_768H_1KV_decayed_1hr.pt"

SHAPE_CFG = {
    "num_hidden_layers": 4,
    "hidden_size": 768,
    "intermediate_size": 2048,
    "num_attention_heads": 12,
    "num_key_value_heads": 1,
    "vocab_size": 8192,
}

def main():
    if not torch.cuda.is_available():
        print("CUDA required.")
        sys.exit(1)
        
    device = torch.device("cuda:0")
    print("=" * 70)
    print("  SANITY CHECK 1.a: KV CACHED VS FULL FORWARD PASS LOGITS")
    print(f"  Checkpoint: {CHECKPOINT_PATH.name}")
    print("=" * 70)
    
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
        max_position_embeddings=2048,
    )
    
    model = LlamaForCausalLM(cfg).to(device=device, dtype=torch.bfloat16)
    state_dict = torch.load(CHECKPOINT_PATH, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    
    # Generate test tokens (128 tokens)
    torch.manual_seed(1234)
    seq_len = 128
    input_ids = torch.randint(0, 8192, (1, seq_len), device=device)
    
    print(f"\n1. Running Full Forward Pass (seq_len = {seq_len})...")
    with torch.no_grad():
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            out_full = model(input_ids)
            logits_full = out_full.logits  # [1, seq_len, vocab_size]
            
    print("2. Running Incremental Cached Forward Pass (token-by-token with past_key_values)...")
    logits_cached = []
    past_key_values = None
    
    with torch.no_grad():
        for t in range(seq_len):
            tok = input_ids[:, t:t+1]
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                if past_key_values is None:
                    out_step = model(tok, use_cache=True)
                else:
                    out_step = model(tok, past_key_values=past_key_values, use_cache=True)
            past_key_values = out_step.past_key_values
            logits_cached.append(out_step.logits)
            
    logits_cached = torch.cat(logits_cached, dim=1)  # [1, seq_len, vocab_size]
    
    # Compare
    diff = torch.abs(logits_full.float() - logits_cached.float())
    max_diff = torch.max(diff).item()
    mean_diff = torch.mean(diff).item()
    top1_match = (torch.argmax(logits_full, dim=-1) == torch.argmax(logits_cached, dim=-1)).float().mean().item()
    
    print("\n--- RESULTS FOR BFLOAT16 WITH SDPA ---")
    print(f"  Max Absolute Difference:  {max_diff:.8e}")
    print(f"  Mean Absolute Difference: {mean_diff:.8e}")
    print(f"  Top-1 Greedy Argmax Match Rate: {top1_match * 100:.2f}%")
    
    # Also test in Float32 (exact math without bf16 rounding accumulation)
    print("\n3. Testing in Float32 mode for precision comparison...")
    model_f32 = LlamaForCausalLM(cfg).to(device=device, dtype=torch.float32)
    model_f32.load_state_dict(state_dict)
    model_f32.eval()
    
    with torch.no_grad():
        out_full_f32 = model_f32(input_ids)
        logits_full_f32 = out_full_f32.logits
        
        logits_cached_f32 = []
        pkv_f32 = None
        for t in range(seq_len):
            tok = input_ids[:, t:t+1]
            out_step = model_f32(tok, past_key_values=pkv_f32, use_cache=True)
            pkv_f32 = out_step.past_key_values
            logits_cached_f32.append(out_step.logits)
        logits_cached_f32 = torch.cat(logits_cached_f32, dim=1)
        
    diff_f32 = torch.abs(logits_full_f32 - logits_cached_f32)
    max_diff_f32 = torch.max(diff_f32).item()
    mean_diff_f32 = torch.mean(diff_f32).item()
    top1_match_f32 = (torch.argmax(logits_full_f32, dim=-1) == torch.argmax(logits_cached_f32, dim=-1)).float().mean().item()
    
    print("\n--- RESULTS FOR FLOAT32 ---")
    print(f"  Max Absolute Difference:  {max_diff_f32:.8e}")
    print(f"  Mean Absolute Difference: {mean_diff_f32:.8e}")
    print(f"  Top-1 Greedy Argmax Match Rate: {top1_match_f32 * 100:.2f}%")
    
    # Also verify non-cached full forward vs incremental full context (the way sample_generations was written)
    print("\n4. Verifying Incremental Full Context (recomputing prefix without KV cache)...")
    logits_prefix = []
    with torch.no_grad():
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            for t in range(1, seq_len + 1):
                pref = input_ids[:, :t]
                out_p = model(pref)
                logits_prefix.append(out_p.logits[:, -1:, :])
    logits_prefix = torch.cat(logits_prefix, dim=1)
    diff_prefix = torch.abs(logits_full.float() - logits_prefix.float())
    max_diff_pref = torch.max(diff_prefix).item()
    print(f"  Max Absolute Difference vs Prefix Recompute: {max_diff_pref:.8e}")
    
    return max_diff, max_diff_f32

if __name__ == "__main__":
    main()
