"""
Long-Context Evaluation Suite:
1. Loss by Position Bucket:
   - Measures token-level cross entropy across context buckets:
     [0-1k], [1k-2k], [2k-4k], [4k-8k], [8k-16k], [16k-32k]
   - Evaluated on continuous held-out documents (PG-19 / Wikipedia Long).
2. Passkey / Needle-in-a-Haystack Retrieval Accuracy:
   - Depths: 1k (1,024), 2k (2,048), 4k (4,096), 8k (8,192), 16k (16,384), 32k (32,768).
   - 20 trials per depth, with needle inserted across depth percentiles (5% to 95%).
   - Exact-match verification on the 5-digit passkey.
"""

import sys
import math
import random
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import LlamaConfig, LlamaForCausalLM
import sentencepiece as spm

ROOT = Path(__file__).resolve().parent.parent
SHARDS_DIR = ROOT / "data" / "shards"
RESULTS_DIR = ROOT / "results"
MODELS_DIR = ROOT / "models"
TOKENIZER_MODEL = MODELS_DIR / "tokenizer_8k" / "tokenizer.model"

BUCKETS = [
    ("0-1k", 0, 1024),
    ("1k-2k", 1024, 2048),
    ("2k-4k", 2048, 4096),
    ("4k-8k", 4096, 8192),
    ("8k-16k", 8192, 16384),
    ("16k-32k", 16384, 32768),
]

PASSKEY_LENGTHS = [1024, 2048, 4096, 8192, 16384, 32768]

# ---------------------------------------------------------------------------
# 1. Position-Bucket Loss Evaluator
# ---------------------------------------------------------------------------
@torch.no_grad()
def evaluate_position_bucket_loss(model, shard, max_ctx=32768, num_sequences=5, device="cpu"):
    """
    Evaluates loss across position buckets by computing per-token cross entropy.
    """
    model.eval()
    max_idx = len(shard) - max_ctx - 1
    rng = np.random.RandomState(42)
    
    # Accumulators for each bucket
    bucket_losses = {name: [] for name, start, _ in BUCKETS if start < max_ctx}
    
    for s_idx in range(num_sequences):
        start_idx = rng.randint(0, max_idx)
        raw_seq = shard[start_idx : start_idx + max_ctx + 1]
        x = torch.from_numpy(raw_seq[:-1].astype(np.int64)).unsqueeze(0).to(device)
        y = torch.from_numpy(raw_seq[1:].astype(np.int64)).unsqueeze(0).to(device)
        
        with torch.autocast(device_type=device if device != "cpu" else "cpu", dtype=torch.bfloat16):
            out = model(x)
            # Logits shape [1, seq_len, vocab_size]
            logits = out.logits[0]  # [seq_len, vocab_size]
            targets = y[0]          # [seq_len]
            
            # Per-token cross entropy
            token_losses = F.cross_entropy(logits, targets, reduction="none").float().cpu().numpy()
            
        for name, b_start, b_end in BUCKETS:
            if b_start < max_ctx:
                actual_end = min(b_end, max_ctx)
                if actual_end > b_start:
                    b_loss = np.mean(token_losses[b_start:actual_end])
                    bucket_losses[name].append(b_loss)
                    
    results = {}
    for name, loss_list in bucket_losses.items():
        mean_ce = float(np.mean(loss_list))
        results[name] = {
            "ce_loss": round(mean_ce, 4),
            "perplexity": round(math.exp(mean_ce), 2),
            "bits_per_token": round(mean_ce / math.log(2.0), 4),
        }
    return results

# ---------------------------------------------------------------------------
# 2. Passkey Needle-in-a-Haystack Retrieval Evaluator
# ---------------------------------------------------------------------------
def generate_filler_text(sp, num_tokens):
    """Generates neutral filler text from repeated generic sentences."""
    filler_sentence = "The trees in the forest swayed gently as the cool evening breeze blew through the quiet valley. "
    sent_tokens = sp.encode_as_ids(filler_sentence)
    reps = (num_tokens // len(sent_tokens)) + 2
    tokens = (sent_tokens * reps)[:num_tokens]
    return tokens

def run_passkey_retrieval_test(model, sp, lengths=PASSKEY_LENGTHS, trials_per_length=20, device="cpu"):
    """
    Executes passkey retrieval across specified context lengths.
    Inserts a 5-digit passkey needle at depth percentiles between 5% and 95%.
    """
    model.eval()
    results = []
    
    depth_percents = np.linspace(0.05, 0.95, trials_per_length)
    
    print(f"\nRunning Passkey Retrieval Benchmark ({trials_per_length} trials per length across {len(lengths)} depths)...", flush=True)
    
    for ctx_len in lengths:
        correct_count = 0
        trial_details = []
        
        for trial_idx, depth_pct in enumerate(depth_percents):
            # Generate random 5-digit passkey
            passkey = f"{random.randint(10000, 99999)}"
            prefix_needle = f"The secret passkey is {passkey}. Remember this passkey. "
            prompt_suffix = " What is the secret passkey? The secret passkey is "
            
            needle_ids = sp.encode_as_ids(prefix_needle)
            suffix_ids = sp.encode_as_ids(prompt_suffix)
            
            total_filler_needed = ctx_len - len(needle_ids) - len(suffix_ids) - 10
            if total_filler_needed < 0:
                total_filler_needed = 10
                
            insert_pos = int(total_filler_needed * depth_pct)
            prefix_filler = generate_filler_text(sp, insert_pos)
            suffix_filler = generate_filler_text(sp, total_filler_needed - insert_pos)
            
            # Assemble full context
            full_ids = prefix_filler + needle_ids + suffix_filler + suffix_ids
            input_tensor = torch.tensor([full_ids], dtype=torch.long, device=device)
            
            # Greedy generate next 10 tokens
            gen_ids = []
            curr_tensor = input_tensor
            with torch.no_grad():
                for _ in range(8):
                    with torch.autocast(device_type=device if device != "cpu" else "cpu", dtype=torch.bfloat16):
                        out = model(curr_tensor)
                        next_tok = torch.argmax(out.logits[0, -1, :]).item()
                    gen_ids.append(next_tok)
                    curr_tensor = torch.cat([curr_tensor, torch.tensor([[next_tok]], device=device)], dim=1)
                    if next_tok == sp.eos_id():
                        break
                        
            gen_text = sp.decode(gen_ids).strip()
            is_correct = passkey in gen_text
            if is_correct:
                correct_count += 1
                
            trial_details.append({
                "trial": trial_idx + 1,
                "depth_pct": round(depth_pct * 100, 1),
                "passkey": passkey,
                "generated": gen_text,
                "correct": is_correct
            })
            
        acc = correct_count / trials_per_length
        print(f"  Context Depth {ctx_len:5d} tokens: Accuracy = {acc*100:5.1f}% ({correct_count}/{trials_per_length})", flush=True)
        results.append({
            "context_length": ctx_len,
            "trials": trials_per_length,
            "correct": correct_count,
            "accuracy_pct": round(acc * 100, 2),
            "details": trial_details
        })
        
    return results

# ---------------------------------------------------------------------------
# Standalone CPU Self-Test
# ---------------------------------------------------------------------------
def self_test_on_cpu():
    print("=== Self-Testing Long-Context Evaluation Suite on CPU ===", flush=True)
    sp = spm.SentencePieceProcessor()
    sp.load(str(TOKENIZER_MODEL))
    
    # Create small 2-layer test model on CPU to verify pipeline logic
    config = LlamaConfig(
        vocab_size=8192,
        hidden_size=256,
        intermediate_size=512,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=1,
        tie_word_embeddings=True,
        torch_dtype="bfloat16",
        _attn_implementation="sdpa",
        max_position_embeddings=4096,
    )
    model = LlamaForCausalLM(config).to("cpu", dtype=torch.bfloat16)
    
    # 1. Test position-bucket logic on 2k slice
    mock_shard = np.random.randint(0, 8192, size=10000, dtype=np.uint16)
    bucket_res = evaluate_position_bucket_loss(model, mock_shard, max_ctx=2048, num_sequences=2, device="cpu")
    print("\nPosition Bucket Results on Mock Data (CPU):")
    for b_name, m in bucket_res.items():
        print(f"  {b_name:8s}: CE = {m['ce_loss']:.4f} | PPL = {m['perplexity']:.2f}")
    assert "0-1k" in bucket_res and "1k-2k" in bucket_res
    print("Position-bucket loss evaluator: PASSED!")
    
    # 2. Test passkey pipeline on 1k and 2k
    passkey_res = run_passkey_retrieval_test(model, sp, lengths=[1024], trials_per_length=2, device="cpu")
    assert len(passkey_res) == 1
    print("Passkey retrieval test pipeline: PASSED!")
    
    del model
    print("\nAll Long-Context Evaluation Tooling Verified Successfully on CPU!\n", flush=True)

if __name__ == "__main__":
    self_test_on_cpu()
