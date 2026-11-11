"""
Step 2: Long-Context Baseline Evaluation on Undecayed Stage A Checkpoint (RoPE base 10,000)
- Evaluates loss by position bucket: 0-1k, 1-2k, 2-4k, 4-8k, 8-16k, 16-32k
  on held-out long sets (val_pg19_long, val_wikipedia_long, val_fineweb_edu_long).
- Evaluates passkey retrieval at 1k, 2k, 4k, 8k, 16k, 32k
  (20 trials each, needle at 5%-95% depth, filler sampled from held-out text).
- Saves results to results/longctx_baseline.csv.
"""

import sys
import math
import random
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import LlamaConfig, LlamaForCausalLM
import sentencepiece as spm

sys.stdout.reconfigure(line_buffering=True)

ROOT = Path(__file__).resolve().parent.parent
SHARDS_DIR = ROOT / "data" / "shards"
RESULTS_DIR = ROOT / "results"
MODELS_DIR = ROOT / "models"
CHECKPOINT_PATH = MODELS_DIR / "checkpoints" / "stageA" / "stageA_final_undecayed.pt"
TOKENIZER_MODEL = MODELS_DIR / "tokenizer_8k" / "tokenizer.model"
OUT_CSV = RESULTS_DIR / "longctx_baseline.csv"

BUCKETS = [
    ("0-1k", 0, 1024),
    ("1k-2k", 1024, 2048),
    ("2k-4k", 2048, 4096),
    ("4k-8k", 4096, 8192),
    ("8k-16k", 8192, 16384),
    ("16k-32k", 16384, 32768),
]

PASSKEY_LENGTHS = [1024, 2048, 4096, 8192, 16384, 32768]

def create_model(device="cuda", rope_theta=10000.0, max_pos=32768):
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
        max_position_embeddings=max_pos,
        rope_theta=rope_theta,
    )
    return LlamaForCausalLM(config).to(device=device, dtype=torch.bfloat16)

@torch.no_grad()
def evaluate_position_buckets(model, shard, max_ctx=32768, num_sequences=10, seed=42, device="cuda"):
    model.eval()
    max_idx = len(shard) - max_ctx - 1
    if max_idx <= 0:
        return {}
    rng = np.random.RandomState(seed)
    
    bucket_losses = {name: [] for name, start, _ in BUCKETS if start < max_ctx}
    
    for s_idx in range(num_sequences):
        start_idx = rng.randint(0, max_idx)
        raw_seq = shard[start_idx : start_idx + max_ctx + 1]
        x = torch.from_numpy(raw_seq[:-1].astype(np.int64)).unsqueeze(0).to(device)
        y = torch.from_numpy(raw_seq[1:].astype(np.int64)).unsqueeze(0).to(device)
        
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            out = model(x)
            logits = out.logits[0]  # [seq_len, vocab_size]
            targets = y[0]          # [seq_len]
            token_losses = F.cross_entropy(logits, targets, reduction="none").float().cpu().numpy()
            
        for name, b_start, b_end in BUCKETS:
            if b_start < max_ctx:
                actual_end = min(b_end, max_ctx)
                if actual_end > b_start:
                    b_loss = float(np.mean(token_losses[b_start:actual_end]))
                    bucket_losses[name].append(b_loss)
                    
    results = {}
    for name, loss_list in bucket_losses.items():
        if loss_list:
            results[name] = round(float(np.mean(loss_list)), 4)
    return results

def get_held_out_filler(shard, num_tokens, rng):
    max_idx = len(shard) - num_tokens - 1
    idx = rng.randint(0, max_idx)
    return list(shard[idx : idx + num_tokens].astype(int))

def run_passkey_retrieval(model, sp, val_shard, lengths=PASSKEY_LENGTHS, trials_per_length=20, device="cuda"):
    model.eval()
    results = {}
    rng = np.random.RandomState(1234)
    depth_percents = np.linspace(0.05, 0.95, trials_per_length)
    
    for ctx_len in lengths:
        correct = 0
        for trial_idx, depth_pct in enumerate(depth_percents):
            passkey = f"{rng.randint(10000, 99999)}"
            prefix_needle = f"The secret passkey is {passkey}. Remember this passkey. "
            prompt_suffix = " What is the secret passkey? The secret passkey is "
            
            needle_ids = sp.encode_as_ids(prefix_needle)
            suffix_ids = sp.encode_as_ids(prompt_suffix)
            
            filler_tokens_needed = ctx_len - len(needle_ids) - len(suffix_ids)
            if filler_tokens_needed < 10:
                filler_tokens_needed = 10
                
            insert_pos = int(filler_tokens_needed * depth_pct)
            prefix_filler = get_held_out_filler(val_shard, insert_pos, rng)
            suffix_filler = get_held_out_filler(val_shard, filler_tokens_needed - insert_pos, rng)
            
            full_ids = prefix_filler + needle_ids + suffix_filler + suffix_ids
            input_tensor = torch.tensor([full_ids], dtype=torch.long, device=device)
            
            curr = input_tensor
            gen_ids = []
            with torch.no_grad():
                for _ in range(6):
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        out = model(curr)
                        next_tok = torch.argmax(out.logits[0, -1, :]).item()
                    gen_ids.append(next_tok)
                    curr = torch.cat([curr, torch.tensor([[next_tok]], device=device)], dim=1)
                    if next_tok == sp.eos_id():
                        break
                        
            gen_text = sp.decode(gen_ids).strip()
            if passkey in gen_text:
                correct += 1
                
        acc = round((correct / trials_per_length) * 100.0, 1)
        results[f"passkey_{ctx_len//1024}k_acc"] = acc
        print(f"  [Baseline] Passkey @ {ctx_len:5d} tokens: {acc:5.1f}% ({correct}/{trials_per_length})")
        
    return results

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== Step 2: Long-Context Baseline Evaluation (RoPE Base 10,000) on {device} ===")
    
    t_start = time.time()
    
    # Load tokenizer
    sp = spm.SentencePieceProcessor()
    sp.load(str(TOKENIZER_MODEL))
    
    # Load model with RoPE base 10,000
    print(f"Loading Stage A Undecayed Checkpoint from {CHECKPOINT_PATH}...")
    model = create_model(device=device, rope_theta=10000.0, max_pos=32768)
    raw = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
    model.load_state_dict(raw["model_state"] if "model_state" in raw else raw)
    
    # Load held-out shards
    val_pg19 = np.memmap(SHARDS_DIR / "val_pg19_long.bin", dtype=np.uint16, mode="r")
    val_wiki = np.memmap(SHARDS_DIR / "val_wikipedia_long.bin", dtype=np.uint16, mode="r")
    val_fineweb = np.memmap(SHARDS_DIR / "val_fineweb_edu_long.bin", dtype=np.uint16, mode="r")
    
    print("\n1. Evaluating Position-Bucket Loss on PG-19 Long (32k depth)...")
    buckets_pg19 = evaluate_position_buckets(model, val_pg19, max_ctx=32768, num_sequences=10, device=device)
    for b, l in buckets_pg19.items():
        print(f"  PG-19 [{b:8s}]: CE = {l:.4f}")
        
    print("\n2. Evaluating Position-Bucket Loss on Wikipedia Long (8k depth)...")
    buckets_wiki = evaluate_position_buckets(model, val_wiki, max_ctx=8192, num_sequences=10, device=device)
    for b, l in buckets_wiki.items():
        print(f"  Wikipedia [{b:8s}]: CE = {l:.4f}")

    print("\n3. Evaluating Position-Bucket Loss on FineWeb-Edu Long (8k depth)...")
    buckets_fw = evaluate_position_buckets(model, val_fineweb, max_ctx=8192, num_sequences=10, device=device)
    for b, l in buckets_fw.items():
        print(f"  FineWeb-Edu [{b:8s}]: CE = {l:.4f}")

    print("\n4. Running Passkey Needle-in-a-Haystack Retrieval (Filler from PG-19 Val)...")
    passkey_results = run_passkey_retrieval(model, sp, val_pg19, lengths=PASSKEY_LENGTHS, trials_per_length=20, device=device)
    
    # Assemble comprehensive baseline table
    record = {
        "model": "StageA_Undecayed_Baseline",
        "rope_theta": 10000.0,
        "pg19_0_1k": buckets_pg19.get("0-1k", None),
        "pg19_1k_2k": buckets_pg19.get("1k-2k", None),
        "pg19_2k_4k": buckets_pg19.get("2k-4k", None),
        "pg19_4k_8k": buckets_pg19.get("4k-8k", None),
        "pg19_8k_16k": buckets_pg19.get("8k-16k", None),
        "pg19_16k_32k": buckets_pg19.get("16k-32k", None),
        "wiki_0_1k": buckets_wiki.get("0-1k", None),
        "wiki_1k_2k": buckets_wiki.get("1k-2k", None),
        "wiki_2k_4k": buckets_wiki.get("2k-4k", None),
        "wiki_4k_8k": buckets_wiki.get("4k-8k", None),
        "fineweb_0_1k": buckets_fw.get("0-1k", None),
        "fineweb_1k_2k": buckets_fw.get("1k-2k", None),
        "fineweb_2k_4k": buckets_fw.get("2k-4k", None),
        "fineweb_4k_8k": buckets_fw.get("4k-8k", None),
        **passkey_results
    }
    
    df = pd.DataFrame([record])
    df.to_csv(OUT_CSV, index=False)
    elapsed = time.time() - t_start
    print(f"\nSaved long-context baseline to {OUT_CSV} (Completed in {elapsed:.1f}s)")

if __name__ == "__main__":
    main()
