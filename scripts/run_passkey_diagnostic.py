"""
Passkey Diagnostic Suite (Step 1)
Evaluates passkey retrieval behavior on both:
  - Undecayed Stage A Checkpoint (RoPE base 10,000, max_ctx 2048)
  - Undecayed Stage B Checkpoint (RoPE base 500,000, max_ctx 8192)

Checks:
1. Scorer Check & 10 Raw Generations with true passkey.
2. Sweep Filler Length (0, 64, 128, 256, 512, 1024, 2048 tokens) across:
   - Held-out prose filler (from val_pg19_long.bin)
   - Simple repetitive filler ("The grass is green. The sky is blue. ")
3. Log-Probability Rank Test:
   - Evaluates true passkey vs 20 random distractor passkeys via teacher-forced log-likelihood.
4. Alternative Prompt Format:
   - Tests sensitivity to wording:
     Format A: "The secret passkey is {passkey}. Remember this passkey. ... What is the secret passkey? The secret passkey is "
     Format B: "The pass code is {passkey}. ... What is the pass code? The pass code is "
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

sys.stdout.reconfigure(line_buffering=True)

ROOT = Path(__file__).resolve().parent.parent
SHARDS_DIR = ROOT / "data" / "shards"
RESULTS_DIR = ROOT / "results"
MODELS_DIR = ROOT / "models"
TOKENIZER_MODEL = MODELS_DIR / "tokenizer_8k" / "tokenizer.model"

CKPT_STAGE_A = MODELS_DIR / "checkpoints" / "stageA" / "stageA_final_undecayed.pt"
CKPT_STAGE_B = MODELS_DIR / "checkpoints" / "stageB" / "stageB_final_undecayed.pt"

OUT_CSV = RESULTS_DIR / "passkey_diagnostic_sweep.csv"
OUT_RAW_GEN_CSV = RESULTS_DIR / "passkey_raw_generations.csv"
OUT_REPORT_MD = RESULTS_DIR / "passkey_diagnostic_report.md"

FILLER_LENGTHS = [0, 64, 128, 256, 512, 1024, 2048]
TRIALS_PER_LEN = 20

def create_model(device, rope_theta=10000.0, max_ctx=2048):
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
        max_position_embeddings=max_ctx,
        rope_theta=rope_theta,
    )
    return LlamaForCausalLM(config).to(device=device, dtype=torch.bfloat16)

def load_checkpoint(model, ckpt_path, device):
    raw = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = raw["model_state"] if "model_state" in raw else raw
    model.load_state_dict(state)
    model.eval()
    print(f"Loaded weights from {ckpt_path.name} cleanly.", flush=True)

def generate_repetitive_filler(sp, num_tokens):
    sent = "The grass is green. The sky is blue. "
    toks = sp.encode_as_ids(sent)
    reps = (num_tokens // len(toks)) + 2
    return (toks * reps)[:num_tokens]

def compute_candidate_logprob(model, context_ids, candidate_ids, device):
    """
    Computes teacher-forced log-likelihood of candidate_ids following context_ids.
    """
    full_ids = torch.tensor([context_ids + candidate_ids], dtype=torch.long, device=device)
    with torch.no_grad():
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            out = model(full_ids)
            logits = out.logits[0]  # [seq_len, vocab_size]
    
    # logits at position i predicts token at i+1
    start_pos = len(context_ids) - 1
    end_pos = start_pos + len(candidate_ids)
    pred_logits = logits[start_pos:end_pos]  # [k, vocab_size]
    target_tokens = torch.tensor(candidate_ids, dtype=torch.long, device=device)
    
    log_probs = F.log_softmax(pred_logits, dim=-1)
    token_log_probs = log_probs[torch.arange(len(candidate_ids)), target_tokens]
    return token_log_probs.sum().item()

def run_diagnostic_for_model(model_name, model, sp, val_pg19, device="cuda"):
    print(f"\n{'='*75}\nRunning Passkey Diagnostics on: {model_name}\n{'='*75}", flush=True)
    rng = np.random.RandomState(42)
    max_pg19_idx = len(val_pg19) - 50000
    
    sweep_records = []
    raw_gen_records = []
    
    depth_percents = np.linspace(0.1, 0.9, TRIALS_PER_LEN)
    
    # -----------------------------------------------------------------------
    # Test 1 & 2: Sweep Filler Lengths with Prose vs Repetitive Filler
    # -----------------------------------------------------------------------
    for filler_type in ["prose", "repetitive"]:
        for f_len in FILLER_LENGTHS:
            correct_greedy = 0
            correct_startswith = 0
            rank1_count = 0
            mean_rank_sum = 0.0
            
            for t_idx, depth_pct in enumerate(depth_percents):
                # Sample random 5-digit passkey
                passkey = f"{rng.randint(10000, 99999)}"
                
                # Standard prompt format
                prefix_needle = f"The secret passkey is {passkey}. Remember this passkey. "
                prompt_suffix = " What is the secret passkey? The secret passkey is "
                needle_ids = sp.encode_as_ids(prefix_needle)
                suffix_ids = sp.encode_as_ids(prompt_suffix)
                
                # Assemble filler
                if f_len == 0:
                    context_ids = needle_ids + suffix_ids
                else:
                    insert_pos = int(f_len * depth_pct)
                    if filler_type == "prose":
                        start_idx = rng.randint(0, max_pg19_idx)
                        pre_filler = list(val_pg19[start_idx : start_idx + insert_pos].astype(int))
                        post_filler = list(val_pg19[start_idx + insert_pos : start_idx + f_len].astype(int))
                    else:
                        pre_filler = generate_repetitive_filler(sp, insert_pos)
                        post_filler = generate_repetitive_filler(sp, f_len - insert_pos)
                    context_ids = pre_filler + needle_ids + post_filler + suffix_ids
                
                # 1. Greedy Generation
                input_tensor = torch.tensor([context_ids], dtype=torch.long, device=device)
                curr = input_tensor
                gen_ids = []
                with torch.no_grad():
                    for _ in range(8):
                        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                            out = model(curr)
                            nxt = torch.argmax(out.logits[0, -1, :]).item()
                        gen_ids.append(nxt)
                        curr = torch.cat([curr, torch.tensor([[nxt]], device=device)], dim=1)
                        if nxt == sp.eos_id():
                            break
                            
                gen_text = sp.decode(gen_ids).strip()
                in_gen = passkey in gen_text
                starts_with = gen_text.startswith(passkey)
                
                if in_gen:
                    correct_greedy += 1
                if starts_with:
                    correct_startswith += 1
                    
                # Record 10 raw generations for inspection at f_len == 1024 prose
                if filler_type == "prose" and f_len == 1024 and t_idx < 10:
                    raw_gen_records.append({
                        "model": model_name,
                        "trial": t_idx + 1,
                        "filler_len": f_len,
                        "depth_pct": round(depth_pct * 100, 1),
                        "true_passkey": passkey,
                        "generated_raw": repr(gen_text),
                        "passkey_in_gen": in_gen,
                        "starts_with_passkey": starts_with
                    })
                    
                # 2. Log-Probability Rank Test vs 20 Distractor Passkeys
                true_ids = sp.encode_as_ids(passkey)
                # Ensure leading space doesn't distort: encode without space and with space
                true_lp = compute_candidate_logprob(model, context_ids, true_ids, device)
                
                distractor_lps = []
                for _ in range(20):
                    dist = f"{rng.randint(10000, 99999)}"
                    while dist == passkey:
                        dist = f"{rng.randint(10000, 99999)}"
                    d_ids = sp.encode_as_ids(dist)
                    d_lp = compute_candidate_logprob(model, context_ids, d_ids, device)
                    distractor_lps.append(d_lp)
                    
                # Rank: how many distractors had strictly higher logprob?
                rank = sum(1 for d in distractor_lps if d > true_lp) + 1
                if rank == 1:
                    rank1_count += 1
                mean_rank_sum += rank
                
            acc_greedy = round((correct_greedy / TRIALS_PER_LEN) * 100.0, 1)
            acc_startswith = round((correct_startswith / TRIALS_PER_LEN) * 100.0, 1)
            acc_rank1 = round((rank1_count / TRIALS_PER_LEN) * 100.0, 1)
            avg_rank = round(mean_rank_sum / TRIALS_PER_LEN, 2)
            
            sweep_records.append({
                "model": model_name,
                "prompt_format": "standard",
                "filler_type": filler_type,
                "filler_tokens": f_len,
                "trials": TRIALS_PER_LEN,
                "greedy_in_acc": acc_greedy,
                "greedy_startswith_acc": acc_startswith,
                "logprob_rank1_acc": acc_rank1,
                "avg_logprob_rank": avg_rank,
            })
            print(f"  [{filler_type:10s} | {f_len:4d} tokens] Greedy Acc: {acc_greedy:5.1f}% | Startswith: {acc_startswith:5.1f}% | LogProb Rank-1: {acc_rank1:5.1f}% | Avg Rank: {avg_rank:4.2f}/21", flush=True)

    # -----------------------------------------------------------------------
    # Test 4: Alternative Prompt Format Sensitivity Check
    # -----------------------------------------------------------------------
    print(f"\n--- Testing Alternative Prompt Format on {model_name} ---", flush=True)
    # Format B: "The pass code is {passkey}. ... What is the pass code? The pass code is "
    for f_len in [0, 128, 512, 1024]:
        correct_greedy = 0
        rank1_count = 0
        mean_rank_sum = 0.0
        
        for t_idx, depth_pct in enumerate(depth_percents):
            passkey = f"{rng.randint(10000, 99999)}"
            prefix_needle = f"The pass code is {passkey}. "
            prompt_suffix = " What is the pass code? The pass code is "
            needle_ids = sp.encode_as_ids(prefix_needle)
            suffix_ids = sp.encode_as_ids(prompt_suffix)
            
            if f_len == 0:
                context_ids = needle_ids + suffix_ids
            else:
                insert_pos = int(f_len * depth_pct)
                start_idx = rng.randint(0, max_pg19_idx)
                pre_filler = list(val_pg19[start_idx : start_idx + insert_pos].astype(int))
                post_filler = list(val_pg19[start_idx + insert_pos : start_idx + f_len].astype(int))
                context_ids = pre_filler + needle_ids + post_filler + suffix_ids
                
            input_tensor = torch.tensor([context_ids], dtype=torch.long, device=device)
            curr = input_tensor
            gen_ids = []
            with torch.no_grad():
                for _ in range(8):
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        out = model(curr)
                        nxt = torch.argmax(out.logits[0, -1, :]).item()
                    gen_ids.append(nxt)
                    curr = torch.cat([curr, torch.tensor([[nxt]], device=device)], dim=1)
                    if nxt == sp.eos_id():
                        break
            gen_text = sp.decode(gen_ids).strip()
            if passkey in gen_text:
                correct_greedy += 1
                
            true_ids = sp.encode_as_ids(passkey)
            true_lp = compute_candidate_logprob(model, context_ids, true_ids, device)
            distractor_lps = []
            for _ in range(20):
                dist = f"{rng.randint(10000, 99999)}"
                d_ids = sp.encode_as_ids(dist)
                distractor_lps.append(compute_candidate_logprob(model, context_ids, d_ids, device))
            rank = sum(1 for d in distractor_lps if d > true_lp) + 1
            if rank == 1:
                rank1_count += 1
            mean_rank_sum += rank
            
        acc_greedy = round((correct_greedy / TRIALS_PER_LEN) * 100.0, 1)
        acc_rank1 = round((rank1_count / TRIALS_PER_LEN) * 100.0, 1)
        avg_rank = round(mean_rank_sum / TRIALS_PER_LEN, 2)
        
        sweep_records.append({
            "model": model_name,
            "prompt_format": "pass_code_alt",
            "filler_type": "prose",
            "filler_tokens": f_len,
            "trials": TRIALS_PER_LEN,
            "greedy_in_acc": acc_greedy,
            "greedy_startswith_acc": acc_greedy,
            "logprob_rank1_acc": acc_rank1,
            "avg_logprob_rank": avg_rank,
        })
        print(f"  [Alt Format | {f_len:4d} tokens] Greedy Acc: {acc_greedy:5.1f}% | LogProb Rank-1: {acc_rank1:5.1f}% | Avg Rank: {avg_rank:4.2f}/21", flush=True)

    return sweep_records, raw_gen_records

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== Starting Step 1: Passkey Diagnostic on {device} ===")
    
    sp = spm.SentencePieceProcessor()
    sp.load(str(TOKENIZER_MODEL))
    val_pg19 = np.memmap(SHARDS_DIR / "val_pg19_long.bin", dtype=np.uint16, mode="r")
    
    all_sweep = []
    all_raw = []
    
    # 1. Undecayed Stage A
    model_a = create_model(device=device, rope_theta=10000.0, max_ctx=2048)
    load_checkpoint(model_a, CKPT_STAGE_A, device=device)
    sw_a, raw_a = run_diagnostic_for_model("StageA_Undecayed", model_a, sp, val_pg19, device=device)
    all_sweep.extend(sw_a)
    all_raw.extend(raw_a)
    del model_a
    torch.cuda.empty_cache()
    
    # 2. Undecayed Stage B
    model_b = create_model(device=device, rope_theta=500000.0, max_ctx=8192)
    load_checkpoint(model_b, CKPT_STAGE_B, device=device)
    sw_b, raw_b = run_diagnostic_for_model("StageB_Undecayed", model_b, sp, val_pg19, device=device)
    all_sweep.extend(sw_b)
    all_raw.extend(raw_b)
    del model_b
    torch.cuda.empty_cache()
    
    df_sweep = pd.DataFrame(all_sweep)
    df_sweep.to_csv(OUT_CSV, index=False)
    print(f"\nSaved passkey diagnostic sweep to {OUT_CSV}")
    
    df_raw = pd.DataFrame(all_raw)
    df_raw.to_csv(OUT_RAW_GEN_CSV, index=False)
    print(f"Saved raw generations to {OUT_RAW_GEN_CSV}")

if __name__ == "__main__":
    main()
