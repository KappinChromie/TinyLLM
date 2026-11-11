"""
Stage B Fallback Decay Branch (8k Context)
- Branched from a copy of stageB_final_undecayed.pt.
- Runs ~25M tokens (96 steps * 262,144 = 25,165,824 tokens) of cosine decay:
  7.0e-4 down to 1.0e-4 on the Stage B mix.
- Evaluates:
  1. 2k Val CE on 4 sets (fixed 2M subsets)
  2. Position-bucket loss up to 8k (0-1k, 1k-2k, 2k-4k, 4k-8k)
  3. Passkey retrieval at 2k, 4k, 8k (20 trials each)
  4. 5 text generation samples
- Saves decayed model to models/checkpoints/stageB/stageB_decay_branch_final.pt.
"""

import sys
import math
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
SRC_CHECKPOINT = MODELS_DIR / "checkpoints" / "stageB" / "stageB_final_undecayed.pt"
DECAY_CKPT = MODELS_DIR / "checkpoints" / "stageB" / "stageB_decay_branch_final.pt"
TOKENIZER_MODEL = MODELS_DIR / "tokenizer_8k" / "tokenizer.model"
OUT_CSV = RESULTS_DIR / "stageB_decay_branch_eval.csv"
OUT_SAMPLES_MD = RESULTS_DIR / "stageB_decay_branch_samples.md"

from train_stage_b import (
    StageBDataLoader, Fixed2MEvaluator, evaluate_position_buckets_8k,
    run_passkey_retrieval_8k, generate_samples, create_model, apply_temperature_governor
)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== Stage B Fallback Decay Branch Execution on {device} ===")
    
    if not SRC_CHECKPOINT.exists():
        raise FileNotFoundError(f"Source checkpoint {SRC_CHECKPOINT} does not exist!")
        
    sp = spm.SentencePieceProcessor()
    sp.load(str(TOKENIZER_MODEL))
    
    eval_2m = Fixed2MEvaluator(SHARDS_DIR, device=device)
    val_pg19 = np.memmap(SHARDS_DIR / "val_pg19_long.bin", dtype=np.uint16, mode="r")
    val_wiki = np.memmap(SHARDS_DIR / "val_wikipedia_long.bin", dtype=np.uint16, mode="r")
    val_fineweb = np.memmap(SHARDS_DIR / "val_fineweb_edu_long.bin", dtype=np.uint16, mode="r")
    
    model = create_model(device=device, rope_theta=500000.0)
    raw = torch.load(SRC_CHECKPOINT, map_location=device, weights_only=False)
    model.load_state_dict(raw["model_state"] if "model_state" in raw else raw)
    print("Loaded undecayed Stage B weights cleanly.")
    
    total_steps = 96  # 96 steps * 262,144 tokens = 25,165,824 tokens (~25M)
    init_lr = 7.0e-4
    min_lr = 1.0e-4
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=init_lr, betas=(0.9, 0.95), weight_decay=0.1, fused=True)
    loader = StageBDataLoader(shards_dir=SHARDS_DIR, seed=999)
    
    t_start = time.time()
    total_pause_time = 0.0
    total_pauses = 0
    
    print(f"\nStarting ~25M token cosine decay ({total_steps} steps: 7.0e-4 -> 1.0e-4)...", flush=True)
    model.train()
    optimizer.zero_grad(set_to_none=True)
    
    for step in range(total_steps):
        coeff = 0.5 * (1.0 + math.cos(math.pi * step / total_steps))
        curr_lr = min_lr + coeff * (init_lr - min_lr)
        for pg in optimizer.param_groups:
            pg["lr"] = curr_lr
            
        step_loss = 0.0
        # 12 long micro-batches (16,384 tok each) -> 196,608 tok (75%)
        for _ in range(12):
            x, y = loader.get_long_batch(batch_size=2, device=device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                out = model(x)
                loss = F.cross_entropy(out.logits.reshape(-1, 8192), y.reshape(-1))
                scaled = loss / 16.0
            scaled.backward()
            step_loss += (loss.item() * 16384)
            
        # 4 short replay micro-batches (16,384 tok each) -> 65,536 tok (25%)
        for _ in range(4):
            x, y = loader.get_replay_batch(batch_size=8, device=device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                out = model(x)
                loss = F.cross_entropy(out.logits.reshape(-1, 8192), y.reshape(-1))
                scaled = loss / 16.0
            scaled.backward()
            step_loss += (loss.item() * 16384)
            
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        
        norm_step_loss = step_loss / 262144.0
        tokens_seen = (step + 1) * 262144
        
        p_dur, p_cnt = apply_temperature_governor(step + 1, tokens_seen, "StageB_Decay")
        total_pause_time += p_dur
        total_pauses += p_cnt
        
        if (step + 1) % 20 == 0 or step == total_steps - 1:
            print(f"[Stage B Decay] Step {step+1:2d}/{total_steps} | LR {curr_lr:.2e} | Loss {norm_step_loss:.4f} | Tokens: {tokens_seen/1e6:.1f}M", flush=True)
            
    wall_sec = time.time() - t_start
    active_sec = wall_sec - total_pause_time
    tok_s = (total_steps * 262144) / active_sec if active_sec > 0 else 0
    print(f"\nStage B Decay Finished in {wall_sec/60:.2f} min (Active: {active_sec/60:.2f} min, Paused: {total_pause_time:.1f}s in {total_pauses} pauses). Speed: {tok_s:.0f} tok/s")
    
    # Save decayed model
    torch.save({
        "step": total_steps,
        "accum_tokens": total_steps * 262144,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
    }, DECAY_CKPT)
    print(f"Saved Stage B Decayed Checkpoint to {DECAY_CKPT}!", flush=True)
    
    # Comprehensive evaluation
    print("\n--- Evaluating Stage B Decayed Fallback Checkpoint ---", flush=True)
    ce_2k = eval_2m.evaluate_all(model)
    print(f"  2k Val CE: Wiki={ce_2k['val_wikipedia_neutral']:.4f}, FW={ce_2k['val_fineweb_edu']:.4f}, Cosmo={ce_2k['val_cosmopedia']:.4f}, Tiny={ce_2k['val_tinystories']:.4f}")
    
    b_pg19 = evaluate_position_buckets_8k(model, val_pg19, num_sequences=10, device=device)
    b_wiki = evaluate_position_buckets_8k(model, val_wiki, num_sequences=10, device=device)
    b_fw = evaluate_position_buckets_8k(model, val_fineweb, num_sequences=10, device=device)
    print(f"  PG-19 [0-1k]: {b_pg19['0-1k']:.4f} | [1k-2k]: {b_pg19['1k-2k']:.4f} | [2k-4k]: {b_pg19['2k-4k']:.4f} | [4k-8k]: {b_pg19['4k-8k']:.4f}")
    
    passkey_res = run_passkey_retrieval_8k(model, sp, val_pg19, lengths=[2048, 4096, 8192], trials_per_length=20, device=device)
    print(f"  Passkey Acc: 2k = {passkey_res['passkey_2k_acc']:.1f}% | 4k = {passkey_res['passkey_4k_acc']:.1f}% | 8k = {passkey_res['passkey_8k_acc']:.1f}%")
    
    samples = generate_samples(model, sp, device=device)
    with open(OUT_SAMPLES_MD, "w", encoding="utf-8") as f:
        f.write("# Stage B Decayed Fallback Model (8k Context) Samples\n\n")
        for s_idx, s in enumerate(samples, 1):
            f.write(f"### Prompt {s_idx}: `{s['prompt']}`\n\n```text\n{s['completion']}\n```\n\n---\n\n")
            
    df_eval = pd.DataFrame([{
        "model": "StageB_Decayed_Fallback_8k",
        "tokens_decayed_m": round((total_steps * 262144) / 1e6, 2),
        "val_wiki_2k_ce": ce_2k['val_wikipedia_neutral'],
        "val_fineweb_2k_ce": ce_2k['val_fineweb_edu'],
        "val_cosmo_2k_ce": ce_2k['val_cosmopedia'],
        "val_tinystories_2k_ce": ce_2k['val_tinystories'],
        "pg19_0_1k": b_pg19['0-1k'],
        "pg19_1k_2k": b_pg19['1k-2k'],
        "pg19_2k_4k": b_pg19['2k-4k'],
        "pg19_4k_8k": b_pg19['4k-8k'],
        "wiki_4k_8k": b_wiki['4k-8k'],
        "fineweb_4k_8k": b_fw['4k-8k'],
        **passkey_res
    }])
    df_eval.to_csv(OUT_CSV, index=False)
    print(f"Saved Stage B decay evaluation to {OUT_CSV}")

if __name__ == "__main__":
    main()
