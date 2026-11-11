"""
Context-Use Test at 8k Context (Step 2)
The new core metric for long-context support:
- On held-out long documents, takes windows of length L = 8,192 tokens.
- Evaluates cross-entropy loss on the exact LAST 1,024 target tokens of each window under four context conditions:
    1. Full 8k window (7,168 tokens of preceding context)
    2. Last 2k window (1,024 tokens of preceding context)
    3. Last 1.5k window (512 tokens of preceding context)
    4. Last 1k window (0 tokens of preceding context; acts as floor)
- Exactly 100 windows per source:
    - val_pg19_long.bin
    - val_wikipedia_long.bin
    - val_fineweb_edu_long.bin
- Evaluates on:
    1. Undecayed Stage B Checkpoint (RoPE base 500,000)
    2. Undecayed Stage A Checkpoint (RoPE base 10,000)
- Computes mean loss, standard error (SE = std / sqrt(N)), and delta (full 8k vs 2k).
- Pass criterion: Full 8k loss <= 2k loss + 0.02 (more context must not hurt).
"""

import sys
import math
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import LlamaConfig, LlamaForCausalLM

sys.stdout.reconfigure(line_buffering=True)

ROOT = Path(__file__).resolve().parent.parent
SHARDS_DIR = ROOT / "data" / "shards"
RESULTS_DIR = ROOT / "results"
MODELS_DIR = ROOT / "models"

CKPT_STAGE_A = MODELS_DIR / "checkpoints" / "stageA" / "stageA_final_undecayed.pt"
CKPT_STAGE_B = MODELS_DIR / "checkpoints" / "stageB" / "stageB_final_undecayed.pt"

OUT_CSV = RESULTS_DIR / "context_use_test_8k.csv"

SOURCES = [
    ("PG-19 Val", "val_pg19_long.bin"),
    ("Wikipedia Long Val", "val_wikipedia_long.bin"),
    ("FineWeb-Edu Long Val", "val_fineweb_edu_long.bin"),
]

TARGET_LEN = 1024
NUM_WINDOWS = 100

def create_model(device, rope_theta=500000.0, max_ctx=8192):
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

@torch.no_grad()
def evaluate_window_loss(model, seq_full, device):
    """
    Given sequence of length 8193:
    Target tokens: seq_full[7169 : 8193] (1,024 tokens)
    Conditions:
      1. Full 8k: input = seq_full[0 : 8192] (8,192 tokens)
         logits[7168 : 8192] predicts seq_full[7169 : 8193]
      2. Last 2k: input = seq_full[6144 : 8192] (2,048 tokens)
         logits[1024 : 2048] predicts seq_full[7169 : 8193]
      3. Last 1.5k (512 ctx): input = seq_full[6656 : 8192] (1,536 tokens)
         logits[512 : 1536] predicts seq_full[7169 : 8193]
      4. Last 1k (0 ctx floor): input = seq_full[7168 : 8192] (1,024 tokens)
         logits[0 : 1024] predicts seq_full[7169 : 8193]
    """
    target = torch.from_numpy(seq_full[7169 : 8193].astype(np.int64)).to(device)
    
    # Condition 1: Full 8k
    inp_8k = torch.from_numpy(seq_full[0 : 8192].astype(np.int64)).unsqueeze(0).to(device)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        out_8k = model(inp_8k)
        pred_8k = out_8k.logits[0, 7168 : 8192]
        loss_8k = F.cross_entropy(pred_8k, target).item()
        
    # Condition 2: Last 2k
    inp_2k = torch.from_numpy(seq_full[6144 : 8192].astype(np.int64)).unsqueeze(0).to(device)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        out_2k = model(inp_2k)
        pred_2k = out_2k.logits[0, 1024 : 2048]
        loss_2k = F.cross_entropy(pred_2k, target).item()
        
    # Condition 3: Last 1.5k (512 ctx)
    inp_512 = torch.from_numpy(seq_full[6656 : 8192].astype(np.int64)).unsqueeze(0).to(device)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        out_512 = model(inp_512)
        pred_512 = out_512.logits[0, 512 : 1536]
        loss_512 = F.cross_entropy(pred_512, target).item()
        
    # Condition 4: Last 1k (0 ctx floor)
    inp_1k = torch.from_numpy(seq_full[7168 : 8192].astype(np.int64)).unsqueeze(0).to(device)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        out_1k = model(inp_1k)
        pred_1k = out_1k.logits[0, 0 : 1024]
        loss_1k = F.cross_entropy(pred_1k, target).item()
        
    return {
        "loss_full_8k": loss_8k,
        "loss_last_2k": loss_2k,
        "loss_ctx_512": loss_512,
        "loss_floor_1k": loss_1k,
    }

def run_context_use_evaluation(model_name, model, device="cuda"):
    print(f"\n{'='*75}\nRunning Context-Use Evaluation on: {model_name}\n{'='*75}", flush=True)
    results = []
    
    for src_label, fname in SOURCES:
        shard_path = SHARDS_DIR / fname
        shard = np.memmap(shard_path, dtype=np.uint16, mode="r")
        max_idx = len(shard) - 8193 - 1
        
        # Deterministic window sampling across runs
        rng = np.random.RandomState(42)
        step_stride = max(1, max_idx // NUM_WINDOWS)
        starts = [min(max_idx, i * step_stride + rng.randint(0, min(500, step_stride))) for i in range(NUM_WINDOWS)]
        
        losses_8k = []
        losses_2k = []
        losses_512 = []
        losses_1k = []
        
        for w_idx, s_idx in enumerate(starts):
            raw_seq = shard[s_idx : s_idx + 8193]
            res = evaluate_window_loss(model, raw_seq, device=device)
            losses_8k.append(res["loss_full_8k"])
            losses_2k.append(res["loss_last_2k"])
            losses_512.append(res["loss_ctx_512"])
            losses_1k.append(res["loss_floor_1k"])
            
        m_8k, se_8k = float(np.mean(losses_8k)), float(np.std(losses_8k) / math.sqrt(NUM_WINDOWS))
        m_2k, se_2k = float(np.mean(losses_2k)), float(np.std(losses_2k) / math.sqrt(NUM_WINDOWS))
        m_512, se_512 = float(np.mean(losses_512)), float(np.std(losses_512) / math.sqrt(NUM_WINDOWS))
        m_1k, se_1k = float(np.mean(losses_1k)), float(np.std(losses_1k) / math.sqrt(NUM_WINDOWS))
        
        delta_8k_vs_2k = m_8k - m_2k
        gain_vs_floor = m_1k - m_8k
        extra_gain = m_2k - m_8k
        passed = delta_8k_vs_2k <= 0.02
        status = "PASS" if passed else "FAIL"
        
        print(f"\n[{src_label}] (100 windows, target = last 1,024 tokens)")
        print(f"  Floor (0 ctx, 1k block) : {m_1k:.4f} +/- {se_1k:.4f} CE")
        print(f"  512 tokens ctx (1.5k)   : {m_512:.4f} +/- {se_512:.4f} CE")
        print(f"  1,024 tokens ctx (2k)   : {m_2k:.4f} +/- {se_2k:.4f} CE")
        print(f"  7,168 tokens ctx (8k)   : {m_8k:.4f} +/- {se_8k:.4f} CE")
        print(f"  Delta (8k - 2k)         : {delta_8k_vs_2k:+.4f} CE (Threshold <= +0.02) -> [{status}]")
        print(f"  Total Context Gain (1k->8k): {gain_vs_floor:+.4f} CE")
        print(f"  Extra Context Gain (2k->8k): {extra_gain:+.4f} CE")
        
        results.append({
            "model": model_name,
            "source": src_label,
            "num_windows": NUM_WINDOWS,
            "target_tokens": TARGET_LEN,
            "loss_floor_1k_mean": round(m_1k, 4),
            "loss_floor_1k_se": round(se_1k, 4),
            "loss_ctx_512_mean": round(m_512, 4),
            "loss_ctx_512_se": round(se_512, 4),
            "loss_last_2k_mean": round(m_2k, 4),
            "loss_last_2k_se": round(se_2k, 4),
            "loss_full_8k_mean": round(m_8k, 4),
            "loss_full_8k_se": round(se_8k, 4),
            "delta_8k_minus_2k": round(delta_8k_vs_2k, 4),
            "total_gain_vs_floor": round(gain_vs_floor, 4),
            "extra_gain_vs_2k": round(extra_gain, 4),
            "gate_criterion_met": passed,
            "gate_status": status,
        })
        
    return results

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== Starting Step 2: Context-Use Evaluation on {device} ===")
    
    all_results = []
    
    # 1. Undecayed Stage B (RoPE base 500,000)
    model_b = create_model(device=device, rope_theta=500000.0, max_ctx=8192)
    load_checkpoint(model_b, CKPT_STAGE_B, device=device)
    res_b = run_context_use_evaluation("StageB_Undecayed_500k", model_b, device=device)
    all_results.extend(res_b)
    del model_b
    torch.cuda.empty_cache()
    
    # 2. Undecayed Stage A (RoPE base 10,000) for direct comparison
    model_a = create_model(device=device, rope_theta=10000.0, max_ctx=8192)
    load_checkpoint(model_a, CKPT_STAGE_A, device=device)
    res_a = run_context_use_evaluation("StageA_Undecayed_10k_Base", model_a, device=device)
    all_results.extend(res_a)
    del model_a
    torch.cuda.empty_cache()
    
    df = pd.DataFrame(all_results)
    df.to_csv(OUT_CSV, index=False)
    print(f"\nSaved 8k Context-Use Test Results to {OUT_CSV}!\n")

if __name__ == "__main__":
    main()
