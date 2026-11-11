import os
import sys
import time
import math
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, LlamaConfig, LlamaForCausalLM

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
sys.stdout.reconfigure(line_buffering=True)

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "shards"
RESULTS_DIR = ROOT / "results"
CKPT_DIR = ROOT / "models" / "checkpoints"

CKPT_BASE = CKPT_DIR / "stageC" / "stageC_decayed_final.pt"
CKPT_INIT = CKPT_DIR / "stageC_chat_init.pt"
CKPT_CHAT = CKPT_DIR / "chat_sft_final.pt"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def load_llama_model(ckpt_path, vocab_size=8256):
    print(f"Loading {ckpt_path.name} (vocab={vocab_size})...", flush=True)
    cfg = LlamaConfig(
        vocab_size=vocab_size,
        hidden_size=768,
        intermediate_size=2048,
        num_hidden_layers=6,
        num_attention_heads=12,
        num_key_value_heads=1,
        tie_word_embeddings=True,
        torch_dtype="bfloat16",
        _attn_implementation="sdpa",
        max_position_embeddings=32768,
        rope_theta=1000000.0,
    )
    model = LlamaForCausalLM(cfg).to(device=DEVICE, dtype=torch.bfloat16)
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    state = ckpt["model_state"] if "model_state" in ckpt else ckpt
    
    # Handle vocab size mismatch if loading 8192-vocab base model into 8256 config or vice versa
    if "model.embed_tokens.weight" in state:
        ckpt_vocab = state["model.embed_tokens.weight"].shape[0]
        if ckpt_vocab != vocab_size:
            print(f"  Note: checkpoint vocab is {ckpt_vocab}, target model config vocab is {vocab_size}")
            # Recreate with exact checkpoint vocab size
            cfg.vocab_size = ckpt_vocab
            model = LlamaForCausalLM(cfg).to(device=DEVICE, dtype=torch.bfloat16)
            
    model.load_state_dict(state)
    model.eval()
    return model

# ---------------------------------------------------------------------------
# 1. Context-Use Side-by-Side (8k vs 2k, 32k vs 8k, 32k vs 2k)
# ---------------------------------------------------------------------------
@torch.no_grad()
def evaluate_context_use_all_models(models_dict):
    """
    Evaluates 8k vs 2k (PG-19, Wiki, FineWeb; 100 windows each)
    and 32k vs 8k, 32k vs 2k (PG-19; 50 windows)
    Target = last 1,024 tokens.
    """
    print("\n" + "="*75)
    print("RUNNING CONTEXT-USE EVALUATION (IDENTICAL PROTOCOL, WINDOWS, TOKENS)")
    print("="*75, flush=True)
    
    shards_8k = [
        ("PG-19 Val", DATA_DIR / "val_pg19_long.bin", 100),
        ("Wikipedia Long Val", DATA_DIR / "val_wikipedia_long.bin", 100),
        ("FineWeb-Edu Long Val", DATA_DIR / "val_fineweb_edu_long.bin", 100),
    ]
    
    results = []
    
    # 8k vs 2k tests
    for name, shard_path, max_win in shards_8k:
        print(f"\n--- 8k vs 2k on {name} ({max_win} windows) ---", flush=True)
        shard = np.memmap(shard_path, dtype=np.uint16, mode="r")
        actual_win = min(max_win, len(shard) // 8193)
        
        # Preload windows into memory to ensure exact identity across models
        windows = [np.array(shard[i*8193 : (i+1)*8193], dtype=np.int64) for i in range(actual_win)]
        
        for m_name, model in models_dict.items():
            diffs_8k_vs_2k = []
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                for raw in windows:
                    # Full 8k context (tokens 0..8192)
                    x_8k = torch.from_numpy(raw[:-1]).unsqueeze(0).to(DEVICE)
                    y_8k = torch.from_numpy(raw[1:]).to(DEVICE)
                    out_8k = model(x_8k)
                    loss_8k = F.cross_entropy(out_8k.logits[0, -1024:], y_8k[-1024:]).item()
                    
                    # 2k context (tokens 6144..8192, length 2048)
                    raw_2k = raw[-2049:]
                    x_2k = torch.from_numpy(raw_2k[:-1]).unsqueeze(0).to(DEVICE)
                    y_2k = torch.from_numpy(raw_2k[1:]).to(DEVICE)
                    out_2k = model(x_2k)
                    loss_2k = F.cross_entropy(out_2k.logits[0, -1024:], y_2k[-1024:]).item()
                    
                    diffs_8k_vs_2k.append(loss_8k - loss_2k)
                    
            arr = np.array(diffs_8k_vs_2k)
            mean_d = float(np.mean(arr))
            se_d = float(np.std(arr, ddof=1) / np.sqrt(len(arr)))
            results.append({
                "model": m_name,
                "dataset": name,
                "comparison": "8k vs 2k (target=last 1k)",
                "windows": len(arr),
                "mean_delta": mean_d,
                "se": se_d,
                "ci_95_low": mean_d - 1.96 * se_d,
                "ci_95_high": mean_d + 1.96 * se_d,
                "status": "PASS" if (mean_d + 1.96 * se_d) <= 0.02 else "FLAG"
            })
            print(f"  {m_name:20s}: Mean Delta = {mean_d:+.4f} (SE: {se_d:.4f}, 95% CI: [{mean_d - 1.96 * se_d:+.4f}, {mean_d + 1.96 * se_d:+.4f}])")

    # 32k vs 8k and 32k vs 2k on PG-19 (50 windows)
    print("\n--- 32k tests on PG-19 Val (50 windows) ---", flush=True)
    pg19_shard = np.memmap(DATA_DIR / "val_pg19_long.bin", dtype=np.uint16, mode="r")
    num_32k_win = 50
    windows_32k = [np.array(pg19_shard[i*32769 : (i+1)*32769], dtype=np.int64) for i in range(num_32k_win)]
    
    for m_name, model in models_dict.items():
        diffs_32k_vs_8k = []
        diffs_32k_vs_2k = []
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            for raw in windows_32k:
                # 32k full
                x_32k = torch.from_numpy(raw[:-1]).unsqueeze(0).to(DEVICE)
                y_32k = torch.from_numpy(raw[1:]).to(DEVICE)
                out_32k = model(x_32k)
                loss_32k = F.cross_entropy(out_32k.logits[0, -1024:], y_32k[-1024:]).item()
                
                # 8k context (last 8192 tokens)
                raw_8k = raw[-8193:]
                x_8k = torch.from_numpy(raw_8k[:-1]).unsqueeze(0).to(DEVICE)
                y_8k = torch.from_numpy(raw_8k[1:]).to(DEVICE)
                out_8k = model(x_8k)
                loss_8k = F.cross_entropy(out_8k.logits[0, -1024:], y_8k[-1024:]).item()
                
                # 2k context (last 2048 tokens)
                raw_2k = raw[-2049:]
                x_2k = torch.from_numpy(raw_2k[:-1]).unsqueeze(0).to(DEVICE)
                y_2k = torch.from_numpy(raw_2k[1:]).to(DEVICE)
                out_2k = model(x_2k)
                loss_2k = F.cross_entropy(out_2k.logits[0, -1024:], y_2k[-1024:]).item()
                
                diffs_32k_vs_8k.append(loss_32k - loss_8k)
                diffs_32k_vs_2k.append(loss_32k - loss_2k)
                
        for comp, diff_list in [("32k vs 8k (target=last 1k)", diffs_32k_vs_8k), ("32k vs 2k (target=last 1k)", diffs_32k_vs_2k)]:
            arr = np.array(diff_list)
            mean_d = float(np.mean(arr))
            se_d = float(np.std(arr, ddof=1) / np.sqrt(len(arr)))
            results.append({
                "model": m_name,
                "dataset": "PG-19 Val (32k)",
                "comparison": comp,
                "windows": len(arr),
                "mean_delta": mean_d,
                "se": se_d,
                "ci_95_low": mean_d - 1.96 * se_d,
                "ci_95_high": mean_d + 1.96 * se_d,
                "status": "PASS" if (mean_d + 1.96 * se_d) <= 0.02 else "FLAG"
            })
            print(f"  {m_name:20s} [{comp:25s}]: Mean Delta = {mean_d:+.4f} (SE: {se_d:.4f}, 95% CI: [{mean_d - 1.96 * se_d:+.4f}, {mean_d + 1.96 * se_d:+.4f}])")

    df = pd.DataFrame(results)
    out_path = RESULTS_DIR / "step2_audited_context_use_side_by_side.csv"
    df.to_csv(out_path, index=False)
    print(f"\nSaved audited context use table to {out_path}")
    return df

# ---------------------------------------------------------------------------
# 2. Position Stability Side-by-Side (50 windows of 32k PG-19)
# ---------------------------------------------------------------------------
@torch.no_grad()
def evaluate_position_stability_side_by_side(base_model, chat_model):
    print("\n" + "="*75)
    print("RUNNING POSITION STABILITY (50 IDENTICAL WINDOWS, BASE VS CHAT)")
    print("="*75, flush=True)
    
    pg19_shard = np.memmap(DATA_DIR / "val_pg19_long.bin", dtype=np.uint16, mode="r")
    num_windows = 50
    windows = [np.array(pg19_shard[i*32769 : (i+1)*32769], dtype=np.int64) for i in range(num_windows)]
    
    models = [("Stage C Base Model", base_model), ("Chat Model (SFT Final)", chat_model)]
    results = []
    
    for m_name, model in models:
        diffs = []
        losses_2_4 = []
        losses_16_32 = []
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            for raw in windows:
                x = torch.from_numpy(raw[:-1]).unsqueeze(0).to(DEVICE)
                y = torch.from_numpy(raw[1:]).to(DEVICE)
                out = model(x)
                logits = out.logits[0]
                
                l_2_4 = F.cross_entropy(logits[2048:4096], y[2048:4096]).item()
                l_16_32 = F.cross_entropy(logits[16384:32768], y[16384:32768]).item()
                
                losses_2_4.append(l_2_4)
                losses_16_32.append(l_16_32)
                diffs.append(l_16_32 - l_2_4)
                
        arr = np.array(diffs)
        mean_d = float(np.mean(arr))
        se_d = float(np.std(arr, ddof=1) / np.sqrt(len(arr)))
        pct_gt_05 = float(np.mean(arr > 0.05) * 100.0)
        
        results.append({
            "model": m_name,
            "mean_ce_2k_4k": float(np.mean(losses_2_4)),
            "mean_ce_16k_32k": float(np.mean(losses_16_32)),
            "mean_delta_16_32_vs_2_4": mean_d,
            "se": se_d,
            "ci_95_low": mean_d - 1.96 * se_d,
            "ci_95_high": mean_d + 1.96 * se_d,
            "pct_windows_gt_005": pct_gt_05,
            "status": "PASS" if (mean_d + 1.96 * se_d) <= 0.02 and pct_gt_05 < 30.0 else "FLAG"
        })
        print(f"  {m_name:25s}: Mean [2k-4k]={np.mean(losses_2_4):.4f} | Mean [16k-32k]={np.mean(losses_16_32):.4f} | Delta={mean_d:+.4f} (95% CI: [{mean_d - 1.96 * se_d:+.4f}, {mean_d + 1.96 * se_d:+.4f}], >0.05: {pct_gt_05:.1f}%)")

    df = pd.DataFrame(results)
    out_path = RESULTS_DIR / "step2_audited_position_stability_side_by_side.csv"
    df.to_csv(out_path, index=False)
    print(f"\nSaved audited position stability table to {out_path}")
    return df

def main():
    print("="*75)
    print("STEP 2 EVALUATIONS: CONTEXT-USE AND POSITION STABILITY")
    print("="*75)
    
    # Load models
    base_model = load_llama_model(CKPT_BASE, vocab_size=8192)
    init_model = load_llama_model(CKPT_INIT, vocab_size=8256)
    chat_model = load_llama_model(CKPT_CHAT, vocab_size=8256)
    
    models_dict = {
        "Stage C Base": base_model,
        "SFT-Init": init_model,
        "Chat Model (Final)": chat_model
    }
    
    evaluate_context_use_all_models(models_dict)
    evaluate_position_stability_side_by_side(base_model, chat_model)

if __name__ == "__main__":
    main()
