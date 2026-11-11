"""
RoPE-Base Mini-Trial for Stage C (Step 3)
Tests RoPE base 1,000,000 vs 4,000,000:
- 15M tokens at 32k context (115 steps * 131,072 tokens = 15,073,280 tokens).
- Global batch: 4 sequences of 32k equivalent = 131,072 tokens:
    - 3 micro-batches of 1 sequence of 32k (PG-19 >=32k continuous) = 98,304 tokens (75%)
    - 4 micro-batches of 4 sequences of 2k replay = 32,768 tokens (25%)
      Replay composition: 40% TinyStories, 30% FineWeb-Edu, 30% Cosmopedia v2.
- Learning rate: flat 5.0e-4 with 1-step micro-ramp (step 1: 2.5e-4, step 2+: 5.0e-4).
- Architecture: 6L_768H_1KV, tied embeddings, gradient checkpointing.
- Evaluations per candidate:
    1. 2k Val CE on all 4 sets (fixed 2M subsets)
    2. Position-bucket loss up to 32k on PG-19 val [0-1k, 1k-2k, 2k-4k, 4k-8k, 8k-16k, 16k-32k]
    3. Context-use test at L = 32,768 (target = last 1,024 tokens; Full 32k, Last 8k, Last 2k) on 50 PG-19 val windows.
- Selection Rule:
    Lower loss in 16k-32k bucket, with checks that full-32k context-use <= 8k-only + 0.02,
    and 2k val CE on FineWeb/Cosmo didn't regress by > 0.05. If within noise, choose 1M.
"""

import sys
import math
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
SHARDS_DIR = ROOT / "data" / "shards"
RESULTS_DIR = ROOT / "results"
MODELS_DIR = ROOT / "models"
SRC_CHECKPOINT = MODELS_DIR / "checkpoints" / "stageB" / "stageB_final_undecayed.pt"
OUT_CSV = RESULTS_DIR / "rope_minitrial_stageC.csv"

BUCKETS_32K = [
    ("0-1k", 0, 1024),
    ("1k-2k", 1024, 2048),
    ("2k-4k", 2048, 4096),
    ("4k-8k", 4096, 8192),
    ("8k-16k", 8192, 16384),
    ("16k-32k", 16384, 32768),
]

# ---------------------------------------------------------------------------
# Temperature Governor
# ---------------------------------------------------------------------------
def get_gpu_temperature():
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader,nounits"],
            encoding="utf-8"
        )
        return float(out.strip().split("\n")[0])
    except Exception:
        return 50.0

def apply_temperature_governor(step, accum_tokens, candidate_name):
    temp = get_gpu_temperature()
    paused_seconds = 0.0
    pause_events = 0
    if temp >= 83.0:
        pause_start = time.time()
        print(f"\n[THERMAL GOVERNOR ACTIVE - {candidate_name}] Temp reached {temp}C (>= 83C limit). Pausing training...", flush=True)
        pause_events += 1
        while temp > 78.0:
            time.sleep(5)
            temp = get_gpu_temperature()
            print(f"  [Cooling {candidate_name} Step {step}] Current temp: {temp}C (target <= 78C)...", flush=True)
        paused_seconds = time.time() - pause_start
        print(f"[THERMAL GOVERNOR RESUMED] Resuming after {paused_seconds:.1f}s cooldown. Current temp: {temp}C\n", flush=True)
    return paused_seconds, pause_events

# ---------------------------------------------------------------------------
# Stage C Data Loader for Mini-Trial
# ---------------------------------------------------------------------------
class StageCMiniTrialLoader:
    def __init__(self, shards_dir=SHARDS_DIR, seed=42):
        self.seq_len_long = 32768
        self.seq_len_short = 2048
        
        # PG-19 continuous >= 32k
        self.pg19 = np.memmap(shards_dir / "train_pg19_long.bin", dtype=np.uint16, mode="r")
        eos_pos = np.where(self.pg19 == 2)[0]
        starts = np.concatenate(([0], eos_pos[:-1] + 1))
        lens = eos_pos - starts + 1
        mask = lens >= self.seq_len_long + 1
        self.pg19_starts = starts[mask]
        self.pg19_lens = lens[mask]
        
        # Replay shards: 40% TinyStories, 30% FineWeb-Edu, 30% Cosmopedia
        self.shards_replay = [
            np.memmap(shards_dir / "train_tinystories.bin", dtype=np.uint16, mode="r"),
            np.memmap(shards_dir / "train_fineweb_edu.bin", dtype=np.uint16, mode="r"),
            np.memmap(shards_dir / "train_cosmopedia.bin", dtype=np.uint16, mode="r"),
        ]
        self.replay_weights = np.array([0.40, 0.30, 0.30])
        
        self.rng_long = np.random.RandomState(seed + 10)
        self.rng_short = np.random.RandomState(seed + 20)

    def get_long_sequence(self, device="cuda"):
        """Draws 1 continuous sequence of 32,768 tokens from PG-19 doc >= 32k."""
        doc_choice = self.rng_long.randint(0, len(self.pg19_starts))
        d_start = self.pg19_starts[doc_choice]
        d_len = self.pg19_lens[doc_choice]
        max_offset = d_len - self.seq_len_long - 1
        offset = self.rng_long.randint(0, max_offset + 1) if max_offset > 0 else 0
        seq_start = d_start + offset
        
        seq = self.pg19[seq_start : seq_start + self.seq_len_long + 1]
        x = torch.from_numpy(seq[:-1].astype(np.int64)).unsqueeze(0).to(device)
        y = torch.from_numpy(seq[1:].astype(np.int64)).unsqueeze(0).to(device)
        return x, y

    def get_replay_batch(self, batch_size=4, device="cuda"):
        """Draws isolated short replay batch (2,048 tokens per sequence)."""
        shard_indices = self.rng_short.choice(3, size=batch_size, p=self.replay_weights)
        x_list, y_list = [], []
        for s_idx in shard_indices:
            shard = self.shards_replay[s_idx]
            max_idx = len(shard) - self.seq_len_short - 1
            idx = self.rng_short.randint(0, max_idx)
            x_list.append(shard[idx : idx + self.seq_len_short])
            y_list.append(shard[idx + 1 : idx + 1 + self.seq_len_short])
        x = torch.from_numpy(np.stack(x_list).astype(np.int64)).to(device)
        y = torch.from_numpy(np.stack(y_list).astype(np.int64)).to(device)
        return x, y

# ---------------------------------------------------------------------------
# Evaluators
# ---------------------------------------------------------------------------
class Fixed2MEvaluator:
    def __init__(self, shards_dir=SHARDS_DIR, device="cuda"):
        self.device = device
        self.seq_len = 2048
        self.num_batches = 244
        self.batch_size = 4
        self.eval_sets = {
            "val_wikipedia_neutral": np.memmap(shards_dir / "val_wikipedia_neutral.bin", dtype=np.uint16, mode="r"),
            "val_fineweb_edu": np.memmap(shards_dir / "val_fineweb_edu.bin", dtype=np.uint16, mode="r"),
            "val_cosmopedia": np.memmap(shards_dir / "val_cosmopedia.bin", dtype=np.uint16, mode="r"),
            "val_tinystories": np.memmap(shards_dir / "val_tinystories.bin", dtype=np.uint16, mode="r"),
        }
        self.fixed_data = {}
        for name, shard in self.eval_sets.items():
            batches = []
            for b_idx in range(self.num_batches):
                x_list, y_list = [], []
                for seq_idx in range(self.batch_size):
                    offset = (b_idx * self.batch_size + seq_idx) * self.seq_len
                    x_list.append(shard[offset : offset + self.seq_len])
                    y_list.append(shard[offset + 1 : offset + 1 + self.seq_len])
                batches.append((
                    torch.from_numpy(np.stack(x_list).astype(np.int64)),
                    torch.from_numpy(np.stack(y_list).astype(np.int64))
                ))
            self.fixed_data[name] = batches

    @torch.no_grad()
    def evaluate_all(self, model):
        model.eval()
        results = {}
        for name, batches in self.fixed_data.items():
            total_loss = 0.0
            total_tokens = 0
            for bx, by in batches:
                bx = bx.to(self.device)
                by = by.to(self.device)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    out = model(bx)
                    loss = F.cross_entropy(out.logits.reshape(-1, 8192), by.reshape(-1))
                total_loss += loss.item() * bx.numel()
                total_tokens += bx.numel()
            results[name] = round(total_loss / total_tokens, 4)
        return results

@torch.no_grad()
def evaluate_position_buckets_32k(model, shard, num_sequences=10, device="cuda"):
    model.eval()
    max_idx = len(shard) - 32769
    rng = np.random.RandomState(42)
    bucket_losses = {name: [] for name, start, _ in BUCKETS_32K}
    
    for _ in range(num_sequences):
        start_idx = rng.randint(0, max_idx)
        raw_seq = shard[start_idx : start_idx + 32769]
        x = torch.from_numpy(raw_seq[:-1].astype(np.int64)).unsqueeze(0).to(device)
        y = torch.from_numpy(raw_seq[1:].astype(np.int64)).unsqueeze(0).to(device)
        
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            out = model(x)
            logits = out.logits[0]
            targets = y[0]
            token_losses = F.cross_entropy(logits, targets, reduction="none").float().cpu().numpy()
            
        for name, b_start, b_end in BUCKETS_32K:
            bucket_losses[name].append(float(np.mean(token_losses[b_start:b_end])))
            
    return {name: round(float(np.mean(loss_list)), 4) for name, loss_list in bucket_losses.items()}

@torch.no_grad()
def evaluate_context_use_32k(model, shard, num_windows=50, device="cuda"):
    """
    Context-use test at L = 32,768 (target = last 1,024 tokens)
    Conditions:
      1. Full 32k: input = seq[0 : 32768], logits[31744 : 32768] predicts seq[31745 : 32769]
      2. Last 8k: input = seq[24576 : 32768], logits[7168 : 8192] predicts seq[31745 : 32769]
      3. Last 2k: input = seq[30720 : 32768], logits[1024 : 2048] predicts seq[31745 : 32769]
    """
    model.eval()
    max_idx = len(shard) - 32769
    rng = np.random.RandomState(42)
    step_stride = max(1, max_idx // num_windows)
    starts = [min(max_idx, i * step_stride + rng.randint(0, min(1000, step_stride))) for i in range(num_windows)]
    
    losses_32k = []
    losses_8k = []
    losses_2k = []
    
    for s_idx in starts:
        raw_seq = shard[s_idx : s_idx + 32769]
        target = torch.from_numpy(raw_seq[31745 : 32769].astype(np.int64)).to(device)
        
        # 1. Full 32k
        x_32k = torch.from_numpy(raw_seq[0 : 32768].astype(np.int64)).unsqueeze(0).to(device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            out_32k = model(x_32k)
            loss_32k = F.cross_entropy(out_32k.logits[0, 31744 : 32768], target).item()
        losses_32k.append(loss_32k)
        
        # 2. Last 8k
        x_8k = torch.from_numpy(raw_seq[24576 : 32768].astype(np.int64)).unsqueeze(0).to(device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            out_8k = model(x_8k)
            loss_8k = F.cross_entropy(out_8k.logits[0, 7168 : 8192], target).item()
        losses_8k.append(loss_8k)
        
        # 3. Last 2k
        x_2k = torch.from_numpy(raw_seq[30720 : 32768].astype(np.int64)).unsqueeze(0).to(device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            out_2k = model(x_2k)
            loss_2k = F.cross_entropy(out_2k.logits[0, 1024 : 2048], target).item()
        losses_2k.append(loss_2k)
        
    m_32k = float(np.mean(losses_32k))
    m_8k = float(np.mean(losses_8k))
    m_2k = float(np.mean(losses_2k))
    se_32k = float(np.std(losses_32k) / math.sqrt(num_windows))
    
    return {
        "ctx_32k_mean": round(m_32k, 4),
        "ctx_32k_se": round(se_32k, 4),
        "ctx_8k_mean": round(m_8k, 4),
        "ctx_2k_mean": round(m_2k, 4),
        "delta_32k_minus_8k": round(m_32k - m_8k, 4),
        "gain_32k_vs_2k": round(m_2k - m_32k, 4),
    }

def create_model(device="cuda", rope_theta=1000000.0):
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
        max_position_embeddings=32768,
        rope_theta=rope_theta,
    )
    model = LlamaForCausalLM(config).to(device=device, dtype=torch.bfloat16)
    model.gradient_checkpointing_enable()
    return model

def run_trial(candidate_name, rope_theta, num_steps=115, device="cuda"):
    print(f"\n{'='*75}\n[STAGE C MINI-TRIAL] Candidate: {candidate_name} (RoPE Base = {rope_theta:,.0f})\n{'='*75}", flush=True)
    
    eval_2m = Fixed2MEvaluator(SHARDS_DIR, device=device)
    val_pg19 = np.memmap(SHARDS_DIR / "val_pg19_long.bin", dtype=np.uint16, mode="r")
    
    model = create_model(device=device, rope_theta=rope_theta)
    raw = torch.load(SRC_CHECKPOINT, map_location=device, weights_only=False)
    model.load_state_dict(raw["model_state"] if "model_state" in raw else raw)
    print("  Loaded undecayed Stage B weights cleanly.", flush=True)
    
    loader = StageCMiniTrialLoader(shards_dir=SHARDS_DIR, seed=42)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5.0e-4, betas=(0.9, 0.95), weight_decay=0.1, fused=True)
    
    total_pause_time = 0.0
    total_pauses = 0
    t_start = time.time()
    
    model.train()
    optimizer.zero_grad(set_to_none=True)
    
    print(f"  Starting {num_steps} steps (15.07M tokens) at flat LR 5.0e-4 (1-step micro-ramp)...", flush=True)
    for step in range(num_steps):
        curr_lr = 2.5e-4 if step == 0 else 5.0e-4
        for pg in optimizer.param_groups:
            pg["lr"] = curr_lr
            
        step_loss = 0.0
        # 3 long micro-batches of 1 sequence of 32k (98,304 tokens = 75%)
        for _ in range(3):
            x, y = loader.get_long_sequence(device=device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                out = model(x)
                loss = F.cross_entropy(out.logits.reshape(-1, 8192), y.reshape(-1))
                scaled = loss / 4.0  # 3 long (98k) + 1 replay equiv (32k) = 4 units of 32k
            scaled.backward()
            step_loss += (loss.item() * 32768)
            
        # 4 replay micro-batches of 4 sequences of 2k (32,768 tokens = 25%)
        for _ in range(4):
            x, y = loader.get_replay_batch(batch_size=4, device=device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                out = model(x)
                loss = F.cross_entropy(out.logits.reshape(-1, 8192), y.reshape(-1))
                scaled = loss / 16.0  # 4 * 8192 = 32768 tokens
            scaled.backward()
            step_loss += (loss.item() * 8192)
            
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        
        norm_step_loss = step_loss / 131072.0
        tokens_seen = (step + 1) * 131072
        
        p_dur, p_cnt = apply_temperature_governor(step + 1, tokens_seen, candidate_name)
        total_pause_time += p_dur
        total_pauses += p_cnt
        
        if (step + 1) % 20 == 0 or step == num_steps - 1:
            print(f"  [{candidate_name}] Step {step+1:3d}/{num_steps} | Loss {norm_step_loss:.4f} | Tokens {tokens_seen/1e6:4.1f}M | LR {curr_lr:.2e}", flush=True)
            
    wall_sec = time.time() - t_start
    active_sec = wall_sec - total_pause_time
    tok_s = (num_steps * 131072) / active_sec if active_sec > 0 else 0
    print(f"\n  Finished {candidate_name} in {wall_sec/60:.2f} min (Active: {active_sec/60:.2f} min, Paused: {total_pause_time:.1f}s in {total_pauses} pauses). Speed: {tok_s:.0f} tok/s", flush=True)
    
    # 1. 2k Val CE
    print("  Evaluating 2k Val CE on 4 sets (fixed 2M subsets)...", flush=True)
    ce_2k = eval_2m.evaluate_all(model)
    print(f"    Wiki: {ce_2k['val_wikipedia_neutral']:.4f} | FW: {ce_2k['val_fineweb_edu']:.4f} | Cosmo: {ce_2k['val_cosmopedia']:.4f} | Tiny: {ce_2k['val_tinystories']:.4f}")
    
    # 2. Position-bucket loss up to 32k on PG-19 val
    print("  Evaluating Position-Bucket Loss up to 32k on PG-19 val...", flush=True)
    b_pg19 = evaluate_position_buckets_32k(model, val_pg19, num_sequences=10, device=device)
    print(f"    PG-19 [1k-2k]: {b_pg19['1k-2k']:.4f} | [4k-8k]: {b_pg19['4k-8k']:.4f} | [8k-16k]: {b_pg19['8k-16k']:.4f} | [16k-32k]: {b_pg19['16k-32k']:.4f}")
    
    # 3. Context-use test at L = 32,768 on 50 PG-19 val windows
    print("  Running Context-Use Test at L = 32,768 (50 PG-19 val windows)...", flush=True)
    cu_32k = evaluate_context_use_32k(model, val_pg19, num_windows=50, device=device)
    print(f"    Full 32k Loss: {cu_32k['ctx_32k_mean']:.4f} +/- {cu_32k['ctx_32k_se']:.4f} CE")
    print(f"    Last 8k Loss : {cu_32k['ctx_8k_mean']:.4f} CE | Last 2k Loss: {cu_32k['ctx_2k_mean']:.4f} CE")
    print(f"    Delta (32k - 8k): {cu_32k['delta_32k_minus_8k']:+.4f} CE (Threshold <= +0.02)")
    
    del model
    del optimizer
    torch.cuda.empty_cache()
    
    return {
        "candidate": candidate_name,
        "rope_theta": rope_theta,
        "tokens_m": round((num_steps * 131072) / 1e6, 2),
        "active_speed_tok_s": round(tok_s, 1),
        "wall_time_min": round(wall_sec / 60.0, 2),
        "paused_time_s": round(total_pause_time, 1),
        "pause_events": total_pauses,
        "val_wiki_2k_ce": ce_2k["val_wikipedia_neutral"],
        "val_fineweb_2k_ce": ce_2k["val_fineweb_edu"],
        "val_cosmo_2k_ce": ce_2k["val_cosmopedia"],
        "val_tinystories_2k_ce": ce_2k["val_tinystories"],
        "pg19_0_1k": b_pg19["0-1k"],
        "pg19_1k_2k": b_pg19["1k-2k"],
        "pg19_2k_4k": b_pg19["2k-4k"],
        "pg19_4k_8k": b_pg19["4k-8k"],
        "pg19_8k_16k": b_pg19["8k-16k"],
        "pg19_16k_32k": b_pg19["16k-32k"],
        **cu_32k,
    }

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== Starting Step 3: RoPE Mini-Trial for Stage C (15M Tokens, 32k Context) on {device} ===")
    
    candidates = [
        ("RoPE_1000k", 1000000.0),
        ("RoPE_4000k", 4000000.0),
    ]
    
    results = []
    for name, theta in candidates:
        res = run_trial(name, theta, num_steps=115, device=device)
        results.append(res)
        
    df = pd.DataFrame(results)
    df.to_csv(OUT_CSV, index=False)
    print(f"\nSaved Stage C RoPE Mini-Trial Results to {OUT_CSV}!\n")
    print(df[["candidate", "rope_theta", "val_fineweb_2k_ce", "val_cosmo_2k_ce", "val_tinystories_2k_ce", "pg19_16k_32k", "ctx_32k_mean", "ctx_8k_mean", "delta_32k_minus_8k"]].to_string())

if __name__ == "__main__":
    main()
