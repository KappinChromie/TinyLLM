"""
Step 3: RoPE-Base Mini-Trial for Stage B (500,000 vs 1,000,000)
- Tests 50M tokens at 8k context for each RoPE base (500k and 1000k).
- From copies of the undecayed Stage A checkpoint.
- Stage B mixture: 35% Wikipedia long, 25% FineWeb long, 15% PG-19 books (8k), 25% short replay (2k).
- Evaluates:
  1. 2k val CE on 4 sets (exact fixed 2M subsets)
  2. Position-bucket loss up to 8k (0-1k, 1k-2k, 2k-4k, 4k-8k)
  3. Passkey retrieval at 2k, 4k, 8k (20 trials each)
- Saves results to results/rope_minitrial_stageB.csv.
"""

import sys
import math
import time
import shutil
import subprocess
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
SRC_CHECKPOINT = MODELS_DIR / "checkpoints" / "stageA" / "stageA_final_undecayed.pt"
TOKENIZER_MODEL = MODELS_DIR / "tokenizer_8k" / "tokenizer.model"
PAUSE_LOG_CSV = RESULTS_DIR / "pause_log.csv"
OUT_CSV = RESULTS_DIR / "rope_minitrial_stageB.csv"

# ---------------------------------------------------------------------------
# GPU Utilities & Temperature Governor (83C / 78C)
# ---------------------------------------------------------------------------
def find_nvidia_smi():
    candidates = [
        "/usr/lib/wsl/lib/nvidia-smi",
        "nvidia-smi",
        "/mnt/c/WINDOWS/system32/nvidia-smi.exe",
        "nvidia-smi.exe"
    ]
    for c in candidates:
        if shutil.which(c) or Path(c).exists():
            return c
    return "nvidia-smi"

SMI_BIN = find_nvidia_smi()

def get_gpu_telemetry():
    try:
        cmd = [SMI_BIN, "--query-gpu=temperature.gpu,power.draw", "--format=csv,noheader,nounits"]
        res = subprocess.run(cmd, capture_output=True, text=True, check=True)
        parts = res.stdout.strip().split(",")
        return float(parts[0]), float(parts[1])
    except Exception:
        return 0.0, 0.0

def apply_temperature_governor(step, accum_tokens, run_id):
    temp, _ = get_gpu_telemetry()
    if temp < 83.0:
        return 0.0, 0
    t_start = time.time()
    t_start_iso = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(t_start))
    temp_initial = temp
    print(f"\n[!WARNING] [{run_id}] Step {step}: GPU temp {temp:.1f}C >= 83C! Pausing...", flush=True)
    while temp > 78.0:
        time.sleep(5)
        temp, _ = get_gpu_telemetry()
        print(f"  Cooling down: current temp = {temp:.1f}C (target <= 78C)...", end="\r", flush=True)
    t_end = time.time()
    t_end_iso = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(t_end))
    pause_duration = t_end - t_start
    print(f"\n[GOVERNOR] Temp dropped to {temp:.1f}C <= 78C. Resuming after {pause_duration:.1f}s pause.", flush=True)
    
    pause_log_exists = PAUSE_LOG_CSV.exists()
    with open(PAUSE_LOG_CSV, "a", encoding="utf-8") as f:
        if not pause_log_exists:
            f.write("timestamp_start,timestamp_end,duration_sec,temp_start,temp_end,optimizer_step,tokens_seen,run_id\n")
        f.write(f"{t_start_iso},{t_end_iso},{pause_duration:.2f},{temp_initial:.1f},{temp:.1f},{step},{accum_tokens},{run_id}\n")
        
    return pause_duration, 1

# ---------------------------------------------------------------------------
# Data Loader (Stage B Mix with Replay Isolation)
# ---------------------------------------------------------------------------
class StageBDataLoader:
    def __init__(self, shards_dir=SHARDS_DIR, seed=42):
        self.seq_len_long = 8192
        self.seq_len_short = 2048
        
        self.wiki_long = np.memmap(shards_dir / "train_wikipedia_long.bin", dtype=np.uint16, mode="r")
        self.fineweb_long = np.memmap(shards_dir / "train_fineweb_edu_long.bin", dtype=np.uint16, mode="r")
        self.pg19_long = np.memmap(shards_dir / "train_pg19_long.bin", dtype=np.uint16, mode="r")
        self.long_shards = [self.wiki_long, self.fineweb_long, self.pg19_long]
        # Normalized shares: 35/75, 25/75, 15/75
        self.long_weights = np.array([0.35, 0.25, 0.15]) / 0.75
        
        self.cosmo_short = np.memmap(shards_dir / "train_cosmopedia.bin", dtype=np.uint16, mode="r")
        self.fineweb_short = np.memmap(shards_dir / "train_fineweb_edu.bin", dtype=np.uint16, mode="r")
        self.short_shards = [self.cosmo_short, self.fineweb_short]
        self.short_weights = np.array([0.50, 0.50])
        
        self.rng_stream = np.random.RandomState(seed)
        self.rng_long = np.random.RandomState(seed + 1001)
        self.rng_short = np.random.RandomState(seed + 2002)

    def get_long_batch(self, batch_size, device="cuda"):
        shard_indices = self.rng_stream.choice(3, size=batch_size, p=self.long_weights)
        x_list, y_list = [], []
        for s_idx in shard_indices:
            shard = self.long_shards[s_idx]
            max_idx = len(shard) - self.seq_len_long - 1
            idx = self.rng_long.randint(0, max_idx)
            x_list.append(shard[idx : idx + self.seq_len_long])
            y_list.append(shard[idx + 1 : idx + 1 + self.seq_len_long])
        x = np.stack(x_list)
        y = np.stack(y_list)
        tx = torch.from_numpy(x.astype(np.int64)).to(device=device, non_blocking=True)
        ty = torch.from_numpy(y.astype(np.int64)).to(device=device, non_blocking=True)
        return tx, ty

    def get_replay_batch(self, batch_size, device="cuda"):
        shard_indices = self.rng_stream.choice(2, size=batch_size, p=self.short_weights)
        x_list, y_list = [], []
        for s_idx in shard_indices:
            shard = self.short_shards[s_idx]
            max_idx = len(shard) - self.seq_len_short - 1
            idx = self.rng_short.randint(0, max_idx)
            x_list.append(shard[idx : idx + self.seq_len_short])
            y_list.append(shard[idx + 1 : idx + 1 + self.seq_len_short])
        x = np.stack(x_list)
        y = np.stack(y_list)
        tx = torch.from_numpy(x.astype(np.int64)).to(device=device, non_blocking=True)
        ty = torch.from_numpy(y.astype(np.int64)).to(device=device, non_blocking=True)
        return tx, ty

# ---------------------------------------------------------------------------
# Evaluators
# ---------------------------------------------------------------------------
class Fixed2MEvaluator:
    def __init__(self, shards_dir, device="cuda"):
        self.device = device
        self.seq_len = 2048
        self.val_shards = {
            "val_wikipedia_neutral": np.memmap(shards_dir / "val_wikipedia_neutral.bin", dtype=np.uint16, mode="r"),
            "val_fineweb_edu": np.memmap(shards_dir / "val_fineweb_edu.bin", dtype=np.uint16, mode="r"),
            "val_cosmopedia": np.memmap(shards_dir / "val_cosmopedia.bin", dtype=np.uint16, mode="r"),
            "val_tinystories": np.memmap(shards_dir / "val_tinystories.bin", dtype=np.uint16, mode="r"),
        }
        self.num_batches = 244
        self.batch_size = 4
        self.fixed_offsets = {}
        for name, shard in self.val_shards.items():
            rng = np.random.RandomState(4242)
            max_idx = len(shard) - self.seq_len - 1
            self.fixed_offsets[name] = [rng.randint(0, max_idx) for _ in range(self.num_batches * self.batch_size)]

    @torch.no_grad()
    def evaluate_set(self, model, name):
        shard = self.val_shards[name]
        offsets = self.fixed_offsets[name]
        total_loss = 0.0
        total_tokens = 0
        for b_idx in range(self.num_batches):
            batch_offsets = offsets[b_idx * self.batch_size : (b_idx + 1) * self.batch_size]
            x = np.stack([shard[i : i + self.seq_len] for i in batch_offsets])
            y = np.stack([shard[i + 1 : i + 1 + self.seq_len] for i in batch_offsets])
            tx = torch.from_numpy(x.astype(np.int64)).to(device=self.device, non_blocking=True)
            ty = torch.from_numpy(y.astype(np.int64)).to(device=self.device, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                outputs = model(tx)
                loss = F.cross_entropy(outputs.logits.reshape(-1, 8192), ty.reshape(-1)).item()
            total_loss += loss * (self.batch_size * self.seq_len)
            total_tokens += self.batch_size * self.seq_len
        return total_loss / total_tokens

    def evaluate_all(self, model):
        model.eval()
        return {name: round(self.evaluate_set(model, name), 4) for name in self.val_shards}

BUCKETS_8K = [
    ("0-1k", 0, 1024),
    ("1k-2k", 1024, 2048),
    ("2k-4k", 2048, 4096),
    ("4k-8k", 4096, 8192),
]

@torch.no_grad()
def evaluate_position_buckets_8k(model, shard, num_sequences=10, seed=42, device="cuda"):
    model.eval()
    max_ctx = 8192
    max_idx = len(shard) - max_ctx - 1
    rng = np.random.RandomState(seed)
    bucket_losses = {name: [] for name, _, _ in BUCKETS_8K}
    for _ in range(num_sequences):
        start_idx = rng.randint(0, max_idx)
        raw_seq = shard[start_idx : start_idx + max_ctx + 1]
        x = torch.from_numpy(raw_seq[:-1].astype(np.int64)).unsqueeze(0).to(device)
        y = torch.from_numpy(raw_seq[1:].astype(np.int64)).unsqueeze(0).to(device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            out = model(x)
            logits = out.logits[0]
            targets = y[0]
            token_losses = F.cross_entropy(logits, targets, reduction="none").float().cpu().numpy()
        for name, b_start, b_end in BUCKETS_8K:
            bucket_losses[name].append(float(np.mean(token_losses[b_start:b_end])))
    return {name: round(float(np.mean(loss_list)), 4) for name, loss_list in bucket_losses.items()}

def run_passkey_retrieval_8k(model, sp, val_shard, lengths=[2048, 4096, 8192], trials_per_length=20, device="cuda"):
    model.eval()
    results = {}
    rng = np.random.RandomState(5432)
    depth_percents = np.linspace(0.05, 0.95, trials_per_length)
    max_idx = len(val_shard) - 100000
    
    for ctx_len in lengths:
        correct = 0
        for trial_idx, depth_pct in enumerate(depth_percents):
            passkey = f"{rng.randint(10000, 99999)}"
            prefix_needle = f"The secret passkey is {passkey}. Remember this passkey. "
            prompt_suffix = " What is the secret passkey? The secret passkey is "
            needle_ids = sp.encode_as_ids(prefix_needle)
            suffix_ids = sp.encode_as_ids(prompt_suffix)
            
            filler_tokens_needed = ctx_len - len(needle_ids) - len(suffix_ids)
            insert_pos = int(filler_tokens_needed * depth_pct)
            
            # Sample filler from held-out shard
            f_start = rng.randint(0, max_idx)
            prefix_filler = list(val_shard[f_start : f_start + insert_pos].astype(int))
            suffix_filler = list(val_shard[f_start + insert_pos : f_start + filler_tokens_needed].astype(int))
            
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
    return results

def create_model(device="cuda", rope_theta=500000.0):
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
        max_position_embeddings=8192,
        rope_theta=rope_theta,
    )
    return LlamaForCausalLM(config).to(device=device, dtype=torch.bfloat16)

def train_minitrial(rope_theta, device="cuda"):
    run_id = f"MiniTrial_RoPE_{int(rope_theta//1000)}k"
    print(f"\n{'='*70}\nStarting 50M Mini-Trial for RoPE base = {rope_theta:.0f} ({run_id})\n{'='*70}", flush=True)
    
    model = create_model(device=device, rope_theta=rope_theta)
    raw = torch.load(SRC_CHECKPOINT, map_location=device, weights_only=False)
    model.load_state_dict(raw["model_state"] if "model_state" in raw else raw)
    
    # Optimizer: AdamW, flat 7e-4 with 1-step micro-ramp
    optimizer = torch.optim.AdamW(model.parameters(), lr=7.0e-4, betas=(0.9, 0.95), weight_decay=0.1, fused=True)
    loader = StageBDataLoader(shards_dir=SHARDS_DIR, seed=42)
    
    total_steps = 191  # 191 steps * 262,144 tokens = 50,069,504 tokens (~50M)
    target_lr = 7.0e-4
    
    t_start = time.time()
    total_pause_time = 0.0
    total_pauses = 0
    
    model.train()
    optimizer.zero_grad(set_to_none=True)
    
    for step in range(total_steps):
        # 1-step micro ramp
        curr_lr = target_lr if step > 0 else (target_lr * 0.5)
        for pg in optimizer.param_groups:
            pg["lr"] = curr_lr
            
        step_loss = 0.0
        
        # Accumulate 12 long micro-batches (2 seqs x 8,192 tok = 16,384 tok each) -> 196,608 tok (75%)
        for _ in range(12):
            x, y = loader.get_long_batch(batch_size=2, device=device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                out = model(x)
                loss = F.cross_entropy(out.logits.reshape(-1, 8192), y.reshape(-1))
                scaled = loss / 16.0
            scaled.backward()
            step_loss += (loss.item() * 16384)
            
        # Accumulate 4 short replay micro-batches (8 seqs x 2,048 tok = 16,384 tok each) -> 65,536 tok (25%)
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
        
        # Thermal governor between optimizer steps
        p_dur, p_cnt = apply_temperature_governor(step + 1, tokens_seen, run_id)
        total_pause_time += p_dur
        total_pauses += p_cnt
        
        if (step + 1) % 25 == 0 or step == total_steps - 1:
            print(f"[{run_id}] Step {step+1:3d}/{total_steps} | Loss: {norm_step_loss:.4f} | Tokens: {tokens_seen/1e6:.1f}M", flush=True)
            
    wall_sec = time.time() - t_start
    active_sec = wall_sec - total_pause_time
    tok_s = (total_steps * 262144) / active_sec if active_sec > 0 else 0
    print(f"\n{run_id} Training Finished in {wall_sec/60:.2f} min (Active: {active_sec/60:.2f} min, Paused: {total_pause_time:.1f}s in {total_pauses} pauses). Speed: {tok_s:.0f} tok/s")
    
    return model, tok_s, active_sec, total_pause_time

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== Step 3: RoPE-Base Mini-Trial (500k vs 1000k) on {device} ===")
    
    # 0. Real Pipeline Pre-Flight Regression Test on Stage B Loader
    print("\n--- Running Pre-Flight Regression Test on Stage B Loader ---")
    test_loader = StageBDataLoader(shards_dir=SHARDS_DIR, seed=999)
    tx_long, ty_long = test_loader.get_long_batch(batch_size=2, device=device)
    assert tx_long.shape == (2, 8192)
    assert (tx_long[:, 1:] == ty_long[:, :-1]).all()
    print("  [Regression Test Check 1]: Long batch y is shifted by exactly 1 relative to x.")
    
    tx_short, ty_short = test_loader.get_replay_batch(batch_size=8, device=device)
    assert tx_short.shape == (8, 2048)
    assert (tx_short[:, 1:] == ty_short[:, :-1]).all()
    print("  [Regression Test Check 2]: Replay batch y is shifted by exactly 1 relative to x.")
    print("  Pre-flight regression test: PASSED!\n")
    
    sp = spm.SentencePieceProcessor()
    sp.load(str(TOKENIZER_MODEL))
    
    eval_2m = Fixed2MEvaluator(SHARDS_DIR, device=device)
    val_pg19 = np.memmap(SHARDS_DIR / "val_pg19_long.bin", dtype=np.uint16, mode="r")
    val_wiki = np.memmap(SHARDS_DIR / "val_wikipedia_long.bin", dtype=np.uint16, mode="r")
    val_fineweb = np.memmap(SHARDS_DIR / "val_fineweb_edu_long.bin", dtype=np.uint16, mode="r")
    
    results = []
    
    # Train and evaluate candidate 1: 500,000
    model_500k, speed_500k, active_500k, pause_500k = train_minitrial(500000.0, device=device)
    print("\nEvaluating Candidate RoPE 500k...")
    ce_2k_500k = eval_2m.evaluate_all(model_500k)
    print(f"  2k Val CE: Wiki={ce_2k_500k['val_wikipedia_neutral']:.4f}, FW={ce_2k_500k['val_fineweb_edu']:.4f}, Cosmo={ce_2k_500k['val_cosmopedia']:.4f}, Tiny={ce_2k_500k['val_tinystories']:.4f}")
    
    b_pg19_500k = evaluate_position_buckets_8k(model_500k, val_pg19, num_sequences=10, device=device)
    b_wiki_500k = evaluate_position_buckets_8k(model_500k, val_wiki, num_sequences=10, device=device)
    b_fw_500k = evaluate_position_buckets_8k(model_500k, val_fineweb, num_sequences=10, device=device)
    print(f"  PG-19 8k Buckets: 0-1k={b_pg19_500k['0-1k']:.4f}, 1k-2k={b_pg19_500k['1k-2k']:.4f}, 2k-4k={b_pg19_500k['2k-4k']:.4f}, 4k-8k={b_pg19_500k['4k-8k']:.4f}")
    
    passkey_500k = run_passkey_retrieval_8k(model_500k, sp, val_pg19, lengths=[2048, 4096, 8192], trials_per_length=20, device=device)
    print(f"  Passkey Retrieval: 2k={passkey_500k['passkey_2k_acc']:.1f}%, 4k={passkey_500k['passkey_4k_acc']:.1f}%, 8k={passkey_500k['passkey_8k_acc']:.1f}%")
    
    results.append({
        "candidate": "RoPE_500k",
        "rope_theta": 500000.0,
        "tokens_m": 50.07,
        "active_speed_tok_s": round(speed_500k, 1),
        "val_wiki_2k_ce": ce_2k_500k['val_wikipedia_neutral'],
        "val_fineweb_2k_ce": ce_2k_500k['val_fineweb_edu'],
        "val_cosmo_2k_ce": ce_2k_500k['val_cosmopedia'],
        "val_tinystories_2k_ce": ce_2k_500k['val_tinystories'],
        "pg19_0_1k": b_pg19_500k["0-1k"],
        "pg19_1k_2k": b_pg19_500k["1k-2k"],
        "pg19_2k_4k": b_pg19_500k["2k-4k"],
        "pg19_4k_8k": b_pg19_500k["4k-8k"],
        "wiki_4k_8k": b_wiki_500k["4k-8k"],
        "fineweb_4k_8k": b_fw_500k["4k-8k"],
        **passkey_500k
    })
    
    del model_500k
    torch.cuda.empty_cache()
    
    # Train and evaluate candidate 2: 1,000,000
    model_1000k, speed_1000k, active_1000k, pause_1000k = train_minitrial(1000000.0, device=device)
    print("\nEvaluating Candidate RoPE 1000k...")
    ce_2k_1000k = eval_2m.evaluate_all(model_1000k)
    print(f"  2k Val CE: Wiki={ce_2k_1000k['val_wikipedia_neutral']:.4f}, FW={ce_2k_1000k['val_fineweb_edu']:.4f}, Cosmo={ce_2k_1000k['val_cosmopedia']:.4f}, Tiny={ce_2k_1000k['val_tinystories']:.4f}")
    
    b_pg19_1000k = evaluate_position_buckets_8k(model_1000k, val_pg19, num_sequences=10, device=device)
    b_wiki_1000k = evaluate_position_buckets_8k(model_1000k, val_wiki, num_sequences=10, device=device)
    b_fw_1000k = evaluate_position_buckets_8k(model_1000k, val_fineweb, num_sequences=10, device=device)
    print(f"  PG-19 8k Buckets: 0-1k={b_pg19_1000k['0-1k']:.4f}, 1k-2k={b_pg19_1000k['1k-2k']:.4f}, 2k-4k={b_pg19_1000k['2k-4k']:.4f}, 4k-8k={b_pg19_1000k['4k-8k']:.4f}")
    
    passkey_1000k = run_passkey_retrieval_8k(model_1000k, sp, val_pg19, lengths=[2048, 4096, 8192], trials_per_length=20, device=device)
    print(f"  Passkey Retrieval: 2k={passkey_1000k['passkey_2k_acc']:.1f}%, 4k={passkey_1000k['passkey_4k_acc']:.1f}%, 8k={passkey_1000k['passkey_8k_acc']:.1f}%")
    
    results.append({
        "candidate": "RoPE_1000k",
        "rope_theta": 1000000.0,
        "tokens_m": 50.07,
        "active_speed_tok_s": round(speed_1000k, 1),
        "val_wiki_2k_ce": ce_2k_1000k['val_wikipedia_neutral'],
        "val_fineweb_2k_ce": ce_2k_1000k['val_fineweb_edu'],
        "val_cosmo_2k_ce": ce_2k_1000k['val_cosmopedia'],
        "val_tinystories_2k_ce": ce_2k_1000k['val_tinystories'],
        "pg19_0_1k": b_pg19_1000k["0-1k"],
        "pg19_1k_2k": b_pg19_1000k["1k-2k"],
        "pg19_2k_4k": b_pg19_1000k["2k-4k"],
        "pg19_4k_8k": b_pg19_1000k["4k-8k"],
        "wiki_4k_8k": b_wiki_1000k["4k-8k"],
        "fineweb_4k_8k": b_fw_1000k["4k-8k"],
        **passkey_1000k
    })
    
    del model_1000k
    torch.cuda.empty_cache()
    
    df = pd.DataFrame(results)
    df.to_csv(OUT_CSV, index=False)
    print(f"\nSaved RoPE mini-trial results to {OUT_CSV}")

if __name__ == "__main__":
    main()
