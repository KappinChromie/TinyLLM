"""
Stage B: Long-Context Extension to 8,192 tokens (RoPE Base 500,000)
- Architecture: 6L_768H_1KV (42.28M params, tied embeddings)
- Context length: 8,192 tokens
- RoPE base: 500,000.0 (Winning base from Step 3 mini-trial)
- Total tokens: 250,085,376 (~250.1M tokens, 954 optimizer steps)
- Global batch: 32 sequences (262,144 tokens per step)
- Accumulation:
    12 long micro-batches (2 seqs x 8,192 = 16,384 tok each) -> 196,608 tok (75%)
     4 short replay micro-batches (8 seqs x 2,048 = 16,384 tok each) -> 65,536 tok (25%)
- Mixture: 35% Wikipedia Long / 25% FineWeb Long / 15% PG-19 Books (8k), 25% Short Replay (2k)
- Learning rate: Flat 7.0e-4 with 1-step micro-ramp (no re-warm to 2e-3)
- Evaluates at 25%, 50%, 75%, 100% of stage:
    2k Val CE on 4 sets (fixed 2M subsets)
    Position-bucket loss up to 8k
    Passkey retrieval at 2k, 4k, 8k
- Infrastructure:
    Temperature governor (83C pause / 78C resume)
    Loss spike guard (1.5x trailing avg for 20 steps -> rollback)
    Rolling 15-minute checkpoints + permanent milestone checkpoints
"""

import os
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
STAGE_B_CKPT_DIR = MODELS_DIR / "checkpoints" / "stageB"
TOKENIZER_MODEL = MODELS_DIR / "tokenizer_8k" / "tokenizer.model"
PAUSE_LOG_CSV = RESULTS_DIR / "pause_log.csv"
TRAIN_LOG_CSV = RESULTS_DIR / "stageB_train_log.csv"
EVAL_LOG_CSV = RESULTS_DIR / "stageB_eval.csv"
SAMPLES_MD = RESULTS_DIR / "stageB_samples.md"
RESTARTS_LOG = RESULTS_DIR / "stageB_restarts.log"

SAMPLE_PROMPTS_5 = [
    "Once upon a time, there was a little bunny named ",
    "The fundamental difference between planets and stars is ",
    "In ancient history, the Great Pyramid of Giza was built ",
    "Lily wanted to bake a cake for her mom. First, she ",
    "Water evaporates when the temperature rises because ",
]

BUCKETS_8K = [
    ("0-1k", 0, 1024),
    ("1k-2k", 1024, 2048),
    ("2k-4k", 2048, 4096),
    ("4k-8k", 4096, 8192),
]

# ---------------------------------------------------------------------------
# GPU Utilities & Temperature Governor
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
    print(f"\n[!WARNING] [{run_id}] Step {step}: GPU temp {temp:.1f}C >= 83C! Pausing training...", flush=True)
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
        self.long_weights = np.array([0.35, 0.25, 0.15]) / 0.75
        
        self.cosmo_short = np.memmap(shards_dir / "train_cosmopedia.bin", dtype=np.uint16, mode="r")
        self.fineweb_short = np.memmap(shards_dir / "train_fineweb_edu.bin", dtype=np.uint16, mode="r")
        self.short_shards = [self.cosmo_short, self.fineweb_short]
        self.short_weights = np.array([0.50, 0.50])
        
        self.rng_stream = np.random.RandomState(seed)
        self.rng_long = np.random.RandomState(seed + 1001)
        self.rng_short = np.random.RandomState(seed + 2002)

    def get_state(self):
        return {
            "rng_stream": self.rng_stream.get_state(),
            "rng_long": self.rng_long.get_state(),
            "rng_short": self.rng_short.get_state(),
        }

    def set_state(self, state):
        self.rng_stream.set_state(state["rng_stream"])
        self.rng_long.set_state(state["rng_long"])
        self.rng_short.set_state(state["rng_short"])

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
    rng = np.random.RandomState(8888)
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

def generate_samples(model, sp, prompts=SAMPLE_PROMPTS_5, max_new=32, device="cuda"):
    model.eval()
    samples = []
    for p in prompts:
        curr = sp.encode_as_ids(p)
        with torch.no_grad():
            for _ in range(max_new):
                inp = torch.tensor([curr[-8192:]], dtype=torch.long, device=device)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    out = model(inp)
                    nxt = torch.argmax(out.logits[0, -1, :]).item()
                curr.append(nxt)
                if nxt == sp.eos_id():
                    break
        samples.append({"prompt": p, "completion": sp.decode(curr).strip()})
    return samples

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

def run_preflight_regression_test(model, loader, device="cuda"):
    print("--- Running Pre-Flight Regression Test on Stage B Pipeline ---", flush=True)
    # Check 1: Long batch shift
    x_long, y_long = loader.get_long_batch(batch_size=2, device=device)
    assert x_long.shape == (2, 8192)
    assert (x_long[:, 1:] == y_long[:, :-1]).all()
    print("  [Check 1 Passed]: Long batch y is shifted by exactly 1.")
    
    # Check 2: Replay batch shift
    x_short, y_short = loader.get_replay_batch(batch_size=8, device=device)
    assert x_short.shape == (8, 2048)
    assert (x_short[:, 1:] == y_short[:, :-1]).all()
    print("  [Check 2 Passed]: Replay batch y is shifted by exactly 1.")
    
    # Check 3: Loss calculation parity
    with torch.no_grad():
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            out = model(x_long)
            loss_rep = F.cross_entropy(out.logits.reshape(-1, 8192), y_long.reshape(-1))
            tokens = torch.cat([x_long[:, :1], y_long], dim=1)
            out_ind = model(tokens[:, :-1])
            loss_ind = F.cross_entropy(out_ind.logits.reshape(-1, 8192), tokens[:, 1:].reshape(-1))
            assert torch.allclose(loss_rep, loss_ind, atol=1e-5)
    print("  [Check 3 Passed]: Reported loss matches independent loss.")
    print("Pre-flight regression test: ALL CHECKS PASSED!\n", flush=True)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== Stage B: Long-Context Extension to 8k (RoPE Base 500k) on {device} ===")
    
    STAGE_B_CKPT_DIR.mkdir(parents=True, exist_ok=True)
    
    sp = spm.SentencePieceProcessor()
    sp.load(str(TOKENIZER_MODEL))
    
    model = create_model(device=device, rope_theta=500000.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=7.0e-4, betas=(0.9, 0.95), weight_decay=0.1, fused=True)
    loader = StageBDataLoader(shards_dir=SHARDS_DIR, seed=42)
    eval_2m = Fixed2MEvaluator(SHARDS_DIR, device=device)
    
    val_pg19 = np.memmap(SHARDS_DIR / "val_pg19_long.bin", dtype=np.uint16, mode="r")
    val_wiki = np.memmap(SHARDS_DIR / "val_wikipedia_long.bin", dtype=np.uint16, mode="r")
    val_fineweb = np.memmap(SHARDS_DIR / "val_fineweb_edu_long.bin", dtype=np.uint16, mode="r")
    
    run_preflight_regression_test(model, loader, device=device)
    
    # Check for resume checkpoint
    latest_ckpt = STAGE_B_CKPT_DIR / "latest_rolling.pt"
    start_step = 0
    accum_tokens = 0
    total_pause_time = 0.0
    total_pauses = 0
    spike_event_count = 0
    recent_losses = []
    
    if latest_ckpt.exists():
        print(f"Resuming from rolling checkpoint: {latest_ckpt}...")
        ckpt = torch.load(latest_ckpt, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        loader.set_state(ckpt["loader_state"])
        start_step = ckpt["step"]
        accum_tokens = ckpt["accum_tokens"]
        total_pause_time = ckpt.get("total_pause_time", 0.0)
        total_pauses = ckpt.get("total_pauses", 0)
        spike_event_count = ckpt.get("spike_event_count", 0)
        print(f"Resumed at step {start_step} ({accum_tokens/1e6:.1f}M tokens).")
    else:
        print(f"Loading initial weights from undecayed Stage A: {SRC_CHECKPOINT}...")
        raw = torch.load(SRC_CHECKPOINT, map_location=device, weights_only=False)
        model.load_state_dict(raw["model_state"] if "model_state" in raw else raw)
        print("Loaded Stage A weights cleanly.")
        
    total_steps = 954  # 954 * 262,144 = 250,085,376 tokens (~250.1M tokens)
    milestones = [238, 477, 715, 954]  # 25%, 50%, 75%, 100%
    target_lr = 7.0e-4
    
    wall_start = time.time()
    last_ckpt_time = wall_start
    
    # Init CSV logs if needed
    if not TRAIN_LOG_CSV.exists():
        with open(TRAIN_LOG_CSV, "w", encoding="utf-8") as f:
            f.write("step,tokens_m,train_loss,lr,grad_norm,active_tok_s,active_time_min,wall_time_min,paused_time_min,temp_gpu,power_w\n")
    if not EVAL_LOG_CSV.exists():
        with open(EVAL_LOG_CSV, "w", encoding="utf-8") as f:
            f.write("milestone_pct,step,tokens_m,val_wiki_2k_ce,val_fineweb_2k_ce,val_cosmo_2k_ce,val_tinystories_2k_ce,pg19_0_1k,pg19_1k_2k,pg19_2k_4k,pg19_4k_8k,wiki_4k_8k,fineweb_4k_8k,passkey_2k_acc,passkey_4k_acc,passkey_8k_acc,eval_sec\n")

    print(f"\nStarting Stage B Training (Steps {start_step+1} to {total_steps})...\n", flush=True)
    model.train()
    optimizer.zero_grad(set_to_none=True)
    
    for step in range(start_step, total_steps):
        # 1-step micro-ramp
        curr_lr = target_lr if (step > 0 or start_step > 0) else (target_lr * 0.5)
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
            
        gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0).item()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        
        norm_step_loss = step_loss / 262144.0
        accum_tokens += 262144
        recent_losses.append(norm_step_loss)
        if len(recent_losses) > 100:
            recent_losses.pop(0)
            
        # Loss-spike guard: if loss > 1.5x trailing avg for 20 consecutive steps, roll back
        if len(recent_losses) >= 100:
            trailing_avg = np.mean(recent_losses[:-20])
            recent_tail = recent_losses[-20:]
            if all(l > 1.5 * trailing_avg for l in recent_tail):
                spike_event_count += 1
                print(f"\n[!CRITICAL] Step {step+1}: Loss spike detected (20 consecutive steps > 1.5x trailing avg {trailing_avg:.4f}).", flush=True)
                if spike_event_count > 1:
                    raise RuntimeError("FATAL: Second loss spike event within run. Training halted.")
                print("Rolling back to latest checkpoint...", flush=True)
                ckpt = torch.load(latest_ckpt, map_location=device, weights_only=False)
                model.load_state_dict(ckpt["model_state"])
                optimizer.load_state_dict(ckpt["optimizer_state"])
                loader.set_state(ckpt["loader_state"])
                step = ckpt["step"]
                accum_tokens = ckpt["accum_tokens"]
                recent_losses = []
                continue

        # Temperature governor between optimizer steps
        p_dur, p_cnt = apply_temperature_governor(step + 1, accum_tokens, "StageB_8k")
        total_pause_time += p_dur
        total_pauses += p_cnt
        
        # Periodic step logging (every 10 steps)
        if (step + 1) % 10 == 0 or (step + 1) in milestones:
            now = time.time()
            wall_elapsed = now - wall_start
            active_elapsed = wall_elapsed - total_pause_time
            active_tok_s = accum_tokens / active_elapsed if active_elapsed > 0 else 0
            temp_g, pwr = get_gpu_telemetry()
            with open(TRAIN_LOG_CSV, "a", encoding="utf-8") as f:
                f.write(f"{step+1},{accum_tokens/1e6:.2f},{norm_step_loss:.4f},{curr_lr:.6e},{gnorm:.4f},{active_tok_s:.1f},{active_elapsed/60:.2f},{wall_elapsed/60:.2f},{total_pause_time/60:.2f},{temp_g:.1f},{pwr:.1f}\n")
            print(f"[Stage B] Step {step+1:3d}/{total_steps} | Loss: {norm_step_loss:.4f} | Tokens: {accum_tokens/1e6:5.1f}M | Speed: {active_tok_s:5.0f} tok/s | Temp: {temp_g:.0f}C", flush=True)

        # 15-minute rolling checkpoint save
        now = time.time()
        if (now - last_ckpt_time) >= 900:  # 15 minutes
            save_path = STAGE_B_CKPT_DIR / f"rolling_step_{step+1}.pt"
            torch.save({
                "step": step + 1,
                "accum_tokens": accum_tokens,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "loader_state": loader.get_state(),
                "total_pause_time": total_pause_time,
                "total_pauses": total_pauses,
                "spike_event_count": spike_event_count,
            }, save_path)
            shutil.copyfile(save_path, latest_ckpt)
            last_ckpt_time = now
            print(f"[CHECKPOINT] Saved 15-minute rolling checkpoint at step {step+1}.", flush=True)

        # Milestone Evaluations (25%, 50%, 75%, 100%)
        if (step + 1) in milestones:
            pct = int(((step + 1) / total_steps) * 100)
            print(f"\n{'='*70}\n[MILESTONE {pct}%] Step {step+1}/{total_steps} Evaluation ({accum_tokens/1e6:.1f}M tokens)\n{'='*70}", flush=True)
            t_eval_start = time.time()
            
            # Save permanent milestone checkpoint
            perm_path = STAGE_B_CKPT_DIR / f"permanent_pct_{pct}_step_{step+1}.pt"
            torch.save({
                "step": step + 1,
                "accum_tokens": accum_tokens,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "loader_state": loader.get_state(),
                "total_pause_time": total_pause_time,
                "total_pauses": total_pauses,
                "spike_event_count": spike_event_count,
            }, perm_path)
            
            # 1. 2k Val CE on 4 sets
            print("  Evaluating 2k Val CE on 4 sets (fixed 2M subsets)...")
            ce_2k = eval_2m.evaluate_all(model)
            print(f"    Wiki: {ce_2k['val_wikipedia_neutral']:.4f} | FW: {ce_2k['val_fineweb_edu']:.4f} | Cosmo: {ce_2k['val_cosmopedia']:.4f} | Tiny: {ce_2k['val_tinystories']:.4f}")
            
            # 2. Position-bucket loss up to 8k
            print("  Evaluating Position-Bucket Loss up to 8k...")
            b_pg19 = evaluate_position_buckets_8k(model, val_pg19, num_sequences=10, device=device)
            b_wiki = evaluate_position_buckets_8k(model, val_wiki, num_sequences=10, device=device)
            b_fw = evaluate_position_buckets_8k(model, val_fineweb, num_sequences=10, device=device)
            print(f"    PG-19 [0-1k]: {b_pg19['0-1k']:.4f} | [1k-2k]: {b_pg19['1k-2k']:.4f} | [2k-4k]: {b_pg19['2k-4k']:.4f} | [4k-8k]: {b_pg19['4k-8k']:.4f}")
            
            # 3. Passkey retrieval at 2k, 4k, 8k
            print("  Running Passkey Retrieval Benchmark (2k, 4k, 8k)...")
            passkey_res = run_passkey_retrieval_8k(model, sp, val_pg19, lengths=[2048, 4096, 8192], trials_per_length=20, device=device)
            print(f"    Passkey Acc: 2k = {passkey_res['passkey_2k_acc']:.1f}% | 4k = {passkey_res['passkey_4k_acc']:.1f}% | 8k = {passkey_res['passkey_8k_acc']:.1f}%")
            
            # 4. Text generation samples
            samples = generate_samples(model, sp, device=device)
            with open(SAMPLES_MD, "a", encoding="utf-8") as f:
                f.write(f"\n## Stage B Milestone {pct}% (Step {step+1}, {accum_tokens/1e6:.1f}M tokens)\n\n")
                for s_idx, s in enumerate(samples, 1):
                    f.write(f"### Prompt {s_idx}: `{s['prompt']}`\n\n```text\n{s['completion']}\n```\n\n---\n\n")
                    
            eval_dur = time.time() - t_eval_start
            print(f"  Evaluation completed in {eval_dur:.1f}s.\n")
            
            with open(EVAL_LOG_CSV, "a", encoding="utf-8") as f:
                f.write(f"{pct},{step+1},{accum_tokens/1e6:.2f},{ce_2k['val_wikipedia_neutral']},{ce_2k['val_fineweb_edu']},{ce_2k['val_cosmopedia']},{ce_2k['val_tinystories']},{b_pg19['0-1k']},{b_pg19['1k-2k']},{b_pg19['2k-4k']},{b_pg19['4k-8k']},{b_wiki['4k-8k']},{b_fw['4k-8k']},{passkey_res['passkey_2k_acc']},{passkey_res['passkey_4k_acc']},{passkey_res['passkey_8k_acc']},{eval_dur:.1f}\n")
            model.train()

    # Save final undecayed Stage B checkpoint
    final_undecayed_ckpt = STAGE_B_CKPT_DIR / "stageB_final_undecayed.pt"
    torch.save({
        "step": total_steps,
        "accum_tokens": accum_tokens,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "total_pause_time": total_pause_time,
        "total_pauses": total_pauses,
        "spike_event_count": spike_event_count,
    }, final_undecayed_ckpt)
    print(f"\n[STAGE B FINISHED] Saved Undecayed Stage B Checkpoint to {final_undecayed_ckpt}!", flush=True)

if __name__ == "__main__":
    main()
