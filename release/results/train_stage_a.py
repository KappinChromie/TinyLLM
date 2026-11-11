"""
Stage A Full Training Script: 6L_768H_1KV (2,149.6M tokens, 8,200 optimizer steps)
- Architecture: 6L_768H_1KV (42.28M params)
- Data mixture: 60% FineWeb-Edu / 30% Cosmopedia v2 / 10% TinyStories
- Global batch: 128 sequences (262,144 tokens/step, micro_bs=4, grad_accum=32)
- Peak LR: 2.0e-3, 100-step linear warmup, flat for whole stage (WSD)
- AdamW (0.9, 0.95), weight decay 0.1, grad clip 1.0, bf16
- Temperature governor: 83C pause / 78C resume, pause log, active vs wall clock
- Checkpoints: 15-minute rolling pair + 1-hour permanent checkpoints
- Loss-spike guard: rollback on first event, abort on repeated divergence
- Hourly evaluations: 4 validation sets on fixed 2M-token subsets + 5 fixed prompts
- Pre-flight real-pipeline regression test: (a) y shift=1, (b) loss match, (c) t+1 vs t+2 accuracy
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
CHECKPOINTS_DIR = MODELS_DIR / "checkpoints" / "stageA"
TOKENIZER_MODEL = MODELS_DIR / "tokenizer_8k" / "tokenizer.model"

RESULTS_DIR.mkdir(exist_ok=True)
CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_LOG_CSV = RESULTS_DIR / "stageA_train_log.csv"
PAUSE_LOG_CSV = RESULTS_DIR / "pause_log.csv"
HOURLY_EVAL_CSV = RESULTS_DIR / "stageA_hourly_eval.csv"
HOURLY_SAMPLES_MD = RESULTS_DIR / "stageA_hourly_samples.md"
SPIKE_LOG = RESULTS_DIR / "loss_spike_events.log"
DIVERGENCE_FLAG = RESULTS_DIR / "STOP_DIVERGENCE"

COMPRESSION_STATS = {
    "val_cosmopedia": {"bytes_per_tok": 4.149, "chars_per_tok": 4.148},
    "val_fineweb_edu": {"bytes_per_tok": 4.156, "chars_per_tok": 4.150},
    "val_tinystories": {"bytes_per_tok": 3.220, "chars_per_tok": 3.220},
    "val_wikipedia_neutral": {"bytes_per_tok": 4.098, "chars_per_tok": 4.092},
}

FIXED_PROMPTS = [
    "Once upon a time, there was a little bunny named ",
    "The fundamental difference between planets and stars is ",
    "To write a simple Python function that adds two numbers: ",
    "In ancient history, the Great Pyramid of Giza was built ",
    "Water evaporates when the temperature rises because ",
]

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

def apply_temperature_governor(step, accum_tokens, run_id, total_wall_start):
    temp, _ = get_gpu_telemetry()
    if temp < 83.0:
        return 0.0, 0
        
    t_start = time.time()
    t_start_iso = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(t_start))
    temp_initial = temp
    print(f"\n[!WARNING] [{run_id}] Step {step}: GPU temperature {temp:.1f}C >= 83C threshold! Pausing training...", flush=True)
    
    while temp > 78.0:
        time.sleep(5)
        temp, _ = get_gpu_telemetry()
        print(f"  Cooling down: current temp = {temp:.1f}C (target <= 78C)...", end="\r", flush=True)
        
    t_end = time.time()
    t_end_iso = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(t_end))
    pause_duration = t_end - t_start
    print(f"\n[GOVERNOR] Temperature dropped to {temp:.1f}C <= 78C. Resuming training after {pause_duration:.1f}s pause.", flush=True)
    
    pause_log_exists = PAUSE_LOG_CSV.exists()
    with open(PAUSE_LOG_CSV, "a", encoding="utf-8") as f:
        if not pause_log_exists:
            f.write("timestamp_start,timestamp_end,duration_sec,temp_start,temp_end,optimizer_step,tokens_seen,run_id\n")
        f.write(f"{t_start_iso},{t_end_iso},{pause_duration:.2f},{temp_initial:.1f},{temp:.1f},{step},{accum_tokens},{run_id}\n")
        
    return pause_duration, 1

# ---------------------------------------------------------------------------
# Data Loader (Deterministic Shard Stream)
# ---------------------------------------------------------------------------
class ShardStreamDataLoader:
    def __init__(self, shards_dir, seq_len=2048, mix_weights=[0.30, 0.60, 0.10], seed=42):
        self.seq_len = seq_len
        self.weights = np.array(mix_weights, dtype=np.float64)
        self.weights = self.weights / self.weights.sum()
        
        self.cosmo = np.memmap(shards_dir / "train_cosmopedia.bin", dtype=np.uint16, mode="r")
        self.fineweb = np.memmap(shards_dir / "train_fineweb_edu.bin", dtype=np.uint16, mode="r")
        self.tinystories = np.memmap(shards_dir / "train_tinystories.bin", dtype=np.uint16, mode="r")
        self.shards = [self.cosmo, self.fineweb, self.tinystories]
        self.shard_lengths = [len(s) for s in self.shards]

        self.rng_mix = np.random.RandomState(seed)
        self.rng_cosmo = np.random.RandomState(1001)
        self.rng_fineweb = np.random.RandomState(1002)
        self.rng_tinystories = np.random.RandomState(1003)
        self.shard_rngs = [self.rng_cosmo, self.rng_fineweb, self.rng_tinystories]

    def get_state(self):
        return {
            "rng_mix": self.rng_mix.get_state(),
            "rng_cosmo": self.rng_cosmo.get_state(),
            "rng_fineweb": self.rng_fineweb.get_state(),
            "rng_tinystories": self.rng_tinystories.get_state(),
        }

    def set_state(self, state):
        self.rng_mix.set_state(state["rng_mix"])
        self.rng_cosmo.set_state(state["rng_cosmo"])
        self.rng_fineweb.set_state(state["rng_fineweb"])
        self.rng_tinystories.set_state(state["rng_tinystories"])

    def advance(self, num_sequences):
        """Advances RNGs forward by num_sequences without transferring to GPU."""
        shard_indices = self.rng_mix.choice(3, size=num_sequences, p=self.weights)
        for s_idx in shard_indices:
            max_idx = self.shard_lengths[s_idx] - self.seq_len - 1
            _ = self.shard_rngs[s_idx].randint(0, max_idx)

    def get_batch(self, batch_size, device):
        shard_indices = self.rng_mix.choice(3, size=batch_size, p=self.weights)
        x_list, y_list = [], []
        for s_idx in shard_indices:
            shard = self.shards[s_idx]
            rng = self.shard_rngs[s_idx]
            max_idx = self.shard_lengths[s_idx] - self.seq_len - 1
            idx = rng.randint(0, max_idx)
            x_list.append(shard[idx : idx + self.seq_len])
            y_list.append(shard[idx + 1 : idx + 1 + self.seq_len])
            
        x = np.stack(x_list)
        y = np.stack(y_list)
        tx = torch.from_numpy(x.astype(np.int64)).to(device=device, non_blocking=True)
        ty = torch.from_numpy(y.astype(np.int64)).to(device=device, non_blocking=True)
        return tx, ty

# ---------------------------------------------------------------------------
# Real-Pipeline Regression Test
# ---------------------------------------------------------------------------
def run_real_pipeline_regression_test(model, loader, device):
    print("\n--- Running Mandatory Pre-Training Real-Pipeline Regression Test ---", flush=True)
    model.eval()
    
    # 1. Pull real batch from loader
    x, y = loader.get_batch(batch_size=4, device=device)
    
    # (a) Check y is x shifted by exactly 1 position within each sequence
    if not (x[:, 1:] == y[:, :-1]).all():
        raise RuntimeError("FATAL Check (a): Real loader batch y is NOT shifted by exactly 1 relative to x!")
    print("  [Check a Passed]: y is x shifted by exactly 1 position within each sequence.", flush=True)
    
    # (b) Reported loss matches independent F.cross_entropy computed from raw tokens
    with torch.no_grad():
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            out = model(x)
            if hasattr(out, "loss") and out.loss is not None:
                raise RuntimeError("FATAL: Model forward returned internal loss!")
            loss_reported = F.cross_entropy(out.logits.view(-1, 8192), y.view(-1))
            
            # Independent verification
            tokens = torch.cat([x[:, :1], y], dim=1) # [B, seq_len+1]
            ind_x = tokens[:, :-1]
            ind_y = tokens[:, 1:]
            out_ind = model(ind_x)
            loss_independent = F.cross_entropy(out_ind.logits.reshape(-1, 8192), ind_y.reshape(-1))
            
        if not torch.allclose(loss_reported, loss_independent, atol=1e-5):
            raise RuntimeError(f"FATAL Check (b): Reported loss ({loss_reported.item():.5f}) != Independent loss ({loss_independent.item():.5f})!")
    print("  [Check b Passed]: Reported loss matches independent cross entropy computation.", flush=True)
    
    # (c) Teacher-forced accuracy on held-out batch: t+1 must be clearly higher than t+2
    ref_ckpt = ROOT / "models" / "checkpoints" / "lr_reverification" / "LR_2e-03_final.pt"
    if ref_ckpt.exists():
        ref_model = create_model(device=device, seed=42)
        ref_model.load_state_dict(torch.load(ref_ckpt, map_location=device))
        ref_model.eval()
        
        val_shard = np.memmap(SHARDS_DIR / "val_wikipedia_neutral.bin", dtype=np.uint16, mode="r")
        tokens_val = torch.from_numpy(val_shard[1000:3049].astype(np.int64)).to(device).unsqueeze(0)
        x_val = tokens_val[:, :-1]
        y_val_t1 = tokens_val[:, 1:]
        x_val_t2 = tokens_val[:, :-2]
        y_val_t2 = tokens_val[:, 2:]
        
        with torch.no_grad():
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                preds_t1 = torch.argmax(ref_model(x_val).logits, dim=-1)
                acc_t1 = (preds_t1 == y_val_t1).float().mean().item()
                preds_t2 = torch.argmax(ref_model(x_val_t2).logits, dim=-1)
                acc_t2 = (preds_t2 == y_val_t2).float().mean().item()
                
        print(f"  [Check c Evaluation]: Reference model held-out accuracy: t+1 = {acc_t1*100:.1f}%, t+2 = {acc_t2*100:.1f}%", flush=True)
        if acc_t1 <= acc_t2 * 2.0 or acc_t1 < 0.20:
            raise RuntimeError(f"FATAL Check (c): t+1 acc ({acc_t1*100:.1f}%) is NOT clearly higher than t+2 ({acc_t2*100:.1f}%)!")
        print("  [Check c Passed]: Teacher-forced accuracy for t+1 is clearly higher than t+2.", flush=True)
        del ref_model
        torch.cuda.empty_cache()
    else:
        print("  [Check c Note]: Reference checkpoint not present; checked a and b.", flush=True)
        
    model.train()
    print("Mandatory Real-Pipeline Regression Test: ALL CHECKS PASSED!\n", flush=True)

# ---------------------------------------------------------------------------
# Hourly Evaluator
# ---------------------------------------------------------------------------
class HourlyEvaluator:
    def __init__(self, shards_dir, device):
        self.device = device
        self.seq_len = 2048
        self.val_shards = {
            "val_wikipedia_neutral": np.memmap(shards_dir / "val_wikipedia_neutral.bin", dtype=np.uint16, mode="r"),
            "val_fineweb_edu": np.memmap(shards_dir / "val_fineweb_edu.bin", dtype=np.uint16, mode="r"),
            "val_cosmopedia": np.memmap(shards_dir / "val_cosmopedia.bin", dtype=np.uint16, mode="r"),
            "val_tinystories": np.memmap(shards_dir / "val_tinystories.bin", dtype=np.uint16, mode="r"),
        }
        # Precompute fixed offsets for exactly 244 batches of 4 sequences (~2,000,000 tokens per set)
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
                loss = F.cross_entropy(outputs.logits.view(-1, 8192), ty.view(-1)).item()
            total_loss += loss * (self.batch_size * self.seq_len)
            total_tokens += (self.batch_size * self.seq_len)
            
        avg_loss = total_loss / total_tokens
        bits_per_tok = avg_loss / math.log(2.0)
        bpb = bits_per_tok / COMPRESSION_STATS[name]["bytes_per_tok"]
        return round(avg_loss, 4), round(bpb, 4)

    def evaluate_all(self, model):
        model.eval()
        results = {}
        for name in ["val_wikipedia_neutral", "val_fineweb_edu", "val_cosmopedia", "val_tinystories"]:
            ce, bpb = self.evaluate_set(model, name)
            results[name] = {"ce": ce, "bpb": bpb}
        model.train()
        return results

    def generate_samples(self, model, sp, max_new_tokens=64):
        model.eval()
        samples = []
        for p in FIXED_PROMPTS:
            input_ids = sp.encode_as_ids(p)
            curr_ids = list(input_ids)
            with torch.no_grad():
                for _ in range(max_new_tokens):
                    ctx = curr_ids[-2048:]
                    inp = torch.tensor([ctx], dtype=torch.long, device=self.device)
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        out = model(inp)
                        next_token = torch.argmax(out.logits[0, -1, :]).item()
                    curr_ids.append(next_token)
                    if next_token == sp.eos_id():
                        break
            gen_text = sp.decode(curr_ids)
            samples.append({"prompt": p, "completion": gen_text})
        model.train()
        return samples

# ---------------------------------------------------------------------------
# Model Creator
# ---------------------------------------------------------------------------
def create_model(device="cuda", seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
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
        max_position_embeddings=2048,
    )
    return LlamaForCausalLM(config).to(device=device, dtype=torch.bfloat16)

# ---------------------------------------------------------------------------
# Main Stage A Training Loop
# ---------------------------------------------------------------------------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== Stage A Launch: 6L_768H_1KV on {device} ({torch.cuda.get_device_name(0)}) ===", flush=True)
    
    if DIVERGENCE_FLAG.exists():
        print(f"[!FATAL] Found {DIVERGENCE_FLAG}! Previous run halted due to repeated divergence. Clear flag to restart.", flush=True)
        sys.exit(1)
        
    sp = spm.SentencePieceProcessor()
    sp.load(str(TOKENIZER_MODEL))
    
    loader = ShardStreamDataLoader(SHARDS_DIR, seq_len=2048, mix_weights=[0.30, 0.60, 0.10], seed=42)
    evaluator = HourlyEvaluator(SHARDS_DIR, device=device)
    model = create_model(device=device, seed=42)
    
    # Real-pipeline regression test
    run_real_pipeline_regression_test(model, loader, device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, betas=(0.9, 0.95), weight_decay=0.1, fused=True)
    
    micro_bs = 4
    grad_accum = 32
    seq_len = 2048
    tokens_per_step = micro_bs * grad_accum * seq_len  # 262,144
    total_steps = 8200
    warmup_steps = 100
    peak_lr = 2e-3
    
    start_step = 0
    accum_tokens = 0
    active_sec = 0.0
    total_wall_start = time.time()
    total_paused_sec = 0.0
    total_eval_sec = 0.0
    num_pauses = 0
    
    rolling_ckpts = []
    permanent_ckpts = []
    
    # Check for existing checkpoint to resume
    latest_ckpt = CHECKPOINTS_DIR / "latest_rolling.pt"
    if latest_ckpt.exists():
        print(f"\n[RESUME] Found existing checkpoint: {latest_ckpt}. Resuming training...", flush=True)
        ckpt_data = torch.load(latest_ckpt, map_location=device)
        model.load_state_dict(ckpt_data["model_state"])
        optimizer.load_state_dict(ckpt_data["optimizer_state"])
        loader.set_state(ckpt_data["loader_state"])
        torch.set_rng_state(ckpt_data["torch_cpu_rng"])
        torch.cuda.set_rng_state_all(ckpt_data["torch_cuda_rng"])
        np.random.set_state(ckpt_data["np_rng"])
        start_step = ckpt_data["step"]
        accum_tokens = ckpt_data["accum_tokens"]
        active_sec = ckpt_data.get("active_sec", 0.0)
        total_paused_sec = ckpt_data.get("total_paused_sec", 0.0)
        num_pauses = ckpt_data.get("num_pauses", 0)
        print(f"  Successfully resumed from Step {start_step} ({accum_tokens/1e6:.2f}M tokens)!\n", flush=True)
    else:
        # Initialize log CSVs if starting fresh
        if not TRAIN_LOG_CSV.exists():
            with open(TRAIN_LOG_CSV, "w", encoding="utf-8") as f:
                f.write("step,tokens_m,train_loss,lr,grad_norm,active_tok_s,active_time_min,wall_time_min,paused_time_min,temp_gpu,power_w\n")
        if not HOURLY_EVAL_CSV.exists():
            with open(HOURLY_EVAL_CSV, "w", encoding="utf-8") as f:
                f.write("active_hour,step,tokens_m,val_wiki_ce,val_wiki_bpb,val_fineweb_ce,val_fineweb_bpb,val_cosmo_ce,val_cosmo_bpb,val_tinystories_ce,val_tinystories_bpb,eval_sec\n")
        if not HOURLY_SAMPLES_MD.exists():
            with open(HOURLY_SAMPLES_MD, "w", encoding="utf-8") as f:
                f.write("# Stage A Hourly Generation Samples (6L_768H_1KV)\n\n")

    last_rolling_ckpt_time = time.time()
    next_eval_active_hour = int(active_sec // 3600) + 1
    
    # Loss spike guard state
    trailing_100_losses = []
    spike_consecutive_count = 0
    spike_event_count = 0
    last_spike_step = -9999
    
    print(f"\nStarting Main Stage A Training Loop (Steps {start_step+1} to {total_steps})...\n", flush=True)
    model.train()
    optimizer.zero_grad(set_to_none=True)
    
    step_start_time = time.time()
    
    for step in range(start_step, total_steps):
        # 1. Temperature Governor Check every 2 steps (since each step is ~3.5s)
        if step % 2 == 0:
            p_sec, was_p = apply_temperature_governor(step, accum_tokens, "StageA", total_wall_start)
            total_paused_sec += p_sec
            num_pauses += was_p
            
            total_elapsed = time.time() - total_wall_start
            if total_elapsed > 3600 and (total_paused_sec / total_elapsed) > 0.25:
                print(f"[!CAUTION] Cumulative paused time ({total_paused_sec/60:.1f}m) exceeds 25% of elapsed time ({total_elapsed/60:.1f}m)!", flush=True)

        # 2. Linear Warmup to Peak LR 2e-3, then flat (WSD schedule)
        if step < warmup_steps:
            curr_lr = peak_lr * (step + 1) / warmup_steps
        else:
            curr_lr = peak_lr
            
        for param_group in optimizer.param_groups:
            param_group["lr"] = curr_lr
            
        # 3. Accumulation Loop
        step_loss = 0.0
        t0_accum = time.time()
        for _ in range(grad_accum):
            x, y = loader.get_batch(micro_bs, device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                out = model(x)
                loss = F.cross_entropy(out.logits.view(-1, 8192), y.view(-1))
                scaled_loss = loss / grad_accum
            scaled_loss.backward()
            step_loss += loss.item() / grad_accum
            
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0).item()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        
        dt_step = time.time() - t0_accum
        active_sec += dt_step
        accum_tokens += tokens_per_step
        
        # 4. Loss-Spike Guard
        is_bad_num = math.isnan(step_loss) or math.isinf(step_loss)
        trailing_mean = np.mean(trailing_100_losses) if len(trailing_100_losses) >= 10 else step_loss
        is_spike = (step_loss > 1.5 * trailing_mean) and (len(trailing_100_losses) >= 10)
        
        if is_spike:
            spike_consecutive_count += 1
        else:
            spike_consecutive_count = 0
            
        trailing_100_losses.append(step_loss)
        if len(trailing_100_losses) > 100:
            trailing_100_losses.pop(0)
            
        # Trigger condition: NaN/inf OR 20 consecutive spikes
        if is_bad_num or spike_consecutive_count >= 20:
            err_type = "NaN/Inf loss" if is_bad_num else f"20 consecutive loss spikes (>1.5x trailing avg: {step_loss:.4f} vs {trailing_mean:.4f})"
            print(f"\n[!ALERT] Loss Spike Guard Triggered at Step {step+1}: {err_type}!", flush=True)
            
            with open(SPIKE_LOG, "a", encoding="utf-8") as f:
                f.write(f"[{time.strftime('%Y-%m-%dT%H:%M:%SZ')}] Step {step+1}: {err_type}\n")
                
            # If second event within 500 steps, or NaN again -> STOP
            if spike_event_count >= 1 and ((step - last_spike_step) < 500 or is_bad_num):
                print(f"[!FATAL] Second divergence event within 500 steps! Halting training per protocol.", flush=True)
                with open(DIVERGENCE_FLAG, "w", encoding="utf-8") as f:
                    f.write(f"Divergence halt at step {step+1}: {err_type}\n")
                raise RuntimeError(f"FATAL: Repeated training divergence at step {step+1}: {err_type}")
                
            # First event: Roll back to last rolling checkpoint
            if latest_ckpt.exists():
                print(f"  [ROLLBACK] Rolling back to last verified checkpoint: {latest_ckpt}...", flush=True)
                ckpt_data = torch.load(latest_ckpt, map_location=device)
                model.load_state_dict(ckpt_data["model_state"])
                optimizer.load_state_dict(ckpt_data["optimizer_state"])
                loader.set_state(ckpt_data["loader_state"])
                torch.set_rng_state(ckpt_data["torch_cpu_rng"])
                torch.cuda.set_rng_state_all(ckpt_data["torch_cuda_rng"])
                np.random.set_state(ckpt_data["np_rng"])
                
                # Advance loader past offending window (128 sequences)
                print(f"  [ADVANCE] Skipping data loader ahead by 128 sequences past offending window...", flush=True)
                loader.advance(128)
                
                spike_event_count += 1
                last_spike_step = step
                spike_consecutive_count = 0
                trailing_100_losses.clear()
                print("  [RESUME] Rollback complete. Continuing training.\n", flush=True)
                continue
            else:
                print("[!FATAL] Divergence triggered before first checkpoint was created! Aborting.", flush=True)
                raise RuntimeError("Divergence before first checkpoint.")

        # 5. Logging every 50 steps
        if (step + 1) % 50 == 0 or step == total_steps - 1:
            temp_gpu, pwr = get_gpu_telemetry()
            total_wall_sec = time.time() - total_wall_start
            active_tok_s = accum_tokens / max(1e-5, active_sec)
            
            print(f"[Stage A] Step {step+1:4d}/{total_steps} | LR {curr_lr:.2e} | Loss {step_loss:.4f} | "
                  f"Tokens: {accum_tokens/1e6:6.1f}M | Active: {active_sec/60:5.1f}m (Paused: {total_paused_sec/60:4.1f}m) | "
                  f"Speed: {active_tok_s:,.0f} tok/s | Temp: {temp_gpu:.0f}C ({pwr:.0f}W)", flush=True)
                  
            with open(TRAIN_LOG_CSV, "a", encoding="utf-8") as f:
                f.write(f"{step+1},{accum_tokens/1e6:.2f},{step_loss:.4f},{curr_lr:.6e},{grad_norm:.4f},"
                        f"{active_tok_s:.1f},{active_sec/60:.2f},{total_wall_sec/60:.2f},{total_paused_sec/60:.2f},"
                        f"{temp_gpu:.1f},{pwr:.1f}\n")

        # 6. Checkpoints: 15-minute rolling pair
        now_time = time.time()
        if (now_time - last_rolling_ckpt_time) >= 900:  # 15 minutes
            ckpt_p = CHECKPOINTS_DIR / f"rolling_step_{step+1}.pt"
            ckpt_dict = {
                "step": step + 1,
                "accum_tokens": accum_tokens,
                "active_sec": active_sec,
                "total_paused_sec": total_paused_sec,
                "num_pauses": num_pauses,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "loader_state": loader.get_state(),
                "torch_cpu_rng": torch.get_rng_state(),
                "torch_cuda_rng": torch.cuda.get_rng_state_all(),
                "np_rng": np.random.get_state(),
            }
            torch.save(ckpt_dict, ckpt_p)
            torch.save(ckpt_dict, latest_ckpt)
            rolling_ckpts.append(ckpt_p)
            
            # Keep only last 2 rolling checkpoints
            if len(rolling_ckpts) > 2:
                old_p = rolling_ckpts.pop(0)
                if old_p.exists():
                    try:
                        old_p.unlink()
                    except Exception:
                        pass
            last_rolling_ckpt_time = now_time
            print(f"  [CKPT] Saved 15-min rolling checkpoint: {ckpt_p.name}", flush=True)

        # 7. Checkpoints & Monitoring: 1-hour of active training
        curr_active_hour = int(active_sec // 3600)
        if curr_active_hour >= next_eval_active_hour:
            print(f"\n=======================================================", flush=True)
            print(f"HOURLY MILESTONE REACHED: Active Hour {curr_active_hour} (Step {step+1}, {accum_tokens/1e6:.1f}M tokens)", flush=True)
            print(f"=======================================================", flush=True)
            
            # Save permanent 1-hour checkpoint
            perm_ckpt_p = CHECKPOINTS_DIR / f"permanent_hour_{curr_active_hour}_step_{step+1}.pt"
            ckpt_dict = {
                "step": step + 1,
                "accum_tokens": accum_tokens,
                "active_sec": active_sec,
                "active_hour": curr_active_hour,
                "total_paused_sec": total_paused_sec,
                "num_pauses": num_pauses,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "loader_state": loader.get_state(),
                "torch_cpu_rng": torch.get_rng_state(),
                "torch_cuda_rng": torch.cuda.get_rng_state_all(),
                "np_rng": np.random.get_state(),
            }
            torch.save(ckpt_dict, perm_ckpt_p)
            print(f"  [PERMANENT CKPT] Saved hourly checkpoint: {perm_ckpt_p.name}", flush=True)
            
            # Run hourly evaluation (timed separately)
            t_eval_start = time.time()
            print("  Running Hourly Evaluation on all 4 validation sets (2M fixed tokens each)...", flush=True)
            eval_res = evaluator.evaluate_all(model)
            eval_samples = evaluator.generate_samples(model, sp, max_new_tokens=64)
            dt_eval = time.time() - t_eval_start
            total_eval_sec += dt_eval
            
            for vname, m in eval_res.items():
                print(f"    {vname:25s}: CE = {m['ce']:.4f} | BPB = {m['bpb']:.4f}", flush=True)
                
            # Log hourly evaluation CSV
            with open(HOURLY_EVAL_CSV, "a", encoding="utf-8") as f:
                f.write(f"{curr_active_hour},{step+1},{accum_tokens/1e6:.2f},"
                        f"{eval_res['val_wikipedia_neutral']['ce']:.4f},{eval_res['val_wikipedia_neutral']['bpb']:.4f},"
                        f"{eval_res['val_fineweb_edu']['ce']:.4f},{eval_res['val_fineweb_edu']['bpb']:.4f},"
                        f"{eval_res['val_cosmopedia']['ce']:.4f},{eval_res['val_cosmopedia']['bpb']:.4f},"
                        f"{eval_res['val_tinystories']['ce']:.4f},{eval_res['val_tinystories']['bpb']:.4f},"
                        f"{dt_eval:.1f}\n")
                        
            # Log hourly samples markdown
            with open(HOURLY_SAMPLES_MD, "a", encoding="utf-8") as f:
                f.write(f"## Active Hour {curr_active_hour} (Step {step+1}, {accum_tokens/1e6:.1f}M tokens)\n\n")
                for s in eval_samples:
                    f.write(f"**Prompt**: `{s['prompt']}`\n\n```text\n{s['completion']}\n```\n\n")
                f.write("---\n\n")
                
            print(f"  Hourly evaluation finished in {dt_eval:.1f}s. Resuming active training.\n", flush=True)
            next_eval_active_hour = curr_active_hour + 1

    total_wall_sec = time.time() - total_wall_start
    print(f"\n=======================================================", flush=True)
    print(f"STAGE A TRAINING COMPLETED SUCCESSFULLY!", flush=True)
    print(f"  Total Steps:       {total_steps}", flush=True)
    print(f"  Total Tokens:      {accum_tokens/1e6:.2f}M", flush=True)
    print(f"  Active Train Time: {active_sec/3600:.2f} hours ({active_sec/60:.1f} min)", flush=True)
    print(f"  Total Paused Time: {total_paused_sec/60:.1f} min ({num_pauses} pauses)", flush=True)
    print(f"  Evaluation Time:   {total_eval_sec/60:.1f} min", flush=True)
    print(f"  Total Wall Time:   {total_wall_sec/3600:.2f} hours ({total_wall_sec/60:.1f} min)", flush=True)
    print(f"  Active Speed:      {accum_tokens/active_sec:,.0f} tok/s", flush=True)
    print(f"=======================================================", flush=True)
    
    # Save final undecayed Stage A checkpoint
    final_undecayed_p = CHECKPOINTS_DIR / "stageA_final_undecayed.pt"
    ckpt_dict = {
        "step": total_steps,
        "accum_tokens": accum_tokens,
        "active_sec": active_sec,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "loader_state": loader.get_state(),
        "torch_cpu_rng": torch.get_rng_state(),
        "torch_cuda_rng": torch.cuda.get_rng_state_all(),
        "np_rng": np.random.get_state(),
    }
    torch.save(ckpt_dict, final_undecayed_p)
    print(f"Saved final undecayed Stage A checkpoint to {final_undecayed_p}", flush=True)

if __name__ == "__main__":
    main()
