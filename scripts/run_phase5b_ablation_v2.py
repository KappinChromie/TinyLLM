"""
Phase 5B Data Mixture Ablation Script (v2 with Temperature Governor & Resumable Checkpoints)
Trains 6L_768H_1KV across candidate Stage A mixtures for 250M tokens (~1 hour each):
- Run A: 100% FineWeb-Edu (executed with micro_bs=8)
- Run B: 100% Cosmopedia v2
- Run C: Default Blend (45% Cosmopedia, 45% FineWeb-Edu, 10% TinyStories)
- Run D: Simpler-Text Blend (40% TinyStories, 40% Cosmopedia, 20% FineWeb-Edu)

Enhancements:
1. Configurable micro-batch size (--micro-bs, default 4) with matching grad_accum to keep global batch = 32 seqs (65,536 tokens).
2. Temperature Governor: polls GPU core temp between optimizer steps. If temp >= 80C, pauses until temp <= 75C.
   Logs all pauses to results/pause_log.csv. Tracks active training time vs wall-clock time.
3. Resumable Checkpoints: saves full-state checkpoint (model, optimizer, scheduler, RNGs, loader state) every 15 minutes,
   keeping the last 2. Supports resuming seamlessly via --resume-from <ckpt_path>.
4. Preserves 100% apples-to-apples ablation design, fixed seed 42, deterministic data order, and WSD schedule.
"""

import os
import sys
import math
import time
import argparse
import shutil
import subprocess
import threading
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
CHECKPOINTS_DIR = MODELS_DIR / "checkpoints" / "phase5b"
TOKENIZER_MODEL = MODELS_DIR / "tokenizer_8k" / "tokenizer.model"

RESULTS_DIR.mkdir(exist_ok=True)
CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)

TELEMETRY_CSV = RESULTS_DIR / "phase5b_telemetry.csv"
PAUSE_LOG_CSV = RESULTS_DIR / "pause_log.csv"
CURVES_CSV = RESULTS_DIR / "phase5b_curves.csv"
SUMMARY_CSV = RESULTS_DIR / "phase5b_summary.csv"
SAMPLES_MD = RESULTS_DIR / "phase5b_sample_generations.md"
REPORT_MD = RESULTS_DIR / "phase5b_report.md"

NOISE_FLOOR_VAL = 0.0076   # 2-seed validation spread
NOISE_FLOOR_TRAIN = 0.0254 # 3-seed training spread

# Tokenizer compression stats
COMPRESSION_STATS = {
    "val_cosmopedia": {"bytes_per_tok": 4.149, "chars_per_tok": 4.148},
    "val_fineweb_edu": {"bytes_per_tok": 4.156, "chars_per_tok": 4.150},
    "val_tinystories": {"bytes_per_tok": 3.220, "chars_per_tok": 3.220},
    "val_wikipedia_neutral": {"bytes_per_tok": 4.098, "chars_per_tok": 4.092},
}

# 15 Evaluation prompts: 10 standard + 5 basic grammar prompts
PROMPTS = [
    # 10 Standard Prompts
    "Once upon a time, there was a little bunny named ",
    "The fundamental difference between planets and stars is ",
    "To write a simple Python function that adds two numbers: ",
    "In ancient history, the Great Pyramid of Giza was built ",
    "Lily wanted to bake a cake for her mom. First, she ",
    "Water evaporates when the temperature rises because ",
    "A computer network is a system of connected computers that ",
    "The dog ran across the park to catch ",
    "Photosynthesis is the biological process by which plants ",
    "When solving a math problem, it is important to ",
    # 5 Basic Grammar Prompts
    "She opened the door and saw ",
    "The dog ran after the ball because ",
    "He went to the store to buy ",
    "They were very happy when they ",
    "Every morning, the boy eats ",
]

# ---------------------------------------------------------------------------
# Temperature Polling & GPU Utilities
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

def get_gpu_temperature():
    try:
        cmd = [SMI_BIN, "--query-gpu=temperature.gpu", "--format=csv,noheader,nounits"]
        res = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return float(res.stdout.strip())
    except Exception:
        return 0.0

def get_nvidia_smi_memory_mb():
    try:
        cmd = [SMI_BIN, "--query-gpu=memory.used", "--format=csv,noheader,nounits"]
        res = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return float(res.stdout.strip())
    except Exception:
        return 0.0

class GPUTelemetry(threading.Thread):
    def __init__(self, log_path, interval_sec=60):
        super().__init__(daemon=True)
        self.log_path = log_path
        self.interval_sec = interval_sec
        self.stop_event = threading.Event()
        self.current_run = "idle"
        
        if not self.log_path.exists():
            with open(self.log_path, "w", encoding="utf-8") as f:
                f.write("timestamp,run_name,temp_gpu,power_w,clock_sm_mhz,clock_mem_mhz,mem_used_mb\n")

    def set_run_name(self, name):
        self.current_run = name

    def run(self):
        while not self.stop_event.is_set():
            try:
                cmd = [
                    SMI_BIN,
                    "--query-gpu=timestamp,temperature.gpu,power.draw,clocks.current.graphics,clocks.current.memory,memory.used",
                    "--format=csv,noheader,nounits"
                ]
                res = subprocess.run(cmd, capture_output=True, text=True, check=True)
                line = res.stdout.strip().replace(", ", ",")
                parts = line.split(",", 1)
                full_line = f"{parts[0]},{self.current_run},{parts[1]}"
                with open(self.log_path, "a", encoding="utf-8") as f:
                    f.write(full_line + "\n")
            except Exception:
                pass
            self.stop_event.wait(self.interval_sec)

    def stop(self):
        self.stop_event.set()

# ---------------------------------------------------------------------------
# Data Loading & Shard Streaming (with Resumable State)
# ---------------------------------------------------------------------------
class ShardStreamDataLoader:
    def __init__(self, shards_dir, seq_len, mix_weights):
        self.seq_len = seq_len
        self.weights = np.array(mix_weights, dtype=np.float64)
        self.weights = self.weights / self.weights.sum()
        
        self.cosmo = np.memmap(shards_dir / "train_cosmopedia.bin", dtype=np.uint16, mode="r")
        self.fineweb = np.memmap(shards_dir / "train_fineweb_edu.bin", dtype=np.uint16, mode="r")
        self.tinystories = np.memmap(shards_dir / "train_tinystories.bin", dtype=np.uint16, mode="r")
        self.shards = [self.cosmo, self.fineweb, self.tinystories]
        self.shard_lengths = [len(s) for s in self.shards]

        self.rng_mix = np.random.RandomState(42)
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
# Validation Evaluator
# ---------------------------------------------------------------------------
class ValidationEvaluator:
    def __init__(self, shards_dir, seq_len, device):
        self.seq_len = seq_len
        self.device = device
        self.val_shards = {
            "val_wikipedia_neutral": np.memmap(shards_dir / "val_wikipedia_neutral.bin", dtype=np.uint16, mode="r"),
            "val_cosmopedia": np.memmap(shards_dir / "val_cosmopedia.bin", dtype=np.uint16, mode="r"),
            "val_fineweb_edu": np.memmap(shards_dir / "val_fineweb_edu.bin", dtype=np.uint16, mode="r"),
            "val_tinystories": np.memmap(shards_dir / "val_tinystories.bin", dtype=np.uint16, mode="r"),
        }
        
    @torch.no_grad()
    def evaluate_set(self, model, name, num_batches=40, batch_size=4):
        shard = self.val_shards[name]
        max_idx = len(shard) - self.seq_len - 1
        rng = np.random.RandomState(4242)
        total_loss = 0.0
        total_tokens = 0
        
        for _ in range(num_batches):
            offsets = [rng.randint(0, max_idx) for _ in range(batch_size)]
            x = np.stack([shard[i : i + self.seq_len] for i in offsets])
            y = np.stack([shard[i + 1 : i + 1 + self.seq_len] for i in offsets])
            tx = torch.from_numpy(x.astype(np.int64)).to(device=self.device, non_blocking=True)
            ty = torch.from_numpy(y.astype(np.int64)).to(device=self.device, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                outputs = model(tx)
                loss = F.cross_entropy(outputs.logits.view(-1, 8192), ty.view(-1)).item()
            total_loss += loss * (batch_size * self.seq_len)
            total_tokens += batch_size * self.seq_len
            
        avg_loss = total_loss / total_tokens
        bits_per_tok = avg_loss / math.log(2.0)
        bpb = bits_per_tok / COMPRESSION_STATS[name]["bytes_per_tok"]
        bpc = bits_per_tok / COMPRESSION_STATS[name]["chars_per_tok"]
        return {
            "ce_loss": round(avg_loss, 4),
            "bits_per_tok": round(bits_per_tok, 4),
            "bpb": round(bpb, 4),
            "bpc": round(bpc, 4),
            "eval_tokens": total_tokens,
        }

    def evaluate_all(self, model, num_batches=40):
        model.eval()
        results = {}
        for name in ["val_wikipedia_neutral", "val_cosmopedia", "val_fineweb_edu", "val_tinystories"]:
            results[name] = self.evaluate_set(model, name, num_batches=num_batches)
        model.train()
        return results

# ---------------------------------------------------------------------------
# Qualitative Generation
# ---------------------------------------------------------------------------
def generate_samples(model, sp, prompts, device, max_new_tokens=64):
    model.eval()
    samples = []
    for p in prompts:
        input_ids = sp.encode_as_ids(p)
        curr_ids = list(input_ids)
        with torch.no_grad():
            for _ in range(max_new_tokens):
                ctx = curr_ids[-2048:]
                inp = torch.tensor([ctx], dtype=torch.long, device=device)
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
# Model Construction
# ---------------------------------------------------------------------------
def create_model(device, seed=42):
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
# Temperature Governor Function
# ---------------------------------------------------------------------------
def apply_temperature_governor(step, accum_tokens, run_id, total_wall_start):
    temp = get_gpu_temperature()
    if temp < 80.0:
        return 0.0, 0
        
    # Trigger cooling pause
    t_start = time.time()
    t_start_iso = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(t_start))
    temp_initial = temp
    print(f"\n[!WARNING] [{run_id}] Step {step}: GPU core temperature {temp:.1f}C >= 80C threshold! Pausing training...", flush=True)
    
    while temp > 75.0:
        time.sleep(5)
        temp = get_gpu_temperature()
        print(f"  Cooling down: current temp = {temp:.1f}C (target <= 75C)...", end="\r", flush=True)
        
    t_end = time.time()
    t_end_iso = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(t_end))
    pause_duration = t_end - t_start
    print(f"\n[GOVERNOR] Temperature dropped to {temp:.1f}C <= 75C. Resuming training after {pause_duration:.1f}s pause.", flush=True)
    
    # Log to pause_log.csv
    pause_log_exists = PAUSE_LOG_CSV.exists()
    with open(PAUSE_LOG_CSV, "a", encoding="utf-8") as f:
        if not pause_log_exists:
            f.write("timestamp_start,timestamp_end,duration_sec,temp_start,temp_end,optimizer_step,tokens_seen,run_id\n")
        f.write(f"{t_start_iso},{t_end_iso},{pause_duration:.2f},{temp_initial:.1f},{temp:.1f},{step},{accum_tokens},{run_id}\n")
        
    return pause_duration, 1

# ---------------------------------------------------------------------------
# Training Loop for Single Run (with Resume & Governor)
# ---------------------------------------------------------------------------
def run_mixture_training(run_id, run_name, mix_weights, total_steps, warmup_steps, decay_steps,
                         peak_lr, min_lr, micro_bs, grad_accum, seq_len, device,
                         evaluator, sp, telemetry, resume_ckpt_path=None, is_test=False):
    global_batch_seqs = micro_bs * grad_accum
    tokens_per_step = global_batch_seqs * seq_len
    total_tokens_target = total_steps * tokens_per_step
    
    print(f"\n=======================================================", flush=True)
    print(f"STARTING {run_id}: {run_name}", flush=True)
    print(f"Mix Proportions [Cosmo, FineWeb, Tiny]: {mix_weights}", flush=True)
    print(f"Batch Config: micro_bs={micro_bs}, grad_accum={grad_accum} -> global_bs={global_batch_seqs} seqs ({tokens_per_step:,} tok/step)", flush=True)
    print(f"Total Steps: {total_steps} ({total_tokens_target / 1e6:.1f}M tokens)", flush=True)
    print(f"=======================================================", flush=True)
    
    telemetry.set_run_name(run_id)
    loader = ShardStreamDataLoader(SHARDS_DIR, seq_len, mix_weights)
    model = create_model(device, seed=42)
    optimizer = torch.optim.AdamW(model.parameters(), lr=peak_lr, betas=(0.9, 0.95), weight_decay=0.1, fused=True)
    
    start_step = 0
    accum_tokens = 0
    total_paused_sec = 0.0
    num_pauses = 0
    rolling_ckpts = []
    
    # Handle Resume
    if resume_ckpt_path and Path(resume_ckpt_path).exists():
        print(f"Loading resume checkpoint from: {resume_ckpt_path}", flush=True)
        ckpt = torch.load(resume_ckpt_path, weights_only=False)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        loader.set_state(ckpt["loader_state"])
        torch.set_rng_state(ckpt["torch_cpu_rng"])
        torch.cuda.set_rng_state_all(ckpt["torch_cuda_rng"])
        np.random.set_state(ckpt["np_rng"])
        start_step = ckpt["step"]
        accum_tokens = ckpt.get("accum_tokens", start_step * tokens_per_step)
        total_paused_sec = ckpt.get("total_paused_sec", 0.0)
        num_pauses = ckpt.get("num_pauses", 0)
        print(f"Resumed from Step {start_step} ({accum_tokens/1e6:.1f}M tokens)!", flush=True)

    stable_steps = total_steps - decay_steps
    wall_start_time = time.time()
    last_checkpoint_time = time.time()
    
    curve_csv_exists = CURVES_CSV.exists()
    if not curve_csv_exists:
        with open(CURVES_CSV, "w", encoding="utf-8") as f:
            f.write("run_id,run_name,step,tokens_m,wall_sec,active_sec,lr,train_loss\n")

    model.train()
    optimizer.zero_grad(set_to_none=True)
    
    for step in range(start_step, total_steps):
        # 1. Temperature Governor check between optimizer steps (every 5 steps)
        if step % 5 == 0:
            pause_sec, was_paused = apply_temperature_governor(step, accum_tokens, run_id, wall_start_time)
            total_paused_sec += pause_sec
            num_pauses += was_paused
            
            # Check 25% threshold
            total_elapsed = time.time() - wall_start_time
            if total_elapsed > 120 and (total_paused_sec / total_elapsed) > 0.25:
                print(f"[!CAUTION] [{run_id}] Total paused time ({total_paused_sec/60:.1f}m) exceeds 25% of elapsed time ({total_elapsed/60:.1f}m)!", flush=True)

        # 2. LR Schedule (strictly keyed to optimizer step)
        if step < warmup_steps:
            lr = peak_lr * (step + 1) / warmup_steps
        elif step < stable_steps:
            lr = peak_lr
        else:
            decay_ratio = (step - stable_steps) / max(1, decay_steps)
            coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
            lr = min_lr + coeff * (peak_lr - min_lr)
            
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
            
        step_loss = 0.0
        for micro_step in range(grad_accum):
            x, y = loader.get_batch(micro_bs, device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                outputs = model(x)
                loss = F.cross_entropy(outputs.logits.view(-1, 8192), y.view(-1))
                scaled_loss = loss / grad_accum
            scaled_loss.backward()
            step_loss += loss.item() / grad_accum

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        
        accum_tokens += tokens_per_step
        total_wall_sec = time.time() - wall_start_time
        active_sec = max(1e-5, total_wall_sec - total_paused_sec)
        
        # Loss sanity guard
        if math.isnan(step_loss) or step_loss > 12.0:
            raise RuntimeError(f"FATAL: Divergence detected in {run_id} at step {step}: loss = {step_loss}")
            
        # Logging
        log_interval = 5 if is_test else 50
        if (step + 1) % log_interval == 0 or step == total_steps - 1:
            tok_s_active = accum_tokens / active_sec
            print(f"[{run_id}] Step {step+1:4d}/{total_steps} | LR {lr:.2e} | Loss {step_loss:.4f} | "
                  f"Tokens: {accum_tokens/1e6:6.2f}M | Active: {active_sec/60:5.1f}m (Paused: {total_paused_sec/60:4.1f}m) | Active Speed: {tok_s_active:,.0f} tok/s", flush=True)
            
            with open(CURVES_CSV, "a", encoding="utf-8") as f:
                f.write(f"{run_id},{run_name},{step+1},{accum_tokens/1e6:.2f},{total_wall_sec:.1f},{active_sec:.1f},{lr:.6e},{step_loss:.4f}\n")

        # 3. Resumable Checkpoint saving (every 15 minutes of active time)
        if (time.time() - last_checkpoint_time) > (900 if not is_test else 30):
            ckpt_save_path = CHECKPOINTS_DIR / f"{run_id}_step_{step+1}.pt"
            ckpt_dict = {
                "step": step + 1,
                "accum_tokens": accum_tokens,
                "total_paused_sec": total_paused_sec,
                "num_pauses": num_pauses,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "loader_state": loader.get_state(),
                "torch_cpu_rng": torch.get_rng_state(),
                "torch_cuda_rng": torch.cuda.get_rng_state_all(),
                "np_rng": np.random.get_state(),
            }
            torch.save(ckpt_dict, ckpt_save_path)
            # Maintain symlink or copy to latest
            latest_path = CHECKPOINTS_DIR / f"{run_id}_latest.pt"
            torch.save(ckpt_dict, latest_path)
            
            rolling_ckpts.append(ckpt_save_path)
            # Keep only last 2
            if len(rolling_ckpts) > 2:
                oldest = rolling_ckpts.pop(0)
                if oldest.exists() and oldest != ckpt_save_path:
                    try:
                        oldest.unlink()
                    except Exception:
                        pass
            print(f"  [CKPT] Saved rolling checkpoint: {ckpt_save_path.name} (Active time: {active_sec/60:.1f}m)", flush=True)
            last_checkpoint_time = time.time()

    total_wall_sec = time.time() - wall_start_time
    total_active_sec = max(1e-5, total_wall_sec - total_paused_sec)
    print(f"\nTraining completed for {run_id}:", flush=True)
    print(f"  Total Wall-Clock Time: {total_wall_sec/60:.2f} minutes", flush=True)
    print(f"  Active Training Time:   {total_active_sec/60:.2f} minutes", flush=True)
    print(f"  Total Paused Time:     {total_paused_sec/60:.2f} minutes ({num_pauses} cooling pauses)", flush=True)
    print(f"  Active Throughput:     {accum_tokens/total_active_sec:,.0f} tok/s", flush=True)
    
    # Save final checkpoint
    ckpt_path = CHECKPOINTS_DIR / f"{run_id}_final.pt"
    torch.save(model.state_dict(), ckpt_path)
    print(f"Final model checkpoint saved to: {ckpt_path}", flush=True)
    
    # Evaluation
    print(f"\n--- Evaluating {run_id} on All 4 Validation Sets ---", flush=True)
    eval_num_batches = 5 if is_test else 50
    val_results = evaluator.evaluate_all(model, num_batches=eval_num_batches)
    for vname, m in val_results.items():
        print(f"  {vname:25s}: CE Loss = {m['ce_loss']:.4f} | Bits/Tok = {m['bits_per_tok']:.4f} | BPB = {m['bpb']:.4f}", flush=True)

    # Generations
    print(f"\n--- Generating 15 Evaluation Samples for {run_id} ---", flush=True)
    sample_tokens = 30 if is_test else 64
    samples = generate_samples(model, sp, PROMPTS, device, max_new_tokens=sample_tokens)

    return {
        "run_id": run_id,
        "run_name": run_name,
        "total_tokens_m": accum_tokens / 1e6,
        "micro_bs": micro_bs,
        "grad_accum": grad_accum,
        "total_wall_min": round(total_wall_sec / 60, 2),
        "active_train_min": round(total_active_sec / 60, 2),
        "total_paused_min": round(total_paused_sec / 60, 2),
        "num_pauses": num_pauses,
        "active_tok_per_sec": round(accum_tokens / total_active_sec, 1),
        "final_train_loss": round(step_loss, 4),
        "val_wiki_ce": val_results["val_wikipedia_neutral"]["ce_loss"],
        "val_wiki_bpb": val_results["val_wikipedia_neutral"]["bpb"],
        "val_cosmo_ce": val_results["val_cosmopedia"]["ce_loss"],
        "val_cosmo_bpb": val_results["val_cosmopedia"]["bpb"],
        "val_fineweb_ce": val_results["val_fineweb_edu"]["ce_loss"],
        "val_fineweb_bpb": val_results["val_fineweb_edu"]["bpb"],
        "val_tinystories_ce": val_results["val_tinystories"]["ce_loss"],
        "val_tinystories_bpb": val_results["val_tinystories"]["bpb"],
        "samples": samples,
    }

# ---------------------------------------------------------------------------
# Main Execution
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true", help="Run quick 10-step verification test")
    parser.add_argument("--runs", nargs="+", default=["Run_B", "Run_C", "Run_D"], help="Which runs to execute")
    parser.add_argument("--micro-bs", type=int, default=4, help="Micro-batch size (e.g. 4 or 2)")
    parser.add_argument("--resume-from", type=str, default=None, help="Path to checkpoint to resume from")
    args = parser.parse_args()

    micro_bs = args.micro_bs
    grad_accum = 32 // micro_bs
    assert micro_bs * grad_accum == 32, "micro_bs * grad_accum must equal 32 sequences!"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running Phase 5B (v2) on device: {device} ({torch.cuda.get_device_name(0)})")
    print(f"Settings: micro_bs={micro_bs}, grad_accum={grad_accum} (global batch = 32 seqs = 65,536 tokens)")
    
    # Load SentencePiece tokenizer
    sp = spm.SentencePieceProcessor()
    sp.load(str(TOKENIZER_MODEL))
    print(f"Loaded Tokenizer: vocab_size = {sp.get_piece_size()}")

    # Setup telemetry monitor
    telemetry = GPUTelemetry(TELEMETRY_CSV, interval_sec=60)
    telemetry.start()

    evaluator = ValidationEvaluator(SHARDS_DIR, seq_len=2048, device=device)

    RUN_CONFIGS = {
        "Run_A": {
            "name": "100% FineWeb-Edu",
            "mix_weights": [0.0, 1.0, 0.0],
        },
        "Run_B": {
            "name": "100% Cosmopedia v2",
            "mix_weights": [1.0, 0.0, 0.0],
        },
        "Run_C": {
            "name": "Default Blend (45% Cosmo, 45% FW-Edu, 10% Tiny)",
            "mix_weights": [0.45, 0.45, 0.10],
        },
        "Run_D": {
            "name": "Simpler-Text Blend (40% Tiny, 40% Cosmo, 20% FW-Edu)",
            "mix_weights": [0.40, 0.20, 0.40],
        },
    }

    if args.test:
        total_steps = 10
        warmup_steps = 2
        decay_steps = 3
    else:
        total_steps = 3815
        warmup_steps = 50
        decay_steps = 381

    # Load existing summaries if present (e.g. Run_A)
    all_summaries = []
    if SUMMARY_CSV.exists():
        try:
            df_old = pd.read_csv(SUMMARY_CSV)
            all_summaries = df_old.to_dict(orient="records")
            print(f"Loaded {len(all_summaries)} existing run summaries from {SUMMARY_CSV}")
        except Exception:
            pass

    all_samples = {}

    try:
        for run_id in args.runs:
            if run_id not in RUN_CONFIGS:
                continue
            cfg = RUN_CONFIGS[run_id]
            res = run_mixture_training(
                run_id=run_id,
                run_name=cfg["name"],
                mix_weights=cfg["mix_weights"],
                total_steps=total_steps,
                warmup_steps=warmup_steps,
                decay_steps=decay_steps,
                peak_lr=1e-3,
                min_lr=1e-4,
                micro_bs=micro_bs,
                grad_accum=grad_accum,
                seq_len=2048,
                device=device,
                evaluator=evaluator,
                sp=sp,
                telemetry=telemetry,
                resume_ckpt_path=args.resume_from,
                is_test=args.test,
            )
            all_samples[run_id] = res.pop("samples")
            
            # Replace existing summary for this run if re-run
            all_summaries = [s for s in all_summaries if s["run_id"] != run_id]
            all_summaries.append(res)
            
            df_summary = pd.DataFrame(all_summaries)
            df_summary.to_csv(SUMMARY_CSV, index=False)
            
    finally:
        telemetry.stop()

    # Append to Sample Generations Report
    if all_samples:
        print("\n--- Updating Sample Generations Report ---", flush=True)
        mode = "a" if SAMPLES_MD.exists() else "w"
        with open(SAMPLES_MD, mode, encoding="utf-8") as f:
            if mode == "w":
                f.write("# Phase 5B Data Mixture Ablation: Sample Generations\n\n")
                f.write("Generated using greedy decoding (`temperature = 0.0`, `max_new_tokens = 64`) from decayed 250M-token checkpoints.\n\n")
            
            for p_idx, prompt in enumerate(PROMPTS):
                category = "Standard Prompt" if p_idx < 10 else "Basic Grammar Prompt"
                f.write(f"### Prompt {p_idx+1} ({category})\n")
                f.write(f"> **Prompt**: `{prompt}`\n\n")
                
                for run_id, samples_list in all_samples.items():
                    sample = samples_list[p_idx]
                    f.write(f"**{run_id} ({RUN_CONFIGS[run_id]['name']})**:\n")
                    f.write(f"```text\n{sample['completion']}\n```\n\n")
                f.write("---\n\n")

    # Generate Markdown Summary Report
    df = pd.DataFrame(all_summaries)
    df.to_csv(SUMMARY_CSV, index=False)
    
    ref_row = df[df["run_id"] == "Run_C"]
    ref_wiki = ref_row["val_wiki_ce"].values[0] if len(ref_row) > 0 else df["val_wiki_ce"].min()
    
    with open(REPORT_MD, "w", encoding="utf-8") as f:
        f.write("# Phase 5B Data Mixture Ablation Report\n\n")
        f.write("## 1. Executive Summary & Results Table\n\n")
        f.write(f"- **Model Architecture**: `6L_768H_1KV` (42.28M parameters)\n")
        f.write(f"- **Volume per Run**: 250M tokens (~1 hour wall-clock each)\n")
        f.write(f"- **Schedule**: Flat WSD with 10% terminal cosine decay to $0.1\\times \\eta_0$\n")
        f.write(f"- **Noise Floor Reference**: $\\Delta_{{\\text{{val}}}} = {NOISE_FLOOR_VAL:.4f}$ (validation spread), $\\Delta_{{\\text{{train}}}} = {NOISE_FLOOR_TRAIN:.4f}$ (training spread)\n\n")
        
        f.write("| Run ID | Mixture Description | Micro BS | Wiki Neutral (CE) | Wiki (BPB) | Cosmo (CE) | FineWeb (CE) | TinyStories (CE) | Active Time | Wall Time | Pauses | Active Tok/s |\n")
        f.write("| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |\n")
        for _, r in df.iterrows():
            mbs = r.get('micro_bs', '8 (initial)')
            pauses = r.get('num_pauses', 0)
            active_t = r.get('active_train_min', r.get('total_time_min', 0))
            wall_t = r.get('total_wall_min', r.get('total_time_min', 0))
            tps = r.get('active_tok_per_sec', r.get('tok_per_sec', 0))
            f.write(f"| **{r['run_id']}** | {r['run_name']} | {mbs} | **{r['val_wiki_ce']:.4f}** | {r['val_wiki_bpb']:.4f} | {r['val_cosmo_ce']:.4f} | {r['val_fineweb_ce']:.4f} | {r['val_tinystories_ce']:.4f} | {active_t:.1f}m | {wall_t:.1f}m | {pauses} | {tps:,.0f} |\n")
        
        f.write("\n## 2. Decision Analysis on Neutral Benchmark\n\n")
        f.write("Gaps against default blend (Run C) on neutral English Wikipedia:\n\n")
        for _, r in df.iterrows():
            gap = r['val_wiki_ce'] - ref_wiki
            sig_val = abs(gap) > NOISE_FLOOR_VAL
            sig_str = "Statistically Significant (> Val Floor)" if sig_val else "Within Noise Floor"
            f.write(f"- **{r['run_id']} ({r['run_name']})**: $\\Delta = {gap:+.4f}$ CE vs Run C ({sig_str})\n")

    print(f"\nPhase 5B Completed Successfully! Outputs saved to {RESULTS_DIR}", flush=True)

if __name__ == "__main__":
    main()
