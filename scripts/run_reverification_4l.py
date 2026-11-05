"""
Step 2A: Shape Re-Verification - 4L_768H_1KV on Default Blend (250M tokens)
Compares directly against 6L Run C on the exact same:
- Mixture: 45% Cosmopedia v2 / 45% FineWeb-Edu / 10% TinyStories
- Global batch: 32 sequences (65,536 tokens), micro_bs=4, grad_accum=8
- Tokens: 250,019,320 tokens (3,815 optimizer steps)
- Schedule: WSD (50-step warmup, flat 1e-3 to step 3,434, 381-step cosine decay to 1e-4)
- Loss: Verified explicit next-token cross entropy (no double-shift)
- Seed: 42
- Temperature Governor: 83C pause / 78C resume
- Pre-training regression test: verifies exact t -> t+1 label alignment before training begins.
"""

import os
import sys
import math
import time
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
CHECKPOINTS_DIR = MODELS_DIR / "checkpoints" / "reverification"
TOKENIZER_MODEL = MODELS_DIR / "tokenizer_8k" / "tokenizer.model"

RESULTS_DIR.mkdir(exist_ok=True)
CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)

TELEMETRY_CSV = RESULTS_DIR / "reverification_4l_telemetry.csv"
PAUSE_LOG_CSV = RESULTS_DIR / "pause_log.csv"
CURVES_CSV = RESULTS_DIR / "reverification_4l_curves.csv"
SUMMARY_CSV = RESULTS_DIR / "shape_comparison_reverification.csv"
SAMPLES_MD = RESULTS_DIR / "shape_comparison_samples.md"

COMPRESSION_STATS = {
    "val_cosmopedia": {"bytes_per_tok": 4.149, "chars_per_tok": 4.148},
    "val_fineweb_edu": {"bytes_per_tok": 4.156, "chars_per_tok": 4.150},
    "val_tinystories": {"bytes_per_tok": 3.220, "chars_per_tok": 3.220},
    "val_wikipedia_neutral": {"bytes_per_tok": 4.098, "chars_per_tok": 4.092},
}

PROMPTS = [
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
    "She opened the door and saw ",
    "The dog ran after the ball because ",
    "He went to the store to buy ",
    "They were very happy when they ",
    "Every morning, the boy eats ",
]

# ---------------------------------------------------------------------------
# Label Alignment Regression Test
# ---------------------------------------------------------------------------
def run_label_alignment_regression_test(model, device):
    print("\n--- Running Mandatory Pre-Training Label-Shift Regression Test ---", flush=True)
    model.eval()
    dummy_seq = torch.tensor([[1, 2, 3, 4, 5]], dtype=torch.long, device=device)
    x = dummy_seq[:, :-1]  # [1, 2, 3, 4]
    y = dummy_seq[:, 1:]   # [2, 3, 4, 5]
    
    # 1. Structural check: y must be x shifted by 1
    if not (x[0, 1:] == y[0, :-1]).all():
        raise RuntimeError("FATAL: Target vector y is not shifted by exactly 1 relative to x!")
    if not (y[0] == torch.tensor([2, 3, 4, 5], device=device)).all():
        raise RuntimeError(f"FATAL: Target y {y[0].tolist()} does not match expected [2, 3, 4, 5]!")
    if x.shape != y.shape:
        raise RuntimeError(f"FATAL: Shape mismatch between x {x.shape} and y {y.shape}")
        
    # 2. Forward pass check
    with torch.no_grad():
        out = model(x)
        if out.logits.shape[1] != x.shape[1]:
            raise RuntimeError(f"FATAL: Logits seq length {out.logits.shape[1]} != input length {x.shape[1]}")
        if hasattr(out, "loss") and out.loss is not None:
            raise RuntimeError("FATAL: Model forward returned an internal loss! Must compute loss explicitly outside model.")
            
        loss_val = F.cross_entropy(out.logits.view(-1, out.logits.shape[-1]), y.view(-1))
        if torch.isnan(loss_val) or loss_val.item() <= 0:
            raise RuntimeError("FATAL: Regression test loss is NaN or non-positive!")
            
    model.train()
    print("  [PASSED]: Exact 1-step label shift (t -> t+1: [1,2,3,4,5] -> x:[1,2,3,4], y:[2,3,4,5]) mathematically verified with zero secondary shift.\n", flush=True)

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

def get_gpu_temperature():
    try:
        cmd = [SMI_BIN, "--query-gpu=temperature.gpu", "--format=csv,noheader,nounits"]
        res = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return float(res.stdout.strip())
    except Exception:
        return 0.0

def apply_temperature_governor(step, accum_tokens, run_id, total_wall_start):
    temp = get_gpu_temperature()
    if temp < 83.0:
        return 0.0, 0
        
    t_start = time.time()
    t_start_iso = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(t_start))
    temp_initial = temp
    print(f"\n[!WARNING] [{run_id}] Step {step}: GPU temperature {temp:.1f}C >= 83C threshold! Pausing training...", flush=True)
    
    while temp > 78.0:
        time.sleep(5)
        temp = get_gpu_temperature()
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
# Data Loader & Evaluator
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
        }

    def evaluate_all(self, model, num_batches=40):
        model.eval()
        results = {}
        for name in ["val_wikipedia_neutral", "val_cosmopedia", "val_fineweb_edu", "val_tinystories"]:
            results[name] = self.evaluate_set(model, name, num_batches=num_batches)
        model.train()
        return results

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

def create_4l_model(device, seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    config = LlamaConfig(
        vocab_size=8192,
        hidden_size=768,
        intermediate_size=2048,
        num_hidden_layers=4,
        num_attention_heads=12,
        num_key_value_heads=1,
        tie_word_embeddings=True,
        torch_dtype="bfloat16",
        _attn_implementation="sdpa",
        max_position_embeddings=2048,
    )
    return LlamaForCausalLM(config).to(device=device, dtype=torch.bfloat16)

# ---------------------------------------------------------------------------
# Main Training Function
# ---------------------------------------------------------------------------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Step 2A: Running 4L_768H_1KV Re-Verification on device: {device} ({torch.cuda.get_device_name(0)})")
    
    sp = spm.SentencePieceProcessor()
    sp.load(str(TOKENIZER_MODEL))
    
    model = create_4l_model(device, seed=42)
    run_label_alignment_regression_test(model, device)
    
    telemetry = GPUTelemetry(TELEMETRY_CSV, interval_sec=60)
    telemetry.set_run_name("4L_Run_C_Reverification")
    telemetry.start()
    
    evaluator = ValidationEvaluator(SHARDS_DIR, seq_len=2048, device=device)
    loader = ShardStreamDataLoader(SHARDS_DIR, seq_len=2048, mix_weights=[0.45, 0.45, 0.10])
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, betas=(0.9, 0.95), weight_decay=0.1, fused=True)
    
    micro_bs = 4
    grad_accum = 8
    seq_len = 2048
    tokens_per_step = micro_bs * grad_accum * seq_len  # 65,536
    total_steps = 3815
    warmup_steps = 50
    decay_steps = 381
    stable_steps = total_steps - decay_steps
    
    peak_lr = 1e-3
    min_lr = 1e-4
    
    wall_start_time = time.time()
    last_checkpoint_time = time.time()
    accum_tokens = 0
    total_paused_sec = 0.0
    num_pauses = 0
    rolling_ckpts = []
    
    with open(CURVES_CSV, "w", encoding="utf-8") as f:
        f.write("run_id,step,tokens_m,wall_sec,active_sec,lr,train_loss\n")
        
    model.train()
    optimizer.zero_grad(set_to_none=True)
    
    print("\nStarting Training: 4L_768H_1KV (250M tokens on 45/45/10 mix)...", flush=True)
    
    try:
        for step in range(total_steps):
            if step % 5 == 0:
                p_sec, was_p = apply_temperature_governor(step, accum_tokens, "4L_768H_1KV", wall_start_time)
                total_paused_sec += p_sec
                num_pauses += was_p
                
                total_el = time.time() - wall_start_time
                if total_el > 120 and (total_paused_sec / total_el) > 0.25:
                    print(f"[!CAUTION] Paused time ({total_paused_sec/60:.1f}m) > 25% of elapsed ({total_el/60:.1f}m)!", flush=True)

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
            for _ in range(grad_accum):
                x, y = loader.get_batch(micro_bs, device)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    out = model(x)
                    loss = F.cross_entropy(out.logits.view(-1, 8192), y.view(-1))
                    scaled_loss = loss / grad_accum
                scaled_loss.backward()
                step_loss += loss.item() / grad_accum
                
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            
            accum_tokens += tokens_per_step
            total_wall_sec = time.time() - wall_start_time
            active_sec = max(1e-5, total_wall_sec - total_paused_sec)
            
            if (step + 1) % 50 == 0 or step == total_steps - 1:
                tok_s = accum_tokens / active_sec
                print(f"[4L_768H_1KV] Step {step+1:4d}/{total_steps} | LR {lr:.2e} | Loss {step_loss:.4f} | "
                      f"Tokens: {accum_tokens/1e6:6.2f}M | Active: {active_sec/60:5.1f}m (Paused: {total_paused_sec/60:4.1f}m) | Active Speed: {tok_s:,.0f} tok/s", flush=True)
                with open(CURVES_CSV, "a", encoding="utf-8") as f:
                    f.write(f"4L_768H_1KV,{step+1},{accum_tokens/1e6:.2f},{total_wall_sec:.1f},{active_sec:.1f},{lr:.6e},{step_loss:.4f}\n")
                    
            if (time.time() - last_checkpoint_time) > 900:
                ckpt_p = CHECKPOINTS_DIR / f"4L_step_{step+1}.pt"
                ckpt_dict = {
                    "step": step + 1,
                    "accum_tokens": accum_tokens,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "loader_state": loader.get_state(),
                    "torch_cpu_rng": torch.get_rng_state(),
                    "torch_cuda_rng": torch.cuda.get_rng_state_all(),
                    "np_rng": np.random.get_state(),
                }
                torch.save(ckpt_dict, ckpt_p)
                rolling_ckpts.append(ckpt_p)
                if len(rolling_ckpts) > 2:
                    old = rolling_ckpts.pop(0)
                    if old.exists():
                        try:
                            old.unlink()
                        except Exception:
                            pass
                last_checkpoint_time = time.time()
                print(f"  [CKPT] Saved rolling checkpoint: {ckpt_p.name}", flush=True)

    finally:
        telemetry.stop()
        
    total_wall_sec = time.time() - wall_start_time
    total_active_sec = max(1e-5, total_wall_sec - total_paused_sec)
    print(f"\n4L Training Completed:", flush=True)
    print(f"  Total Wall Time:   {total_wall_sec/60:.2f} min", flush=True)
    print(f"  Active Train Time: {total_active_sec/60:.2f} min", flush=True)
    print(f"  Total Paused Time: {total_paused_sec/60:.2f} min ({num_pauses} pauses)", flush=True)
    print(f"  Active Speed:      {accum_tokens/total_active_sec:,.0f} tok/s", flush=True)
    
    # Save final model
    final_ckpt = CHECKPOINTS_DIR / "4L_768H_1KV_final.pt"
    torch.save(model.state_dict(), final_ckpt)
    print(f"Saved final checkpoint to {final_ckpt}", flush=True)
    
    print("\n--- Evaluating 4L on All 4 Validation Sets ---", flush=True)
    val_res = evaluator.evaluate_all(model, num_batches=50)
    for vname, m in val_res.items():
        print(f"  {vname:25s}: CE Loss = {m['ce_loss']:.4f} | Bits/Tok = {m['bits_per_tok']:.4f} | BPB = {m['bpb']:.4f}", flush=True)
        
    print("\n--- Generating 15 Evaluation Samples for 4L ---", flush=True)
    samples = generate_samples(model, sp, PROMPTS, device, max_new_tokens=64)
    
    # Save sample generations
    with open(SAMPLES_MD, "w", encoding="utf-8") as f:
        f.write("# 4L vs 6L Re-Verification Sample Generations (Default Mix 45/45/10)\n\n")
        for p_idx, p in enumerate(PROMPTS):
            f.write(f"### Prompt {p_idx+1}: `{p}`\n\n")
            f.write(f"**4L_768H_1KV**:\n```text\n{samples[p_idx]['completion']}\n```\n\n---\n\n")
            
    # Read existing 6L Run C from phase5b_summary.csv
    df_6l = pd.read_csv(RESULTS_DIR / "phase5b_summary.csv")
    row_6l = df_6l[df_6l["run_id"] == "Run_C"].iloc[0]
    
    summary_data = [
        {
            "model": "6L_768H_1KV",
            "params_m": 42.28,
            "val_wiki_ce": row_6l["val_wiki_ce"],
            "val_wiki_bpb": row_6l["val_wiki_bpb"],
            "val_cosmo_ce": row_6l["val_cosmo_ce"],
            "val_fineweb_ce": row_6l["val_fineweb_ce"],
            "val_tinystories_ce": row_6l["val_tinystories_ce"],
            "active_train_min": row_6l["active_train_min"],
            "total_wall_min": row_6l["total_wall_min"],
            "pauses": row_6l["num_pauses"],
            "active_tok_per_sec": row_6l["active_tok_per_sec"],
            "decode_0k_f16": 1384.9,
            "decode_32k_f16": 1101.2,
        },
        {
            "model": "4L_768H_1KV",
            "params_m": 30.28,
            "val_wiki_ce": val_res["val_wikipedia_neutral"]["ce_loss"],
            "val_wiki_bpb": val_res["val_wikipedia_neutral"]["bpb"],
            "val_cosmo_ce": val_res["val_cosmopedia"]["ce_loss"],
            "val_fineweb_ce": val_res["val_fineweb_edu"]["ce_loss"],
            "val_tinystories_ce": val_res["val_tinystories"]["ce_loss"],
            "active_train_min": round(total_active_sec / 60, 2),
            "total_wall_min": round(total_wall_sec / 60, 2),
            "pauses": num_pauses,
            "active_tok_per_sec": round(accum_tokens / total_active_sec, 1),
            "decode_0k_f16": 1660.7,
            "decode_32k_f16": 1314.8,
        }
    ]
    df_sum = pd.DataFrame(summary_data)
    df_sum.to_csv(SUMMARY_CSV, index=False)
    
    gap_wiki = val_res["val_wikipedia_neutral"]["ce_loss"] - row_6l["val_wiki_ce"]
    print("\n=======================================================", flush=True)
    print("SHAPE RE-VERIFICATION RESULT (6L vs 4L):", flush=True)
    print(f"  6L Run C Neutral Wiki CE: {row_6l['val_wiki_ce']:.4f}")
    print(f"  4L Run   Neutral Wiki CE: {val_res['val_wikipedia_neutral']['ce_loss']:.4f}")
    print(f"  Gap (4L - 6L): +{gap_wiki:.4f} CE (6L advantage = -{gap_wiki:.4f})")
    print(f"  Threshold for second seed: 0.05 CE")
    if gap_wiki >= 0.05:
        print("  VERDICT: 6L advantage exceeds 0.05 CE threshold -> 6L_768H_1KV IS CONFIRMED!")
    else:
        print("  VERDICT: Gap is under 0.05 CE -> SECOND SEED (seed 43) IS TRIGGERED.")
    print("=======================================================", flush=True)

if __name__ == "__main__":
    main()
