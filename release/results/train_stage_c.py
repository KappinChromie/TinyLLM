"""
Stage C Training Script: 32,768 Context Extension & Quality Anneal
Locked Architecture: 6L_768H_1KV (42.28M parameters, tied embeddings).
- Total Token Budget: Exactly 114.69M tokens (875 optimizer steps * 131,072 tokens = 114,688,000 tokens).
- Global Batch: 4 sequences of 32k equivalent = 131,072 tokens per optimizer step:
    - Micro-batch long: 1 sequence of 32k (gradient checkpointing enabled)
    - Micro-batch replay: 4 sequences of 2,048 tokens
- Sub-stages:
    C1 (Steps 1 to 437, 57.28M tokens):
        - 75% PG-19 Long (>=32k continuous) + 25% Short Replay (2k)
        - Replay Mix: 40% TinyStories, 30% FineWeb-Edu, 30% Cosmopedia v2 (TinyStories is 10% overall)
        - Flat LR: 5.0e-4 (step 1 micro-ramp at 2.5e-4)
        - End of C1: Full evaluation, C1 Rule verification (stop if 32k context-use > 8k + 0.02 or 16k-32k dropping steeply)
        - Saves models/checkpoints/stageC/stageC1_undecayed_final.pt
    C2 (Steps 438 to 875, 57.41M tokens):
        - 50% PG-19 Long (>=32k continuous) + 50% Short Text (2k)
        - Replay Mix: 20% TinyStories, 40% FineWeb-Edu, 40% Cosmopedia v2 (TinyStories is 10% overall)
        - Cosine Anneal: 5.0e-4 -> 1.0e-4
        - Saves models/checkpoints/stageC/stageC_decayed_final.pt
- Temperature Governor: Pause >= 83C, Resume <= 78C.
- Loss Spike Guard: 1 rollback on spike (> 3x 20-step MA), stops on 2nd or NaN.
"""

import sys
import os
import math
import time
import shutil
import argparse
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
STAGE_C_CKPT_DIR = MODELS_DIR / "checkpoints" / "stageC"
TOKENIZER_MODEL = MODELS_DIR / "tokenizer_8k" / "tokenizer.model"
SRC_CHECKPOINT = MODELS_DIR / "checkpoints" / "stageB" / "stageB_final_undecayed.pt"

TRAIN_LOG_CSV = RESULTS_DIR / "stageC_train_log.csv"
EVAL_LOG_CSV = RESULTS_DIR / "stageC_eval.csv"
PAUSE_LOG_CSV = RESULTS_DIR / "pause_log.csv"
SAMPLES_MD = RESULTS_DIR / "stageC_samples.md"

SAMPLE_PROMPTS_5 = [
    "Once upon a time in a faraway forest, there lived a tiny",
    "The fundamental difference between planets and stars is",
    "To solve the system of linear equations, first compute the",
    "The history of the Roman Empire demonstrates that political power",
    "Deep neural networks learn hierarchical representations by",
]

BUCKETS_32K = [
    ("0-1k", 0, 1024),
    ("1k-2k", 1024, 2048),
    ("2k-4k", 2048, 4096),
    ("4k-8k", 4096, 8192),
    ("8k-16k", 8192, 16384),
    ("16k-32k", 16384, 32768),
]

BUCKETS_8K = [
    ("0-1k", 0, 1024),
    ("1k-2k", 1024, 2048),
    ("2k-4k", 2048, 4096),
    ("4k-8k", 4096, 8192),
]

PASSKEY_LENGTHS = [1024, 2048, 4096, 8192, 16384, 32768]

# ---------------------------------------------------------------------------
# GPU Telemetry & Temperature Governor
# ---------------------------------------------------------------------------
def get_gpu_telemetry():
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=temperature.gpu,memory.used,power.draw", "--format=csv,noheader,nounits"],
            encoding="utf-8"
        )
        parts = out.strip().split("\n")[0].split(",")
        return float(parts[0]), float(parts[1]), float(parts[2])
    except Exception:
        return 50.0, 4000.0, 150.0

def apply_temperature_governor(step, accum_tokens, run_id="StageC"):
    temp, mem, pwr = get_gpu_telemetry()
    if temp < 83.0:
        return 0.0, 0
    t_start = time.time()
    t_start_iso = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(t_start))
    print(f"\n[!WARNING] [{run_id}] Step {step}: GPU temp {temp:.1f}C >= 83C! Pausing training...", flush=True)
    while temp > 78.0:
        time.sleep(5)
        temp, mem, pwr = get_gpu_telemetry()
        print(f"  Cooling down: current temp = {temp:.1f}C (target <= 78C)...", end="\r", flush=True)
    t_end = time.time()
    t_end_iso = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(t_end))
    pause_dur = t_end - t_start
    print(f"\n[GOVERNOR] Temp dropped to {temp:.1f}C <= 78C. Resuming after {pause_dur:.1f}s pause.", flush=True)
    
    pause_exists = PAUSE_LOG_CSV.exists()
    with open(PAUSE_LOG_CSV, "a", encoding="utf-8") as f:
        if not pause_exists:
            f.write("timestamp_start,timestamp_end,duration_sec,temp_start,temp_end,optimizer_step,tokens_seen,run_id\n")
        f.write(f"{t_start_iso},{t_end_iso},{pause_dur:.2f},83.0,{temp:.1f},{step},{accum_tokens},{run_id}\n")
    return pause_dur, 1

# ---------------------------------------------------------------------------
# Stage C Data Loaders (C1 and C2 Mixtures)
# ---------------------------------------------------------------------------
class StageCLoader:
    def __init__(self, shards_dir=SHARDS_DIR, seed=1234):
        self.seq_len_long = 32768
        self.seq_len_short = 2048
        
        # 1. Long PG-19 books continuous >= 32k
        self.pg19 = np.memmap(shards_dir / "train_pg19_long.bin", dtype=np.uint16, mode="r")
        eos_pos = np.where(self.pg19 == 2)[0]
        starts = np.concatenate(([0], eos_pos[:-1] + 1))
        lens = eos_pos - starts + 1
        mask = lens >= self.seq_len_long + 1
        self.pg19_starts = starts[mask]
        self.pg19_lens = lens[mask]
        
        # 2. Short Replay Sources
        self.tinystories = np.memmap(shards_dir / "train_tinystories.bin", dtype=np.uint16, mode="r")
        self.fineweb_edu = np.memmap(shards_dir / "train_fineweb_edu.bin", dtype=np.uint16, mode="r")
        self.cosmopedia = np.memmap(shards_dir / "train_cosmopedia.bin", dtype=np.uint16, mode="r")
        self.replay_shards = [self.tinystories, self.fineweb_edu, self.cosmopedia]
        
        # C1 Replay Weights: 40% TinyStories, 30% FineWeb, 30% Cosmo
        self.c1_replay_weights = np.array([0.40, 0.30, 0.30])
        # C2 Short Weights: 20% TinyStories, 40% FineWeb, 40% Cosmo
        self.c2_replay_weights = np.array([0.20, 0.40, 0.40])
        
        self.rng_long = np.random.RandomState(seed + 101)
        self.rng_short = np.random.RandomState(seed + 202)

    def get_state(self):
        return {
            "rng_long": self.rng_long.get_state(),
            "rng_short": self.rng_short.get_state(),
        }

    def set_state(self, state):
        self.rng_long.set_state(state["rng_long"])
        self.rng_short.set_state(state["rng_short"])

    def get_long_sequence(self, device="cuda"):
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

    def get_replay_batch(self, batch_size=4, sub_stage="C1", device="cuda"):
        weights = self.c1_replay_weights if sub_stage == "C1" else self.c2_replay_weights
        shard_indices = self.rng_short.choice(3, size=batch_size, p=weights)
        x_list, y_list = [], []
        for s_idx in shard_indices:
            shard = self.replay_shards[s_idx]
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
def evaluate_position_buckets(model, shard, max_ctx=32768, num_sequences=10, device="cuda"):
    model.eval()
    max_idx = len(shard) - max_ctx - 1
    rng = np.random.RandomState(42)
    buckets = BUCKETS_32K if max_ctx == 32768 else BUCKETS_8K
    bucket_losses = {name: [] for name, start, _ in buckets if start < max_ctx}
    
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
            
        for name, b_start, b_end in buckets:
            if b_start < max_ctx:
                actual_end = min(b_end, max_ctx)
                if actual_end > b_start:
                    bucket_losses[name].append(float(np.mean(token_losses[b_start:actual_end])))
                    
    return {name: round(float(np.mean(loss_list)), 4) for name, loss_list in bucket_losses.items()}

@torch.no_grad()
def evaluate_context_use_8k(model, shards_dict, num_windows=50, device="cuda"):
    results = {}
    for src_name, shard in shards_dict.items():
        max_idx = len(shard) - 8193 - 1
        rng = np.random.RandomState(42)
        step_stride = max(1, max_idx // num_windows)
        starts = [min(max_idx, i * step_stride + rng.randint(0, min(500, step_stride))) for i in range(num_windows)]
        
        losses_8k, losses_2k = [], []
        for s_idx in starts:
            seq = shard[s_idx : s_idx + 8193]
            target = torch.from_numpy(seq[7169 : 8193].astype(np.int64)).to(device)
            
            # Full 8k
            inp_8k = torch.from_numpy(seq[0 : 8192].astype(np.int64)).unsqueeze(0).to(device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                out = model(inp_8k)
                loss_8k = F.cross_entropy(out.logits[0, 7168 : 8192], target).item()
            losses_8k.append(loss_8k)
            
            # Last 2k
            inp_2k = torch.from_numpy(seq[6144 : 8192].astype(np.int64)).unsqueeze(0).to(device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                out = model(inp_2k)
                loss_2k = F.cross_entropy(out.logits[0, 1024 : 2048], target).item()
            losses_2k.append(loss_2k)
            
        m_8k = float(np.mean(losses_8k))
        m_2k = float(np.mean(losses_2k))
        results[f"{src_name}_8k"] = round(m_8k, 4)
        results[f"{src_name}_2k"] = round(m_2k, 4)
        results[f"{src_name}_delta"] = round(m_8k - m_2k, 4)
    return results

@torch.no_grad()
def evaluate_context_use_32k(model, shard, num_windows=50, device="cuda"):
    max_idx = len(shard) - 32769
    rng = np.random.RandomState(42)
    step_stride = max(1, max_idx // num_windows)
    starts = [min(max_idx, i * step_stride + rng.randint(0, min(1000, step_stride))) for i in range(num_windows)]
    
    losses_32k, losses_8k, losses_2k = [], [], []
    for s_idx in starts:
        raw_seq = shard[s_idx : s_idx + 32769]
        target = torch.from_numpy(raw_seq[31745 : 32769].astype(np.int64)).to(device)
        
        # 32k
        x_32k = torch.from_numpy(raw_seq[0 : 32768].astype(np.int64)).unsqueeze(0).to(device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            out_32k = model(x_32k)
            loss_32k = F.cross_entropy(out_32k.logits[0, 31744 : 32768], target).item()
        losses_32k.append(loss_32k)
        
        # 8k
        x_8k = torch.from_numpy(raw_seq[24576 : 32768].astype(np.int64)).unsqueeze(0).to(device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            out_8k = model(x_8k)
            loss_8k = F.cross_entropy(out_8k.logits[0, 7168 : 8192], target).item()
        losses_8k.append(loss_8k)
        
        # 2k
        x_2k = torch.from_numpy(raw_seq[30720 : 32768].astype(np.int64)).unsqueeze(0).to(device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            out_2k = model(x_2k)
            loss_2k = F.cross_entropy(out_2k.logits[0, 1024 : 2048], target).item()
        losses_2k.append(loss_2k)
        
    m_32k = float(np.mean(losses_32k))
    m_8k = float(np.mean(losses_8k))
    m_2k = float(np.mean(losses_2k))
    return {
        "pg19_ctx_32k": round(m_32k, 4),
        "pg19_ctx_8k": round(m_8k, 4),
        "pg19_ctx_2k": round(m_2k, 4),
        "pg19_delta_32k_8k": round(m_32k - m_8k, 4),
        "pg19_gain_32k_2k": round(m_2k - m_32k, 4),
    }

def run_passkey_retrieval(model, sp, val_shard, lengths=PASSKEY_LENGTHS, trials_per_length=20, device="cuda"):
    model.eval()
    results = {}
    rng = np.random.RandomState(8888)
    depth_percents = np.linspace(0.1, 0.9, trials_per_length)
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
        k_val = ctx_len // 1024
        k_str = f"{k_val}k" if k_val >= 1 else f"{ctx_len}tok"
        results[f"passkey_{k_str}_acc"] = acc
    return results

def generate_samples(model, sp, prompts=SAMPLE_PROMPTS_5, max_new=32, device="cuda"):
    model.eval()
    samples = []
    for p in prompts:
        curr = sp.encode_as_ids(p)
        with torch.no_grad():
            for _ in range(max_new):
                inp = torch.tensor([curr[-32768:]], dtype=torch.long, device=device)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    out = model(inp)
                    nxt = torch.argmax(out.logits[0, -1, :]).item()
                curr.append(nxt)
                if nxt == sp.eos_id():
                    break
        samples.append({"prompt": p, "completion": sp.decode(curr).strip()})
    return samples

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

def run_preflight_regression_test(model, loader, device="cuda"):
    print("\n--- Running Pre-Flight Regression Test on Stage C Pipeline ---", flush=True)
    # Check 1: Long batch shift
    x_long, y_long = loader.get_long_sequence(device=device)
    assert x_long.shape == (1, 32768)
    assert y_long.shape == (1, 32768)
    assert (x_long[:, 1:] == y_long[:, :-1]).all()
    print("  [Check 1 Passed]: Long batch y is shifted by exactly 1 (32,768 tokens).")
    
    # Check 2: Replay batch shift
    x_short, y_short = loader.get_replay_batch(batch_size=4, sub_stage="C1", device=device)
    assert x_short.shape == (4, 2048)
    assert (x_short[:, 1:] == y_short[:, :-1]).all()
    print("  [Check 2 Passed]: Replay batch y is shifted by exactly 1 (4 seqs x 2,048 tokens).")
    
    # Check 3: Loss calculation parity
    with torch.no_grad():
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            out = model(x_short)
            loss_rep = F.cross_entropy(out.logits.reshape(-1, 8192), y_short.reshape(-1))
            tokens = torch.cat([x_short[:, :1], y_short], dim=1)
            out_ind = model(tokens[:, :-1])
            loss_ind = F.cross_entropy(out_ind.logits.reshape(-1, 8192), tokens[:, 1:].reshape(-1))
            assert torch.allclose(loss_rep, loss_ind, atol=1e-5)
    print("  [Check 3 Passed]: Reported loss matches independent cross-entropy loss.")
    print("Pre-flight regression test: ALL CHECKS PASSED!\n", flush=True)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rope-theta", type=float, default=1000000.0)
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== Stage C: 32,768 Context Extension & Quality Anneal on {device} ===")
    print(f"  RoPE Base Theta: {args.rope_theta:,.0f} | Global Batch: 131,072 tokens (4 seqs 32k eq)")
    
    STAGE_C_CKPT_DIR.mkdir(parents=True, exist_ok=True)
    sp = spm.SentencePieceProcessor()
    sp.load(str(TOKENIZER_MODEL))
    
    eval_2m = Fixed2MEvaluator(SHARDS_DIR, device=device)
    val_pg19 = np.memmap(SHARDS_DIR / "val_pg19_long.bin", dtype=np.uint16, mode="r")
    val_wiki = np.memmap(SHARDS_DIR / "val_wikipedia_long.bin", dtype=np.uint16, mode="r")
    val_fw = np.memmap(SHARDS_DIR / "val_fineweb_edu_long.bin", dtype=np.uint16, mode="r")
    val_8k_shards = {"pg19": val_pg19, "wiki": val_wiki, "fineweb": val_fw}
    
    model = create_model(device=device, rope_theta=args.rope_theta)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5.0e-4, betas=(0.9, 0.95), weight_decay=0.1, fused=True)
    loader = StageCLoader(shards_dir=SHARDS_DIR, seed=1234)
    
    run_preflight_regression_test(model, loader, device=device)
    
    # Check for resume
    latest_ckpt = STAGE_C_CKPT_DIR / "rolling_stageC_latest.pt"
    start_step = 0
    accum_tokens = 0
    total_pause_time = 0.0
    total_pauses = 0
    spike_event_count = 0
    
    if latest_ckpt.exists():
        print(f"Found rolling checkpoint {latest_ckpt}. Resuming Stage C...", flush=True)
        ckpt = torch.load(latest_ckpt, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        loader.set_state(ckpt["loader_state"])
        start_step = ckpt["step"]
        accum_tokens = ckpt["accum_tokens"]
        total_pause_time = ckpt.get("total_pause_time", 0.0)
        total_pauses = ckpt.get("total_pauses", 0)
        spike_event_count = ckpt.get("spike_event_count", 0)
        print(f"Resumed from Step {start_step} ({accum_tokens/1e6:.1f}M tokens).")
    else:
        print(f"Loading initial weights from undecayed Stage B: {SRC_CHECKPOINT}...")
        raw = torch.load(SRC_CHECKPOINT, map_location=device, weights_only=False)
        model.load_state_dict(raw["model_state"] if "model_state" in raw else raw)
        print("Loaded Stage B weights cleanly.")
        
    total_steps = 875  # 875 steps * 131,072 tokens = 114,688,000 tokens (~114.7M)
    c1_steps = 437     # C1: steps 0 to 436 (57.28M tokens)
    c2_steps = 438     # C2: steps 437 to 874 (57.41M tokens)
    
    if not TRAIN_LOG_CSV.exists():
        with open(TRAIN_LOG_CSV, "w", encoding="utf-8") as f:
            f.write("step,tokens_m,train_loss,lr,grad_norm,active_tok_s,active_time_min,wall_time_min,paused_time_min,temp_gpu,mem_used_mb,power_w\n")
            
    recent_losses = []
    bucket_16_32k_history = []
    last_ckpt_time = time.time()
    t_start_wall = time.time()
    t_active_accum = 0.0
    
    model.train()
    optimizer.zero_grad(set_to_none=True)
    
    print(f"\nStarting Stage C Training (Steps {start_step + 1} to {total_steps})...\n", flush=True)
    
    for step in range(start_step, total_steps):
        t_step_start = time.time()
        
        # Sub-stage configuration
        if step < c1_steps:
            sub_stage = "C1"
            curr_lr = 2.5e-4 if step == 0 else 5.0e-4
            # C1: 3 long micro-batches (98,304 tok = 75%) + 4 replay micro-batches (32,768 tok = 25%)
            num_long_mb = 3
            num_replay_mb = 4
        else:
            sub_stage = "C2"
            c2_step = step - c1_steps
            coeff = 0.5 * (1.0 + math.cos(math.pi * c2_step / c2_steps))
            curr_lr = 1.0e-4 + coeff * (5.0e-4 - 1.0e-4)
            # C2: 2 long micro-batches (65,536 tok = 50%) + 8 replay micro-batches (65,536 tok = 50%)
            num_long_mb = 2
            num_replay_mb = 8
            
        for pg in optimizer.param_groups:
            pg["lr"] = curr_lr
            
        step_loss = 0.0
        # Long micro-batches (32,768 tokens each)
        for _ in range(num_long_mb):
            x, y = loader.get_long_sequence(device=device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                out = model(x)
                loss = F.cross_entropy(out.logits.reshape(-1, 8192), y.reshape(-1))
                scaled = loss / 4.0  # 4 sequence units of 32k per step
            scaled.backward()
            step_loss += (loss.item() * 32768)
            
        # Short replay micro-batches (4 seqs x 2,048 tokens = 8,192 tokens each)
        for _ in range(num_replay_mb):
            x, y = loader.get_replay_batch(batch_size=4, sub_stage=sub_stage, device=device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                out = model(x)
                loss = F.cross_entropy(out.logits.reshape(-1, 8192), y.reshape(-1))
                scaled = loss / 16.0  # 16 units of 8k = 131,072
            scaled.backward()
            step_loss += (loss.item() * 8192)
            
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        
        t_step_active = time.time() - t_step_start
        t_active_accum += t_step_active
        norm_step_loss = step_loss / 131072.0
        accum_tokens += 131072
        
        # Loss Spike Guard
        if math.isnan(norm_step_loss) or math.isinf(norm_step_loss):
            print(f"FATAL: NaN/Inf loss encountered at step {step+1}. Halting run!", flush=True)
            sys.exit(1)
            
        if len(recent_losses) >= 10:
            ma = np.mean(recent_losses[-20:])
            if norm_step_loss > 3.0 * ma or norm_step_loss > 8.0:
                spike_event_count += 1
                print(f"\n[LOSS SPIKE DETECTED] Step {step+1}: Loss {norm_step_loss:.4f} > 3x MA ({ma:.4f}). Spike event #{spike_event_count}.", flush=True)
                if spike_event_count >= 2:
                    print("FATAL: Second loss spike encountered within Stage C. Halting as per protocol!", flush=True)
                    sys.exit(1)
                else:
                    print("Rolling back to latest rolling checkpoint...", flush=True)
                    if latest_ckpt.exists():
                        ckpt = torch.load(latest_ckpt, map_location=device, weights_only=False)
                        model.load_state_dict(ckpt["model_state"])
                        optimizer.load_state_dict(ckpt["optimizer_state"])
                        loader.set_state(ckpt["loader_state"])
                        continue
        recent_losses.append(norm_step_loss)
        
        # Thermal Governor
        p_dur, p_cnt = apply_temperature_governor(step + 1, accum_tokens, "StageC")
        total_pause_time += p_dur
        total_pauses += p_cnt
        
        # Logging every 10 steps
        if (step + 1) % 10 == 0 or step == total_steps - 1 or (step + 1) == c1_steps:
            step_speed = 131072 / t_step_active if t_step_active > 0 else 0
            cur_temp, cur_mem, cur_pwr = get_gpu_telemetry()
            wall_m = (time.time() - t_start_wall) / 60.0
            act_m = t_active_accum / 60.0
            pause_m = total_pause_time / 60.0
            print(f"[Stage {sub_stage}] Step {step+1:3d}/{total_steps} | Loss: {norm_step_loss:.4f} | LR: {curr_lr:.2e} | Tokens: {accum_tokens/1e6:5.1f}M | Speed: {step_speed:5.0f} tok/s | VRAM: {cur_mem:.0f}MB | Temp: {cur_temp:.0f}C", flush=True)
            with open(TRAIN_LOG_CSV, "a", encoding="utf-8") as f:
                f.write(f"{step+1},{accum_tokens/1e6:.2f},{norm_step_loss:.4f},{curr_lr:.6e},{grad_norm:.4f},{step_speed:.1f},{act_m:.2f},{wall_m:.2f},{pause_m:.2f},{cur_temp:.1f},{cur_mem:.1f},{cur_pwr:.1f}\n")
                
        # 15-minute rolling checkpoint
        now = time.time()
        if (now - last_ckpt_time) >= 900 or (step + 1) == total_steps or (step + 1) == c1_steps:
            save_path = STAGE_C_CKPT_DIR / f"rolling_stageC_step_{step+1}.pt"
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

        # Track 16k-32k bucket over last quarter of C1 (steps 328 to 437)
        if step >= 328 and (step + 1) % 25 == 0 and step < c1_steps:
            b_probe = evaluate_position_buckets(model, val_pg19, max_ctx=32768, num_sequences=5, device=device)
            bucket_16_32k_history.append((step + 1, b_probe["16k-32k"]))
            print(f"  [Probe Step {step+1}] PG-19 16k-32k loss: {b_probe['16k-32k']:.4f}", flush=True)

        # -------------------------------------------------------------------
        # End of Sub-stage C1: Milestone Evaluation & C1 Rule Verification
        # -------------------------------------------------------------------
        if (step + 1) == c1_steps:
            print(f"\n{'='*75}\n[END OF STAGE C1] Step {c1_steps}/{total_steps} ({accum_tokens/1e6:.1f}M tokens)\n{'='*75}", flush=True)
            c1_ckpt = STAGE_C_CKPT_DIR / "stageC1_undecayed_final.pt"
            torch.save({
                "step": c1_steps,
                "accum_tokens": accum_tokens,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "loader_state": loader.get_state(),
            }, c1_ckpt)
            print(f"Saved undecayed C1 checkpoint to {c1_ckpt}!", flush=True)
            
            print("\n--- Running Comprehensive C1 Evaluation ---", flush=True)
            # 1. 2k Val CE
            c1_ce2k = eval_2m.evaluate_all(model)
            print(f"  2k Val CE: Wiki={c1_ce2k['val_wikipedia_neutral']:.4f}, FW={c1_ce2k['val_fineweb_edu']:.4f}, Cosmo={c1_ce2k['val_cosmopedia']:.4f}, Tiny={c1_ce2k['val_tinystories']:.4f}")
            
            # 2. Position buckets up to 32k
            c1_b_pg19 = evaluate_position_buckets(model, val_pg19, max_ctx=32768, num_sequences=10, device=device)
            c1_b_wiki = evaluate_position_buckets(model, val_wiki, max_ctx=8192, num_sequences=10, device=device)
            c1_b_fw = evaluate_position_buckets(model, val_fw, max_ctx=8192, num_sequences=10, device=device)
            bucket_16_32k_history.append((c1_steps, c1_b_pg19["16k-32k"]))
            print(f"  PG-19 Buckets: [1k-2k]={c1_b_pg19['1k-2k']:.4f} | [2k-4k]={c1_b_pg19['2k-4k']:.4f} | [4k-8k]={c1_b_pg19['4k-8k']:.4f} | [8k-16k]={c1_b_pg19['8k-16k']:.4f} | [16k-32k]={c1_b_pg19['16k-32k']:.4f}")
            
            # 3. Context-use at 8k and 32k
            c1_cu8k = evaluate_context_use_8k(model, val_8k_shards, num_windows=50, device=device)
            print(f"  Context-Use 8k Deltas: PG-19={c1_cu8k['pg19_delta']:.4f}, Wiki={c1_cu8k['wiki_delta']:.4f}, FW={c1_cu8k['fineweb_delta']:.4f}")
            
            c1_cu32k = evaluate_context_use_32k(model, val_pg19, num_windows=50, device=device)
            print(f"  Context-Use 32k: 32k={c1_cu32k['pg19_ctx_32k']:.4f}, 8k={c1_cu32k['pg19_ctx_8k']:.4f}, 2k={c1_cu32k['pg19_ctx_2k']:.4f} | Delta (32k - 8k)={c1_cu32k['pg19_delta_32k_8k']:+.4f} CE")
            
            # 4. Passkey (informational)
            c1_passkey = run_passkey_retrieval(model, sp, val_pg19, lengths=PASSKEY_LENGTHS, trials_per_length=20, device=device)
            print(f"  Passkey (Informational): 1k={c1_passkey['passkey_1k_acc']}%, 2k={c1_passkey['passkey_2k_acc']}%, 4k={c1_passkey['passkey_4k_acc']}%, 8k={c1_passkey['passkey_8k_acc']}%, 16k={c1_passkey['passkey_16k_acc']}%, 32k={c1_passkey['passkey_32k_acc']}%")
            
            # Save C1 Milestone Metrics
            df_c1 = pd.DataFrame([{
                "model": "StageC1_Undecayed_Milestone",
                "tokens_m": round(accum_tokens / 1e6, 2),
                "val_wiki_2k_ce": c1_ce2k["val_wikipedia_neutral"],
                "val_fineweb_2k_ce": c1_ce2k["val_fineweb_edu"],
                "val_cosmo_2k_ce": c1_ce2k["val_cosmopedia"],
                "val_tinystories_2k_ce": c1_ce2k["val_tinystories"],
                "pg19_1k_2k": c1_b_pg19["1k-2k"],
                "pg19_2k_4k": c1_b_pg19["2k-4k"],
                "pg19_4k_8k": c1_b_pg19["4k-8k"],
                "pg19_8k_16k": c1_b_pg19["8k-16k"],
                "pg19_16k_32k": c1_b_pg19["16k-32k"],
                **c1_cu8k,
                **c1_cu32k,
                **c1_passkey,
            }])
            df_c1.to_csv(RESULTS_DIR / "stageC1_eval.csv", index=False)
            print(f"  Saved C1 milestone metrics to {RESULTS_DIR / 'stageC1_eval.csv'}")
            
            # C1 Rule Verification:
            # Rule: if full-32k context-use > 8k-only + 0.02, OR 16k-32k bucket still dropping steeply over last quarter of C1 -> STOP!
            delta_32k_8k = c1_cu32k["pg19_delta_32k_8k"]
            is_delta_violated = delta_32k_8k > 0.02
            
            # Check slope over last quarter of C1
            is_slope_steep = False
            if len(bucket_16_32k_history) >= 2:
                first_probe = bucket_16_32k_history[0][1]
                last_probe = bucket_16_32k_history[-1][1]
                slope_drop = first_probe - last_probe
                # Steep drop defined as > 0.08 CE drop over last quarter (~14M tokens)
                if slope_drop > 0.08:
                    is_slope_steep = True
                    print(f"  Notice: 16k-32k loss dropped by {slope_drop:.4f} CE over last quarter of C1.", flush=True)
                    
            if is_delta_violated or is_slope_steep:
                reason = "32k Context-Use Delta > +0.02" if is_delta_violated else "16k-32k Loss Still Dropping Steeply"
                print(f"\n[!CRITICAL] C1 Dynamic Extension Rule Triggered ({reason})!", flush=True)
                print(f"  Delta (32k - 8k): {delta_32k_8k:+.4f} CE (Ceiling: +0.02)")
                print("  Halting Stage C as per protocol. Checkpoint saved. Reporting to user for decision.\n", flush=True)
                sys.exit(0)
            else:
                print(f"\n[C1 RULE PASSED] 32k Context-Use Delta ({delta_32k_8k:+.4f} CE <= +0.02) and Loss Slope Stable. Proceeding to C2 Quality Anneal!\n", flush=True)
                
            model.train()

    # -----------------------------------------------------------------------
    # End of Stage C (C2 Complete): Final Decayed Checkpoint & Full Evaluation
    # -----------------------------------------------------------------------
    print(f"\n{'='*75}\n[STAGE C TRAINING COMPLETE] Step {total_steps}/{total_steps} (114.69M tokens)\n{'='*75}", flush=True)
    final_decayed_ckpt = STAGE_C_CKPT_DIR / "stageC_decayed_final.pt"
    torch.save({
        "step": total_steps,
        "accum_tokens": accum_tokens,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
    }, final_decayed_ckpt)
    print(f"Saved Final Decayed Stage C Checkpoint to {final_decayed_ckpt}!\n", flush=True)
    
    print("--- Running Final Comprehensive Evaluation Suite on Decayed Stage C Model ---", flush=True)
    ce_2k = eval_2m.evaluate_all(model)
    print(f"  2k Val CE: Wiki={ce_2k['val_wikipedia_neutral']:.4f}, FW={ce_2k['val_fineweb_edu']:.4f}, Cosmo={ce_2k['val_cosmopedia']:.4f}, Tiny={ce_2k['val_tinystories']:.4f}")
    
    b_pg19 = evaluate_position_buckets(model, val_pg19, max_ctx=32768, num_sequences=10, device=device)
    b_wiki = evaluate_position_buckets(model, val_wiki, max_ctx=8192, num_sequences=10, device=device)
    b_fw = evaluate_position_buckets(model, val_fw, max_ctx=8192, num_sequences=10, device=device)
    print(f"  PG-19 Buckets: [1k-2k]={b_pg19['1k-2k']:.4f} | [2k-4k]={b_pg19['2k-4k']:.4f} | [4k-8k]={b_pg19['4k-8k']:.4f} | [8k-16k]={b_pg19['8k-16k']:.4f} | [16k-32k]={b_pg19['16k-32k']:.4f}")
    
    cu_8k = evaluate_context_use_8k(model, val_8k_shards, num_windows=50, device=device)
    print(f"  Context-Use 8k Deltas: PG-19={cu_8k['pg19_delta']:.4f}, Wiki={cu_8k['wiki_delta']:.4f}, FW={cu_8k['fineweb_delta']:.4f}")
    
    cu_32k = evaluate_context_use_32k(model, val_pg19, num_windows=50, device=device)
    print(f"  Context-Use 32k: 32k={cu_32k['pg19_ctx_32k']:.4f}, 8k={cu_32k['pg19_ctx_8k']:.4f}, 2k={cu_32k['pg19_ctx_2k']:.4f} | Delta (32k - 8k)={cu_32k['pg19_delta_32k_8k']:+.4f} CE")
    
    passkey_full = run_passkey_retrieval(model, sp, val_pg19, lengths=PASSKEY_LENGTHS, trials_per_length=20, device=device)
    print(f"  Passkey (Informational): 1k={passkey_full['passkey_1k_acc']}%, 2k={passkey_full['passkey_2k_acc']}%, 4k={passkey_full['passkey_4k_acc']}%, 8k={passkey_full['passkey_8k_acc']}%, 16k={passkey_full['passkey_16k_acc']}%, 32k={passkey_full['passkey_32k_acc']}%")
    
    samples = generate_samples(model, sp, device=device)
    with open(SAMPLES_MD, "w", encoding="utf-8") as f:
        f.write("# Stage C Decayed Model (32k Context) Generations\n\n")
        for s_idx, s in enumerate(samples, 1):
            f.write(f"### Prompt {s_idx}: `{s['prompt']}`\n\n```text\n{s['completion']}\n```\n\n---\n\n")
            
    df_eval = pd.DataFrame([{
        "model": "StageC_Decayed_Final_32k",
        "tokens_m": round(accum_tokens / 1e6, 2),
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
        "wiki_4k_8k": b_wiki["4k-8k"],
        "fineweb_4k_8k": b_fw["4k-8k"],
        **cu_8k,
        **cu_32k,
        **passkey_full,
    }])
    df_eval.to_csv(EVAL_LOG_CSV, index=False)
    print(f"Saved Stage C evaluation metrics to {EVAL_LOG_CSV}!\n")

if __name__ == "__main__":
    main()
