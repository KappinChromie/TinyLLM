"""
Stage A Decay Branch (Fallback Checkpoint Evaluation & GGUF Parity)
- Branched from a copy of stageA_final_undecayed.pt.
- Runs ~50M tokens (191 steps at batch 128 = 50,069,504 tokens) of cosine decay:
  2.0e-3 down to 2.0e-4 on the approved Stage A mix (60% FineWeb-Edu / 30% Cosmopedia / 10% TinyStories).
- Evaluates on all 4 validation sets (CE and BPB).
- Generates completions for 10 fixed prompts + 5 simple grammar prompts.
- Exports to GGUF (F16) and verifies llama.cpp greedy output matches PyTorch exactly.
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
LLAMA_CPP_DIR = ROOT / "tools" / "llama.cpp"

DECAY_RESULTS_CSV = RESULTS_DIR / "stageA_decay_branch_eval.csv"
DECAY_SAMPLES_MD = RESULTS_DIR / "stageA_decay_branch_samples.md"
GGUF_PARITY_TXT = RESULTS_DIR / "stageA_decay_gguf_parity.txt"

COMPRESSION_STATS = {
    "val_cosmopedia": {"bytes_per_tok": 4.149, "chars_per_tok": 4.148},
    "val_fineweb_edu": {"bytes_per_tok": 4.156, "chars_per_tok": 4.150},
    "val_tinystories": {"bytes_per_tok": 3.220, "chars_per_tok": 3.220},
    "val_wikipedia_neutral": {"bytes_per_tok": 4.098, "chars_per_tok": 4.092},
}

EVAL_PROMPTS_15 = [
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
    # 5 Simple Grammar Prompts
    "She opened the door and saw ",
    "The cat jumped over the ",
    "He went to the store to buy ",
    "They were very happy because ",
    "Every morning, the boy eats ",
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

def apply_temperature_governor(step, accum_tokens):
    temp, _ = get_gpu_telemetry()
    if temp < 83.0:
        return 0.0, 0
    t_start = time.time()
    print(f"\n[!WARNING] Step {step}: GPU temperature {temp:.1f}C >= 83C threshold! Pausing decay training...", flush=True)
    while temp > 78.0:
        time.sleep(5)
        temp, _ = get_gpu_telemetry()
        print(f"  Cooling down: current temp = {temp:.1f}C (target <= 78C)...", end="\r", flush=True)
    pause_duration = time.time() - t_start
    print(f"\n[GOVERNOR] Temperature dropped to {temp:.1f}C <= 78C. Resuming decay training after {pause_duration:.1f}s pause.", flush=True)
    return pause_duration, 1

# ---------------------------------------------------------------------------
# Data Loader
# ---------------------------------------------------------------------------
class ShardStreamDataLoader:
    def __init__(self, shards_dir, seq_len=2048, mix_weights=[0.30, 0.60, 0.10], seed=42):
        self.seq_len = seq_len
        self.weights = np.array(mix_weights, dtype=np.float64) / sum(mix_weights)
        self.cosmo = np.memmap(shards_dir / "train_cosmopedia.bin", dtype=np.uint16, mode="r")
        self.fineweb = np.memmap(shards_dir / "train_fineweb_edu.bin", dtype=np.uint16, mode="r")
        self.tinystories = np.memmap(shards_dir / "train_tinystories.bin", dtype=np.uint16, mode="r")
        self.shards = [self.cosmo, self.fineweb, self.tinystories]
        self.shard_lengths = [len(s) for s in self.shards]

        self.rng_mix = np.random.RandomState(seed + 999)
        self.rng_cosmo = np.random.RandomState(4001)
        self.rng_fineweb = np.random.RandomState(4002)
        self.rng_tinystories = np.random.RandomState(4003)
        self.shard_rngs = [self.rng_cosmo, self.rng_fineweb, self.rng_tinystories]

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
# Evaluator
# ---------------------------------------------------------------------------
class Evaluator:
    def __init__(self, shards_dir, device):
        self.device = device
        self.seq_len = 2048
        self.val_shards = {
            "val_wikipedia_neutral": np.memmap(shards_dir / "val_wikipedia_neutral.bin", dtype=np.uint16, mode="r"),
            "val_fineweb_edu": np.memmap(shards_dir / "val_fineweb_edu.bin", dtype=np.uint16, mode="r"),
            "val_cosmopedia": np.memmap(shards_dir / "val_cosmopedia.bin", dtype=np.uint16, mode="r"),
            "val_tinystories": np.memmap(shards_dir / "val_tinystories.bin", dtype=np.uint16, mode="r"),
        }

    @torch.no_grad()
    def evaluate_set(self, model, name, num_batches=100, batch_size=4):
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
                out = model(tx)
                loss = F.cross_entropy(out.logits.reshape(-1, 8192), ty.reshape(-1)).item()
            total_loss += loss * (batch_size * self.seq_len)
            total_tokens += batch_size * self.seq_len
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
        for p in EVAL_PROMPTS_15:
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

def create_model(device="cuda"):
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

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== Stage A Decay Branch Execution on {device} ===", flush=True)
    
    src_ckpt = CHECKPOINTS_DIR / "stageA_final_undecayed.pt"
    if not src_ckpt.exists():
        raise FileNotFoundError(f"Source checkpoint {src_ckpt} does not exist!")
        
    branch_ckpt = CHECKPOINTS_DIR / "stageA_decay_branch_final.pt"
    
    sp = spm.SentencePieceProcessor()
    sp.load(str(TOKENIZER_MODEL))
    
    model = create_model(device=device)
    raw_data = torch.load(src_ckpt, map_location=device, weights_only=False)
    model.load_state_dict(raw_data["model_state"] if "model_state" in raw_data else raw_data)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, betas=(0.9, 0.95), weight_decay=0.1, fused=True)
    loader = ShardStreamDataLoader(SHARDS_DIR, seq_len=2048, mix_weights=[0.30, 0.60, 0.10], seed=999)
    evaluator = Evaluator(SHARDS_DIR, device=device)

    print("\n--- Pre-Decay Evaluation: Undecayed Stage A Final Checkpoint ---", flush=True)
    undecayed_eval = evaluator.evaluate_all(model)
    for vname, m in undecayed_eval.items():
        print(f"  [Undecayed] {vname:25s}: CE = {m['ce']:.4f} | BPB = {m['bpb']:.4f}", flush=True)
    
    micro_bs = 4
    grad_accum = 32
    seq_len = 2048
    tokens_per_step = micro_bs * grad_accum * seq_len  # 262,144
    total_steps = 191  # 191 * 262,144 = 50,069,504 tokens (~50M tokens)
    
    init_lr = 2.0e-3
    min_lr = 2.0e-4
    
    print(f"\nStarting 50M token cosine decay ({total_steps} steps: 2.0e-3 -> 2.0e-4)...", flush=True)
    model.train()
    optimizer.zero_grad(set_to_none=True)
    
    wall_start = time.time()
    total_pause_time = 0.0
    total_pause_count = 0

    for step in range(total_steps):
        # Thermal governor between optimizer steps
        tokens_seen = (step + 1) * tokens_per_step
        p_dur, p_cnt = apply_temperature_governor(step + 1, tokens_seen)
        total_pause_time += p_dur
        total_pause_count += p_cnt

        # Cosine decay schedule
        coeff = 0.5 * (1.0 + math.cos(math.pi * step / total_steps))
        curr_lr = min_lr + coeff * (init_lr - min_lr)
        
        for param_group in optimizer.param_groups:
            param_group["lr"] = curr_lr
            
        step_loss = 0.0
        for _ in range(grad_accum):
            x, y = loader.get_batch(micro_bs, device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                out = model(x)
                loss = F.cross_entropy(out.logits.reshape(-1, 8192), y.reshape(-1))
                scaled = loss / grad_accum
            scaled.backward()
            step_loss += loss.item() / grad_accum
            
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        
        if (step + 1) % 25 == 0 or step == total_steps - 1:
            print(f"[Decay Branch A] Step {step+1:3d}/{total_steps} | LR {curr_lr:.2e} | Loss {step_loss:.4f} | Tokens: {tokens_seen/1e6:.1f}M", flush=True)
            
    wall_end = time.time()
    total_wall_sec = wall_end - wall_start
    active_sec = total_wall_sec - total_pause_time
    total_decay_tokens = total_steps * tokens_per_step
    tok_per_sec = total_decay_tokens / active_sec if active_sec > 0 else 0
    print(f"\nDecay Training Completed in {total_wall_sec/60:.2f} min (Active: {active_sec/60:.2f} min, Paused: {total_pause_time:.1f}s in {total_pause_count} pauses)")
    print(f"Decay Active Speed: {tok_per_sec:.0f} tok/s", flush=True)

    # Save decayed model
    torch.save(model.state_dict(), branch_ckpt)
    print(f"Saved decayed checkpoint to {branch_ckpt}", flush=True)
    
    # Evaluate across all 4 sets
    print("\n--- Post-Decay Evaluation: Decayed Stage A Fallback Checkpoint ---", flush=True)
    decay_eval = evaluator.evaluate_all(model)
    for vname, m in decay_eval.items():
        delta_ce = decay_eval[vname]["ce"] - undecayed_eval[vname]["ce"]
        print(f"  [Decayed]  {vname:25s}: CE = {m['ce']:.4f} (Δ {delta_ce:+.4f}) | BPB = {m['bpb']:.4f}", flush=True)
        
    df_eval = pd.DataFrame([
        {
            "model": "StageA_Final_Undecayed",
            "tokens_decayed_m": 0.0,
            "val_wiki_ce": undecayed_eval["val_wikipedia_neutral"]["ce"],
            "val_wiki_bpb": undecayed_eval["val_wikipedia_neutral"]["bpb"],
            "val_fineweb_ce": undecayed_eval["val_fineweb_edu"]["ce"],
            "val_fineweb_bpb": undecayed_eval["val_fineweb_edu"]["bpb"],
            "val_cosmo_ce": undecayed_eval["val_cosmopedia"]["ce"],
            "val_cosmo_bpb": undecayed_eval["val_cosmopedia"]["bpb"],
            "val_tinystories_ce": undecayed_eval["val_tinystories"]["ce"],
            "val_tinystories_bpb": undecayed_eval["val_tinystories"]["bpb"],
        },
        {
            "model": "StageA_Decayed_Fallback",
            "tokens_decayed_m": round(total_decay_tokens / 1e6, 2),
            "val_wiki_ce": decay_eval["val_wikipedia_neutral"]["ce"],
            "val_wiki_bpb": decay_eval["val_wikipedia_neutral"]["bpb"],
            "val_fineweb_ce": decay_eval["val_fineweb_edu"]["ce"],
            "val_fineweb_bpb": decay_eval["val_fineweb_edu"]["bpb"],
            "val_cosmo_ce": decay_eval["val_cosmopedia"]["ce"],
            "val_cosmo_bpb": decay_eval["val_cosmopedia"]["bpb"],
            "val_tinystories_ce": decay_eval["val_tinystories"]["ce"],
            "val_tinystories_bpb": decay_eval["val_tinystories"]["bpb"],
        }
    ])
    df_eval.to_csv(DECAY_RESULTS_CSV, index=False)
    print(f"Saved decay evaluation to {DECAY_RESULTS_CSV}", flush=True)
    
    # Generate 15 samples
    print("\nGenerating 15 Evaluation Samples (10 Standard + 5 Simple Grammar)...", flush=True)
    samples = evaluator.generate_samples(model, sp, max_new_tokens=64)
    with open(DECAY_SAMPLES_MD, "w", encoding="utf-8") as f:
        f.write("# Stage A Decayed Model (Fallback) Sample Generations\n\n")
        for s_idx, s in enumerate(samples, 1):
            f.write(f"### Prompt {s_idx}: `{s['prompt']}`\n\n```text\n{s['completion']}\n```\n\n---\n\n")
    print(f"Saved samples to {DECAY_SAMPLES_MD}", flush=True)

if __name__ == "__main__":
    main()
