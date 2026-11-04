"""
Step 2: 3-Hour Head-to-Head Comparison: 4L_768H_1KV vs 6L_768H_1KV
Curriculum: Stage A Mix (45% Cosmopedia, 45% FineWeb-Edu, 10% TinyStories)
Context: 2,048 tokens, Global Batch: 32 seqs (65,536 tokens)
Optimizer: AdamW (lr=1e-3, betas=(0.9, 0.95), weight_decay=0.1)
Schedule: WSD (100-step warmup, flat 1e-3 for 3 hours)
Milestones: Checkpoints saved at 250M, 500M, 750M tokens, plus 3h end-of-run
Decay Branches: From each milestone + 3h end, branch short cosine decay (381 steps / ~25M tokens) to 0.1x LR
Telemetry: Polls GPU clocks, power, temperature every 60s
Validation: Evaluates on val_cosmopedia, val_fineweb_edu, val_tinystories, val_wikipedia_neutral
Outputs: Summary table, gap table, curves CSV, Pareto plots, and sample generations.
"""

import os
import sys
import copy
import math
import time
import shutil
import subprocess
import threading
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from transformers import LlamaConfig, LlamaForCausalLM
import sentencepiece as spm

sys.stdout.reconfigure(line_buffering=True)

ROOT = Path(__file__).resolve().parent.parent
SHARDS_DIR = ROOT / "data" / "shards"
RESULTS_DIR = ROOT / "results"
MODELS_DIR = ROOT / "models"
CHECKPOINTS_DIR = MODELS_DIR / "checkpoints"
TOKENIZER_MODEL = MODELS_DIR / "tokenizer_8k" / "tokenizer.model"
RESULTS_DIR.mkdir(exist_ok=True)
CHECKPOINTS_DIR.mkdir(exist_ok=True)

# Output paths
TELEMETRY_CSV = RESULTS_DIR / "head_to_head_telemetry.csv"
CURVES_CSV = RESULTS_DIR / "head_to_head_curves.csv"
SUMMARY_CSV = RESULTS_DIR / "head_to_head_summary.csv"
GAP_TABLE_CSV = RESULTS_DIR / "head_to_head_gap_table.csv"
SAMPLES_MD = RESULTS_DIR / "head_to_head_sample_generations.md"
REPORT_MD = RESULTS_DIR / "head_to_head_report.md"

PLOT_LOSS_TOKENS = RESULTS_DIR / "head_to_head_loss_vs_tokens.png"
PLOT_LOSS_TIME = RESULTS_DIR / "head_to_head_loss_vs_time.png"
PLOT_PARETO_0K = RESULTS_DIR / "head_to_head_pareto_0k.png"
PLOT_PARETO_32K = RESULTS_DIR / "head_to_head_pareto_32k.png"

# Empirical noise thresholds established in Phase 5A
NOISE_FLOOR_VAL = 0.0076   # 2-seed validation spread on Stage A mix
NOISE_FLOOR_TRAIN = 0.0254 # 3-seed training spread

# Baseline decode throughputs (tokens/sec) measured in Phase 1 & Phase 5A
SPEED_BENCHMARKS = {
    "4L_768H_1KV": {"tg_0k": 1514.8, "tg_32k": 1356.4},
    "6L_768H_1KV": {"tg_0k": 1326.5, "tg_32k": 1180.7},
}

# Model shape definitions
MODELS_TO_RUN = {
    "4L_768H_1KV": {
        "num_hidden_layers": 4,
        "hidden_size": 768,
        "intermediate_size": 2048,
        "num_attention_heads": 12,
        "num_key_value_heads": 1,
        "vocab_size": 8192,
        "params": 30.28,
    },
    "6L_768H_1KV": {
        "num_hidden_layers": 6,
        "hidden_size": 768,
        "intermediate_size": 2048,
        "num_attention_heads": 12,
        "num_key_value_heads": 1,
        "vocab_size": 8192,
        "params": 42.28,
    },
}

# Training Hyperparameters
MAIN_RUN_DURATION_SEC = 10800  # 3 hours = 10,800 seconds
SEQ_LEN = 2048
MICRO_BS = 8
GRAD_ACCUM_STEPS = 4
GLOBAL_BS_TOKENS = MICRO_BS * GRAD_ACCUM_STEPS * SEQ_LEN  # 65,536 tokens/step
PEAK_LR = 1e-3
MIN_LR = 1e-4
WARMUP_STEPS = 100
SHARED_MILESTONES = [250_000_000, 500_000_000, 750_000_000]
DECAY_BRANCH_STEPS = 381  # 381 steps * 65,536 tokens = ~25M tokens (cosine decay to 0.1x LR)

COMPRESSION_STATS = {
    "val_cosmopedia": {"bytes_per_tok": 3.478, "chars_per_tok": 3.474},
    "val_fineweb_edu": {"bytes_per_tok": 3.160, "chars_per_tok": 3.148},
    "val_tinystories": {"bytes_per_tok": 3.269, "chars_per_tok": 3.266},
    "val_wikipedia_neutral": {"bytes_per_tok": 2.813, "chars_per_tok": 2.797},
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
]

# ---------------------------------------------------------------------------
# Telemetry Monitor
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

class GPUTelemetry(threading.Thread):
    def __init__(self, log_path, interval_sec=60):
        super().__init__(daemon=True)
        self.log_path = log_path
        self.interval_sec = interval_sec
        self.stop_event = threading.Event()
        self.smi_bin = find_nvidia_smi()
        print(f"GPUTelemetry using binary: {self.smi_bin}")
        
        # Write CSV header if not exists
        if not self.log_path.exists():
            with open(self.log_path, "w", encoding="utf-8") as f:
                f.write("timestamp,temp_gpu,power_w,clock_sm_mhz,clock_mem_mhz,mem_used_mb\n")

    def run(self):
        while not self.stop_event.is_set():
            try:
                cmd = [
                    self.smi_bin,
                    "--query-gpu=timestamp,temperature.gpu,power.draw,clocks.current.graphics,clocks.current.memory,memory.used",
                    "--format=csv,noheader,nounits"
                ]
                res = subprocess.run(cmd, capture_output=True, text=True, check=True)
                line = res.stdout.strip().replace(", ", ",")
                with open(self.log_path, "a", encoding="utf-8") as f:
                    f.write(line + "\n")
            except Exception as e:
                pass
            self.stop_event.wait(self.interval_sec)

    def stop(self):
        self.stop_event.set()

# ---------------------------------------------------------------------------
# Data Loading & Validation
# ---------------------------------------------------------------------------
class StageAMixDataLoader:
    def __init__(self, shards_dir, seq_len):
        self.seq_len = seq_len
        cosmo_path = shards_dir / "train_cosmopedia.bin"
        fineweb_path = shards_dir / "train_fineweb_edu.bin"
        tinystories_path = shards_dir / "train_tinystories.bin"
        
        self.shards = [
            np.memmap(cosmo_path, dtype=np.uint16, mode="r"),
            np.memmap(fineweb_path, dtype=np.uint16, mode="r"),
            np.memmap(tinystories_path, dtype=np.uint16, mode="r"),
        ]
        self.weights = np.array([0.45, 0.45, 0.10], dtype=np.float64)
        self.shard_lengths = [len(s) for s in self.shards]
        print(f"Loaded Stage A Training Shards: Cosmopedia ({len(self.shards[0])/1e6:.1f}M), "
              f"FineWeb-Edu ({len(self.shards[1])/1e6:.1f}M), TinyStories ({len(self.shards[2])/1e6:.1f}M)")

    def get_batch(self, batch_size, rng, device):
        shard_indices = rng.choice(3, size=batch_size, p=self.weights)
        x_list, y_list = [], []
        for s_idx in shard_indices:
            shard = self.shards[s_idx]
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
            "val_cosmopedia": np.memmap(shards_dir / "val_cosmopedia.bin", dtype=np.uint16, mode="r"),
            "val_fineweb_edu": np.memmap(shards_dir / "val_fineweb_edu.bin", dtype=np.uint16, mode="r"),
            "val_tinystories": np.memmap(shards_dir / "val_tinystories.bin", dtype=np.uint16, mode="r"),
            "val_wikipedia_neutral": np.memmap(shards_dir / "val_wikipedia_neutral.bin", dtype=np.uint16, mode="r"),
        }
        
    @torch.no_grad()
    def evaluate_set(self, model, name, num_batches=30):
        shard = self.val_shards[name]
        max_idx = len(shard) - self.seq_len - 1
        rng = np.random.RandomState(4242)  # Consistent validation sample
        total_loss = 0.0
        total_tokens = 0
        batch_size = 8
        for _ in range(num_batches):
            offsets = [rng.randint(0, max_idx) for _ in range(batch_size)]
            x = np.stack([shard[i : i + self.seq_len] for i in offsets])
            y = np.stack([shard[i + 1 : i + 1 + self.seq_len] for i in offsets])
            tx = torch.from_numpy(x.astype(np.int64)).to(device=self.device, non_blocking=True)
            ty = torch.from_numpy(y.astype(np.int64)).to(device=self.device, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                outputs = model(tx, labels=ty)
                loss = outputs.loss.item()
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
        
    def evaluate_all(self, model):
        was_training = model.training
        model.eval()
        results = {}
        for name in ["val_cosmopedia", "val_fineweb_edu", "val_tinystories", "val_wikipedia_neutral"]:
            results[name] = self.evaluate_set(model, name)
            
        stage_a_val_loss = (
            0.45 * results["val_cosmopedia"]["ce_loss"] +
            0.45 * results["val_fineweb_edu"]["ce_loss"] +
            0.10 * results["val_tinystories"]["ce_loss"]
        )
        stage_a_bpb = (
            0.45 * results["val_cosmopedia"]["bpb"] +
            0.45 * results["val_fineweb_edu"]["bpb"] +
            0.10 * results["val_tinystories"]["bpb"]
        )
        results["stage_a_val_loss"] = round(stage_a_val_loss, 4)
        results["stage_a_bpb"] = round(stage_a_bpb, 4)
        if was_training:
            model.train()
        return results

# ---------------------------------------------------------------------------
# Branch Decay Routine
# ---------------------------------------------------------------------------
def run_decay_branch(base_model, loader, evaluator, device, steps=DECAY_BRANCH_STEPS):
    print(f"      [Decay Branch] Branching cosine decay ({steps} steps / ~{steps*GLOBAL_BS_TOKENS/1e6:.1f}M tokens to {MIN_LR:.1e})...", flush=True)
    branch_model = copy.deepcopy(base_model).to(device)
    branch_opt = torch.optim.AdamW(branch_model.parameters(), lr=PEAK_LR, betas=(0.9, 0.95), weight_decay=0.1, fused=True)
    branch_rng = np.random.RandomState(9999)  # Independent RNG so main run stream is pristine
    branch_model.train()
    
    t0 = time.time()
    for s in range(steps):
        prog = s / max(1, steps)
        curr_lr = MIN_LR + 0.5 * (PEAK_LR - MIN_LR) * (1.0 + math.cos(math.pi * prog))
        for pg in branch_opt.param_groups:
            pg["lr"] = curr_lr
            
        branch_opt.zero_grad(set_to_none=True)
        for _ in range(GRAD_ACCUM_STEPS):
            x, y = loader.get_batch(MICRO_BS, branch_rng, device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                outputs = branch_model(x, labels=y)
                loss = outputs.loss / GRAD_ACCUM_STEPS
            loss.backward()
        torch.nn.utils.clip_grad_norm_(branch_model.parameters(), 1.0)
        branch_opt.step()
        
    branch_time = time.time() - t0
    print(f"      [Decay Branch] Finished in {branch_time:.1f}s. Evaluating...", flush=True)
    metrics = evaluator.evaluate_all(branch_model)
    return branch_model, metrics

# ---------------------------------------------------------------------------
# Generation Helper
# ---------------------------------------------------------------------------
def generate_samples(model, sp, prompts, device, max_new_tokens=64):
    model.eval()
    completions = []
    with torch.no_grad():
        for p in prompts:
            toks = [1] + list(sp.encode(p))
            curr = list(toks)
            for _ in range(max_new_tokens):
                inp = torch.tensor([curr], dtype=torch.long, device=device)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    out = model(inp)
                tok = torch.argmax(out.logits[0, -1, :]).item()
                if tok == 2:
                    break
                curr.append(tok)
            gen_text = sp.decode(curr[1:])
            completions.append((p, gen_text))
    return completions

# ---------------------------------------------------------------------------
# Main Training Function for a Model
# ---------------------------------------------------------------------------
def train_model_head_to_head(model_name, shape_cfg, loader, evaluator, device, curves_records, summary_records):
    print("\n" + "=" * 78)
    print(f"  STARTING 3-HOUR HEAD-TO-HEAD TRIAL: {model_name} ({shape_cfg['params']:.2f}M Params)")
    print("=" * 78)
    
    cfg = LlamaConfig(
        vocab_size=shape_cfg["vocab_size"],
        hidden_size=shape_cfg["hidden_size"],
        intermediate_size=shape_cfg["intermediate_size"],
        num_hidden_layers=shape_cfg["num_hidden_layers"],
        num_attention_heads=shape_cfg["num_attention_heads"],
        num_key_value_heads=shape_cfg["num_key_value_heads"],
        tie_word_embeddings=True,
        torch_dtype="bfloat16",
        max_position_embeddings=2048,
        attn_implementation="sdpa",
    )
    
    # Strictly identical initial seed
    torch.manual_seed(42)
    np.random.seed(42)
    train_rng = np.random.RandomState(42)
    
    model = LlamaForCausalLM(cfg).to(device=device, dtype=torch.bfloat16)
    optimizer = torch.optim.AdamW(model.parameters(), lr=PEAK_LR, betas=(0.9, 0.95), weight_decay=0.1, fused=True)
    
    model.train()
    
    milestones_pending = list(SHARED_MILESTONES)
    total_tokens = 0
    step = 0
    elapsed_main = 0.0
    last_print = time.time()
    saved_3hr = False
    
    while (elapsed_main < MAIN_RUN_DURATION_SEC) or (milestones_pending and total_tokens < SHARED_MILESTONES[-1]):
        iter_start = time.time()
        
        # WSD Schedule: Warmup then completely flat PEAK_LR
        if step < WARMUP_STEPS:
            curr_lr = PEAK_LR * (step + 1) / WARMUP_STEPS
        else:
            curr_lr = PEAK_LR
            
        for pg in optimizer.param_groups:
            pg["lr"] = curr_lr
            
        optimizer.zero_grad(set_to_none=True)
        accum_loss = 0.0
        
        for _ in range(GRAD_ACCUM_STEPS):
            x, y = loader.get_batch(MICRO_BS, train_rng, device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                outputs = model(x, labels=y)
                loss = outputs.loss / GRAD_ACCUM_STEPS
            loss.backward()
            accum_loss += loss.item()
            
        # Stability monitoring: check for loss divergence
        if math.isnan(accum_loss) or accum_loss > 15.0:
            print(f"\n[FATAL] Divergence detected on {model_name} at step {step}: loss = {accum_loss:.4f}!", flush=True)
            break
        if step > 200 and accum_loss > 7.5:
            print(f"\n[WARNING] Loss spike on {model_name} at step {step}: loss = {accum_loss:.4f}!", flush=True)
            
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        iter_time = time.time() - iter_start
        elapsed_main += iter_time
        step += 1
        total_tokens += GLOBAL_BS_TOKENS
        
        # Log curve data
        if step % 25 == 0 or step == 1:
            tps = GLOBAL_BS_TOKENS / max(0.001, iter_time)
            curves_records.append({
                "model": model_name,
                "step": step,
                "elapsed_main_sec": round(elapsed_main, 1),
                "total_tokens": total_tokens,
                "train_loss": round(accum_loss, 4),
                "lr": curr_lr,
                "tok_per_sec": round(tps, 1),
            })
            if time.time() - last_print > 15.0 or step == 1:
                print(f"[{model_name}] Step {step:5d} | Time: {elapsed_main/60:5.1f}m/{MAIN_RUN_DURATION_SEC/60:.0f}m | "
                      f"Tokens: {total_tokens/1e6:6.1f}M | Loss: {accum_loss:.4f} | Tok/s: {tps:6.0f}", flush=True)
                last_print = time.time()
                
        # Check 3-Hour Wall-Clock Milestone
        if elapsed_main >= MAIN_RUN_DURATION_SEC and not saved_3hr:
            print(f"\n>>> Reached 3-Hour Wall-Clock Mark ({elapsed_main/3600:.2f}h, {total_tokens/1e6:.1f}M tokens) <<<", flush=True)
            undecayed_3h_ckpt = CHECKPOINTS_DIR / f"{model_name}_3hr_undecayed.pt"
            torch.save(model.state_dict(), undecayed_3h_ckpt)
            print(f"    Evaluating 3-Hour Undecayed Checkpoint...", flush=True)
            undec_3h_eval = evaluator.evaluate_all(model)
            summary_records.append({
                "model": model_name,
                "checkpoint": "3hr_undecayed",
                "tokens": total_tokens,
                "wall_clock_sec": round(elapsed_main, 1),
                "is_decayed": False,
                "stage_a_ce": undec_3h_eval["stage_a_val_loss"],
                "stage_a_bpb": undec_3h_eval["stage_a_bpb"],
                "wiki_ce": undec_3h_eval["val_wikipedia_neutral"]["ce_loss"],
                "wiki_bpb": undec_3h_eval["val_wikipedia_neutral"]["bpb"],
                "cosmo_ce": undec_3h_eval["val_cosmopedia"]["ce_loss"],
                "fineweb_ce": undec_3h_eval["val_fineweb_edu"]["ce_loss"],
                "tinystories_ce": undec_3h_eval["val_tinystories"]["ce_loss"],
            })
            
            decayed_3h_model, dec_3h_eval = run_decay_branch(model, loader, evaluator, device)
            decayed_3h_ckpt = CHECKPOINTS_DIR / f"{model_name}_3hr_decayed.pt"
            torch.save(decayed_3h_model.state_dict(), decayed_3h_ckpt)
            print(f"    Saved 3-Hour Decayed Checkpoint: {decayed_3h_ckpt.name}", flush=True)
            summary_records.append({
                "model": model_name,
                "checkpoint": "3hr_decayed",
                "tokens": total_tokens + DECAY_BRANCH_STEPS * GLOBAL_BS_TOKENS,
                "wall_clock_sec": round(elapsed_main, 1),
                "is_decayed": True,
                "stage_a_ce": dec_3h_eval["stage_a_val_loss"],
                "stage_a_bpb": dec_3h_eval["stage_a_bpb"],
                "wiki_ce": dec_3h_eval["val_wikipedia_neutral"]["ce_loss"],
                "wiki_bpb": dec_3h_eval["val_wikipedia_neutral"]["bpb"],
                "cosmo_ce": dec_3h_eval["val_cosmopedia"]["ce_loss"],
                "fineweb_ce": dec_3h_eval["val_fineweb_edu"]["ce_loss"],
                "tinystories_ce": dec_3h_eval["val_tinystories"]["ce_loss"],
            })
            saved_3hr = True
            
            # If 3h is the final milestone (750M already done), break now
            if not milestones_pending:
                print(f"    All milestones complete! Finalizing {model_name}...", flush=True)
                break
            else:
                print(f"    Continuing main run to complete remaining milestone ({milestones_pending[0]/1e6:.0f}M tokens)...\n", flush=True)
                del decayed_3h_model
                torch.cuda.empty_cache()

        # Check shared token milestones (250M, 500M, 750M)
        if milestones_pending and total_tokens >= milestones_pending[0]:
            target_tokens = milestones_pending.pop(0)
            tag = f"{int(target_tokens/1e6)}M"
            print(f"\n>>> Reached Milestone: {tag} ({total_tokens:,} tokens) at {elapsed_main/60:.2f}m <<<", flush=True)
            
            # Save undecayed checkpoint
            undecayed_ckpt = CHECKPOINTS_DIR / f"{model_name}_{tag}_undecayed.pt"
            torch.save(model.state_dict(), undecayed_ckpt)
            print(f"    Saved undecayed checkpoint: {undecayed_ckpt.name}", flush=True)
            
            # Evaluate undecayed
            print(f"    Evaluating undecayed model at {tag}...", flush=True)
            undec_eval = evaluator.evaluate_all(model)
            summary_records.append({
                "model": model_name,
                "checkpoint": f"{tag}_undecayed",
                "tokens": total_tokens,
                "wall_clock_sec": round(elapsed_main, 1),
                "is_decayed": False,
                "stage_a_ce": undec_eval["stage_a_val_loss"],
                "stage_a_bpb": undec_eval["stage_a_bpb"],
                "wiki_ce": undec_eval["val_wikipedia_neutral"]["ce_loss"],
                "wiki_bpb": undec_eval["val_wikipedia_neutral"]["bpb"],
                "cosmo_ce": undec_eval["val_cosmopedia"]["ce_loss"],
                "fineweb_ce": undec_eval["val_fineweb_edu"]["ce_loss"],
                "tinystories_ce": undec_eval["val_tinystories"]["ce_loss"],
            })
            
            # Branch decay
            decayed_model, dec_eval = run_decay_branch(model, loader, evaluator, device)
            decayed_ckpt = CHECKPOINTS_DIR / f"{model_name}_{tag}_decayed.pt"
            torch.save(decayed_model.state_dict(), decayed_ckpt)
            print(f"    Saved decayed checkpoint: {decayed_ckpt.name}", flush=True)
            summary_records.append({
                "model": model_name,
                "checkpoint": f"{tag}_decayed",
                "tokens": total_tokens + DECAY_BRANCH_STEPS * GLOBAL_BS_TOKENS,
                "wall_clock_sec": round(elapsed_main, 1),
                "is_decayed": True,
                "stage_a_ce": dec_eval["stage_a_val_loss"],
                "stage_a_bpb": dec_eval["stage_a_bpb"],
                "wiki_ce": dec_eval["val_wikipedia_neutral"]["ce_loss"],
                "wiki_bpb": dec_eval["val_wikipedia_neutral"]["bpb"],
                "cosmo_ce": dec_eval["val_cosmopedia"]["ce_loss"],
                "fineweb_ce": dec_eval["val_fineweb_edu"]["ce_loss"],
                "tinystories_ce": dec_eval["val_tinystories"]["ce_loss"],
            })
            del decayed_model
            torch.cuda.empty_cache()
            
            # If 3hr wall-clock was already reached earlier, and this was the last milestone, we are done
            if saved_3hr and not milestones_pending:
                print(f"    Milestone {tag} was final pending milestone. Finishing {model_name}...", flush=True)
                break
            else:
                print(f"    Resuming main flat training run...\n", flush=True)
            
    print(f"\nCompleted Main Run for {model_name}: {elapsed_main/3600:.2f} hours, {step} steps, {total_tokens/1e6:.1f}M tokens.")
    
    # In case 3hr was not triggered inside loop (e.g. slight timing boundary)
    if not saved_3hr:
        tag_3h = "3hr"
        undecayed_3h_ckpt = CHECKPOINTS_DIR / f"{model_name}_{tag_3h}_undecayed.pt"
        torch.save(model.state_dict(), undecayed_3h_ckpt)
        print(f"Evaluating 3-Hour Undecayed Checkpoint...", flush=True)
        undec_3h_eval = evaluator.evaluate_all(model)
        summary_records.append({
            "model": model_name,
            "checkpoint": f"{tag_3h}_undecayed",
            "tokens": total_tokens,
            "wall_clock_sec": round(elapsed_main, 1),
            "is_decayed": False,
            "stage_a_ce": undec_3h_eval["stage_a_val_loss"],
            "stage_a_bpb": undec_3h_eval["stage_a_bpb"],
            "wiki_ce": undec_3h_eval["val_wikipedia_neutral"]["ce_loss"],
            "wiki_bpb": undec_3h_eval["val_wikipedia_neutral"]["bpb"],
            "cosmo_ce": undec_3h_eval["val_cosmopedia"]["ce_loss"],
            "fineweb_ce": undec_3h_eval["val_fineweb_edu"]["ce_loss"],
            "tinystories_ce": undec_3h_eval["val_tinystories"]["ce_loss"],
        })
        
        decayed_3h_model, dec_3h_eval = run_decay_branch(model, loader, evaluator, device)
        decayed_3h_ckpt = CHECKPOINTS_DIR / f"{model_name}_{tag_3h}_decayed.pt"
        torch.save(decayed_3h_model.state_dict(), decayed_3h_ckpt)
        print(f"Saved 3-Hour Decayed Checkpoint: {decayed_3h_ckpt.name}", flush=True)
        summary_records.append({
            "model": model_name,
            "checkpoint": f"{tag_3h}_decayed",
            "tokens": total_tokens + DECAY_BRANCH_STEPS * GLOBAL_BS_TOKENS,
            "wall_clock_sec": round(elapsed_main, 1),
            "is_decayed": True,
            "stage_a_ce": dec_3h_eval["stage_a_val_loss"],
            "stage_a_bpb": dec_3h_eval["stage_a_bpb"],
            "wiki_ce": dec_3h_eval["val_wikipedia_neutral"]["ce_loss"],
            "wiki_bpb": dec_3h_eval["val_wikipedia_neutral"]["bpb"],
            "cosmo_ce": dec_3h_eval["val_cosmopedia"]["ce_loss"],
            "fineweb_ce": dec_3h_eval["val_fineweb_edu"]["ce_loss"],
            "tinystories_ce": dec_3h_eval["val_tinystories"]["ce_loss"],
        })
    else:
        # Load the 3hr decayed checkpoint to generate samples
        decayed_3h_model = LlamaForCausalLM(cfg).to(device=device, dtype=torch.bfloat16)
        decayed_3h_model.load_state_dict(torch.load(CHECKPOINTS_DIR / f"{model_name}_3hr_decayed.pt", map_location=device))
    
    # Generate 10 samples from the 3h decayed checkpoint
    sp = spm.SentencePieceProcessor(model_file=str(TOKENIZER_MODEL))
    samples = generate_samples(decayed_3h_model, sp, PROMPTS, device)
    
    del model, optimizer, decayed_3h_model
    torch.cuda.empty_cache()
    
    return samples

# ---------------------------------------------------------------------------
# Gap Analysis & Plots
# ---------------------------------------------------------------------------
def compute_gap_analysis(summary_df):
    gap_rows = []
    milestones = ["250M_undecayed", "250M_decayed", "500M_undecayed", "500M_decayed", 
                  "750M_undecayed", "750M_decayed", "3hr_undecayed", "3hr_decayed"]
    
    for m in milestones:
        sub = summary_df[summary_df["checkpoint"] == m]
        if len(sub) == 2:
            r4 = sub[sub["model"] == "4L_768H_1KV"].iloc[0]
            r6 = sub[sub["model"] == "6L_768H_1KV"].iloc[0]
            
            diff_stage_a_ce = r6["stage_a_ce"] - r4["stage_a_ce"]
            diff_stage_a_bpb = r6["stage_a_bpb"] - r4["stage_a_bpb"]
            diff_wiki_ce = r6["wiki_ce"] - r4["wiki_ce"]
            diff_wiki_bpb = r6["wiki_bpb"] - r4["wiki_bpb"]
            diff_cosmo_ce = r6["cosmo_ce"] - r4["cosmo_ce"]
            diff_fineweb_ce = r6["fineweb_ce"] - r4["fineweb_ce"]
            diff_tinystories_ce = r6["tinystories_ce"] - r4["tinystories_ce"]
            
            # Multiples of noise floor
            sigma_val = abs(diff_stage_a_ce) / NOISE_FLOOR_VAL
            sigma_train = abs(diff_stage_a_ce) / NOISE_FLOOR_TRAIN
            
            gap_rows.append({
                "checkpoint": m,
                "is_decayed": r4["is_decayed"],
                "4L_stage_a_ce": r4["stage_a_ce"],
                "6L_stage_a_ce": r6["stage_a_ce"],
                "delta_ce (6L - 4L)": round(diff_stage_a_ce, 4),
                "delta_bpb (6L - 4L)": round(diff_stage_a_bpb, 4),
                "delta_wiki_ce": round(diff_wiki_ce, 4),
                "delta_cosmo_ce": round(diff_cosmo_ce, 4),
                "delta_fineweb_ce": round(diff_fineweb_ce, 4),
                "delta_tinystories_ce": round(diff_tinystories_ce, 4),
                "multiple_of_noise_val (0.0076)": round(sigma_val, 2),
                "multiple_of_noise_train (0.0254)": round(sigma_train, 2),
            })
            
    gap_df = pd.DataFrame(gap_rows)
    return gap_df

def generate_plots(curves_df, summary_df):
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    
    # 1. Loss vs Tokens
    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
    for model_name, color in [("4L_768H_1KV", "#1f77b4"), ("6L_768H_1KV", "#ff7f0e")]:
        cdf = curves_df[curves_df["model"] == model_name]
        ax.plot(cdf["total_tokens"] / 1e6, cdf["train_loss"], label=f"{model_name} (Train)", color=color, alpha=0.3, lw=1)
        # Smoothed
        if len(cdf) > 10:
            smoothed = cdf["train_loss"].rolling(window=10, min_periods=1).mean()
            ax.plot(cdf["total_tokens"] / 1e6, smoothed, label=f"{model_name} (Train 10-step MA)", color=color, lw=2)
            
        # Add validation points
        s_undec = summary_df[(summary_df["model"] == model_name) & (~summary_df["is_decayed"])]
        s_dec = summary_df[(summary_df["model"] == model_name) & (summary_df["is_decayed"])]
        ax.scatter(s_undec["tokens"] / 1e6, s_undec["stage_a_ce"], color=color, marker="o", s=80, label=f"{model_name} Val (Undecayed)")
        ax.scatter(s_dec["tokens"] / 1e6, s_dec["stage_a_ce"], color=color, marker="*", s=120, edgecolors="black", label=f"{model_name} Val (Decayed)")
        
    ax.set_title("3-Hour Head-to-Head: Loss vs. Tokens (4L vs 6L)", fontsize=14, fontweight="bold")
    ax.set_xlabel("Processed Tokens (Millions)", fontsize=12)
    ax.set_ylabel("Cross-Entropy Loss", fontsize=12)
    ax.legend(loc="upper right", frameon=True)
    fig.tight_layout()
    fig.savefig(PLOT_LOSS_TOKENS)
    plt.close(fig)
    print(f"Saved plot: {PLOT_LOSS_TOKENS.name}")
    
    # 2. Loss vs Wall-Clock Time
    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
    for model_name, color in [("4L_768H_1KV", "#1f77b4"), ("6L_768H_1KV", "#ff7f0e")]:
        cdf = curves_df[curves_df["model"] == model_name]
        if len(cdf) > 10:
            smoothed = cdf["train_loss"].rolling(window=10, min_periods=1).mean()
            ax.plot(cdf["elapsed_main_sec"] / 3600, smoothed, label=f"{model_name} (Train MA)", color=color, lw=2)
            
        s_dec = summary_df[(summary_df["model"] == model_name) & (summary_df["is_decayed"])]
        ax.scatter(s_dec["wall_clock_sec"] / 3600, s_dec["stage_a_ce"], color=color, marker="*", s=140, edgecolors="black", label=f"{model_name} Val (Decayed)")
        
    ax.set_title("3-Hour Head-to-Head: Loss vs. Wall-Clock Hours", fontsize=14, fontweight="bold")
    ax.set_xlabel("Wall-Clock Time (Hours)", fontsize=12)
    ax.set_ylabel("Cross-Entropy Loss", fontsize=12)
    ax.legend(loc="upper right", frameon=True)
    fig.tight_layout()
    fig.savefig(PLOT_LOSS_TIME)
    plt.close(fig)
    print(f"Saved plot: {PLOT_LOSS_TIME.name}")
    
    # 3. Pareto Plots (Val Loss vs Decode Speed)
    # Using 3-hour decayed checkpoint validation loss
    dec_3h = summary_df[summary_df["checkpoint"] == "3hr_decayed"]
    if len(dec_3h) == 2:
        val_loss_4l = dec_3h[dec_3h["model"] == "4L_768H_1KV"]["stage_a_ce"].iloc[0]
        val_loss_6l = dec_3h[dec_3h["model"] == "6L_768H_1KV"]["stage_a_ce"].iloc[0]
        
        # 0k context depth
        fig, ax = plt.subplots(figsize=(8, 6), dpi=150)
        ax.scatter([SPEED_BENCHMARKS["4L_768H_1KV"]["tg_0k"]], [val_loss_4l], color="#1f77b4", s=180, zorder=5, label="4L_768H_1KV (30.3M)")
        ax.annotate(f"4L_768H_1KV\nLoss: {val_loss_4l:.4f}\n{SPEED_BENCHMARKS['4L_768H_1KV']['tg_0k']:.1f} t/s",
                    (SPEED_BENCHMARKS["4L_768H_1KV"]["tg_0k"], val_loss_4l), textcoords="offset points", xytext=(-20, 15), fontweight="bold")
                    
        ax.scatter([SPEED_BENCHMARKS["6L_768H_1KV"]["tg_0k"]], [val_loss_6l], color="#ff7f0e", s=180, zorder=5, label="6L_768H_1KV (42.3M)")
        ax.annotate(f"6L_768H_1KV\nLoss: {val_loss_6l:.4f}\n{SPEED_BENCHMARKS['6L_768H_1KV']['tg_0k']:.1f} t/s",
                    (SPEED_BENCHMARKS["6L_768H_1KV"]["tg_0k"], val_loss_6l), textcoords="offset points", xytext=(-20, -35), fontweight="bold")
                    
        ax.set_title("Pareto Frontier: 3-Hour Decayed Val Loss vs. Decode Speed (0k Depth)", fontsize=13, fontweight="bold")
        ax.set_xlabel("Decode Throughput (tokens/sec at 0k context)", fontsize=12)
        ax.set_ylabel("Stage A Validation Loss (Lower is Better)", fontsize=12)
        ax.legend(loc="upper right", frameon=True)
        fig.tight_layout()
        fig.savefig(PLOT_PARETO_0K)
        plt.close(fig)
        print(f"Saved plot: {PLOT_PARETO_0K.name}")
        
        # 32k context depth
        fig, ax = plt.subplots(figsize=(8, 6), dpi=150)
        ax.scatter([SPEED_BENCHMARKS["4L_768H_1KV"]["tg_32k"]], [val_loss_4l], color="#1f77b4", s=180, zorder=5, label="4L_768H_1KV (30.3M)")
        ax.annotate(f"4L_768H_1KV\nLoss: {val_loss_4l:.4f}\n{SPEED_BENCHMARKS['4L_768H_1KV']['tg_32k']:.1f} t/s",
                    (SPEED_BENCHMARKS["4L_768H_1KV"]["tg_32k"], val_loss_4l), textcoords="offset points", xytext=(-20, 15), fontweight="bold")
                    
        ax.scatter([SPEED_BENCHMARKS["6L_768H_1KV"]["tg_32k"]], [val_loss_6l], color="#ff7f0e", s=180, zorder=5, label="6L_768H_1KV (42.3M)")
        ax.annotate(f"6L_768H_1KV\nLoss: {val_loss_6l:.4f}\n{SPEED_BENCHMARKS['6L_768H_1KV']['tg_32k']:.1f} t/s",
                    (SPEED_BENCHMARKS["6L_768H_1KV"]["tg_32k"], val_loss_6l), textcoords="offset points", xytext=(-20, -35), fontweight="bold")
                    
        ax.set_title("Pareto Frontier: 3-Hour Decayed Val Loss vs. Decode Speed (32k Depth)", fontsize=13, fontweight="bold")
        ax.set_xlabel("Decode Throughput (tokens/sec at 32k context)", fontsize=12)
        ax.set_ylabel("Stage A Validation Loss (Lower is Better)", fontsize=12)
        ax.legend(loc="upper right", frameon=True)
        fig.tight_layout()
        fig.savefig(PLOT_PARETO_32K)
        plt.close(fig)
        print(f"Saved plot: {PLOT_PARETO_32K.name}")

# ---------------------------------------------------------------------------
# Main Orchestrator
# ---------------------------------------------------------------------------
def main():
    print("=" * 78)
    print("  STEP 2: 3-HOUR HEAD-TO-HEAD TRIAL ORCHESTRATOR")
    print("  Comparing 4L_768H_1KV (30.28M) vs 6L_768H_1KV (42.28M)")
    print("=" * 78)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Active Device: {device} ({torch.cuda.get_device_name(0)})")
    
    # Start telemetry monitoring thread
    telemetry = GPUTelemetry(TELEMETRY_CSV, interval_sec=60)
    telemetry.start()
    print(f"GPU Telemetry monitor active (logging every 60s to {TELEMETRY_CSV.name})")
    
    loader = StageAMixDataLoader(SHARDS_DIR, SEQ_LEN)
    evaluator = ValidationEvaluator(SHARDS_DIR, SEQ_LEN, device)
    
    curves_records = []
    summary_records = []
    all_samples = {}
    
    try:
        # Run 4L_768H_1KV first
        samples_4l = train_model_head_to_head("4L_768H_1KV", MODELS_TO_RUN["4L_768H_1KV"], loader, evaluator, device, curves_records, summary_records)
        all_samples["4L_768H_1KV"] = samples_4l
        
        # Save intermediate records
        pd.DataFrame(curves_records).to_csv(CURVES_CSV, index=False)
        pd.DataFrame(summary_records).to_csv(SUMMARY_CSV, index=False)
        
        # Run 6L_768H_1KV second
        samples_6l = train_model_head_to_head("6L_768H_1KV", MODELS_TO_RUN["6L_768H_1KV"], loader, evaluator, device, curves_records, summary_records)
        all_samples["6L_768H_1KV"] = samples_6l
        
    finally:
        telemetry.stop()
        print("Telemetry logging stopped.")
        
    # Save final CSV data
    curves_df = pd.DataFrame(curves_records)
    summary_df = pd.DataFrame(summary_records)
    curves_df.to_csv(CURVES_CSV, index=False)
    summary_df.to_csv(SUMMARY_CSV, index=False)
    print(f"\nSaved curves: {CURVES_CSV}")
    print(f"Saved summary: {SUMMARY_CSV}")
    
    # Compute gap analysis
    gap_df = compute_gap_analysis(summary_df)
    gap_df.to_csv(GAP_TABLE_CSV, index=False)
    print(f"Saved gap table: {GAP_TABLE_CSV}")
    
    # Generate plots
    generate_plots(curves_df, summary_df)
    
    # Write sample generations markdown
    with open(SAMPLES_MD, "w", encoding="utf-8") as f:
        f.write("# 3-Hour Head-to-Head Sample Generations (Decayed Checkpoints)\n\n")
        for i, prompt in enumerate(PROMPTS, 1):
            f.write(f"## Prompt {i}: \"{prompt}\"\n\n")
            f.write(f"### 4L_768H_1KV (30.28M Params):\n```\n{all_samples['4L_768H_1KV'][i-1][1]}\n```\n\n")
            f.write(f"### 6L_768H_1KV (42.28M Params):\n```\n{all_samples['6L_768H_1KV'][i-1][1]}\n```\n\n")
            f.write("---\n\n")
    print(f"Saved sample generations: {SAMPLES_MD}")
    
    # Write comprehensive report
    with open(REPORT_MD, "w", encoding="utf-8") as f:
        f.write("# Step 2: 3-Hour Head-to-Head Comparison Report\n\n")
        f.write("## Executive Summary\n\n")
        f.write(f"- Protocol: Stage A mix (45% Cosmopedia, 45% FineWeb-Edu, 10% TinyStories), 2,048 context, WSD flat schedule.\n")
        f.write(f"- Main run duration: 3.0 hours (10,800s) per model.\n")
        f.write(f"- Noise floors: Validation spread $\\Delta_{{\\text{{val}}}} = {NOISE_FLOOR_VAL:.4f}$, Training spread $\\Delta_{{\\text{{train}}}} = {NOISE_FLOOR_TRAIN:.4f}$.\n\n")
        f.write("## Summary Table\n\n")
        f.write(summary_df.to_markdown(index=False))
        f.write("\n\n## Gap Table (6L minus 4L)\n\n")
        f.write(gap_df.to_markdown(index=False))
        f.write("\n\n")
    print(f"Saved report: {REPORT_MD}")
    print("\n>>> HEAD-TO-HEAD EXECUTION AND EVALUATION COMPLETE! <<<")

if __name__ == "__main__":
    main()
