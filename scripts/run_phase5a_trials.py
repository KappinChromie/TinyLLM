"""
Phase 5A: Architecture Selection Trials (4 Shapes x 1 Hour at 2k Context)
- Finalist 1: 4L_768H_1KV (30.28M params)
- Finalist 2: 6L_512H_1KV (19.54M params)
- Finalist 3: 6L_768H_1KV (42.28M params)
- Control:    4L_768H_2KV (30.68M params)

Recipe:
- Duration: Exactly 1 hour (3,600s) per shape.
- Context: 2,048 tokens.
- Global batch: 32 sequences (65,536 tokens, micro-batch 8, grad accum 4).
- Learning rate: eta_0 = 1e-3, AdamW (weight_decay 0.1, grad clip 1.0).
- Schedule: 100-step warmup, flat until 54m (3,240s), 6m cosine decay to 0.1x (1e-4).
- Iso-token checkpoint at 220M tokens (undecayed) + short decay branch (168 steps to 0.1x).
- 1-hour final decayed checkpoint.
- Comprehensive evaluation on all 4 clean validation sets (CE, bits/tok, BPB, BPC).
- 10 fixed sample generations per shape.
- Pareto plots vs decode speed at 0 depth and 32k depth.
- Comparison against empirical noise floor Delta_noise = 0.0254.
"""

import os
import sys
import copy
import math
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from transformers import LlamaConfig, LlamaForCausalLM
import sentencepiece as spm

sys.stdout.reconfigure(line_buffering=True)

ROOT = Path(__file__).resolve().parent.parent
SHARDS_DIR = ROOT / "data" / "shards"
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)
CHECKPOINTS_DIR = ROOT / "models" / "checkpoints"
CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
TOKENIZER_MODEL = ROOT / "models" / "tokenizer_8k" / "tokenizer.model"

SUMMARY_CSV = RESULTS_DIR / "trials_1hr_summary.csv"
CURVES_CSV = RESULTS_DIR / "trials_1hr_curves.csv"
GENERATIONS_MD = RESULTS_DIR / "phase5a_sample_generations.md"
REPORT_MD = RESULTS_DIR / "phase5a_architecture_report.md"
PARETO_0K_PNG = RESULTS_DIR / "pareto_val_loss_vs_speed_0k.png"
PARETO_32K_PNG = RESULTS_DIR / "pareto_val_loss_vs_speed_32k.png"

PEAK_LR = 1e-3
MIN_LR = 1e-4
TARGET_DURATION_SEC = 3600  # 1 hour
FLAT_DURATION_SEC = 3240    # 54 minutes
DECAY_DURATION_SEC = 360    # 6 minutes
SEQ_LEN = 2048
MICRO_BS = 8
GRAD_ACCUM_STEPS = 4
GLOBAL_BS_TOKENS = MICRO_BS * GRAD_ACCUM_STEPS * SEQ_LEN  # 65,536 tokens
WARMUP_STEPS = 100
ISO_TOKEN_TARGET = 220_000_000  # 220M tokens
ISO_DECAY_STEPS = 168           # ~11M tokens (5% of 220M)
NOISE_FLOOR = 0.0254            # Empirical noise floor from multi-seed trials

# Decode speeds from long_context_sweep.csv
SPEED_PROFILES = {
    "4L_768H_1KV": {"tg_0k": 1662.8, "tg_32k": 1356.4, "params": 30.28},
    "6L_512H_1KV": {"tg_0k": 1546.1, "tg_32k": 1211.6, "params": 19.54},
    "6L_768H_1KV": {"tg_0k": 1384.7, "tg_32k": 1121.9, "params": 42.28},
    "4L_768H_2KV": {"tg_0k": 1672.7, "tg_32k": 1254.3, "params": 30.68},
}

SHAPES = {
    "4L_768H_1KV": {
        "num_hidden_layers": 4, "hidden_size": 768, "intermediate_size": 2048,
        "num_attention_heads": 12, "num_key_value_heads": 1, "vocab_size": 8192
    },
    "6L_512H_1KV": {
        "num_hidden_layers": 6, "hidden_size": 512, "intermediate_size": 1280,
        "num_attention_heads": 8, "num_key_value_heads": 1, "vocab_size": 8192
    },
    "6L_768H_1KV": {
        "num_hidden_layers": 6, "hidden_size": 768, "intermediate_size": 2048,
        "num_attention_heads": 12, "num_key_value_heads": 1, "vocab_size": 8192
    },
    "4L_768H_2KV": {
        "num_hidden_layers": 4, "hidden_size": 768, "intermediate_size": 2048,
        "num_attention_heads": 12, "num_key_value_heads": 2, "vocab_size": 8192
    },
}

SAMPLE_PROMPTS = [
    "Once upon a time, there was a little bunny named",
    "The fundamental difference between planets and stars is",
    "To write a simple Python function that adds two numbers:",
    "In ancient history, the Great Pyramid of Giza was built",
    "Lily wanted to bake a cake for her mom. First, she",
    "Water evaporates when the temperature rises because",
    "A computer network is a system of connected computers that",
    "The dog ran across the park to catch",
    "Photosynthesis is the biological process by which plants",
    "When solving a math problem, it is important to",
]

COMPRESSION_STATS = {
    "val_cosmopedia": {"bytes_per_tok": 3.478, "chars_per_tok": 3.474},
    "val_fineweb_edu": {"bytes_per_tok": 3.160, "chars_per_tok": 3.148},
    "val_tinystories": {"bytes_per_tok": 3.269, "chars_per_tok": 3.266},
    "val_wikipedia_neutral": {"bytes_per_tok": 2.813, "chars_per_tok": 2.797},
}

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
    def evaluate_set(self, model, name, num_batches=40):
        shard = self.val_shards[name]
        max_idx = len(shard) - self.seq_len - 1
        rng = np.random.RandomState(4242)
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

def run_decay_branch(base_model, loader, evaluator, device, steps=ISO_DECAY_STEPS):
    print(f"    Running 220M Decayed Branch ({steps} steps cosine decay to {MIN_LR:.1e})...", flush=True)
    branch_model = copy.deepcopy(base_model).to(device)
    branch_opt = torch.optim.AdamW(branch_model.parameters(), lr=PEAK_LR, weight_decay=0.1, fused=True)
    branch_rng = np.random.RandomState(9999)
    branch_model.train()
    
    for s in range(steps):
        # Cosine decay from PEAK_LR to MIN_LR
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
        
    val_metrics = evaluator.evaluate_all(branch_model)
    return branch_model, val_metrics

def generate_samples(model, sp, prompts, device, max_new_tokens=96):
    model.eval()
    completions = []
    for p in prompts:
        token_ids = sp.encode(p)
        tx = torch.tensor([token_ids], dtype=torch.long, device=device)
        with torch.no_grad():
            curr_ids = list(token_ids)
            for _ in range(max_new_tokens):
                inp = torch.tensor([curr_ids[-SEQ_LEN:]], dtype=torch.long, device=device)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    out = model(inp)
                next_logits = out.logits[0, -1, :] / 0.7  # temp 0.7
                # Top-p sampling (0.9)
                sorted_logits, sorted_indices = torch.sort(next_logits, descending=True)
                cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > 0.9
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                next_logits[indices_to_remove] = -float('Inf')
                probs = torch.softmax(next_logits, dim=-1)
                next_tok = torch.multinomial(probs, num_samples=1).item()
                if next_tok == 2:  # EOS
                    break
                curr_ids.append(next_tok)
        gen_text = sp.decode(curr_ids)
        completions.append({"prompt": p, "completion": gen_text})
    return completions

def train_trial(shape_name, shape_cfg, loader, evaluator, sp, device):
    print("\n" + "=" * 80, flush=True)
    print(f"  STARTING 1-HOUR TRIAL: {shape_name}", flush=True)
    print(f"  Layers: {shape_cfg['num_hidden_layers']} | Hidden: {shape_cfg['hidden_size']} | FFN: {shape_cfg['intermediate_size']} | KV Heads: {shape_cfg['num_key_value_heads']}", flush=True)
    print(f"  Duration: 3,600s (54m stable + 6m cosine decay) | Batch: 65,536 tok", flush=True)
    print("=" * 80, flush=True)
    
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.manual_seed(42)
    np.random.seed(42)
    rng = np.random.RandomState(42)
    
    cfg = LlamaConfig(
        vocab_size=shape_cfg["vocab_size"],
        hidden_size=shape_cfg["hidden_size"],
        intermediate_size=shape_cfg["intermediate_size"],
        num_hidden_layers=shape_cfg["num_hidden_layers"],
        num_attention_heads=shape_cfg["num_attention_heads"],
        num_key_value_heads=shape_cfg["num_key_value_heads"],
        tie_word_embeddings=True,
        torch_dtype="bfloat16",
        _attn_implementation="sdpa",
        max_position_embeddings=SEQ_LEN,
    )
    
    model = LlamaForCausalLM(cfg).to(device=device, dtype=torch.bfloat16)
    optimizer = torch.optim.AdamW(model.parameters(), lr=PEAK_LR, weight_decay=0.1, fused=True)
    
    step = 0
    total_tokens_seen = 0
    step_losses = []
    curve_records = []
    
    iso_saved = False
    iso_val_undecayed = None
    iso_val_decayed = None
    
    t_start = time.perf_counter()
    last_print = t_start
    
    while True:
        t_elapsed = time.perf_counter() - t_start
        if t_elapsed >= TARGET_DURATION_SEC:
            break
            
        # Determine current learning rate
        if t_elapsed < FLAT_DURATION_SEC:
            # Warmup then flat
            if step < WARMUP_STEPS:
                curr_lr = PEAK_LR * (step + 1) / WARMUP_STEPS
            else:
                curr_lr = PEAK_LR
        else:
            # Terminal cosine decay (last 6 minutes)
            decay_prog = (t_elapsed - FLAT_DURATION_SEC) / DECAY_DURATION_SEC
            decay_prog = min(1.0, max(0.0, decay_prog))
            curr_lr = MIN_LR + 0.5 * (PEAK_LR - MIN_LR) * (1.0 + math.cos(math.pi * decay_prog))
            
        for pg in optimizer.param_groups:
            pg["lr"] = curr_lr
            
        optimizer.zero_grad(set_to_none=True)
        accum_loss = 0.0
        
        for micro_step in range(GRAD_ACCUM_STEPS):
            x, y = loader.get_batch(MICRO_BS, rng, device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                outputs = model(x, labels=y)
                loss = outputs.loss / GRAD_ACCUM_STEPS
            loss.backward()
            accum_loss += loss.item()
            total_tokens_seen += MICRO_BS * SEQ_LEN
            
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        step += 1
        step_losses.append(accum_loss)
        
        # Check stability guard (Fix 6)
        if math.isnan(accum_loss) or accum_loss > 15.0:
            print(f"  [CRITICAL WARNING] Loss spike detected on {shape_name}: loss = {accum_loss}!", flush=True)
            
        # Iso-token checkpoint trigger (220M tokens)
        if total_tokens_seen >= ISO_TOKEN_TARGET and not iso_saved:
            iso_saved = True
            print(f"\n  >>> REACHED ISO-TOKEN THRESHOLD (220M Tokens at Step {step}, Elapsed: {t_elapsed:.1f}s) <<<", flush=True)
            print("  Evaluating undecayed 220M checkpoint...", flush=True)
            iso_val_undecayed = evaluator.evaluate_all(model)
            torch.save(model.state_dict(), CHECKPOINTS_DIR / f"{shape_name}_undecayed_220m.pt")
            print(f"  Saved {shape_name}_undecayed_220m.pt (Stage A Val Loss: {iso_val_undecayed['stage_a_val_loss']:.4f})", flush=True)
            
            # Run short decay branch from 220M checkpoint
            branch_model, iso_val_decayed = run_decay_branch(model, loader, evaluator, device)
            torch.save(branch_model.state_dict(), CHECKPOINTS_DIR / f"{shape_name}_decayed_220m_branch.pt")
            print(f"  Saved {shape_name}_decayed_220m_branch.pt (Stage A Val Loss: {iso_val_decayed['stage_a_val_loss']:.4f})", flush=True)
            del branch_model
            torch.cuda.empty_cache()
            print("  Continuing main 1-hour training run...\n", flush=True)
            
        if (time.perf_counter() - last_print) >= 30.0 or step == 10:
            last_print = time.perf_counter()
            t_now = time.perf_counter() - t_start
            window = step_losses[-50:] if len(step_losses) >= 50 else step_losses
            smooth = float(np.mean(window))
            spd = total_tokens_seen / max(1e-3, t_now)
            phase_tag = "STABLE" if t_now < FLAT_DURATION_SEC else "DECAY"
            print(f"  [{phase_tag}] Step {step:4d} | Elapsed: {t_now:6.1f}s / {TARGET_DURATION_SEC}s | Loss: {accum_loss:.4f} (smooth: {smooth:.4f}) | LR: {curr_lr:.2e} | Speed: {spd:,.0f} tok/s", flush=True)
            
            curve_records.append({
                "shape": shape_name,
                "step": step,
                "wall_clock_sec": round(t_now, 1),
                "tokens_seen": total_tokens_seen,
                "loss": round(accum_loss, 4),
                "smoothed_loss": round(smooth, 4),
                "lr": curr_lr
            })
            
    torch.cuda.synchronize()
    total_time = time.perf_counter() - t_start
    final_tokens_per_sec = total_tokens_seen / total_time
    peak_vram = torch.cuda.max_memory_allocated(device) / (1024 * 1024)
    
    print(f"\n--- 1-Hour Run Completed for {shape_name} ---", flush=True)
    print(f"  Total Steps: {step:,} | Elapsed: {total_time:.1f}s | Tokens Seen: {total_tokens_seen:,} ({final_tokens_per_sec:,.0f} tok/s)", flush=True)
    
    # Save final 1-hour decayed checkpoint
    torch.save(model.state_dict(), CHECKPOINTS_DIR / f"{shape_name}_decayed_1hr.pt")
    
    print("Evaluating 1-hour fully decayed model across all 4 validation sets...", flush=True)
    final_val_metrics = evaluator.evaluate_all(model)
    print(f"  1-Hour Final Val Loss (Stage A Mix): {final_val_metrics['stage_a_val_loss']:.4f} (BPB: {final_val_metrics['stage_a_bpb']:.4f})", flush=True)
    print(f"  Cosmopedia: {final_val_metrics['val_cosmopedia']['ce_loss']:.4f} (BPB: {final_val_metrics['val_cosmopedia']['bpb']:.4f})", flush=True)
    print(f"  FineWeb-Edu: {final_val_metrics['val_fineweb_edu']['ce_loss']:.4f} (BPB: {final_val_metrics['val_fineweb_edu']['bpb']:.4f})", flush=True)
    print(f"  TinyStories: {final_val_metrics['val_tinystories']['ce_loss']:.4f} (BPB: {final_val_metrics['val_tinystories']['bpb']:.4f})", flush=True)
    print(f"  Wikipedia Neutral: {final_val_metrics['val_wikipedia_neutral']['ce_loss']:.4f} (BPB: {final_val_metrics['val_wikipedia_neutral']['bpb']:.4f})", flush=True)
    
    print("\nGenerating 10 sample completions...", flush=True)
    sample_outputs = generate_samples(model, sp, SAMPLE_PROMPTS, device)
    
    del model, optimizer
    torch.cuda.empty_cache()
    
    result = {
        "shape": shape_name,
        "params_m": SPEED_PROFILES[shape_name]["params"],
        "decode_0k_ts": SPEED_PROFILES[shape_name]["tg_0k"],
        "decode_32k_ts": SPEED_PROFILES[shape_name]["tg_32k"],
        "train_tok_per_sec": round(final_tokens_per_sec, 1),
        "total_tokens_1hr": total_tokens_seen,
        "peak_vram_mib": round(peak_vram, 1),
        # 1-Hour Decayed
        "val_loss_1hr_decayed": final_val_metrics["stage_a_val_loss"],
        "bpb_1hr_decayed": final_val_metrics["stage_a_bpb"],
        "cosmo_1hr_decayed": final_val_metrics["val_cosmopedia"]["ce_loss"],
        "fineweb_1hr_decayed": final_val_metrics["val_fineweb_edu"]["ce_loss"],
        "tinystories_1hr_decayed": final_val_metrics["val_tinystories"]["ce_loss"],
        "wiki_neutral_1hr_decayed": final_val_metrics["val_wikipedia_neutral"]["ce_loss"],
        # 220M Iso-Token Decayed Branch
        "val_loss_220m_decayed": iso_val_decayed["stage_a_val_loss"] if iso_val_decayed else None,
        "bpb_220m_decayed": iso_val_decayed["stage_a_bpb"] if iso_val_decayed else None,
        "cosmo_220m_decayed": iso_val_decayed["val_cosmopedia"]["ce_loss"] if iso_val_decayed else None,
        "fineweb_220m_decayed": iso_val_decayed["val_fineweb_edu"]["ce_loss"] if iso_val_decayed else None,
        "wiki_neutral_220m_decayed": iso_val_decayed["val_wikipedia_neutral"]["ce_loss"] if iso_val_decayed else None,
        # 220M Iso-Token Undecayed
        "val_loss_220m_undecayed": iso_val_undecayed["stage_a_val_loss"] if iso_val_undecayed else None,
        "bpb_220m_undecayed": iso_val_undecayed["stage_a_bpb"] if iso_val_undecayed else None,
        "cosmo_220m_undecayed": iso_val_undecayed["val_cosmopedia"]["ce_loss"] if iso_val_undecayed else None,
        "fineweb_220m_undecayed": iso_val_undecayed["val_fineweb_edu"]["ce_loss"] if iso_val_undecayed else None,
        "wiki_neutral_220m_undecayed": iso_val_undecayed["val_wikipedia_neutral"]["ce_loss"] if iso_val_undecayed else None,
    }
    
    return result, curve_records, sample_outputs

def main():
    if not torch.cuda.is_available():
        print("CUDA not available.")
        sys.exit(1)
        
    device = torch.device("cuda:0")
    print("=" * 80, flush=True)
    print("  PHASE 5A: ARCHITECTURE SELECTION TRIALS (4 SHAPES x 1 HOUR)", flush=True)
    print(f"  Device: {torch.cuda.get_device_name(device)} | Noise Floor Delta_noise = {NOISE_FLOOR:.4f}", flush=True)
    print("=" * 80, flush=True)
    
    sp = spm.SentencePieceProcessor(model_file=str(TOKENIZER_MODEL))
    loader = StageAMixDataLoader(SHARDS_DIR, SEQ_LEN)
    evaluator = ValidationEvaluator(SHARDS_DIR, SEQ_LEN, device)
    
    all_results = []
    all_curves = []
    all_generations = {}
    
    # Run the 4 finalist trials sequentially
    for shape_name in ["4L_768H_1KV", "6L_512H_1KV", "6L_768H_1KV", "4L_768H_2KV"]:
        res, curves, gens = train_trial(shape_name, SHAPES[shape_name], loader, evaluator, sp, device)
        all_results.append(res)
        all_curves.extend(curves)
        all_generations[shape_name] = gens
        
        # Save intermediate checkpoint tables
        df_res = pd.DataFrame(all_results)
        df_res.to_csv(SUMMARY_CSV, index=False)
        df_curves = pd.DataFrame(all_curves)
        df_curves.to_csv(CURVES_CSV, index=False)
        print(f"\n[CHECKPOINT] Updated {SUMMARY_CSV} and {CURVES_CSV}", flush=True)
        
    # Write sample generations markdown
    with open(GENERATIONS_MD, "w", encoding="utf-8") as f:
        f.write("# Phase 5A: 1-Hour Finalist Sample Generations (10 Fixed Prompts)\n\n")
        for i, prompt in enumerate(SAMPLE_PROMPTS):
            f.write(f"### Prompt {i+1}: *\"{prompt}\"*\n\n")
            for shape_name in ["4L_768H_1KV", "6L_512H_1KV", "6L_768H_1KV", "4L_768H_2KV"]:
                gen_text = all_generations[shape_name][i]["completion"]
                f.write(f"**{shape_name}:**\n> {gen_text}\n\n")
            f.write("---\n\n")
    print(f"Generations saved to {GENERATIONS_MD}", flush=True)
    
    # Generate Pareto Plots
    df_res = pd.DataFrame(all_results)
    
    # 1. Pareto at 0k context depth
    fig, ax = plt.subplots(figsize=(8.5, 6), dpi=150)
    for _, row in df_res.iterrows():
        s = row["shape"]
        x = row["decode_0k_ts"]
        y_1hr = row["val_loss_1hr_decayed"]
        y_220m = row["val_loss_220m_decayed"]
        ax.scatter(x, y_1hr, s=120, label=f"{s} (1-hr decayed)", zorder=4)
        ax.annotate(f"{s}\n({y_1hr:.3f})", (x, y_1hr), textcoords="offset points", xytext=(0, 10), ha='center', fontsize=9, fontweight='bold')
        ax.scatter(x, y_220m, s=80, marker="^", alpha=0.7, zorder=3)
    ax.set_xlabel("Single-Stream Decode Speed at 0k Context (tokens/sec)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Validation Loss (Lower is Better)", fontsize=11, fontweight="bold")
    ax.set_title("Pareto Frontier: Validation Loss vs. Decode Speed (0 Context Depth)", fontsize=12, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(loc="upper right", fontsize=9)
    plt.tight_layout()
    plt.savefig(PARETO_0K_PNG)
    plt.close()
    
    # 2. Pareto at 32k context depth
    fig, ax = plt.subplots(figsize=(8.5, 6), dpi=150)
    for _, row in df_res.iterrows():
        s = row["shape"]
        x = row["decode_32k_ts"]
        y_1hr = row["val_loss_1hr_decayed"]
        ax.scatter(x, y_1hr, s=120, label=f"{s}", zorder=4)
        ax.annotate(f"{s}\n({x:,.0f} t/s, {y_1hr:.3f})", (x, y_1hr), textcoords="offset points", xytext=(0, 10), ha='center', fontsize=9, fontweight='bold')
    ax.set_xlabel("Single-Stream Decode Speed at 32k Depth (tokens/sec)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Validation Loss (Lower is Better)", fontsize=11, fontweight="bold")
    ax.set_title("Pareto Frontier: Validation Loss vs. Decode Speed (32k Context Depth)", fontsize=12, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(loc="upper right", fontsize=9)
    plt.tight_layout()
    plt.savefig(PARETO_32K_PNG)
    plt.close()
    
    print("\nAll 4 trials and plots completed successfully!", flush=True)

if __name__ == "__main__":
    main()
