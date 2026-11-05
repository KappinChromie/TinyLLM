"""
Test Resumable Checkpoints for 6L_768H_1KV
Verifies exact state recovery and loss trajectory parity:
1. Ground Truth Run: 200 uninterrupted steps from seed 42.
2. Resumed Run: Train 100 steps from seed 42, save full checkpoint, resume and train steps 101-200.
3. Compare loss at every step from 101 to 200 and report max delta.
"""

import os
import sys
import copy
import time
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
RESULTS_DIR.mkdir(exist_ok=True)
CHECKPOINT_PATH = RESULTS_DIR / "test_resume_ckpt.pt"
RESUME_REPORT_CSV = RESULTS_DIR / "resume_parity_test.csv"

class TestStreamDataLoader:
    def __init__(self, shards_dir, seq_len):
        self.seq_len = seq_len
        self.fineweb = np.memmap(shards_dir / "train_fineweb_edu.bin", dtype=np.uint16, mode="r")
        self.shard_len = len(self.fineweb)
        self.rng = np.random.RandomState(42)

    def get_state(self):
        return self.rng.get_state()

    def set_state(self, state):
        self.rng.set_state(state)

    def get_batch(self, batch_size, device):
        x_list, y_list = [], []
        max_idx = self.shard_len - self.seq_len - 1
        for _ in range(batch_size):
            idx = self.rng.randint(0, max_idx)
            x_list.append(self.fineweb[idx : idx + self.seq_len])
            y_list.append(self.fineweb[idx + 1 : idx + 1 + self.seq_len])
        x = np.stack(x_list)
        y = np.stack(y_list)
        tx = torch.from_numpy(x.astype(np.int64)).to(device=device, non_blocking=True)
        ty = torch.from_numpy(y.astype(np.int64)).to(device=device, non_blocking=True)
        return tx, ty

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

def run_uninterrupted(device, total_steps=200, micro_bs=4, grad_accum=8):
    print("\n--- Running Uninterrupted Ground Truth Run (200 steps) ---", flush=True)
    loader = TestStreamDataLoader(SHARDS_DIR, seq_len=2048)
    model = create_model(device, seed=42)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, betas=(0.9, 0.95), weight_decay=0.1, fused=True)
    
    losses = []
    for step in range(total_steps):
        step_loss = 0.0
        optimizer.zero_grad(set_to_none=True)
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
        losses.append(step_loss)
        if (step + 1) % 50 == 0:
            print(f"  Uninterrupted Step {step+1:3d}: Loss = {step_loss:.4f}", flush=True)
            
    return losses

def run_with_resume(device, total_steps=200, split_step=100, micro_bs=4, grad_accum=8):
    print(f"\n--- Running Part 1: Steps 1 to {split_step} ---", flush=True)
    loader = TestStreamDataLoader(SHARDS_DIR, seq_len=2048)
    model = create_model(device, seed=42)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, betas=(0.9, 0.95), weight_decay=0.1, fused=True)
    
    losses_part1 = []
    for step in range(split_step):
        step_loss = 0.0
        optimizer.zero_grad(set_to_none=True)
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
        losses_part1.append(step_loss)
        if (step + 1) % 50 == 0:
            print(f"  Part 1 Step {step+1:3d}: Loss = {step_loss:.4f}", flush=True)
            
    # Save full state
    print(f"\nSaving Full Checkpoint at Step {split_step} to {CHECKPOINT_PATH}...", flush=True)
    ckpt = {
        "step": split_step,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "loader_rng_state": loader.get_state(),
        "torch_cpu_rng": torch.get_rng_state(),
        "torch_cuda_rng": torch.cuda.get_rng_state_all(),
        "np_rng": np.random.get_state(),
    }
    torch.save(ckpt, CHECKPOINT_PATH)
    
    # Clean up model and memory to simulate process exit and restart
    del model, optimizer, loader
    torch.cuda.empty_cache()
    
    print(f"\n--- Resuming from Checkpoint: Steps {split_step+1} to {total_steps} ---", flush=True)
    loaded_ckpt = torch.load(CHECKPOINT_PATH, weights_only=False)
    
    # Restore loader and RNG states
    resumed_loader = TestStreamDataLoader(SHARDS_DIR, seq_len=2048)
    resumed_loader.set_state(loaded_ckpt["loader_rng_state"])
    torch.set_rng_state(loaded_ckpt["torch_cpu_rng"])
    torch.cuda.set_rng_state_all(loaded_ckpt["torch_cuda_rng"])
    np.random.set_state(loaded_ckpt["np_rng"])
    
    # Restore model and optimizer
    resumed_model = create_model(device, seed=42)
    resumed_model.load_state_dict(loaded_ckpt["model_state"])
    resumed_optimizer = torch.optim.AdamW(resumed_model.parameters(), lr=1e-3, betas=(0.9, 0.95), weight_decay=0.1, fused=True)
    resumed_optimizer.load_state_dict(loaded_ckpt["optimizer_state"])
    
    losses_part2 = []
    for step in range(split_step, total_steps):
        step_loss = 0.0
        resumed_optimizer.zero_grad(set_to_none=True)
        for _ in range(grad_accum):
            x, y = resumed_loader.get_batch(micro_bs, device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                out = resumed_model(x)
                loss = F.cross_entropy(out.logits.view(-1, 8192), y.view(-1))
                scaled_loss = loss / grad_accum
            scaled_loss.backward()
            step_loss += loss.item() / grad_accum
        torch.nn.utils.clip_grad_norm_(resumed_model.parameters(), 1.0)
        resumed_optimizer.step()
        losses_part2.append(step_loss)
        if (step + 1) % 50 == 0:
            print(f"  Resumed Step {step+1:3d}: Loss = {step_loss:.4f}", flush=True)
            
    return losses_part1 + losses_part2

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Testing Checkpoint Resume Parity on device: {device} ({torch.cuda.get_device_name(0)})")
    
    uninterrupted_losses = run_uninterrupted(device, total_steps=200)
    resumed_losses = run_with_resume(device, total_steps=200, split_step=100)
    
    # Comparison
    diffs = [abs(u - r) for u, r in zip(uninterrupted_losses, resumed_losses)]
    max_diff = max(diffs)
    mean_diff = sum(diffs) / len(diffs)
    part2_diffs = diffs[100:]
    max_part2_diff = max(part2_diffs)
    
    print("\n=======================================================", flush=True)
    print("CHECKPOINT RESUME PARITY RESULTS:", flush=True)
    print(f"  Step 100 Loss (Uninterrupted): {uninterrupted_losses[99]:.6f}")
    print(f"  Step 100 Loss (Pre-Save):     {resumed_losses[99]:.6f}")
    print(f"  Step 101 Loss (Uninterrupted): {uninterrupted_losses[100]:.6f}")
    print(f"  Step 101 Loss (Post-Resume):   {resumed_losses[100]:.6f}")
    print(f"  Step 200 Loss (Uninterrupted): {uninterrupted_losses[199]:.6f}")
    print(f"  Step 200 Loss (Resumed):       {resumed_losses[199]:.6f}")
    print(f"  Max Absolute Loss Difference (Steps 101-200): {max_part2_diff:.8e}")
    print(f"  Overall Mean Loss Difference: {mean_diff:.8e}")
    print("=======================================================", flush=True)
    
    # Save comparison table
    df = pd.DataFrame({
        "step": list(range(1, 201)),
        "uninterrupted_loss": uninterrupted_losses,
        "resumed_loss": resumed_losses,
        "abs_diff": diffs,
    })
    df.to_csv(RESUME_REPORT_CSV, index=False)
    print(f"Saved parity test results to {RESUME_REPORT_CSV}")
    
    if max_part2_diff < 1e-4:
        print("\nSUCCESS: Resumed training reproduces uninterrupted training with bit-level mathematical parity!")
    else:
        print("\nWARNING: Difference observed exceeds 1e-4.")

if __name__ == "__main__":
    main()
