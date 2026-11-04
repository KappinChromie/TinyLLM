"""
Sanity Check 1.c: Train 4L_768H_1KV on TinyStories alone for 15 minutes.
Verifies that the architecture, tokenizer, and training pipeline produce coherent English
and recognizable grammar quickly when trained on simple, consistent data.
Reports validation loss on val_tinystories and generates 10 story samples.
"""

import os
import sys
import math
import time
from pathlib import Path
import numpy as np
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

OUT_REPORT = RESULTS_DIR / "sanity_check_tinystories_15m.md"
CKPT_PATH = CHECKPOINTS_DIR / "4L_768H_1KV_tinystories_15m.pt"

# Architecture: 4L_768H_1KV
SHAPE_CFG = {
    "num_hidden_layers": 4,
    "hidden_size": 768,
    "intermediate_size": 2048,
    "num_attention_heads": 12,
    "num_key_value_heads": 1,
    "vocab_size": 8192,
}

PEAK_LR = 1e-3
MIN_LR = 1e-4
TARGET_DURATION_SEC = 900  # 15 minutes
SEQ_LEN = 2048
MICRO_BS = 8
GRAD_ACCUM_STEPS = 4
GLOBAL_BS_TOKENS = MICRO_BS * GRAD_ACCUM_STEPS * SEQ_LEN  # 65,536 tokens
WARMUP_STEPS = 50

# Compression stats for TinyStories (measured in Phase 2)
BYTES_PER_TOK = 3.269

class TinyStoriesDataLoader:
    def __init__(self, shard_path, seq_len):
        self.seq_len = seq_len
        self.data = np.memmap(shard_path, dtype=np.uint16, mode="r")
        self.max_idx = len(self.data) - self.seq_len - 1
        print(f"Loaded {shard_path.name}: {len(self.data):,} tokens (~{len(self.data)/1e6:.1f}M)")

    def get_batch(self, batch_size, rng, device):
        indices = rng.randint(0, self.max_idx, size=batch_size)
        x_list = [self.data[idx : idx + self.seq_len] for idx in indices]
        y_list = [self.data[idx + 1 : idx + 1 + self.seq_len] for idx in indices]
        x = np.stack(x_list)
        y = np.stack(y_list)
        tx = torch.from_numpy(x.astype(np.int64)).to(device=device, non_blocking=True)
        ty = torch.from_numpy(y.astype(np.int64)).to(device=device, non_blocking=True)
        return tx, ty

def evaluate_val_loss(model, val_loader, eval_steps=20, device="cuda"):
    model.eval()
    val_rng = np.random.RandomState(999)
    total_loss = 0.0
    with torch.no_grad():
        for _ in range(eval_steps):
            x, y = val_loader.get_batch(MICRO_BS * GRAD_ACCUM_STEPS, val_rng, device)
            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                out = model(x, labels=y)
                loss = out.loss
            total_loss += loss.item()
    model.train()
    mean_ce = total_loss / eval_steps
    bpb = (mean_ce / math.log(2.0)) / BYTES_PER_TOK
    return mean_ce, bpb

def generate_samples(model, sp, prompts, max_new_tokens=64, temperature=0.7, top_p=0.9, device="cuda"):
    model.eval()
    results = []
    with torch.no_grad():
        for prompt in prompts:
            toks = [1] + list(sp.encode(prompt))  # Prepend BOS
            curr_ids = list(toks)
            for _ in range(max_new_tokens):
                inp = torch.tensor([curr_ids], dtype=torch.long, device=device)
                with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                    out = model(inp)
                logits = out.logits[0, -1, :].float()
                if temperature == 0.0:
                    next_tok = torch.argmax(logits).item()
                else:
                    logits = logits / temperature
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                    cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    indices_to_remove = sorted_indices[sorted_indices_to_remove]
                    logits[indices_to_remove] = -float("Inf")
                    probs = torch.softmax(logits, dim=-1)
                    next_tok = torch.multinomial(probs, num_samples=1).item()
                if next_tok == 2:  # EOS
                    break
                curr_ids.append(next_tok)
            gen_text = sp.decode(curr_ids[1:])
            results.append((prompt, gen_text))
    return results

def main():
    print("=" * 70)
    print("  SANITY CHECK 1.c: 15-MINUTE TINYSTORIES TRAINING (4L_768H_1KV)")
    print("=" * 70)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} ({torch.cuda.get_device_name(0)})")
    
    train_path = SHARDS_DIR / "train_tinystories.bin"
    val_path = SHARDS_DIR / "val_tinystories.bin"
    assert train_path.exists(), f"Train shard missing: {train_path}"
    assert val_path.exists(), f"Val shard missing: {val_path}"
    assert TOKENIZER_MODEL.exists(), f"Tokenizer missing: {TOKENIZER_MODEL}"
    
    train_loader = TinyStoriesDataLoader(train_path, SEQ_LEN)
    val_loader = TinyStoriesDataLoader(val_path, SEQ_LEN)
    sp = spm.SentencePieceProcessor(model_file=str(TOKENIZER_MODEL))
    
    # Model configuration
    cfg = LlamaConfig(
        vocab_size=SHAPE_CFG["vocab_size"],
        hidden_size=SHAPE_CFG["hidden_size"],
        intermediate_size=SHAPE_CFG["intermediate_size"],
        num_hidden_layers=SHAPE_CFG["num_hidden_layers"],
        num_attention_heads=SHAPE_CFG["num_attention_heads"],
        num_key_value_heads=SHAPE_CFG["num_key_value_heads"],
        tie_word_embeddings=True,
        torch_dtype="bfloat16",
        max_position_embeddings=2048,
        attn_implementation="sdpa",
    )
    
    torch.manual_seed(42)
    np.random.seed(42)
    train_rng = np.random.RandomState(42)
    
    model = LlamaForCausalLM(cfg).to(device=device, dtype=torch.bfloat16)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model: 4L_768H_1KV ({total_params/1e6:.2f}M parameters)")
    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=PEAK_LR,
        betas=(0.9, 0.95),
        weight_decay=0.1,
    )
    
    # Estimate total steps in 900 seconds: ~1200 steps
    # We use a cosine schedule that ramps up for 50 steps, decays towards MIN_LR by ~1200 steps
    ESTIMATED_TOTAL_STEPS = 1250
    
    def get_lr(step):
        if step < WARMUP_STEPS:
            return PEAK_LR * (step + 1) / WARMUP_STEPS
        progress = min(1.0, (step - WARMUP_STEPS) / max(1, (ESTIMATED_TOTAL_STEPS - WARMUP_STEPS)))
        return MIN_LR + 0.5 * (PEAK_LR - MIN_LR) * (1.0 + math.cos(math.pi * progress))
        
    start_time = time.time()
    step = 0
    total_tokens = 0
    step_losses = []
    
    print("\nStarting training loop (target: 900 seconds / 15 minutes)...")
    while True:
        step_start = time.time()
        elapsed = step_start - start_time
        if elapsed >= TARGET_DURATION_SEC:
            print(f"\nTarget duration of {TARGET_DURATION_SEC}s reached at step {step} ({elapsed:.1f}s).")
            break
            
        lr = get_lr(step)
        for pg in optimizer.param_groups:
            pg["lr"] = lr
            
        optimizer.zero_grad(set_to_none=True)
        accum_loss = 0.0
        
        for micro in range(GRAD_ACCUM_STEPS):
            x, y = train_loader.get_batch(MICRO_BS, train_rng, device)
            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                out = model(x, labels=y)
                loss = out.loss / GRAD_ACCUM_STEPS
            loss.backward()
            accum_loss += loss.item()
            
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        step += 1
        total_tokens += GLOBAL_BS_TOKENS
        step_losses.append(accum_loss)
        
        if step % 25 == 0 or step == 1:
            step_time = time.time() - step_start
            tps = GLOBAL_BS_TOKENS / step_time
            print(f"Step {step:4d} | Elapsed: {elapsed:5.1f}s | Loss: {accum_loss:.4f} | LR: {lr:.2e} | Tokens: {total_tokens/1e6:5.1f}M | Tok/s: {tps:6.0f}")
            
    wall_clock = time.time() - start_time
    print(f"\nTraining Complete: {step} steps in {wall_clock:.1f}s ({total_tokens/1e6:.1f}M tokens).")
    
    # Save checkpoint
    torch.save(model.state_dict(), CKPT_PATH)
    print(f"Saved checkpoint to {CKPT_PATH}")
    
    # Evaluate validation loss
    print("\nEvaluating on val_tinystories...")
    val_ce, val_bpb = evaluate_val_loss(model, val_loader, eval_steps=30, device=device)
    print(f"Validation Cross-Entropy: {val_ce:.4f}")
    print(f"Validation BPB:          {val_bpb:.4f}")
    
    # Generate 10 samples
    sample_prompts = [
        "Once upon a time, there was a little bunny named ",
        "One sunny day, a little boy named Timmy went to ",
        "Lily wanted to bake a cake for her mom. First, she ",
        "The little puppy saw a big red ball and ",
        "Once there was an old wise tree that could ",
        "Mia loved to paint pictures of ",
        "Jack found a magical key under ",
        "The stars were shining brightly when the little owl ",
        "Sara was sad because she lost her ",
        "A friendly cat named Whiskers decided to ",
    ]
    
    print("\nGenerating 10 Story Samples (Temperature = 0.7, Top-p = 0.9):")
    samples_t07 = generate_samples(model, sp, sample_prompts, max_new_tokens=64, temperature=0.7, top_p=0.9, device=device)
    
    print("\nGenerating 10 Story Samples (Greedy / Temperature = 0.0):")
    samples_greedy = generate_samples(model, sp, sample_prompts, max_new_tokens=64, temperature=0.0, device=device)
    
    # Write report
    with open(OUT_REPORT, "w", encoding="utf-8") as f:
        f.write("# Sanity Check 1.c: 15-Minute TinyStories Training Report\n\n")
        f.write(f"- **Model**: `4L_768H_1KV` (30.28M parameters)\n")
        f.write(f"- **Data**: `train_tinystories.bin` exclusively (100% TinyStories)\n")
        f.write(f"- **Duration**: {wall_clock:.1f} seconds ({wall_clock/60:.1f} minutes)\n")
        f.write(f"- **Steps Completed**: {step}\n")
        f.write(f"- **Tokens Processed**: {total_tokens:,} ({total_tokens/1e6:.1f}M tokens, ~{total_tokens/30.28e6:.1f} tokens/param)\n")
        f.write(f"- **Final Training Loss**: {step_losses[-1]:.4f}\n")
        f.write(f"- **Validation Loss (`val_tinystories`)**: **{val_ce:.4f}**\n")
        f.write(f"- **Validation BPB**: **{val_bpb:.4f}**\n\n")
        
        f.write("## 10 Sample Generations (Temperature = 0.7, Top-p = 0.9)\n\n")
        for i, (prompt, text) in enumerate(samples_t07, 1):
            f.write(f"### Prompt {i}: \"{prompt}\"\n\n")
            f.write(f"```\n{text}\n```\n\n")
            
        f.write("## 10 Sample Generations (Greedy / Temperature = 0.0)\n\n")
        for i, (prompt, text) in enumerate(samples_greedy, 1):
            f.write(f"### Prompt {i}: \"{prompt}\"\n\n")
            f.write(f"```\n{text}\n```\n\n")
            
    print(f"\nReport written to {OUT_REPORT}")

if __name__ == "__main__":
    main()
