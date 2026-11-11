"""
Step 3: Learning Rate Trial for SFT (1e-4, 3e-4, 1e-3)
- Trains 20M tokens for each LR on SFT mix (85%) + replay mix (10% short 2k, 5% long 8k)
- Global batch: 32 sequences of 2,048 tokens = 65,536 tokens per step (305 steps = 19.99M tokens)
- Warmup: 50 steps, cosine decay to 10% of peak
- Seed 42, AdamW (0.9, 0.95), weight decay 0.1, grad clip 1.0, bf16
- Temperature governor: 83C pause / 78C resume
- Evaluates:
  1. Held-out assistant loss on val_conversations.jsonl
  2. 2k val CE on FineWeb-Edu and Cosmopedia v2
  3. 5 sample chat generations
"""

import os
import sys
import math
import time
import json
import shutil
import subprocess
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, LlamaConfig, LlamaForCausalLM

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent.parent
SFT_DATA_DIR = ROOT / "data" / "sft"
DATA_DIR = ROOT / "data" / "shards"
CHAT_TOKENIZER_DIR = ROOT / "models" / "tokenizer_chat"
RESULTS_DIR = ROOT / "results"
CHAT_INIT_CKPT = ROOT / "models" / "checkpoints" / "stageC_chat_init.pt"
PAUSE_LOG_CSV = RESULTS_DIR / "pause_log.csv"
OUT_CSV = RESULTS_DIR / "sft_lr_trials.csv"
OUT_SAMPLES_MD = RESULTS_DIR / "sft_lr_trial_samples.md"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SAMPLE_PROMPTS = [
    [{"role": "user", "content": "Hello! Can you introduce yourself?"}],
    [{"role": "user", "content": "What is the capital of Australia?"}],
    [{"role": "user", "content": "Rewrite this sentence to sound polite: I want you to send me the report right now."}],
    [{"role": "user", "content": "Summarize in one sentence: Photosynthesis is a chemical process that occurs in plants, algae, and some types of bacteria, when they are exposed to sunlight. During photosynthesis, water and carbon dioxide combine to form carbohydrates (sugars) and give off oxygen."}],
    [{"role": "user", "content": "Give me exactly three bullet points on why regular exercise is beneficial."}]
]

# ---------------------------------------------------------------------------
# GPU Utilities & Temperature Governor (83C / 78C)
# ---------------------------------------------------------------------------
def get_gpu_temp():
    try:
        res = subprocess.run(
            ["nvidia-smi", "--query-gpu=temperature.gpu,power.draw", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=True
        )
        parts = res.stdout.strip().split(",")
        return float(parts[0]), float(parts[1]) if len(parts) > 1 else 0.0
    except Exception:
        return 50.0, 0.0

def thermal_governor_check(run_name, step, tokens_seen, t_pause_threshold=83.0, t_resume_threshold=78.0):
    temp, power = get_gpu_temp()
    if temp >= t_pause_threshold:
        pause_start = time.time()
        pause_start_iso = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(pause_start))
        print(f"\n[GOVERNOR ALERT] GPU Temp reached {temp:.1f}C >= {t_pause_threshold}C! Pausing run {run_name} at step {step}...", flush=True)
        while temp > t_resume_threshold:
            time.sleep(5)
            temp, power = get_gpu_temp()
        pause_end = time.time()
        pause_end_iso = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(pause_end))
        duration = pause_end - pause_start
        print(f"[GOVERNOR RESUME] GPU Temp cooled to {temp:.1f}C <= {t_resume_threshold}C. Paused for {duration:.2f}s. Resuming training.", flush=True)
        # Log pause
        with open(PAUSE_LOG_CSV, "a", encoding="utf-8") as f:
            f.write(f"{pause_start_iso},{pause_end_iso},{duration:.2f},{temp:.1f},{power:.1f},{step},{tokens_seen},{run_name}\n")
        return duration
    return 0.0

# ---------------------------------------------------------------------------
# SFT + Replay Multi-Loader
# ---------------------------------------------------------------------------
class CombinedSFTReplayLoader:
    def __init__(self, sft_file, tokenizer, batch_size=4, seed=42):
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.seq_len = 2048
        self.rng = np.random.default_rng(seed)
        
        # 1. Load SFT conversations
        print(f"Loading SFT data from {sft_file}...")
        self.conversations = []
        with open(sft_file, "r", encoding="utf-8") as f:
            for line in f:
                self.conversations.append(json.loads(line))
        self.indices = list(range(len(self.conversations)))
        self.rng.shuffle(self.indices)
        self.cur_sft_idx = 0
        
        # 2. Pretraining Shards
        self.fineweb = np.memmap(DATA_DIR / "train_fineweb_edu.bin", dtype=np.uint16, mode="r")
        self.cosmopedia = np.memmap(DATA_DIR / "train_cosmopedia.bin", dtype=np.uint16, mode="r")
        self.tinystories = np.memmap(DATA_DIR / "train_tinystories.bin", dtype=np.uint16, mode="r")
        self.short_shards = [self.fineweb, self.cosmopedia, self.tinystories]
        self.short_weights = np.array([0.60, 0.30, 0.10])
        
        # Long shards (8k context)
        self.pg19 = np.memmap(DATA_DIR / "train_pg19_long.bin", dtype=np.uint16, mode="r")
        self.wiki_long = np.memmap(DATA_DIR / "train_wikipedia_long.bin", dtype=np.uint16, mode="r")
        self.fw_long = np.memmap(DATA_DIR / "train_fineweb_edu_long.bin", dtype=np.uint16, mode="r")
        self.long_shards = [self.pg19, self.wiki_long, self.fw_long]
        self.long_weights = np.array([0.50, 0.25, 0.25])
        
        self.step_counter = 0

    def get_batch(self, device=DEVICE):
        """
        Cycle of 20 steps:
          - Steps 0..16 (17 steps = 85%): SFT packed conversations with block-diagonal attention mask
          - Steps 17..18 (2 steps = 10%): Short pretraining replay (2k context, full loss)
          - Step 19 (1 step = 5%): Long pretraining replay (8k context, batch_size=1, full loss)
        """
        cycle_idx = self.step_counter % 20
        self.step_counter += 1
        
        if cycle_idx < 17:
            # SFT Batch
            return self._get_sft_batch(device)
        elif cycle_idx < 19:
            # Short Pretraining Replay Batch (60/30/10)
            return self._get_short_replay_batch(device)
        else:
            # Long Pretraining Replay Batch (8k context)
            return self._get_long_replay_batch(device)

    def _get_sft_batch(self, device):
        batch_x = []
        batch_doc_ids = []
        
        for b in range(self.batch_size):
            seq_x = []
            seq_docs = []
            cur_doc = 0
            while len(seq_x) < self.seq_len:
                if self.cur_sft_idx >= len(self.indices):
                    self.cur_sft_idx = 0
                    self.rng.shuffle(self.indices)
                conv = self.conversations[self.indices[self.cur_sft_idx]]
                self.cur_sft_idx += 1
                
                # Format
                toks = [1] # BOS
                for m in conv["messages"]:
                    role = m["role"]
                    content = m["content"]
                    h_toks = self.tokenizer.encode(f"<|{role}|>\n", add_special_tokens=False)
                    c_toks = self.tokenizer.encode(content, add_special_tokens=False)
                    toks.extend(h_toks)
                    toks.extend(c_toks)
                    toks.append(8195) # <|end|>
                    toks.append(14)   # \n
                    
                needed = self.seq_len - len(seq_x)
                take = toks[:needed]
                seq_x.extend(take)
                seq_docs.extend([cur_doc] * len(take))
                cur_doc += 1
                
            batch_x.append(seq_x[:self.seq_len])
            batch_doc_ids.append(seq_docs[:self.seq_len])
            
        x_np = np.array(batch_x, dtype=np.int64)
        x = torch.from_numpy(x_np[:, :-1]).to(device)
        y_raw = torch.from_numpy(x_np[:, 1:]).to(device)
        
        # Build assistant loss mask on y
        y = y_raw.clone()
        for b in range(self.batch_size):
            toks = x_np[b]
            in_asst = False
            for t in range(len(toks) - 1):
                cur_t = toks[t]
                next_t = toks[t+1]
                if cur_t == 8194: # <|assistant|>
                    in_asst = True
                elif cur_t == 8195: # <|end|>
                    in_asst = False
                    
                if in_asst and cur_t != 8194:
                    y[b, t] = next_t
                elif in_asst and cur_t == 8194 and next_t == 14:
                    y[b, t] = -100
                elif not in_asst:
                    y[b, t] = -100
                    
        # Block diagonal mask
        doc_arr = np.array(batch_doc_ids)[:, :-1]
        T = doc_arr.shape[1]
        same_doc = (doc_arr[:, :, None] == doc_arr[:, None, :])
        causal = np.tril(np.ones((T, T), dtype=bool))
        allowed = same_doc & causal[None, :, :]
        attn_mask = torch.from_numpy(allowed).unsqueeze(1).to(device)
        return x, y, attn_mask, "sft"

    def _get_short_replay_batch(self, device):
        # 4 sequences of 2,048 tokens from FineWeb (60%), Cosmo (30%), TinyStories (10%)
        shard_indices = self.rng.choice(3, size=self.batch_size, p=self.short_weights)
        x_list, y_list = [], []
        for s_idx in shard_indices:
            shard = self.short_shards[s_idx]
            max_idx = len(shard) - self.seq_len - 1
            idx = int(self.rng.integers(0, max_idx))
            raw = np.array(shard[idx : idx + self.seq_len + 1], dtype=np.int64)
            x_list.append(raw[:-1])
            y_list.append(raw[1:])
        x = torch.from_numpy(np.array(x_list)).to(device)
        y = torch.from_numpy(np.array(y_list)).to(device)
        return x, y, None, "replay_short"

    def _get_long_replay_batch(self, device):
        # 1 sequence of 8,192 tokens (4x tokens of 2k)
        s_idx = self.rng.choice(3, p=self.long_weights)
        shard = self.long_shards[s_idx]
        max_idx = len(shard) - 8193
        idx = int(self.rng.integers(0, max_idx))
        raw = np.array(shard[idx : idx + 8193], dtype=np.int64)
        x = torch.from_numpy(raw[:-1]).unsqueeze(0).to(device)
        y = torch.from_numpy(raw[1:]).unsqueeze(0).to(device)
        return x, y, None, "replay_long"

# ---------------------------------------------------------------------------
# Evaluation Functions
# ---------------------------------------------------------------------------
@torch.no_grad()
def evaluate_held_out_assistant_loss(model, val_file, tokenizer, num_convs=200, device=DEVICE):
    model.eval()
    convs = []
    with open(val_file, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= num_convs:
                break
            convs.append(json.loads(line))
            
    losses = []
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        for item in convs:
            messages = item["messages"]
            # Render tokens
            all_tokens = [1]
            loss_mask = [0]
            for m in messages:
                role = m["role"]
                content = m["content"]
                h_toks = tokenizer.encode(f"<|{role}|>\n", add_special_tokens=False)
                c_toks = tokenizer.encode(content, add_special_tokens=False)
                all_tokens.extend(h_toks)
                loss_mask.extend([0]*len(h_toks))
                all_tokens.extend(c_toks)
                if role == "assistant":
                    loss_mask.extend([1]*len(c_toks))
                    all_tokens.append(8195)
                    loss_mask.append(1)
                    all_tokens.append(14)
                    loss_mask.append(0)
                else:
                    loss_mask.extend([0]*len(c_toks))
                    all_tokens.append(8195)
                    loss_mask.append(0)
                    all_tokens.append(14)
                    loss_mask.append(0)
                    
            tok_np = np.array(all_tokens, dtype=np.int64)
            mask_np = np.array(loss_mask, dtype=np.int64)
            
            x = torch.from_numpy(tok_np[:-1]).unsqueeze(0).to(device)
            y_raw = torch.from_numpy(tok_np[1:]).to(device)
            # Apply mask to targets
            target_mask = torch.from_numpy(mask_np[1:]).bool().to(device)
            
            out = model(x)
            logits = out.logits[0]
            if target_mask.sum() > 0:
                loss = F.cross_entropy(logits[target_mask], y_raw[target_mask]).item()
                losses.append(loss)
                
    model.train()
    return float(np.mean(losses))

@torch.no_grad()
def evaluate_2k_val_subsets(model, device=DEVICE):
    model.eval()
    val_shards = {
        "FineWeb-Edu": DATA_DIR / "val_fineweb_edu.bin",
        "Cosmopedia v2": DATA_DIR / "val_cosmopedia.bin",
    }
    results = {}
    seq_len = 2048
    num_seqs = 2000000 // seq_len # 976 seqs
    
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        for name, p in val_shards.items():
            shard = np.memmap(p, dtype=np.uint16, mode="r")
            losses = []
            for i in range(num_seqs):
                raw = np.array(shard[i*seq_len : (i+1)*seq_len + 1], dtype=np.int64)
                x = torch.from_numpy(raw[:-1]).unsqueeze(0).to(device)
                y = torch.from_numpy(raw[1:]).to(device)
                out = model(x)
                l = F.cross_entropy(out.logits[0], y).item()
                losses.append(l)
            results[name] = float(np.mean(losses))
    model.train()
    return results

@torch.no_grad()
def generate_samples(model, tokenizer, device=DEVICE, max_new_tokens=48):
    model.eval()
    samples = []
    for prompt_msgs in SAMPLE_PROMPTS:
        prompt_txt = tokenizer.apply_chat_template(prompt_msgs, tokenize=False, add_generation_prompt=True)
        inp_ids = tokenizer.apply_chat_template(prompt_msgs, tokenize=True, add_generation_prompt=True)
        input_ids = inp_ids["input_ids"] if isinstance(inp_ids, dict) or hasattr(inp_ids, "input_ids") else inp_ids
        
        curr_ids = list(input_ids)
        for _ in range(max_new_tokens):
            inp = torch.tensor([curr_ids], dtype=torch.long, device=device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                out = model(inp)
                next_tok = int(out.logits[0, -1].argmax().item())
            curr_ids.append(next_tok)
            if next_tok in [8195, 2]: # <|end|> or </s>
                break
        gen_toks = curr_ids[len(input_ids):]
        gen_txt = tokenizer.decode(gen_toks, skip_special_tokens=False)
        samples.append({
            "prompt": prompt_msgs[0]["content"],
            "generation": gen_txt.strip()
        })
    model.train()
    return samples

# ---------------------------------------------------------------------------
# LR Trial Runner
# ---------------------------------------------------------------------------
def run_trial(peak_lr, trial_name="LR_Trial", init_asst_loss=None, init_ce=None):
    print(f"\n{'='*75}\nSTARTING SFT LR TRIAL: Peak LR = {peak_lr:e}\n{'='*75}", flush=True)
    
    tokenizer = AutoTokenizer.from_pretrained(str(CHAT_TOKENIZER_DIR))
    loader = CombinedSFTReplayLoader(SFT_DATA_DIR / "train_conversations.jsonl", tokenizer, batch_size=4, seed=42)
    
    # Model config
    cfg = LlamaConfig(
        vocab_size=8256,
        hidden_size=768,
        intermediate_size=2048,
        num_hidden_layers=6,
        num_attention_heads=12,
        num_key_value_heads=1,
        tie_word_embeddings=True,
        torch_dtype="bfloat16",
        _attn_implementation="sdpa",
        max_position_embeddings=32768,
        rope_theta=1000000.0,
    )
    model = LlamaForCausalLM(cfg).to(device=DEVICE, dtype=torch.bfloat16)
    
    # Load weights from chat init checkpoint
    ckpt = torch.load(CHAT_INIT_CKPT, map_location=DEVICE, weights_only=False)
    state = ckpt["model_state"] if "model_state" in ckpt else ckpt
    model.load_state_dict(state)
    model.train()
    
    # Baseline evaluations before training
    if init_asst_loss is None or init_ce is None:
        print("Evaluating initial baseline before training...")
        init_asst_loss = evaluate_held_out_assistant_loss(model, SFT_DATA_DIR / "val_conversations.jsonl", tokenizer)
        init_ce = evaluate_2k_val_subsets(model)
        print(f"  Initial Assistant Loss: {init_asst_loss:.4f}")
        print(f"  Initial FineWeb-Edu CE : {init_ce['FineWeb-Edu']:.4f}")
        print(f"  Initial Cosmopedia CE  : {init_ce['Cosmopedia v2']:.4f}")
    else:
        print(f"Reusing initial baseline: Asst Loss {init_asst_loss:.4f}, FineWeb {init_ce['FineWeb-Edu']:.4f}, Cosmo {init_ce['Cosmopedia v2']:.4f}")
    
    # Setup Optimizer & Schedule
    # Global batch = 32 seqs = 65,536 tokens
    # Micro-batch = 4 seqs (accum = 8)
    # Total steps for 20M tokens = 20,000,000 / 65,536 = 305 steps
    total_steps = 305
    warmup_steps = 50
    min_lr = peak_lr * 0.10
    grad_accum_steps = 8
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=min_lr, betas=(0.9, 0.95), weight_decay=0.1)
    
    def get_lr(step):
        if step < warmup_steps:
            return min_lr + (peak_lr - min_lr) * (step / warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return min_lr + 0.5 * (peak_lr - min_lr) * (1.0 + math.cos(math.pi * progress))
        
    start_time = time.time()
    active_time = 0.0
    accum_tokens = 0
    step = 0
    
    while step < total_steps:
        optimizer.zero_grad()
        t_step_start = time.time()
        step_loss = 0.0
        
        for micro in range(grad_accum_steps):
            x, y, attn_mask, btype = loader.get_batch(DEVICE)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                if attn_mask is not None:
                    out = model(x, attention_mask=attn_mask)
                else:
                    out = model(x)
                loss = F.cross_entropy(out.logits.view(-1, 8256), y.view(-1), ignore_index=-100)
                loss_scaled = loss / grad_accum_steps
            loss_scaled.backward()
            step_loss += loss.item() / grad_accum_steps
            
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        # LR update
        cur_lr = get_lr(step)
        for pg in optimizer.param_groups:
            pg["lr"] = cur_lr
        optimizer.step()
        
        step += 1
        accum_tokens += 65536
        step_duration = time.time() - t_step_start
        active_time += step_duration
        
        # Thermal check
        pause_dur = thermal_governor_check(trial_name, step, accum_tokens)
        
        if step % 25 == 0 or step == total_steps:
            speed = accum_tokens / max(0.1, active_time)
            print(f"[{trial_name}] Step {step:3d}/{total_steps} ({accum_tokens/1e6:5.2f}M tok) | Loss: {step_loss:.4f} | LR: {cur_lr:.2e} | Speed: {speed:,.1f} tok/s", flush=True)
            
    total_time = time.time() - start_time
    print(f"\n{trial_name} Complete! Active: {active_time/60:.2f} min, Total: {total_time/60:.2f} min. Speed: {accum_tokens/active_time:,.1f} tok/s")
    
    # Final Evaluations
    print("Evaluating post-trial metrics...")
    final_asst_loss = evaluate_held_out_assistant_loss(model, SFT_DATA_DIR / "val_conversations.jsonl", tokenizer)
    final_ce = evaluate_2k_val_subsets(model)
    delta_fw = final_ce["FineWeb-Edu"] - init_ce["FineWeb-Edu"]
    delta_cosmo = final_ce["Cosmopedia v2"] - init_ce["Cosmopedia v2"]
    
    samples = generate_samples(model, tokenizer)
    print(f"  Final Assistant Loss: {final_asst_loss:.4f} (Delta vs init: {final_asst_loss - init_asst_loss:+.4f})")
    print(f"  Final FineWeb-Edu CE : {final_ce['FineWeb-Edu']:.4f} (Delta: {delta_fw:+.4f})")
    print(f"  Final Cosmopedia CE  : {final_ce['Cosmopedia v2']:.4f} (Delta: {delta_cosmo:+.4f})")
    
    return {
        "trial_name": trial_name,
        "peak_lr": peak_lr,
        "tokens_m": accum_tokens / 1e6,
        "active_min": active_time / 60,
        "speed_tok_s": accum_tokens / active_time,
        "init_asst_loss": init_asst_loss,
        "final_asst_loss": final_asst_loss,
        "asst_loss_delta": final_asst_loss - init_asst_loss,
        "fineweb_2k_ce": final_ce["FineWeb-Edu"],
        "fineweb_delta": delta_fw,
        "cosmo_2k_ce": final_ce["Cosmopedia v2"],
        "cosmo_delta": delta_cosmo,
        "samples": samples
    }

def main():
    trials = [
        (1.0e-4, "LR_1e-4"),
        (3.0e-4, "LR_3e-4"),
        (1.0e-3, "LR_1e-3"),
    ]
    
    all_results = []
    cached_asst_loss = None
    cached_ce = None
    
    for lr, name in trials:
        res = run_trial(lr, name, init_asst_loss=cached_asst_loss, init_ce=cached_ce)
        if cached_asst_loss is None:
            cached_asst_loss = res["init_asst_loss"]
            cached_ce = {"FineWeb-Edu": res["fineweb_2k_ce"] - res["fineweb_delta"], "Cosmopedia v2": res["cosmo_2k_ce"] - res["cosmo_delta"]}
        all_results.append(res)
        
    df = pd.DataFrame([{k: v for k, v in r.items() if k != "samples"} for r in all_results])
    df.to_csv(OUT_CSV, index=False)
    print(f"\nSaved LR trials summary to {OUT_CSV}")
    print(df.to_string())
    
    # Write samples markdown
    with open(OUT_SAMPLES_MD, "w", encoding="utf-8") as f:
        f.write("# SFT Learning Rate Trial: Sample Generations\n\n")
        for r in all_results:
            f.write(f"## Trial {r['trial_name']} (Peak LR: {r['peak_lr']:e})\n")
            f.write(f"- Assistant Loss: {r['final_asst_loss']:.4f} (Delta: {r['asst_loss_delta']:+.4f})\n")
            f.write(f"- FineWeb 2k CE: {r['fineweb_2k_ce']:.4f} (Delta: {r['fineweb_delta']:+.4f})\n")
            f.write(f"- Cosmopedia 2k CE: {r['cosmo_2k_ce']:.4f} (Delta: {r['cosmo_delta']:+.4f})\n\n")
            f.write("### Sample Outputs:\n\n")
            for s in r["samples"]:
                f.write(f"**Prompt**: {s['prompt']}\n\n")
                f.write(f"**Generation**: {s['generation']}\n\n")
                f.write("---\n\n")
                
    print(f"Saved sample generations to {OUT_SAMPLES_MD}")
    print("\n" + "="*70)
    print("STEP 3 LR TRIALS COMPLETED!")
    print("="*70)

if __name__ == "__main__":
    main()
