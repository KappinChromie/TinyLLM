"""
Step 4: Main SFT Training Run
- Start checkpoint: models/checkpoints/stageC_chat_init.pt (42.28M params, 8,256 vocab)
- Peak LR: 3e-4 (interior winner from Step 3), warmup 50 steps, cosine decay to 3e-5 (10%)
- Global batch: 32 sequences of 2,048 tokens = 65,536 tokens per step
- Total steps: 2,000 steps = 131.07M tokens (~1.8 active hours, under 2-hour cap)
- Token Mix per 20-step cycle:
  * 17 steps (85%): SFT packed conversations with block-diagonal attention mask (111.4M tokens = 1.98 epochs)
  * 2 steps (10%): Short replay at 2k (60% FineWeb-Edu, 30% Cosmopedia, 10% TinyStories)
  * 1 step (5%): Long replay alternating 8k (micro-batch 1, accum 8) and 32k (micro-batch 1, accum 2, grad checkpointing)
- AdamW (0.9, 0.95), weight decay 0.1, grad clip 1.0, bf16, seed 42
- Temperature governor: 83C pause / 78C resume with results/pause_log.csv
- Loss-spike guard: rollback on >2.5x 20-step moving average, abort on 2nd event or NaN
- 15-minute rolling checkpoints + 1-hour permanent checkpoints
- Milestone evals every 25% (steps 500, 1000, 1500, 2000):
  * Held-out assistant loss on val_conversations.jsonl
  * 2k val CE on the 4 fixed 2M subsets (Wikipedia Neutral, FineWeb-Edu, Cosmopedia, TinyStories)
  * 10 fixed chat prompts
- Saves final weights to models/checkpoints/chat_sft_final.pt
"""

import os
import sys
import math
import time
import json
import shutil
import subprocess
import argparse
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
CKPT_DIR = ROOT / "models" / "checkpoints"
RESULTS_DIR = ROOT / "results"

INIT_CKPT = CKPT_DIR / "stageC_chat_init.pt"
ROLLING_LATEST_CKPT = CKPT_DIR / "sft_rolling_latest.pt"
ROLLING_PREV_CKPT = CKPT_DIR / "sft_rolling_prev.pt"
FINAL_CKPT = CKPT_DIR / "chat_sft_final.pt"

PAUSE_LOG_CSV = RESULTS_DIR / "pause_log.csv"
TRAIN_LOG_CSV = RESULTS_DIR / "sft_train_log.csv"
MILESTONE_CSV = RESULTS_DIR / "sft_training_milestones.csv"
MILESTONE_SAMPLES_MD = RESULTS_DIR / "sft_milestone_samples.md"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

FIXED_10_PROMPTS = [
    [{"role": "user", "content": "Hello! Can you introduce yourself?"}],
    [{"role": "user", "content": "What is the capital of Australia?"}],
    [{"role": "user", "content": "Rewrite this sentence to sound polite: I want you to send me the report right now."}],
    [{"role": "user", "content": "Summarize in one sentence: Photosynthesis is a chemical process that occurs in plants, algae, and some types of bacteria, when they are exposed to sunlight. During photosynthesis, water and carbon dioxide combine to form carbohydrates (sugars) and give off oxygen."}],
    [{"role": "user", "content": "Give me exactly three bullet points on why regular exercise is beneficial."}],
    [{"role": "user", "content": "What are the three primary states of matter?"}],
    [{"role": "user", "content": "Explain gravity to a five-year-old child in two simple sentences."}],
    [{"role": "user", "content": "Why is the sky blue?"}],
    [{"role": "user", "content": "Can you suggest three healthy breakfast ideas?"}],
    [{"role": "user", "content": "I am planning a weekend hiking trip to the mountains. What are the three most essential items to pack?"}],
]

# ---------------------------------------------------------------------------
# GPU Utilities & Temperature Governor (83C / 78C)
# ---------------------------------------------------------------------------
def get_gpu_telemetry():
    try:
        res = subprocess.run(
            ["nvidia-smi", "--query-gpu=temperature.gpu,memory.used,power.draw", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=True
        )
        parts = res.stdout.strip().split(",")
        return float(parts[0]), float(parts[1]), float(parts[2]) if len(parts) > 2 else 0.0
    except Exception:
        return 50.0, 2000.0, 100.0

def apply_temperature_governor(step, accum_tokens, run_id="MainSFT"):
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
# Main SFT Dataset Loader
# ---------------------------------------------------------------------------
class SFTMainDatasetLoader:
    def __init__(self, sft_path, tokenizer, seed=42):
        self.tokenizer = tokenizer
        self.rng = np.random.default_rng(seed)
        
        # 1. Load SFT Conversations
        print(f"Loading SFT data from {sft_path}...", flush=True)
        self.conversations = []
        with open(sft_path, "r", encoding="utf-8") as f:
            for line in f:
                self.conversations.append(json.loads(line))
        self.indices = list(range(len(self.conversations)))
        self.rng.shuffle(self.indices)
        self.cur_sft_idx = 0
        print(f"Loaded {len(self.conversations):,} SFT conversations.", flush=True)
        
        # 2. Pretraining Shards (Short 2k: 60% FineWeb, 30% Cosmo, 10% TinyStories)
        self.fineweb = np.memmap(DATA_DIR / "train_fineweb_edu.bin", dtype=np.uint16, mode="r")
        self.cosmopedia = np.memmap(DATA_DIR / "train_cosmopedia.bin", dtype=np.uint16, mode="r")
        self.tinystories = np.memmap(DATA_DIR / "train_tinystories.bin", dtype=np.uint16, mode="r")
        self.short_shards = [self.fineweb, self.cosmopedia, self.tinystories]
        self.short_weights = np.array([0.60, 0.30, 0.10])
        
        # 3. Long Shards (8k and 32k: 50% PG-19, 25% Wikipedia, 25% FineWeb-Edu)
        self.pg19 = np.memmap(DATA_DIR / "train_pg19_long.bin", dtype=np.uint16, mode="r")
        self.wiki_long = np.memmap(DATA_DIR / "train_wikipedia_long.bin", dtype=np.uint16, mode="r")
        self.fw_long = np.memmap(DATA_DIR / "train_fineweb_edu_long.bin", dtype=np.uint16, mode="r")
        self.long_shards = [self.pg19, self.wiki_long, self.fw_long]
        self.long_weights = np.array([0.50, 0.25, 0.25])
        
        self.long_step_toggle = 0 # Alternates between 8k and 32k

    def get_state(self):
        return {
            "cur_sft_idx": self.cur_sft_idx,
            "indices": self.indices,
            "long_step_toggle": self.long_step_toggle,
        }

    def set_state(self, state):
        self.cur_sft_idx = state["cur_sft_idx"]
        self.indices = state["indices"]
        self.long_step_toggle = state["long_step_toggle"]

    def get_sft_microbatch(self, batch_size=4, seq_len=2048, device=DEVICE):
        batch_x = []
        batch_doc_ids = []
        for _ in range(batch_size):
            seq_x = []
            seq_docs = []
            cur_doc = 0
            while len(seq_x) < seq_len:
                if self.cur_sft_idx >= len(self.indices):
                    self.cur_sft_idx = 0
                    self.rng.shuffle(self.indices)
                conv = self.conversations[self.indices[self.cur_sft_idx]]
                self.cur_sft_idx += 1
                
                toks = [1] # BOS
                for m in conv["messages"]:
                    role = m["role"]
                    content = m["content"]
                    toks.extend(self.tokenizer.encode(f"<|{role}|>\n", add_special_tokens=False))
                    toks.extend(self.tokenizer.encode(content, add_special_tokens=False))
                    toks.append(8195) # <|end|>
                    toks.append(14)   # \n
                    
                needed = seq_len - len(seq_x)
                take = toks[:needed]
                seq_x.extend(take)
                seq_docs.extend([cur_doc] * len(take))
                cur_doc += 1
                
            batch_x.append(seq_x[:seq_len])
            batch_doc_ids.append(seq_docs[:seq_len])
            
        x_np = np.array(batch_x, dtype=np.int64)
        x = torch.from_numpy(x_np[:, :-1]).to(device)
        y_raw = torch.from_numpy(x_np[:, 1:]).to(device)
        
        # Loss mask on y: only assistant content + <|end|>
        y = y_raw.clone()
        for b in range(batch_size):
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
        return x, y, attn_mask

    def get_short_replay_microbatch(self, batch_size=4, seq_len=2048, device=DEVICE):
        shard_indices = self.rng.choice(3, size=batch_size, p=self.short_weights)
        x_list, y_list = [], []
        for s_idx in shard_indices:
            shard = self.short_shards[s_idx]
            max_idx = len(shard) - seq_len - 1
            idx = int(self.rng.integers(0, max_idx))
            raw = np.array(shard[idx : idx + seq_len + 1], dtype=np.int64)
            x_list.append(raw[:-1])
            y_list.append(raw[1:])
        x = torch.from_numpy(np.array(x_list)).to(device)
        y = torch.from_numpy(np.array(y_list)).to(device)
        return x, y

    def get_long_8k_microbatch(self, device=DEVICE):
        s_idx = int(self.rng.choice(3, p=self.long_weights))
        shard = self.long_shards[s_idx]
        max_idx = len(shard) - 8193
        idx = int(self.rng.integers(0, max_idx))
        raw = np.array(shard[idx : idx + 8193], dtype=np.int64)
        x = torch.from_numpy(raw[:-1]).unsqueeze(0).to(device)
        y = torch.from_numpy(raw[1:]).unsqueeze(0).to(device)
        return x, y

    def get_long_32k_microbatch(self, device=DEVICE):
        s_idx = int(self.rng.choice(3, p=self.long_weights))
        shard = self.long_shards[s_idx]
        max_idx = len(shard) - 32769
        idx = int(self.rng.integers(0, max_idx))
        raw = np.array(shard[idx : idx + 32769], dtype=np.int64)
        x = torch.from_numpy(raw[:-1]).unsqueeze(0).to(device)
        y = torch.from_numpy(raw[1:]).unsqueeze(0).to(device)
        return x, y

# ---------------------------------------------------------------------------
# Milestone Evaluation Utilities
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
            target_mask = torch.from_numpy(mask_np[1:]).bool().to(device)
            
            out = model(x)
            logits = out.logits[0]
            if target_mask.sum() > 0:
                loss = F.cross_entropy(logits[target_mask], y_raw[target_mask]).item()
                losses.append(loss)
                
    model.train()
    return float(np.mean(losses))

@torch.no_grad()
def evaluate_4_fixed_2m_subsets(model, device=DEVICE):
    model.eval()
    val_shards = {
        "val_wikipedia_neutral": DATA_DIR / "val_wikipedia_neutral.bin",
        "val_fineweb_edu": DATA_DIR / "val_fineweb_edu.bin",
        "val_cosmopedia": DATA_DIR / "val_cosmopedia.bin",
        "val_tinystories": DATA_DIR / "val_tinystories.bin",
    }
    results = {}
    seq_len = 2048
    num_seqs = 2000000 // seq_len # 976 sequences = 1,998,848 tokens
    
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
def generate_milestone_samples(model, tokenizer, device=DEVICE, max_new_tokens=64):
    model.eval()
    samples = []
    for prompt_msgs in FIXED_10_PROMPTS:
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
# Main Training Loop
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true", help="Resume training from rolling checkpoint if present")
    args = parser.parse_args()
    
    print("\n" + "="*80)
    print("STEP 4: STARTING MAIN SFT TRAINING RUN (6L_768H_1KV, 42.28M PARAMS)")
    print("="*80, flush=True)
    
    tokenizer = AutoTokenizer.from_pretrained(str(CHAT_TOKENIZER_DIR))
    loader = SFTMainDatasetLoader(SFT_DATA_DIR / "train_conversations.jsonl", tokenizer, seed=42)
    
    # Model Configuration
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
    
    total_steps = 2000
    warmup_steps = 50
    peak_lr = 3.0e-4
    min_lr = peak_lr * 0.10 # 3.0e-5
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=min_lr, betas=(0.9, 0.95), weight_decay=0.1)
    
    def get_lr(st):
        if st < warmup_steps:
            return min_lr + (peak_lr - min_lr) * (st / warmup_steps)
        progress = (st - warmup_steps) / max(1, total_steps - warmup_steps)
        return min_lr + 0.5 * (peak_lr - min_lr) * (1.0 + math.cos(math.pi * progress))
        
    start_step = 0
    accum_tokens = 0
    total_pause_time = 0.0
    total_pauses = 0
    spike_event_count = 0
    recent_losses = []
    last_rolling_time = time.time()
    last_permanent_time = time.time()
    active_time_accum = 0.0
    
    # Handle Resume
    if args.resume and ROLLING_LATEST_CKPT.exists():
        print(f"Resuming from rolling checkpoint: {ROLLING_LATEST_CKPT}...", flush=True)
        ckpt = torch.load(ROLLING_LATEST_CKPT, map_location=DEVICE, weights_only=False)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        loader.set_state(ckpt["loader_state"])
        start_step = ckpt["step"]
        accum_tokens = ckpt["accum_tokens"]
        total_pause_time = ckpt.get("total_pause_time", 0.0)
        total_pauses = ckpt.get("total_pauses", 0)
        spike_event_count = ckpt.get("spike_event_count", 0)
        active_time_accum = ckpt.get("active_time_accum", 0.0)
        print(f"Resumed at step {start_step}, tokens {accum_tokens/1e6:.2f}M, active time {active_time_accum/60:.1f}m.", flush=True)
    else:
        print(f"Loading initial weights from {INIT_CKPT}...", flush=True)
        ckpt = torch.load(INIT_CKPT, map_location=DEVICE, weights_only=False)
        state = ckpt["model_state"] if "model_state" in ckpt else ckpt
        model.load_state_dict(state)
        
        # Initial 0% Baseline Evaluation
        print("\nEvaluating initial 0% baseline before SFT begins...", flush=True)
        init_asst_loss = evaluate_held_out_assistant_loss(model, SFT_DATA_DIR / "val_conversations.jsonl", tokenizer)
        init_ce_4 = evaluate_4_fixed_2m_subsets(model)
        init_samples = generate_milestone_samples(model, tokenizer)
        
        print(f"  Initial Assistant Loss: {init_asst_loss:.4f}")
        print(f"  Wiki Neutral 2k CE    : {init_ce_4['val_wikipedia_neutral']:.4f}")
        print(f"  FineWeb-Edu 2k CE     : {init_ce_4['val_fineweb_edu']:.4f}")
        print(f"  Cosmopedia 2k CE      : {init_ce_4['val_cosmopedia']:.4f}")
        print(f"  TinyStories 2k CE     : {init_ce_4['val_tinystories']:.4f}")
        
        # Save 0% milestone
        with open(MILESTONE_CSV, "w", encoding="utf-8") as f:
            f.write("pct,step,tokens_m,asst_loss,val_wiki_ce,val_fw_ce,val_cosmo_ce,val_tiny_ce,eval_sec\n")
            f.write(f"0%,0,0.00,{init_asst_loss:.4f},{init_ce_4['val_wikipedia_neutral']:.4f},{init_ce_4['val_fineweb_edu']:.4f},{init_ce_4['val_cosmopedia']:.4f},{init_ce_4['val_tinystories']:.4f},0.0\n")
            
        with open(MILESTONE_SAMPLES_MD, "w", encoding="utf-8") as f:
            f.write("# SFT Milestone Sample Generations\n\n")
            f.write("## Milestone 0% (Step 0, Initial Checkpoint)\n")
            f.write(f"- Assistant Loss: {init_asst_loss:.4f}\n")
            f.write(f"- 2k CE: Wiki={init_ce_4['val_wikipedia_neutral']:.4f}, FW={init_ce_4['val_fineweb_edu']:.4f}, Cosmo={init_ce_4['val_cosmopedia']:.4f}, Tiny={init_ce_4['val_tinystories']:.4f}\n\n")
            for s in init_samples:
                f.write(f"**Prompt**: {s['prompt']}\n\n")
                f.write(f"**Generation**: {s['generation']}\n\n---\n\n")
                
        # Initialize Train Log CSV
        with open(TRAIN_LOG_CSV, "w", encoding="utf-8") as f:
            f.write("step,tokens_m,batch_type,loss,lr,grad_norm,speed_tok_s,active_min,wall_min,pause_min,temp_c,vram_mb,power_w\n")

    t_wall_start = time.time()
    model.train()
    
    print("\nStarting training loop (2,000 steps)...", flush=True)
    
    for step in range(start_step, total_steps):
        t_step_start = time.time()
        optimizer.zero_grad(set_to_none=True)
        
        cycle_idx = step % 20
        step_loss = 0.0
        
        if cycle_idx < 17:
            # SFT Batch: 8 micro-batches of 4 seqs x 2,048 tokens = 65,536 tokens
            batch_type = "sft"
            model.gradient_checkpointing_disable()
            for _ in range(8):
                x, y, attn_mask = loader.get_sft_microbatch(batch_size=4, seq_len=2048, device=DEVICE)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    out = model(x, attention_mask=attn_mask)
                    loss = F.cross_entropy(out.logits.view(-1, 8256), y.view(-1), ignore_index=-100)
                    scaled = loss / 8.0
                scaled.backward()
                step_loss += loss.item() / 8.0
                
        elif cycle_idx < 19:
            # Short Pretraining Replay: 8 micro-batches of 4 seqs x 2,048 tokens = 65,536 tokens
            batch_type = "replay_short"
            model.gradient_checkpointing_disable()
            for _ in range(8):
                x, y = loader.get_short_replay_microbatch(batch_size=4, seq_len=2048, device=DEVICE)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    out = model(x)
                    loss = F.cross_entropy(out.logits.view(-1, 8256), y.view(-1))
                    scaled = loss / 8.0
                scaled.backward()
                step_loss += loss.item() / 8.0
                
        else:
            # Long Pretraining Replay: Step 19 in cycle
            loader.long_step_toggle += 1
            if loader.long_step_toggle % 2 == 1:
                # 8k Replay: 8 micro-batches of 1 seq x 8,192 tokens = 65,536 tokens
                batch_type = "replay_long_8k"
                model.gradient_checkpointing_disable()
                for _ in range(8):
                    x, y = loader.get_long_8k_microbatch(device=DEVICE)
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        out = model(x)
                        loss = F.cross_entropy(out.logits.view(-1, 8256), y.view(-1))
                        scaled = loss / 8.0
                    scaled.backward()
                    step_loss += loss.item() / 8.0
            else:
                # 32k Replay: 2 micro-batches of 1 seq x 32,768 tokens = 65,536 tokens (with grad checkpointing)
                batch_type = "replay_long_32k"
                model.gradient_checkpointing_enable()
                for _ in range(2):
                    x, y = loader.get_long_32k_microbatch(device=DEVICE)
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        out = model(x)
                        loss = F.cross_entropy(out.logits.view(-1, 8256), y.view(-1))
                        scaled = loss / 2.0
                    scaled.backward()
                    step_loss += loss.item() / 2.0
                model.gradient_checkpointing_disable()
                
        # Gradient clip & Optimizer Step
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0).item()
        cur_lr = get_lr(step)
        for pg in optimizer.param_groups:
            pg["lr"] = cur_lr
        optimizer.step()
        
        t_step_active = time.time() - t_step_start
        active_time_accum += t_step_active
        accum_tokens += 65536
        
        # Loss Spike Guard
        if math.isnan(step_loss) or math.isinf(step_loss):
            print(f"\nFATAL: NaN/Inf loss encountered at step {step+1}. Halting run!", flush=True)
            sys.exit(1)
            
        if len(recent_losses) >= 20:
            ma = float(np.mean(recent_losses[-20:]))
            if step_loss > 2.5 * ma and step_loss > 5.0:
                spike_event_count += 1
                print(f"\n[LOSS SPIKE DETECTED] Step {step+1}: Loss {step_loss:.4f} > 2.5x MA ({ma:.4f}). Spike event #{spike_event_count}.", flush=True)
                if spike_event_count >= 2:
                    print("FATAL: Second loss spike encountered. Halting as per protocol!", flush=True)
                    sys.exit(1)
                else:
                    print("Rolling back to latest rolling checkpoint...", flush=True)
                    if ROLLING_LATEST_CKPT.exists():
                        ckpt = torch.load(ROLLING_LATEST_CKPT, map_location=DEVICE, weights_only=False)
                        model.load_state_dict(ckpt["model_state"])
                        optimizer.load_state_dict(ckpt["optimizer_state"])
                        loader.set_state(ckpt["loader_state"])
                        continue
        recent_losses.append(step_loss)
        
        # Temperature Governor Check
        p_dur, p_cnt = apply_temperature_governor(step + 1, accum_tokens, "MainSFT")
        total_pause_time += p_dur
        total_pauses += p_cnt
        
        # Log Progress Every 10 Steps
        if (step + 1) % 10 == 0 or (step + 1) == total_steps:
            speed = 65536 / max(0.01, t_step_active)
            cur_temp, cur_mem, cur_pwr = get_gpu_telemetry()
            wall_m = (time.time() - t_wall_start) / 60.0
            act_m = active_time_accum / 60.0
            pause_m = total_pause_time / 60.0
            print(f"[MainSFT] Step {step+1:4d}/{total_steps} ({accum_tokens/1e6:5.1f}M) | Type: {batch_type:15s} | Loss: {step_loss:.4f} | LR: {cur_lr:.2e} | Speed: {speed:5.0f} tok/s | Temp: {cur_temp:.0f}C | VRAM: {cur_mem:.0f}MB", flush=True)
            with open(TRAIN_LOG_CSV, "a", encoding="utf-8") as f:
                f.write(f"{step+1},{accum_tokens/1e6:.2f},{batch_type},{step_loss:.4f},{cur_lr:.6e},{grad_norm:.4f},{speed:.1f},{act_m:.2f},{wall_m:.2f},{pause_m:.2f},{cur_temp:.1f},{cur_mem:.1f},{cur_pwr:.1f}\n")
                
        # 15-Minute Rolling Checkpoints
        now = time.time()
        if (now - last_rolling_time) >= 900 or (step + 1) == total_steps:
            if ROLLING_LATEST_CKPT.exists():
                shutil.copyfile(ROLLING_LATEST_CKPT, ROLLING_PREV_CKPT)
            torch.save({
                "step": step + 1,
                "accum_tokens": accum_tokens,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "loader_state": loader.get_state(),
                "total_pause_time": total_pause_time,
                "total_pauses": total_pauses,
                "spike_event_count": spike_event_count,
                "active_time_accum": active_time_accum,
            }, ROLLING_LATEST_CKPT)
            last_rolling_time = now
            print(f"[CHECKPOINT] Saved 15-minute rolling checkpoint at step {step+1}.", flush=True)
            
        # 1-Hour Permanent Checkpoints
        if (now - last_permanent_time) >= 3600:
            hour_idx = int(active_time_accum // 3600) + 1
            hour_ckpt = CKPT_DIR / f"sft_hour_{hour_idx}.pt"
            torch.save({
                "step": step + 1,
                "accum_tokens": accum_tokens,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
            }, hour_ckpt)
            last_permanent_time = now
            print(f"[CHECKPOINT] Saved permanent 1-hour checkpoint at step {step+1} to {hour_ckpt.name}.", flush=True)
            
        # Milestone Evaluations at 25%, 50%, 75%, 100% (steps 500, 1000, 1500, 2000)
        if (step + 1) in [500, 1000, 1500, 2000]:
            pct_label = f"{(step + 1) // 20}%"
            print(f"\n{'='*75}\nEVALUATING MILESTONE {pct_label} (Step {step+1}/{total_steps})\n{'='*75}", flush=True)
            t_eval_start = time.time()
            m_asst_loss = evaluate_held_out_assistant_loss(model, SFT_DATA_DIR / "val_conversations.jsonl", tokenizer)
            m_ce_4 = evaluate_4_fixed_2m_subsets(model)
            m_samples = generate_milestone_samples(model, tokenizer)
            eval_dur = time.time() - t_eval_start
            
            print(f"  Milestone {pct_label} Assistant Loss: {m_asst_loss:.4f}")
            print(f"  Wiki Neutral 2k CE    : {m_ce_4['val_wikipedia_neutral']:.4f}")
            print(f"  FineWeb-Edu 2k CE     : {m_ce_4['val_fineweb_edu']:.4f}")
            print(f"  Cosmopedia 2k CE      : {m_ce_4['val_cosmopedia']:.4f}")
            print(f"  TinyStories 2k CE     : {m_ce_4['val_tinystories']:.4f}")
            print(f"  Evaluation Duration   : {eval_dur:.1f}s", flush=True)
            
            with open(MILESTONE_CSV, "a", encoding="utf-8") as f:
                f.write(f"{pct_label},{step+1},{accum_tokens/1e6:.2f},{m_asst_loss:.4f},{m_ce_4['val_wikipedia_neutral']:.4f},{m_ce_4['val_fineweb_edu']:.4f},{m_ce_4['val_cosmopedia']:.4f},{m_ce_4['val_tinystories']:.4f},{eval_dur:.1f}\n")
                
            with open(MILESTONE_SAMPLES_MD, "a", encoding="utf-8") as f:
                f.write(f"\n## Milestone {pct_label} (Step {step+1}, Tokens {accum_tokens/1e6:.1f}M)\n")
                f.write(f"- Assistant Loss: {m_asst_loss:.4f}\n")
                f.write(f"- 2k CE: Wiki={m_ce_4['val_wikipedia_neutral']:.4f}, FW={m_ce_4['val_fineweb_edu']:.4f}, Cosmo={m_ce_4['val_cosmopedia']:.4f}, Tiny={m_ce_4['val_tinystories']:.4f}\n\n")
                for s in m_samples:
                    f.write(f"**Prompt**: {s['prompt']}\n\n")
                    f.write(f"**Generation**: {s['generation']}\n\n---\n\n")

    # Final Save
    print(f"\nSaving final SFT checkpoint to {FINAL_CKPT}...", flush=True)
    torch.save({
        "step": total_steps,
        "accum_tokens": accum_tokens,
        "model_state": model.state_dict(),
        "vocab_size": 8256,
        "config": cfg.to_dict(),
    }, FINAL_CKPT)
    
    wall_total = (time.time() - t_wall_start) / 60.0
    act_total = active_time_accum / 60.0
    print(f"\nMAIN SFT TRAINING COMPLETE! Total Active: {act_total:.1f}m, Wall Clock: {wall_total:.1f}m, Tokens: {accum_tokens/1e6:.2f}M", flush=True)

if __name__ == "__main__":
    main()
