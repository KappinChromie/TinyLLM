import os
import sys
import random
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, LlamaConfig, LlamaForCausalLM
import sentencepiece as spm

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
sys.stdout.reconfigure(line_buffering=True)

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "shards"
RESULTS_DIR = ROOT / "results"
MODELS_DIR = ROOT / "models"
CHAT_TOKENIZER_DIR = MODELS_DIR / "tokenizer_chat"
CHAT_CKPT = MODELS_DIR / "checkpoints" / "chat_sft_final.pt"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def create_chat_model(device):
    config = LlamaConfig(
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
    model = LlamaForCausalLM(config).to(device=device, dtype=torch.bfloat16)
    ckpt = torch.load(CHAT_CKPT, map_location=device, weights_only=False)
    state = ckpt["model_state"] if "model_state" in ckpt else ckpt
    model.load_state_dict(state)
    model.eval()
    return model

def generate_repetitive_filler(tokenizer, num_tokens):
    sent = "The grass is green. The sky is blue. "
    toks = tokenizer.encode(sent, add_special_tokens=False)
    reps = (num_tokens // len(toks)) + 2
    return (toks * reps)[:num_tokens]

def compute_candidate_logprob(model, context_ids, candidate_ids, device):
    full_ids = torch.tensor([context_ids + candidate_ids], dtype=torch.long, device=device)
    with torch.no_grad():
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            out = model(full_ids)
            logits = out.logits[0]
    
    start_pos = len(context_ids) - 1
    end_pos = start_pos + len(candidate_ids)
    pred_logits = logits[start_pos:end_pos]
    target_tokens = torch.tensor(candidate_ids, dtype=torch.long, device=device)
    
    log_probs = F.log_softmax(pred_logits, dim=-1)
    token_log_probs = log_probs[torch.arange(len(candidate_ids)), target_tokens]
    return token_log_probs.sum().item()

def run_passkey_eval():
    print("="*75)
    print("RUNNING PASSKEY EVALUATION ON FINAL CHAT MODEL")
    print("="*75, flush=True)
    
    tokenizer = AutoTokenizer.from_pretrained(str(CHAT_TOKENIZER_DIR))
    model = create_chat_model(DEVICE)
    val_pg19 = np.memmap(DATA_DIR / "val_pg19_long.bin", dtype=np.uint16, mode="r")
    
    lengths = [2048, 4096, 8192]
    filler_types = ["prose", "repetitive"]
    trials_per_setting = 20
    
    results = []
    
    for flen in lengths:
        for ftype in filler_types:
            rng = np.random.RandomState(42) # Exact same seed as Stage B/C evaluation
            cand_21_correct = 0
            vocab_ranks = []
            greedy_correct = 0
            
            for trial in range(trials_per_setting):
                passkey = f"{rng.randint(10000, 99999)}"
                passkey_tokens = tokenizer.encode(passkey, add_special_tokens=False)
                
                prefix = f"The secret passkey is {passkey}. Remember this passkey. "
                prefix_tokens = tokenizer.encode(prefix, add_special_tokens=False)
                
                suffix = "What is the secret passkey? The secret passkey is "
                suffix_tokens = tokenizer.encode(suffix, add_special_tokens=False)
                
                overhead = len(prefix_tokens) + len(suffix_tokens)
                needed_filler = flen - overhead
                if needed_filler < 0:
                    needed_filler = 0
                    
                if ftype == "prose":
                    max_start = len(val_pg19) - needed_filler - 1
                    f_start = rng.randint(0, max_start)
                    filler_tokens = val_pg19[f_start : f_start + needed_filler].tolist()
                else:
                    filler_tokens = generate_repetitive_filler(tokenizer, needed_filler)
                    
                context_tokens = prefix_tokens + filler_tokens + suffix_tokens
                
                # 20 distractor passkeys
                distractors = [f"{rng.randint(10000, 99999)}" for _ in range(20)]
                candidates = [passkey] + distractors
                
                cand_scores = []
                for cand in candidates:
                    c_toks = tokenizer.encode(cand, add_special_tokens=False)
                    score = compute_candidate_logprob(model, context_tokens, c_toks, DEVICE)
                    cand_scores.append(score)
                    
                best_cand_idx = int(np.argmax(cand_scores))
                if best_cand_idx == 0:
                    cand_21_correct += 1
                    
                # Full vocab rank on first token of passkey
                full_ids = torch.tensor([context_tokens], dtype=torch.long, device=DEVICE)
                with torch.no_grad():
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        out = model(full_ids)
                        next_token_logits = out.logits[0, -1]
                
                sorted_tokens = torch.argsort(next_token_logits, descending=True).cpu().tolist()
                target_first_tok = passkey_tokens[0]
                rank = sorted_tokens.index(target_first_tok) + 1
                vocab_ranks.append(rank)
                
                # Greedy check (decode up to len(passkey_tokens))
                with torch.no_grad():
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        gen = model.generate(full_ids, max_new_tokens=len(passkey_tokens), do_sample=False, pad_token_id=0)
                gen_toks = gen[0, len(context_tokens):].cpu().tolist()
                if gen_toks == passkey_tokens:
                    greedy_correct += 1
                    
            cand_acc = cand_21_correct / trials_per_setting * 100.0
            vocab_acc = sum(1 for r in vocab_ranks if r == 1) / trials_per_setting * 100.0
            greedy_acc = greedy_correct / trials_per_setting * 100.0
            mean_rank = float(np.mean(vocab_ranks))
            median_rank = float(np.median(vocab_ranks))
            
            results.append({
                "model": "Chat Model (SFT Final)",
                "context_depth": flen,
                "filler_type": ftype,
                "trials": trials_per_setting,
                "cand_21_rank1_acc": cand_acc,
                "vocab_8192_rank1_acc": vocab_acc,
                "greedy_acc": greedy_acc,
                "mean_vocab_rank": mean_rank,
                "median_vocab_rank": median_rank
            })
            print(f"Chat Model | Depth: {flen:4d} | Filler: {ftype:10s} | Cand-21 Acc: {cand_acc:5.1f}% | Vocab Rank-1: {vocab_acc:4.1f}% | Greedy: {greedy_acc:4.1f}% | Mean Rank: {mean_rank:6.1f} | Med Rank: {median_rank:6.1f}")

    df = pd.DataFrame(results)
    out_path = RESULTS_DIR / "chat_passkey_results.csv"
    df.to_csv(out_path, index=False)
    print(f"\nSaved chat passkey results to {out_path}")

if __name__ == "__main__":
    run_passkey_eval()
