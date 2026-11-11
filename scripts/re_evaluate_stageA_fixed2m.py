"""
Re-evaluate Stage A Final Undecayed Checkpoint and Stage A Decayed Fallback Checkpoint
on the EXACT fixed 2M-token subsets (244 batches x 4 sequences x 2048 tokens = 1,998,848 tokens)
used in HourlyEvaluator during Stage A training.
"""

import math
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import LlamaConfig, LlamaForCausalLM

ROOT = Path(__file__).resolve().parent.parent
SHARDS_DIR = ROOT / "data" / "shards"
CHECKPOINTS_DIR = ROOT / "models" / "checkpoints" / "stageA"
RESULTS_DIR = ROOT / "results"

COMPRESSION_STATS = {
    "val_cosmopedia": {"bytes_per_tok": 4.149, "chars_per_tok": 4.148},
    "val_fineweb_edu": {"bytes_per_tok": 4.156, "chars_per_tok": 4.150},
    "val_tinystories": {"bytes_per_tok": 3.220, "chars_per_tok": 3.220},
    "val_wikipedia_neutral": {"bytes_per_tok": 4.098, "chars_per_tok": 4.092},
}

class Fixed2MEvaluator:
    def __init__(self, shards_dir, device):
        self.device = device
        self.seq_len = 2048
        self.val_shards = {
            "val_wikipedia_neutral": np.memmap(shards_dir / "val_wikipedia_neutral.bin", dtype=np.uint16, mode="r"),
            "val_fineweb_edu": np.memmap(shards_dir / "val_fineweb_edu.bin", dtype=np.uint16, mode="r"),
            "val_cosmopedia": np.memmap(shards_dir / "val_cosmopedia.bin", dtype=np.uint16, mode="r"),
            "val_tinystories": np.memmap(shards_dir / "val_tinystories.bin", dtype=np.uint16, mode="r"),
        }
        # Precompute fixed offsets for exactly 244 batches of 4 sequences (~2,000,000 tokens per set)
        self.num_batches = 244
        self.batch_size = 4
        self.fixed_offsets = {}
        for name, shard in self.val_shards.items():
            rng = np.random.RandomState(4242)
            max_idx = len(shard) - self.seq_len - 1
            self.fixed_offsets[name] = [rng.randint(0, max_idx) for _ in range(self.num_batches * self.batch_size)]

    @torch.no_grad()
    def evaluate_set(self, model, name):
        shard = self.val_shards[name]
        offsets = self.fixed_offsets[name]
        total_loss = 0.0
        total_tokens = 0
        
        for b_idx in range(self.num_batches):
            batch_offsets = offsets[b_idx * self.batch_size : (b_idx + 1) * self.batch_size]
            x = np.stack([shard[i : i + self.seq_len] for i in batch_offsets])
            y = np.stack([shard[i + 1 : i + 1 + self.seq_len] for i in batch_offsets])
            tx = torch.from_numpy(x.astype(np.int64)).to(device=self.device, non_blocking=True)
            ty = torch.from_numpy(y.astype(np.int64)).to(device=self.device, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                outputs = model(tx)
                loss = F.cross_entropy(outputs.logits.reshape(-1, 8192), ty.reshape(-1)).item()
            total_loss += loss * (self.batch_size * self.seq_len)
            total_tokens += self.batch_size * self.seq_len
            
        avg_loss = total_loss / total_tokens
        bits_per_tok = avg_loss / math.log(2.0)
        bpb = bits_per_tok / COMPRESSION_STATS[name]["bytes_per_tok"]
        return avg_loss, bpb

    def evaluate_all(self, model):
        model.eval()
        results = {}
        for name in ["val_wikipedia_neutral", "val_fineweb_edu", "val_cosmopedia", "val_tinystories"]:
            ce, bpb = self.evaluate_set(model, name)
            results[name] = {"ce": ce, "bpb": bpb}
        return results

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
        rope_theta=10000.0,
    )
    return LlamaForCausalLM(config).to(device=device, dtype=torch.bfloat16)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== Re-evaluating Stage A Models on Exact Fixed 2M Subsets on {device} ===")
    
    evaluator = Fixed2MEvaluator(SHARDS_DIR, device=device)
    model = create_model(device=device)
    
    # 1. Undecayed Final Checkpoint (Step 8200)
    undecayed_ckpt = CHECKPOINTS_DIR / "stageA_final_undecayed.pt"
    print(f"\nLoading Undecayed Final Checkpoint from {undecayed_ckpt}...")
    raw_data = torch.load(undecayed_ckpt, map_location=device, weights_only=False)
    model.load_state_dict(raw_data["model_state"] if "model_state" in raw_data else raw_data)
    
    print("Evaluating Undecayed Final Checkpoint...")
    res_undecayed = evaluator.evaluate_all(model)
    for name, r in res_undecayed.items():
        print(f"  [Undecayed Final] {name:25s}: CE = {r['ce']:.4f} | BPB = {r['bpb']:.4f}")
        
    # 2. Decayed Fallback Checkpoint
    decayed_ckpt = CHECKPOINTS_DIR / "stageA_decay_branch_final.pt"
    print(f"\nLoading Decayed Fallback Checkpoint from {decayed_ckpt}...")
    raw_decay = torch.load(decayed_ckpt, map_location=device, weights_only=False)
    model.load_state_dict(raw_decay["model_state"] if "model_state" in raw_decay else raw_decay)
    
    print("Evaluating Decayed Fallback Checkpoint...")
    res_decayed = evaluator.evaluate_all(model)
    for name, r in res_decayed.items():
        delta_ce = r['ce'] - res_undecayed[name]['ce']
        print(f"  [Decayed Fallback] {name:25s}: CE = {r['ce']:.4f} (Δ {delta_ce:+.4f}) | BPB = {r['bpb']:.4f}")
        
    df_eval = pd.DataFrame([
        {
            "model": "StageA_Final_Undecayed",
            "tokens_decayed_m": 0.0,
            "eval_tokens_per_set": 1998848,
            "val_wiki_ce": round(res_undecayed["val_wikipedia_neutral"]["ce"], 4),
            "val_wiki_bpb": round(res_undecayed["val_wikipedia_neutral"]["bpb"], 4),
            "val_fineweb_ce": round(res_undecayed["val_fineweb_edu"]["ce"], 4),
            "val_fineweb_bpb": round(res_undecayed["val_fineweb_edu"]["bpb"], 4),
            "val_cosmo_ce": round(res_undecayed["val_cosmopedia"]["ce"], 4),
            "val_cosmo_bpb": round(res_undecayed["val_cosmopedia"]["bpb"], 4),
            "val_tinystories_ce": round(res_undecayed["val_tinystories"]["ce"], 4),
            "val_tinystories_bpb": round(res_undecayed["val_tinystories"]["bpb"], 4),
        },
        {
            "model": "StageA_Decayed_Fallback",
            "tokens_decayed_m": 50.07,
            "eval_tokens_per_set": 1998848,
            "val_wiki_ce": round(res_decayed["val_wikipedia_neutral"]["ce"], 4),
            "val_wiki_bpb": round(res_decayed["val_wikipedia_neutral"]["bpb"], 4),
            "val_fineweb_ce": round(res_decayed["val_fineweb_edu"]["ce"], 4),
            "val_fineweb_bpb": round(res_decayed["val_fineweb_edu"]["bpb"], 4),
            "val_cosmo_ce": round(res_decayed["val_cosmopedia"]["ce"], 4),
            "val_cosmo_bpb": round(res_decayed["val_cosmopedia"]["bpb"], 4),
            "val_tinystories_ce": round(res_decayed["val_tinystories"]["ce"], 4),
            "val_tinystories_bpb": round(res_decayed["val_tinystories"]["bpb"], 4),
        }
    ])
    out_csv = RESULTS_DIR / "stageA_decay_branch_eval.csv"
    df_eval.to_csv(out_csv, index=False)
    print(f"\nUpdated {out_csv} with exact fixed 2M eval numbers.")
    
    # Also update stageA_reference_2k.csv for 2k-regression checks in Stage B and C
    df_ref = pd.DataFrame([{
        "source": "StageA_Final_Undecayed_Reference",
        "val_wiki_ce": round(res_undecayed["val_wikipedia_neutral"]["ce"], 4),
        "val_wiki_bpb": round(res_undecayed["val_wikipedia_neutral"]["bpb"], 4),
        "val_fineweb_ce": round(res_undecayed["val_fineweb_edu"]["ce"], 4),
        "val_fineweb_bpb": round(res_undecayed["val_fineweb_edu"]["bpb"], 4),
        "val_cosmo_ce": round(res_undecayed["val_cosmopedia"]["ce"], 4),
        "val_cosmo_bpb": round(res_undecayed["val_cosmopedia"]["bpb"], 4),
        "val_tinystories_ce": round(res_undecayed["val_tinystories"]["ce"], 4),
        "val_tinystories_bpb": round(res_undecayed["val_tinystories"]["bpb"], 4),
    }])
    ref_csv = RESULTS_DIR / "stageA_reference_2k.csv"
    df_ref.to_csv(ref_csv, index=False)
    print(f"Saved reference 2k metrics to {ref_csv}")

if __name__ == "__main__":
    main()
