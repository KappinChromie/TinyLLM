"""
Token budget calculator and curriculum planner for TinyLLM.
Calculates parameter counts, wall-clock hours, token yields, and staged allocations
including decay-branch costs and sequence counts.
"""

import sys
import pandas as pd

def count_params(vocab, d, layers, ffn, heads, kv_heads, tied=True):
    head_dim = d // heads
    attn = d * d + 2 * d * (kv_heads * head_dim) + d * d  # q, k, v, o
    mlp = 3 * d * ffn                                     # gate, up, down
    norms = 2 * d
    per_layer = attn + mlp + norms
    emb = vocab * d
    total = layers * per_layer + emb * (1 if tied else 2) + d
    return total, total - emb

def plan(tok_per_sec_measured, hours, params):
    tokens = tok_per_sec_measured * hours * 3600
    return tokens, tokens / params

def plan_staged_run(tps_2k, tps_8k, tps_32k, hours_stage_a=10.0, hours_stage_b=1.5, hours_stage_c=1.0, decay_branch_pct=0.075):
    """
    Computes exact tokens and sequence counts across stages given measured throughput.
    """
    # Main stages
    toks_a_main = tps_2k * (hours_stage_a * (1.0 - decay_branch_pct)) * 3600
    toks_a_decay_branch = tps_2k * (hours_stage_a * decay_branch_pct) * 3600
    toks_a_total = toks_a_main + toks_a_decay_branch
    
    toks_b_main = tps_8k * (hours_stage_b * (1.0 - decay_branch_pct)) * 3600
    toks_b_decay_branch = tps_8k * (hours_stage_b * decay_branch_pct) * 3600
    toks_b_total = toks_b_main + toks_b_decay_branch
    
    toks_c_total = tps_32k * hours_stage_c * 3600
    
    total_tokens = toks_a_total + toks_b_total + toks_c_total
    total_hours = hours_stage_a + hours_stage_b + hours_stage_c
    
    seqs_a = int(toks_a_total / 2048)
    seqs_b = int(toks_b_total / 8192)
    seqs_c = int(toks_c_total / 32768)
    
    return {
        "Stage A (2k, main)": {"tokens": toks_a_main, "hours": hours_stage_a * (1.0 - decay_branch_pct), "seqs": int(toks_a_main / 2048)},
        "Stage A (2k, decay branch)": {"tokens": toks_a_decay_branch, "hours": hours_stage_a * decay_branch_pct, "seqs": int(toks_a_decay_branch / 2048)},
        "Stage B (8k, main)": {"tokens": toks_b_main, "hours": hours_stage_b * (1.0 - decay_branch_pct), "seqs": int(toks_b_main / 8192)},
        "Stage B (8k, decay branch)": {"tokens": toks_b_decay_branch, "hours": hours_stage_b * decay_branch_pct, "seqs": int(toks_b_decay_branch / 8192)},
        "Stage C (32k, full)": {"tokens": toks_c_total, "hours": hours_stage_c, "seqs": seqs_c},
        "Total Run": {"tokens": total_tokens, "hours": total_hours, "seqs": seqs_a + seqs_b + seqs_c}
    }

if __name__ == "__main__":
    p_tot, p_non = count_params(8192, 768, 4, 2048, 12, 1)
    print(f"4L_768H_1KV: Total={p_tot/1e6:.2f}M, Non-emb={p_non/1e6:.2f}M")
