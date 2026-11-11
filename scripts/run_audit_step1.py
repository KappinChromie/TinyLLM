import os
import sys
import json
import re
from pathlib import Path
from collections import defaultdict, Counter
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, LlamaConfig, LlamaForCausalLM

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results"
PROMPTS_FILE = ROOT / "data" / "chat_eval_prompts.json"
SAMPLES_FILE = RESULTS_DIR / "chat_eval_samples.md"
TRAIN_CONVS_FILE = ROOT / "data" / "sft" / "train_conversations.jsonl"
CHAT_TOKENIZER_DIR = ROOT / "models" / "tokenizer_chat"
CHAT_CKPT = ROOT / "models" / "checkpoints" / "chat_sft_final.pt"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def repetition_ratio(tokens, n):
    if len(tokens) < n * 2:
        return 0.0
    ngrams = [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]
    counts = Counter(ngrams)
    if not counts:
        return 0.0
    most_common_count = counts.most_common(1)[0][1]
    return (most_common_count * n) / len(tokens)

def main():
    print("="*75)
    print("AUDITING STEP 1: GREEDY <|end|> EMISSION AND DIAGNOSIS")
    print("="*75)
    
    tokenizer = AutoTokenizer.from_pretrained(str(CHAT_TOKENIZER_DIR))
    
    with open(PROMPTS_FILE, "r", encoding="utf-8") as f:
        prompts = json.load(f)
    print(f"Loaded {len(prompts)} prompts from {PROMPTS_FILE}.")

    with open(SAMPLES_FILE, "r", encoding="utf-8") as f:
        samples_text = f.read()

    # Parse sections from chat_eval_samples.md
    prompt_sections = samples_text.split("## Prompt ")
    parsed_samples = {}
    for sec in prompt_sections[1:]:
        header_line = sec.split("\n")[0] # e.g. greet_01 (greetings)
        p_id = header_line.split()[0]
        
        # greedy output
        greedy_match = re.search(r"### Chat Model \(Greedy\):\n(.*?)\n\n### Base Model", sec, re.DOTALL)
        greedy_txt = greedy_match.group(1) if greedy_match else ""
        
        # samples
        s1 = re.search(r"- \*\*Sample 1\*\*:\s*(.*?)(?=\n- \*\*Sample 2\*\*|\n\n---|\Z)", sec, re.DOTALL)
        s2 = re.search(r"- \*\*Sample 2\*\*:\s*(.*?)(?=\n- \*\*Sample 3\*\*|\n\n---|\Z)", sec, re.DOTALL)
        s3 = re.search(r"- \*\*Sample 3\*\*:\s*(.*?)(?=\n\n---|\Z)", sec, re.DOTALL)
        
        sample_texts = []
        for s in [s1, s2, s3]:
            if s:
                sample_texts.append(s.group(1).strip())
                
        parsed_samples[p_id] = {
            "greedy": greedy_txt,
            "samples": sample_texts
        }

    # Evaluate emission and repetition
    categories = defaultdict(list)
    cat_greedy_end = defaultdict(int)
    cat_sample_end = defaultdict(int)
    cat_sample_total = defaultdict(int)
    
    rep3_count = 0
    rep4_count = 0
    failing_prompts = []
    
    for p in prompts:
        p_id = p["id"]
        cat = p["category"]
        categories[cat].append(p)
        
        res = parsed_samples.get(p_id)
        if not res:
            continue
            
        g_txt = res["greedy"]
        g_toks = tokenizer.encode(g_txt, add_special_tokens=False)
        has_end_greedy = (8195 in g_toks) or ("<|end|>" in g_txt)
        if has_end_greedy:
            cat_greedy_end[cat] += 1
        else:
            failing_prompts.append({
                "id": p_id,
                "category": cat,
                "prompt": p,
                "greedy_txt": g_txt,
                "greedy_toks": g_toks
            })
            
        r3 = repetition_ratio(g_toks, 3)
        r4 = repetition_ratio(g_toks, 4)
        if r3 > 0.30:
            rep3_count += 1
        if r4 > 0.30:
            rep4_count += 1
            
        for s_txt in res["samples"]:
            s_toks = tokenizer.encode(s_txt, add_special_tokens=False)
            cat_sample_total[cat] += 1
            if (8195 in s_toks) or ("<|end|>" in s_txt):
                cat_sample_end[cat] += 1

    total_prompts = len(prompts)
    total_greedy_end = sum(cat_greedy_end.values())
    total_samples = sum(cat_sample_total.values())
    total_sample_end = sum(cat_sample_end.values())

    print(f"\nOverall Greedy <|end|> Emission: {total_greedy_end}/{total_prompts} ({total_greedy_end/total_prompts*100:.2f}%)")
    print(f"Overall Sampling <|end|> Emission: {total_sample_end}/{total_samples} ({total_sample_end/total_samples*100:.2f}%)")
    print(f"3-gram repetition (>30% output): {rep3_count}/{total_prompts} ({rep3_count/total_prompts*100:.2f}%)")
    print(f"4-gram repetition (>30% output): {rep4_count}/{total_prompts} ({rep4_count/total_prompts*100:.2f}%)")

    print("\n--- EMISSION BY CATEGORY ---")
    print(f"{'Category':22s} | {'Count':5s} | {'Greedy End':15s} | {'Sampling End':15s}")
    print("-" * 65)
    for cat in sorted(categories.keys()):
        tot = len(categories[cat])
        g_cnt = cat_greedy_end[cat]
        s_cnt = cat_sample_end[cat]
        s_tot = cat_sample_total[cat]
        g_pct = g_cnt / tot * 100
        s_pct = s_cnt / s_tot * 100 if s_tot > 0 else 0
        print(f"{cat:22s} | {tot:5d} | {g_cnt:2d}/{tot:2d} ({g_pct:5.1f}%) | {s_cnt:2d}/{s_tot:2d} ({s_pct:5.1f}%)")

    print(f"\nTotal failing greedy prompts: {len(failing_prompts)} / {total_prompts}")
    
    # Analyze the 15 failing generations
    print("\n" + "="*75)
    print("ANALYSIS OF 10 FAILING GENERATIONS (FULL TEXT)")
    print("="*75)
    for idx, item in enumerate(failing_prompts[:10]):
        print(f"\n--- [FAIL {idx+1}/{len(failing_prompts)}] Prompt ID: {item['id']} | Category: {item['category']} ---")
        user_msg = [m['content'] for m in item['prompt']['messages'] if m['role'] == 'user']
        print(f"USER INPUT:\n{user_msg[0] if user_msg else ''}")
        print(f"\nMODEL GREEDY GENERATION (length: {len(item['greedy_toks'])} tokens):\n{item['greedy_txt']}")
        
        # Check failure mode
        txt = item['greedy_txt']
        has_stray_user = ("<|user|>" in txt) or ("User:" in txt) or ("Human:" in txt)
        has_rep = repetition_ratio(item['greedy_toks'], 4) > 0.30
        print(f"\nFailure diagnosis: Loop/Repetition={has_rep}, Stray user/role={has_stray_user}, Hit 256 tokens={len(item['greedy_toks']) >= 256}")

    # Check SFT training dataset assistant turn lengths
    print("\n" + "="*75)
    print("SFT TRAINING SET ASSISTANT TURN LENGTH DISTRIBUTION")
    print("="*75)
    asst_lengths = []
    total_convs = 0
    with open(TRAIN_CONVS_FILE, "r", encoding="utf-8") as f:
        for line in f:
            total_convs += 1
            conv = json.loads(line)
            for m in conv["messages"]:
                if m["role"] == "assistant":
                    toks = tokenizer.encode(m["content"], add_special_tokens=False)
                    asst_lengths.append(len(toks))
                    
    arr_len = np.array(asst_lengths)
    print(f"Total assistant turns in training set: {len(arr_len):,}")
    print(f"Mean length: {np.mean(arr_len):.1f} tokens | Median: {np.median(arr_len):.1f} tokens")
    print(f"Min: {np.min(arr_len)} | Max: {np.max(arr_len)}")
    print(f"P25: {np.percentile(arr_len, 25):.1f} | P50: {np.percentile(arr_len, 50):.1f} | P75: {np.percentile(arr_len, 75):.1f} | P90: {np.percentile(arr_len, 90):.1f} | P95: {np.percentile(arr_len, 95):.1f}")
    pct_lte_256 = np.mean(arr_len <= 256) * 100.0
    pct_gt_256 = np.mean(arr_len > 256) * 100.0
    print(f"Turns <= 256 tokens: {pct_lte_256:.2f}%")
    print(f"Turns > 256 tokens : {pct_gt_256:.2f}%")

if __name__ == "__main__":
    main()
