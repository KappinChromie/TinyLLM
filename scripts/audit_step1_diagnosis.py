import re
import json
from pathlib import Path
from collections import defaultdict
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SAMPLES_FILE = ROOT / "results" / "chat_eval_samples.md"

def parse_samples():
    with open(SAMPLES_FILE, "r", encoding="utf-8") as f:
        text = f.read()

    # Split by prompt headers
    # e.g., ## Prompt greet_01 (Category: greetings)
    pattern = re.compile(r"## Prompt (\S+) \(Category: ([^\)]+)\)")
    splits = list(pattern.finditer(text))
    
    prompts_data = []
    for i, match in enumerate(splits):
        p_id = match.group(1)
        cat = match.group(2)
        start_pos = match.start()
        end_pos = splits[i+1].start() if i+1 < len(splits) else len(text)
        chunk = text[start_pos:end_pos]
        
        # Extract user text
        user_match = re.search(r"\*\*User Input\*\*:\n([^\n]+(?:\n[^\n]+)*?)\n\n\*\*Target Reference\*\*:", chunk)
        user_text = user_match.group(1).strip() if user_match else ""
        
        # Extract Chat Model Greedy
        chat_greedy_match = re.search(r"### Chat Model \(SFT Final\) - Greedy Output \(256 tokens max\):\n```\n(.*?)\n```\n- End token emitted: (True|False)", chunk, re.DOTALL)
        greedy_gen = chat_greedy_match.group(1) if chat_greedy_match else ""
        greedy_end = chat_greedy_match.group(2) == "True" if chat_greedy_match else False
        
        # Extract Chat Model Samples
        # Look for Sample 1, 2, 3
        samples_matches = re.findall(r"\*\*Sample \d\*\*:\n```\n(.*?)\n```\n- End token emitted: (True|False)", chunk, re.DOTALL)
        sample_ends = [m[1] == "True" for m in samples_matches]
        sample_gens = [m[0] for m in samples_matches]
        
        prompts_data.append({
            "id": p_id,
            "category": cat,
            "user_text": user_text,
            "greedy_gen": greedy_gen,
            "greedy_end": greedy_end,
            "sample_ends": sample_ends,
            "sample_gens": sample_gens
        })
    return prompts_data

def repetition_ratio(text, n):
    words = text.split()
    if len(words) < n:
        return 0.0
    ngrams = [tuple(words[i:i+n]) for i in range(len(words)-n+1)]
    freq = defaultdict(int)
    for ng in ngrams:
        freq[ng] += 1
    max_freq = max(freq.values())
    return (max_freq * n) / len(words)

def main():
    data = parse_samples()
    print(f"Parsed {len(data)} prompts.")
    
    cat_counts = defaultdict(lambda: {"total": 0, "greedy_end": 0, "sample_total": 0, "sample_end": 0})
    failing_greedy = []
    
    total_greedy_end = 0
    total_samples = 0
    total_sample_end = 0
    
    # Repetition stats
    greedy_rep3_count = 0
    greedy_rep4_count = 0
    
    for item in data:
        cat = item["category"]
        cat_counts[cat]["total"] += 1
        if item["greedy_end"]:
            cat_counts[cat]["greedy_end"] += 1
            total_greedy_end += 1
        else:
            failing_greedy.append(item)
            
        r3 = repetition_ratio(item["greedy_gen"], 3)
        r4 = repetition_ratio(item["greedy_gen"], 4)
        if r3 > 0.30:
            greedy_rep3_count += 1
        if r4 > 0.30:
            greedy_rep4_count += 1
            
        for se in item["sample_ends"]:
            cat_counts[cat]["sample_total"] += 1
            total_samples += 1
            if se:
                cat_counts[cat]["sample_end"] += 1
                total_sample_end += 1

    print("\n--- EMISSION BY CATEGORY ---")
    print(f"{'Category':20s} | {'Greedy End':15s} | {'Sample End':15s}")
    print("-" * 55)
    for cat in sorted(cat_counts.keys()):
        c = cat_counts[cat]
        g_pct = c['greedy_end'] / c['total'] * 100
        s_pct = c['sample_end'] / c['sample_total'] * 100 if c['sample_total'] > 0 else 0
        print(f"{cat:20s} | {c['greedy_end']:2d}/{c['total']:2d} ({g_pct:5.1f}%) | {c['sample_end']:2d}/{c['sample_total']:2d} ({s_pct:5.1f}%)")
    
    print("-" * 55)
    print(f"{'TOTAL':20s} | {total_greedy_end:2d}/{len(data):2d} ({total_greedy_end/len(data)*100:5.1f}%) | {total_sample_end:2d}/{total_samples:2d} ({total_sample_end/total_samples*100:5.1f}%)")
    
    print("\n--- REPETITION DEFINITION COMPARISON ---")
    print(f"3-gram (>30% output): {greedy_rep3_count}/{len(data)} ({greedy_rep3_count/len(data)*100:.2f}%)")
    print(f"4-gram (>30% output): {greedy_rep4_count}/{len(data)} ({greedy_rep4_count/len(data)*100:.2f}%)")
    
    print(f"\nTotal failing greedy prompts: {len(failing_greedy)}")
    print("Listing all 15 failing prompts:\n")
    for idx, f in enumerate(failing_greedy):
        print(f"[{idx+1}] ID: {f['id']} | Category: {f['category']}")
        print(f"User: {f['user_text']}")
        print(f"Generation (first 250 chars): {f['greedy_gen'][:250]}...")
        print("-" * 50)

if __name__ == "__main__":
    main()
